"""Live enforcement — gate a tool call before it executes.

Everywhere else, `screening.guard` produces a verdict for a step that already
happened, read back from a recorded trace: the shape a benchmark needs, where
the agent has already run to completion and the question is only "would we
have caught this." An agent that wants the middleware to actually stop a call
needs the verdict *before* the call runs. This is that entrypoint — the
`@guard(...)` decorator the project brief describes in §7.

A `Session` holds the running state one agent turn needs (the task
description, the tool outputs seen so far) and exposes `protect(fn)`, which
wraps a tool function so that calling it screens, checks policy, and escalates
first, and only calls the real function if the verdict says to.
"""

from __future__ import annotations

import contextlib
import contextvars
import functools
from collections.abc import Callable
from dataclasses import dataclass

from middleware.melon.cache import ToolCallCache
from middleware.melon.engine import AgentCallFn, make_escalate_fn
from middleware.melon.types import ToolCall
from middleware.screening.guard import check_calls, screen_step
from middleware.screening.screener import JudgeFn
from middleware.trace.logger import TraceLogger
from middleware.trace.schema import StepTrace


class Blocked(Exception):
    """Raised in place of calling the wrapped function. The tool call never
    ran — this is not "the call ran and then failed," the function body
    itself never executed."""

    def __init__(self, trace: StepTrace) -> None:
        super().__init__(trace.explanation)
        self.trace = trace


class NeedsConfirmation(Exception):
    """Raised when the verdict is ask_user and no confirmation callback was
    given. A caller that wants to handle this itself should pass
    on_ask_user to Session instead of catching this."""

    def __init__(self, trace: StepTrace) -> None:
        super().__init__(trace.explanation)
        self.trace = trace


@dataclass
class Session:
    """One agent turn's worth of state: the task it's working on, and every
    tool output it has seen so far. Construct one per user request, not one
    per process — task_description and the accumulated outputs are specific
    to a single run.
    """

    task_description: str
    judge_fn: JudgeFn
    # Builds the masked run's next decision from a list of messages, exactly
    # what middleware.melon.engine.run_melon_check expects. None disables
    # Stage 3 — escalations fall back to on_ask_user or NeedsConfirmation,
    # same as an unwired eval run.
    melon_agent_call_fn: AgentCallFn | None = None
    system_message: str | None = None
    # Called with (explanation, ToolCall) when Stage 3 can't resolve a step
    # on its own; return True to proceed, False to decline. Left unset, an
    # ask_user verdict raises NeedsConfirmation instead.
    on_ask_user: Callable[[str, ToolCall], bool] | None = None
    logger: TraceLogger | None = None
    # Authors whose content is treated as high-integrity — the user's own
    # domain, known colleagues. Without this every region from one tool call
    # carries the same label, so redaction can only mask a whole tool
    # response at a time and never one poisoned message inside an inbox.
    trusted_authors: frozenset[str] = frozenset()
    # Also block private data reaching an outward channel, not just
    # untrusted data reaching a sensitive action. Off by default: it is a
    # leak policy, and enforcing it against integrity-only labels turns
    # every legitimate outbound email into a violation.
    enforce_confidentiality: bool = False

    def __post_init__(self) -> None:
        self._tool_outputs: list[tuple[str, str]] = []
        self._step = 0
        # MELON's H, accumulated across the whole session: an agent that
        # completes the real task first and acts on the injection later is
        # only caught if earlier masked calls are still being compared.
        self._masked_call_cache = ToolCallCache()

    def observe(self, tool_name: str, output) -> None:
        """Record a tool's output so later calls are screened against it.
        Call this after any tool execution the wrapped functions didn't
        themselves perform (e.g. a read the agent issued directly)."""
        self._tool_outputs.append((tool_name, str(output)))

    def redacted_context(self) -> str:
        """The tool history with regions the next decision must not depend on
        replaced by a redaction marker.

        Feed this to the agent instead of the raw tool outputs. A decorator
        alone cannot enforce this: by the time a wrapped tool function is
        called the model has already generated, so blocking is the only lever
        left at that point. Redaction has to happen earlier, when the prompt
        is built, which means the caller has to ask for it — hence a method
        rather than something `protect` can do on its own.
        """
        screened = screen_step(
            self._tool_outputs,
            self.task_description,
            self.judge_fn,
            trusted_authors=self.trusted_authors,
        )
        return screened.redaction.text

    def protect(self, fn):
        """Wrap a tool function so it only runs after clearing Stages 1-3.

        The wrapped function must be called with keyword arguments — that is
        the shape every tool-calling API (OpenAI, Anthropic, Gemini) already
        hands back, and it is what lets the call be rendered and screened as
        `name(arg=value, ...)` without guessing parameter names from
        positional args.
        """

        @functools.wraps(fn)
        def wrapper(**kwargs):
            self._step += 1
            call = ToolCall(name=fn.__name__, arguments=kwargs)

            screened = screen_step(
                self._tool_outputs,
                self.task_description,
                self.judge_fn,
                trusted_authors=self.trusted_authors,
            )
            escalate_fn = None
            if self.melon_agent_call_fn is not None:
                escalate_fn = make_escalate_fn(
                    tool_output_text="\n\n".join(
                        content for _, content in self._tool_outputs
                    ),
                    agent_call_fn=self.melon_agent_call_fn,
                    system_message=self.system_message,
                    cache=self._masked_call_cache,
                )

            result = check_calls(
                self._step,
                screened,
                [call],
                escalate_fn=escalate_fn,
                enforce_confidentiality=self.enforce_confidentiality,
            )
            trace = result.trace
            if self.logger is not None:
                self.logger.log(trace)

            if trace.final_action == "block":
                raise Blocked(trace)

            if trace.final_action == "ask_user":
                if self.on_ask_user is None:
                    raise NeedsConfirmation(trace)
                if not self.on_ask_user(trace.explanation, call):
                    raise Blocked(trace)

            output = fn(**kwargs)
            self.observe(fn.__name__, output)
            return output

        return wrapper


# The brief (§7) specifies a module-level decorator that nonetheless "has
# access to conversation state". Those two only reconcile if the decorator
# resolves its session when the call happens rather than when the function is
# defined — tool functions are defined once at import, while conversation
# state is per-request. A context variable does that, and unlike a module
# global it stays correct across threads and concurrent async tasks.
_CURRENT_SESSION: contextvars.ContextVar[Session | None] = contextvars.ContextVar(
    "prompt_safe_current_session", default=None
)


class NoActiveSession(RuntimeError):
    """A guarded tool was called with no session bound.

    Raised rather than defaulting to permissive: a guarded function running
    unguarded is the one failure mode this module exists to prevent, and it
    would be silent.
    """


@contextlib.contextmanager
def session_scope(session: Session):
    """Bind `session` for the duration of the block.

    with session_scope(Session(task, judge_fn=openai_judge())):
        agent.run()
    """
    token = _CURRENT_SESSION.set(session)
    try:
        yield session
    finally:
        _CURRENT_SESSION.reset(token)


def current_session() -> Session:
    session = _CURRENT_SESSION.get()
    if session is None:
        raise NoActiveSession(
            "This tool is decorated with @guard but no session is bound. Wrap "
            "the agent run in `with session_scope(Session(...)):`."
        )
    return session


def guard(policy: str = "default"):
    """Decorate a tool function so it is screened against the active session.

    The `policy` argument is accepted because the brief's interface names it,
    and only "default" is implemented — passing anything else raises rather
    than silently applying the default, since a caller who asks for a stricter
    policy and quietly gets the lenient one is worse off than one who gets an
    error.
    """
    if policy != "default":
        raise ValueError(
            f"Unknown policy {policy!r}. Only 'default' is implemented; "
            "configure the policy on the Session instead."
        )

    def decorate(fn):
        @functools.wraps(fn)
        def wrapper(**kwargs):
            return current_session().protect(fn)(**kwargs)

        return wrapper

    return decorate
