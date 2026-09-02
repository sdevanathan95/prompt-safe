"""LangGraph integration.

The project brief (§7) asks for a framework adapter for LangGraph, on the
grounds that the tool-calling boundary has converged to roughly the same shape
everywhere and that LangGraph is the likely stack for a demo comparison.

There is genuinely little to adapt. LangGraph tool nodes call plain Python
functions, and `Session.protect` already wraps plain Python functions, so the
integration is a loop over the tool list rather than a new mechanism. What
this module adds is the one piece that is not automatic: LangGraph decides
what to call from message state the middleware never sees, so the tool outputs
have to be fed back to the Session for later steps to be screened against, and
the redacted context has to be pulled when the prompt is built.

Importing this module does not require langgraph to be installed — nothing
here imports it. It operates on the callables you already hand to a
ToolNode.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from middleware.screening.live import Blocked, NeedsConfirmation, Session


def protect_tools(session: Session, tools: Iterable[Callable]) -> list[Callable]:
    """Wrap each tool so it clears Stages 1-3 before its body runs.

    Pass the result where the unprotected list went:

        session = Session(user_task, judge_fn=openai_judge())
        graph.add_node("tools", ToolNode(protect_tools(session, my_tools)))

    A blocked call raises Blocked out of the tool node. Whether that should
    end the run or come back to the model as an error message is a policy
    decision the graph owns, not this adapter — see blocked_as_tool_message
    for the second option.
    """
    return [session.protect(tool) for tool in tools]


def blocked_as_tool_message(fn: Callable) -> Callable:
    """Turn a Blocked/NeedsConfirmation into a returned string instead of a
    raised exception.

    Some graphs would rather hand the refusal back to the model as an ordinary
    tool result and let it re-plan than tear down the run. The explanation is
    written for a person, so it reads sensibly either way.

    Note what this does *not* change: the wrapped function's body still never
    executed. This only alters how the refusal is reported.
    """

    def wrapper(**kwargs):
        try:
            return fn(**kwargs)
        except (Blocked, NeedsConfirmation) as refusal:
            return f"BLOCKED BY SECURITY POLICY: {refusal.trace.explanation}"

    wrapper.__name__ = getattr(fn, "__name__", "tool")
    wrapper.__doc__ = fn.__doc__
    return wrapper


def observe_tool_messages(session: Session, messages: Iterable) -> None:
    """Feed tool results the middleware did not itself produce back into the
    session.

    Only needed when something other than a protected tool put a result into
    the graph state — a cached result, a node that calls an API directly, a
    resumed thread. Tools wrapped by `protect_tools` record their own output
    already, so double-feeding them would duplicate regions.

    Accepts LangChain ToolMessage objects or plain dicts; anything without a
    name and content is skipped rather than guessed at.
    """
    for message in messages:
        name = getattr(message, "name", None)
        content = getattr(message, "content", None)
        if name is None and isinstance(message, dict):
            name, content = message.get("name"), message.get("content")
        if name and content:
            session.observe(str(name), content)
