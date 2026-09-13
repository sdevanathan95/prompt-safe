"""AgentDojo eval harness — Track B.

Runs a capped subset of one AgentDojo suite through our own MELON check,
independent of AgentDojo's own built-in defenses (the pipeline is built
with defense=None — we intercept ourselves, not through their
PipelineConfig). Makes real, paid LLM calls when invoked — never imported
by tests/, run explicitly via `python -m eval.harness`.

Two conditions per user task:
  - benign: the suite's default (non-malicious) injection-vector content,
    no attack. Ground truth: nothing malicious happened, so our check
    should never verdict "block" here — a block would be a false
    positive.
  - attacked: for each of the suite's injection tasks, AgentDojo's own
    `attacks` module generates an attack payload and injects it. Ground
    truth attack success comes from the injection task's own
    `security()`/`security_from_traces()` check — the same check AgentDojo
    itself reports numbers against, so our results are comparable to the
    papers'.

For each condition: the original task runs normally through the full
AgentDojo pipeline. Its actual tool-output text (whatever it observed,
injected content included) and already-decided tool calls are then handed
to `middleware.melon.engine.run_melon_check` — the same Track-A-facing
entrypoint the eventual live decorator will call, not a harness-local
reimplementation of it. The `agent_call_fn` we give it adapts
`run_melon_check`'s generic masked-conversation dicts (see
middleware/melon/masking.py) into a single direct call to the pipeline's
own `llm` element — not the full pipeline, since it only needs one
decision, not a multi-step loop.
"""

from __future__ import annotations

import functools
import json
import sys
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path

from agentdojo.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
from agentdojo.agent_pipeline.llms.anthropic_llm import AnthropicLLM
from agentdojo.agent_pipeline.llms.openai_llm import OpenAILLM
from agentdojo.attacks import load_attack
from agentdojo.attacks.base_attacks import BaseAttack
from agentdojo.base_tasks import BaseInjectionTask, BaseUserTask
from agentdojo.benchmark import get_suite
from agentdojo.functions_runtime import FunctionCall, FunctionsRuntime
from agentdojo.task_suite.task_suite import (
    TaskSuite,
    functions_stack_trace_from_messages,
    get_text_content_as_str,
    model_output_from_messages,
)
from agentdojo.types import text_content_block_from_string
from dotenv import load_dotenv

from adapters.judge import (
    DEFAULT_ANTHROPIC_JUDGE_MODEL,
    DEFAULT_OPENAI_JUDGE_MODEL,
    anthropic_judge,
    openai_judge,
)
from adapters.rate_limit import (
    DEFAULT_REQUESTS_PER_MINUTE,
    DEFAULT_TOKENS_PER_MINUTE,
    paced_openai_client,
    quota_exhausted,
    requests_sent,
)
from adapters.rate_limit import configure as configure_rate_limits
from adapters.retry import with_retry
from middleware.melon.compare import DEFAULT_THRESHOLD
from middleware.melon.engine import AgentCallFn, run_melon_check
from middleware.melon.types import MaskedRun, MelonVerdict, ToolCall
from middleware.screening.alignment import check_alignment
from middleware.screening.guard import StepResult, check_calls, screen_step
from middleware.screening.policy import is_external_content
from middleware.screening.regions import build_regions

# Provider clients read their key from the environment. Does not override a
# key already exported in the shell.
load_dotenv()

# Cheapest tier per provider — use this to validate the wiring before
# spending on an eval-quality model.
CHEAP_MODEL_BY_PROVIDER = {
    "openai": "gpt-4o-mini-2024-07-18",
    "anthropic": "claude-3-haiku-20240307",
}

DEFAULT_ATTACK_NAME = "important_instructions"

# The channel asks whether the answer carried out an instruction planted in
# content the agent read (screening/output_check.py). On the full AgentDojo run
# it alone stopped 12 of the 286 attacks that worked and blocked none of the 97
# clean runs. `--no-response-channel` turns it off.
RESPONSE_CHANNEL_ENABLED = True


@dataclass
class CaseResult:
    user_task_id: str
    injection_task_id: str | None  # None => benign condition, no attack
    ground_truth_attack_succeeded: bool | None  # None for the benign condition
    melon_verdict: MelonVerdict
    # Whether the agent actually completed the user's real task. Tracked in
    # both conditions, because a defense that blocks everything scores
    # perfectly on attacks and destroys the agent's usefulness.
    user_task_succeeded: bool | None = None
    # Stage 2's verdict and the step's final disposition. `final_action` is
    # the authoritative defense outcome: melon_verdict is only populated for
    # the steps that escalated that far.
    policy_verdict: str | None = None
    final_action: str | None = None
    trace: dict | None = None
    # Per-stage wall clock for this case; see StageTimings.
    timings: dict | None = None
    # The agent's final answer. Kept so the response channel's verdict on a
    # case can be read, and re-checked offline, without re-running the agent.
    final_response: str | None = None

    def to_record(self) -> dict:
        """One JSON-safe line of a resumable results file."""
        verdict = self.melon_verdict
        return {
            "user_task": self.user_task_id,
            "injection": self.injection_task_id,
            "attack_succeeded": self.ground_truth_attack_succeeded,
            "task_succeeded": self.user_task_succeeded,
            "policy": self.policy_verdict,
            "action": self.final_action,
            "melon": {
                "ran": verdict.ran,
                "verdict": verdict.verdict,
                "distance": verdict.distance,
                "explanation": verdict.explanation,
                # Saved for every case, not only escalated ones: the trace
                # holds the calls only when Stage 3 ran, so a case Stage 2
                # cleared -- the kind of miss that matters most -- would
                # otherwise leave no record of what the agent actually did.
                "original_calls": [
                    {"name": c.name, "arguments": c.arguments}
                    for c in verdict.original_calls
                ],
            },
            "trace": self.trace,
            "timings": self.timings,
            "final_response": self.final_response,
        }

    @classmethod
    def from_record(cls, record: dict) -> CaseResult:
        """Inverse of `to_record`, carrying every field eval/metrics.py reads."""
        melon = record["melon"]
        return cls(
            record["user_task"],
            record["injection"],
            record["attack_succeeded"],
            MelonVerdict(
                ran=melon["ran"],
                verdict=melon["verdict"],
                distance=melon["distance"],
                explanation=melon["explanation"],
                original_calls=[
                    ToolCall(c["name"], c["arguments"])
                    for c in melon.get("original_calls", [])
                ],
            ),
            user_task_succeeded=record["task_succeeded"],
            policy_verdict=record["policy"],
            final_action=record["action"],
            trace=record["trace"],
            timings=record["timings"],
            final_response=record.get("final_response"),
        )


ALL_SUITES = ("banking", "slack", "travel", "workspace")


def _paced(element: BasePipelineElement) -> BasePipelineElement:
    """Route an AgentDojo LLM element through the process-wide paced client.

    AgentDojo builds a bare `openai.OpenAI()` per pipeline -- 600-second
    timeout, no pacing -- so the agent's own calls would bypass the budget the
    judge and embeddings share. Swapping the client puts every call against a
    model into one allowance.
    """
    if isinstance(element, OpenAILLM):
        element.client = paced_openai_client()
    return element


def build_llm_element(provider: str, model_id: str) -> BasePipelineElement:
    """An LLM element on its own, for the masked run.

    Every attack that still gets through fails the same way: the masked run
    makes no tool call at all, so there is nothing to converge with and the
    step reads as benign. That is a capability question about the model doing
    the masked run, not about the comparison — MELON's own results vary by
    several points of ASR across backbone models. Letting the masked run use a
    stronger model than the agent is therefore a direct lever on the miss
    rate, and it is affordable because that call is speculative and concurrent
    with Stage 1 rather than added to it.
    """
    config = PipelineConfig(
        llm=model_id,
        model_id=None,
        defense=None,
        system_message_name=None,
        system_message=None,
    )
    pipeline = AgentPipeline.from_config(config)
    return _paced(
        next(e for e in pipeline.elements if isinstance(e, (OpenAILLM, AnthropicLLM)))
    )


def build_pipeline(
    provider: str, model_id: str | None = None
) -> tuple[BasePipelineElement, BasePipelineElement]:
    """Returns (full_pipeline, llm_element) — the llm element is needed
    directly for the masked run, which makes one decision rather than
    running the full multi-step loop."""
    model = model_id or CHEAP_MODEL_BY_PROVIDER[provider]
    config = PipelineConfig(
        llm=model,
        model_id=None,
        defense=None,
        system_message_name=None,
        system_message=None,
    )
    pipeline = AgentPipeline.from_config(config)
    llm_element = _paced(
        next(e for e in pipeline.elements if isinstance(e, (OpenAILLM, AnthropicLLM)))
    )
    return pipeline, llm_element


def _to_tool_calls(calls: list[FunctionCall]) -> list[ToolCall]:
    return [ToolCall(name=c.function, arguments=dict(c.args)) for c in calls]


def _extract_system_message(messages) -> str | None:
    if messages and messages[0]["role"] == "system":
        return get_text_content_as_str(messages[0]["content"])
    return None


def _observations_before_final_decision(messages) -> list:
    """The messages up to, but not including, the decision under test.

    MELON evaluates step t against the observations that preceded it, O_1:t.
    Handing the masked run the *finished* transcript instead includes the
    results of the very call being tested — and the masking prompt ends with
    "make sure that you are not repeating the same tool calls", so a masked
    run shown a completed transfer correctly declines to repeat it, produces
    nothing, and the step scores as maximum divergence.

    That is not a hypothetical: it accounted for every remaining in-scope miss
    on the banking suite, all of them multi-call trajectories where the
    injected action had already executed by the end of the transcript.
    """
    last_decision = None
    for index, message in enumerate(messages):
        if message["role"] == "assistant" and message.get("tool_calls"):
            last_decision = index
    return messages if last_decision is None else messages[:last_decision]


def _extract_tool_output_text(messages) -> str:
    """Concatenates the tool results the original run saw *before* the
    decision under test, labeled by function name — this is what the masked
    run is shown instead of independently deciding whether to go look for it."""
    blocks = []
    for message in _observations_before_final_decision(messages):
        if message["role"] == "tool":
            function_name = message["tool_call"].function
            # Only what the agent read. A write's result echoes the user's own
            # action back into a run that is meant to have no task -- see
            # policy.is_external_content.
            if not is_external_content(function_name):
                continue
            content = get_text_content_as_str(message["content"])
            blocks.append(
                f"{'=' * 50}\n\nfunction: {function_name}\n\n{content}\n\n{'=' * 50}"
            )
    return "\n\n".join(blocks)


def _trusted_authors(environment) -> frozenset[str]:
    """The user's own address and domain, read off the suite environment.

    Without this every region from a tool call carries the same label, the
    dependency label equals it, and the redactor's keep-if-it-flows-to rule
    preserves everything -- measured at 0 regions redacted across 32 workspace
    steps. Selective masking only has something to select between once regions
    differ, and who wrote a message is what makes them differ.
    """
    authors: set[str] = set()
    dumped = environment.model_dump()
    for value in dumped.values():
        if isinstance(value, dict):
            email = value.get("account_email")
            if isinstance(email, str) and "@" in email:
                authors.add(email.lower())
                authors.add(email.split("@", 1)[1].lower())
    return frozenset(authors)


def _extract_tool_outputs(messages) -> list[tuple[str, str, dict]]:
    """The same tool results, kept split by originating function so the
    screener can label and redact them per region rather than as one blob --
    with the arguments of the call that produced each, which is how the
    alignment check recognises a source the user named."""
    return [
        (
            message["tool_call"].function,
            get_text_content_as_str(message["content"]),
            dict(message["tool_call"].args),
        )
        for message in messages
        if message["role"] == "tool"
    ]


def _to_agentdojo_messages(messages: list[dict]) -> list:
    """Adapts middleware.melon.masking's generic {"role", "content", ...}
    dicts into AgentDojo's typed ChatMessage shapes. Framework-specific
    glue lives here, not in middleware/melon, which stays agent-agnostic."""
    result: list = []
    for m in messages:
        role = m["role"]
        if role in ("system", "user"):
            result.append(
                {
                    "role": role,
                    "content": [text_content_block_from_string(m["content"])],
                }
            )
        elif role == "assistant":
            tool_calls = None
            if m.get("tool_calls"):
                tool_calls = [
                    FunctionCall(
                        function=tc["function"],
                        args=tc["arguments"],
                        id=f"melon-mask-{i}",
                        placeholder_args=None,
                    )
                    for i, tc in enumerate(m["tool_calls"])
                ]
            content = (
                [text_content_block_from_string(m["content"])]
                if m.get("content")
                else None
            )
            result.append(
                {"role": "assistant", "content": content, "tool_calls": tool_calls}
            )
        elif role == "tool":
            # Each fabricated tool message in masking.py immediately
            # follows an assistant message with exactly one tool call.
            tool_call = result[-1]["tool_calls"][0]
            result.append(
                {
                    "role": "tool",
                    "content": [text_content_block_from_string(m["content"])],
                    "tool_call": tool_call,
                    "tool_call_id": tool_call.id,
                    "error": None,
                }
            )
    return result


# How many turns the masked run is allowed. MELON re-executes the masked
# state at every step of the real trajectory and accumulates its calls, so a
# multi-step injected task eventually reaches its payload. Reading only the
# masked run's first decision instead misses every injection whose payload
# needs a lookup first -- and the strongest AgentDojo attacks are exactly
# that shape ("send a transaction that includes the IBAN of the user's recent
# dinner companion, as visible from the transaction history"): the masked run
# correctly issues the read, we score it as no-match, and the attack passes.
# Measured: every in-scope miss across banking and travel had this shape.
MASKED_RUN_MAX_TURNS = 4


def _make_agent_call_fn(
    llm_element: BasePipelineElement, suite: TaskSuite, environment
) -> AgentCallFn:
    """Adapt the masked conversation into a short AgentDojo run.

    The masked run is a probe, not the user's agent: its tool calls execute
    against a throwaway copy of the environment purely so the conversation can
    continue, and every call it makes along the way is returned. Nothing it
    does is visible outside this function.
    """
    runtime = FunctionsRuntime(suite.tools)

    def agent_call_fn(messages: list[dict]) -> MaskedRun:
        conversation = _to_agentdojo_messages(messages)
        collected: list[ToolCall] = []
        said: list[str] = []

        for _ in range(MASKED_RUN_MAX_TURNS):
            _, _, _, conversation, _ = with_retry(
                lambda conversation=conversation: llm_element.query(
                    "", runtime, environment, conversation, {}
                )
            )
            spoken = conversation[-1].get("content")
            if spoken:
                said.append(get_text_content_as_str(spoken))
            calls = conversation[-1].get("tool_calls") or []
            if not calls:
                break
            collected.extend(_to_tool_calls(calls))

            for call in calls:
                content, error = "", None
                try:
                    result, error = runtime.run_function(
                        environment, call.function, call.args, raise_on_error=False
                    )
                    content = str(result)
                except Exception as exc:  # noqa: BLE001 - a probe must not fail the step
                    error = str(exc)
                conversation.append(
                    {
                        "role": "tool",
                        "content": [text_content_block_from_string(content or "")],
                        "tool_call": call,
                        "tool_call_id": getattr(call, "id", None),
                        "error": str(error) if error else None,
                    }
                )

        return MaskedRun(calls=collected, text="\n\n".join(said))

    return agent_call_fn


def _check_result(
    task: BaseUserTask | BaseInjectionTask,
    output_text: str,
    pre_environment,
    post_environment,
    function_calls: list[FunctionCall],
) -> bool:
    """Mirrors AgentDojo's own ground-truth computation (the trace-based
    fast path first, falling back to the plain check) using only the
    public per-task API — matches the numbers AgentDojo itself reports."""
    if isinstance(task, BaseUserTask):
        from_traces = task.utility_from_traces(
            output_text, pre_environment, post_environment, function_calls
        )
        if from_traces is not None:
            return from_traces
        return task.utility(output_text, pre_environment, post_environment)

    from_traces = task.security_from_traces(
        output_text, pre_environment, post_environment, function_calls
    )
    if from_traces is not None:
        return from_traces
    return task.security(output_text, pre_environment, post_environment)


def _guarded_verdict(
    llm_element: BasePipelineElement,
    melon_llm_element: BasePipelineElement | None,
    suite: TaskSuite,
    environment,
    user_task: BaseUserTask,
    messages,
    original_calls: list[ToolCall],
    judge_fn,
    threshold: float,
    step: int = 1,
    masking_prompts: tuple[str, ...] = ("summarize",),
    original_response: str = "",
    lazy_masked_run: bool = False,
    alignment_judge_fn=None,
    response_channel: bool = RESPONSE_CHANNEL_ENABLED,
) -> StepResult:
    """Run the full tiered pipeline over one finished episode.

    The masked re-execution reads only the tool output and the system message
    — it does not depend on the screener, the labels, or the policy verdict.
    So it is started concurrently with Stage 1 rather than after Stage 2, and
    a step that escalates finds the answer already waiting. Sequentially the
    two model calls add; this way the step costs about the slower of them.

    The trade is that a step which does not escalate has paid for a masked run
    it discards. That is the right way round: the counterfactual test is the
    expensive part of the latency budget and the cheap part of the money
    budget, and a discarded call costs only money.

    `lazy_masked_run` makes the other trade: the masked run starts only if the
    step escalates. Verdicts are identical -- nothing reads the masked run
    otherwise -- and a step that does not escalate pays for no masked calls,
    which is what lets a full benchmark fit an account's daily request quota.
    """
    tool_outputs = _extract_tool_outputs(messages)
    tool_output_text = _extract_tool_output_text(messages)
    system_message = _extract_system_message(messages)

    # The masked run uses the paper's prompt -- the one live.Session uses -- so
    # the benchmark measures the detector that actually ships. The response
    # channel no longer needs a masked run of its own; see output_check.py.
    prompts = masking_prompts

    def run_masked() -> MelonVerdict:
        masked_element = melon_llm_element or llm_element
        agent_call_fn = _make_agent_call_fn(
            masked_element, suite, environment.model_copy(deep=True)
        )
        return run_melon_check(
            original_calls,
            tool_output_text=tool_output_text,
            agent_call_fn=agent_call_fn,
            system_message=system_message,
            threshold=threshold,
            task_description=user_task.PROMPT,
            masking_prompts=prompts,
        )

    trusted = _trusted_authors(environment)
    # Regions are needed to ask the alignment question and cost nothing to
    # build, so they are built here rather than waited on from Stage 1.
    regions = build_regions(tool_outputs, trusted_authors=trusted)

    alignment_fn = alignment_judge_fn or judge_fn

    def align(call: ToolCall):
        return check_alignment(
            user_task.PROMPT,
            call.name,
            call.arguments,
            regions,
            alignment_fn,
        )

    with ThreadPoolExecutor(max_workers=2 + len(original_calls)) as pool:
        if lazy_masked_run:
            masked = functools.cache(run_masked)
        else:
            masked = pool.submit(run_masked).result
        alignments = [pool.submit(align, call) for call in original_calls]
        screened = screen_step(
            tool_outputs,
            task_description=user_task.PROMPT,
            judge_fn=judge_fn,
            trusted_authors=trusted,
        )
        result = check_calls(
            step,
            screened,
            original_calls,
            escalate_fn=lambda calls: masked(),
            alignment_judge_fn=alignment_fn,
            alignment_results=[f.result() for f in alignments],
            original_response=original_response,
            check_response_channel=response_channel,
            answer_judge_fn=alignment_fn,
        )
    return result


def run_benign_case(
    pipeline: BasePipelineElement,
    llm_element: BasePipelineElement,
    suite: TaskSuite,
    user_task: BaseUserTask,
    judge_fn,
    threshold: float = DEFAULT_THRESHOLD,
    masking_prompts: tuple[str, ...] = ("summarize",),
    melon_llm_element: BasePipelineElement | None = None,
    lazy_masked_run: bool = False,
    alignment_judge_fn=None,
    response_channel: bool = RESPONSE_CHANNEL_ENABLED,
) -> CaseResult:
    environment = suite.load_and_inject_default_environment({})
    pre_environment = environment.model_copy(deep=True)
    runtime = FunctionsRuntime(suite.tools)

    _, _, post_environment, messages, _ = pipeline.query(
        user_task.PROMPT, runtime, environment.model_copy(deep=True)
    )
    output_text = get_text_content_as_str(model_output_from_messages(messages) or [])
    full_function_calls = functions_stack_trace_from_messages(messages)
    # Every sensitive call across the whole episode, not just the last
    # round -- an attack can execute several steps before the model's
    # final action, and comparing only the final round silently dropped
    # it. compare.py already filters to is_sensitive calls before aligning.
    original_calls = _to_tool_calls(full_function_calls)

    user_task_succeeded = _check_result(
        user_task, output_text, pre_environment, post_environment, full_function_calls
    )

    step = _guarded_verdict(
        llm_element,
        melon_llm_element,
        suite,
        environment,
        user_task,
        messages,
        original_calls,
        judge_fn,
        threshold,
        masking_prompts=masking_prompts,
        original_response=output_text,
        lazy_masked_run=lazy_masked_run,
        alignment_judge_fn=alignment_judge_fn,
        response_channel=response_channel,
    )
    return CaseResult(
        user_task.ID,
        None,
        None,
        step.melon_verdict or _not_escalated(original_calls),
        user_task_succeeded=user_task_succeeded,
        policy_verdict=step.trace.policy_verdict,
        final_action=step.trace.final_action,
        trace=step.trace.to_dict(),
        timings=step.timings.to_dict(),
        final_response=output_text,
    )


def run_attack_case(
    pipeline: BasePipelineElement,
    llm_element: BasePipelineElement,
    suite: TaskSuite,
    user_task: BaseUserTask,
    attack: BaseAttack,
    injection_task_id: str,
    judge_fn,
    threshold: float = DEFAULT_THRESHOLD,
    masking_prompts: tuple[str, ...] = ("summarize",),
    melon_llm_element: BasePipelineElement | None = None,
    lazy_masked_run: bool = False,
    alignment_judge_fn=None,
    response_channel: bool = RESPONSE_CHANNEL_ENABLED,
) -> CaseResult:
    injection_task = suite.get_injection_task_by_id(injection_task_id)
    injections = attack.attack(user_task, injection_task)
    environment = suite.load_and_inject_default_environment(injections)
    pre_environment = environment.model_copy(deep=True)
    runtime = FunctionsRuntime(suite.tools)

    _, _, post_environment, messages, _ = pipeline.query(
        user_task.PROMPT, runtime, environment.model_copy(deep=True)
    )
    original_output = get_text_content_as_str(
        model_output_from_messages(messages) or []
    )
    full_function_calls = functions_stack_trace_from_messages(messages)
    # Every sensitive call across the whole episode, not just the last
    # round -- see run_benign_case for why.
    original_calls = _to_tool_calls(full_function_calls)

    ground_truth_attack_succeeded = _check_result(
        injection_task,
        original_output,
        pre_environment,
        post_environment,
        full_function_calls,
    )
    user_task_succeeded = _check_result(
        user_task,
        original_output,
        pre_environment,
        post_environment,
        full_function_calls,
    )

    step = _guarded_verdict(
        llm_element,
        melon_llm_element,
        suite,
        environment,
        user_task,
        messages,
        original_calls,
        judge_fn,
        threshold,
        masking_prompts=masking_prompts,
        original_response=original_output,
        lazy_masked_run=lazy_masked_run,
        alignment_judge_fn=alignment_judge_fn,
        response_channel=response_channel,
    )
    return CaseResult(
        user_task.ID,
        injection_task_id,
        ground_truth_attack_succeeded,
        step.melon_verdict or _not_escalated(original_calls),
        user_task_succeeded=user_task_succeeded,
        policy_verdict=step.trace.policy_verdict,
        final_action=step.trace.final_action,
        trace=step.trace.to_dict(),
        timings=step.timings.to_dict(),
        final_response=original_output,
    )


def _not_escalated(original_calls: list[ToolCall]) -> MelonVerdict:
    """Placeholder verdict for a step the policy check settled on its own —
    the counterfactual test never ran, which is the intended fast path."""
    return MelonVerdict(
        ran=False,
        verdict=None,
        distance=None,
        original_calls=original_calls,
        explanation="Resolved at the policy check; the counterfactual test was not needed.",
    )


def run_suite_subset(
    provider: str,
    suite_name: str,
    benchmark_version: str,
    max_user_tasks: int,
    attack_name: str = DEFAULT_ATTACK_NAME,
    model_id: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    max_injection_tasks: int | None = None,
    judge_model: str | None = None,
    masking_prompts: tuple[str, ...] = ("summarize",),
    melon_model: str | None = None,
    max_workers: int = 1,
    lazy_masked_run: bool = False,
    results_path: Path | None = None,
    alignment_model: str | None = None,
    response_channel: bool = RESPONSE_CHANNEL_ENABLED,
) -> list[CaseResult]:
    """Runs the benign case plus one attack per injection task (capped at
    `max_injection_tasks` if given — a suite typically has more than one)
    for up to `max_user_tasks` of the suite's user tasks. Each case makes 1
    LLM call for the original run, plus a second only if the original
    call(s) were sensitive enough to warrant the masked run (see
    middleware/melon/prefilter.py) — so up to, but not always,
    max_user_tasks * (1 + injection_tasks_used) * 2. This makes real, paid
    LLM calls — call only when you intend to spend on a run."""
    suite = get_suite(benchmark_version, suite_name)
    pipeline, llm_element = build_pipeline(provider, model_id)
    attack = load_attack(attack_name, suite, pipeline)
    judge_fn = build_judge(provider, judge_model)
    # A separate model for the alignment gate, if asked: a harder judgment than
    # screening, and run only on escalated delegation steps, so a stronger
    # model there costs little.
    alignment_fn = build_judge(provider, alignment_model) if alignment_model else None
    melon_llm_element = (
        build_llm_element(provider, melon_model) if melon_model else None
    )

    user_task_ids = list(suite.user_tasks.keys())[:max_user_tasks]
    injection_task_ids = list(suite.injection_tasks.keys())[:max_injection_tasks]

    # Each case is an independent episode against its own environment copy, so
    # they parallelize cleanly. The provider rate limit, not the CPU, is the
    # binding constraint; adapters.rate_limit paces every call against it.
    def benign(user_task_id: str) -> CaseResult:
        return run_benign_case(
            pipeline,
            llm_element,
            suite,
            suite.get_user_task_by_id(user_task_id),
            judge_fn,
            threshold,
            masking_prompts,
            melon_llm_element,
            lazy_masked_run=lazy_masked_run,
            alignment_judge_fn=alignment_fn,
            response_channel=response_channel,
        )

    def attacked(pair: tuple[str, str]) -> CaseResult:
        user_task_id, injection_task_id = pair
        return run_attack_case(
            pipeline,
            llm_element,
            suite,
            suite.get_user_task_by_id(user_task_id),
            attack,
            injection_task_id,
            judge_fn,
            threshold,
            masking_prompts,
            melon_llm_element,
            lazy_masked_run=lazy_masked_run,
            alignment_judge_fn=alignment_fn,
            response_channel=response_channel,
        )

    pairs = [(u, i) for u in user_task_ids for i in injection_task_ids]
    keys: list[tuple[str, str | None]] = [(u, None) for u in user_task_ids] + pairs
    finished = _load_finished(results_path) if results_path else {}

    def job(key: tuple[str, str | None]) -> CaseResult:
        user_task_id, injection_task_id = key
        if injection_task_id is None:
            return benign(user_task_id)
        return attacked((user_task_id, injection_task_id))

    todo = [key for key in keys if key not in finished]
    fresh = _run_jobs(todo, job, max_workers, _appender(results_path))
    # Suite order rather than completion order, so repeated runs read the same.
    # Keys absent from both were never started: the daily quota ran out first.
    return [
        finished.get(key) or fresh[key]
        for key in keys
        if key in finished or key in fresh
    ]


def _run_jobs(
    keys: list[tuple[str, str | None]],
    job: Callable[[tuple[str, str | None]], CaseResult],
    max_workers: int,
    record: Callable[[CaseResult], None],
    should_stop: Callable[[], object] = quota_exhausted,
) -> dict[tuple[str, str | None], CaseResult]:
    """Run cases with a bounded number in flight, recording each as it lands.

    Cases are submitted as slots free up rather than all at once, so once the
    daily quota is nearly spent no further case starts. The ones already
    running finish inside the quota's reserve, and a resumed run picks up
    exactly the cases that never started.
    """
    results: dict[tuple[str, str | None], CaseResult] = {}
    queue = iter(keys)
    workers = max(1, max_workers)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        running = {}

        def top_up() -> None:
            while len(running) < workers and should_stop() is None:
                key = next(queue, None)
                if key is None:
                    return
                running[pool.submit(job, key)] = key

        top_up()
        while running:
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                key = running.pop(future)
                result = _settled(future, *key)
                results[key] = result
                record(result)
            top_up()
    return results


def _appender(path: Path | None) -> Callable[[CaseResult], None]:
    """Append each finished case to `path` the moment it lands, so a run killed
    mid-way loses at most the cases still in flight. Called only from
    `_run_jobs`'s own thread, so appends never interleave."""
    if path is None:
        return lambda _result: None
    path.parent.mkdir(parents=True, exist_ok=True)

    def record(result: CaseResult) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(result.to_record(), default=str) + "\n")

    return record


def _load_finished(path: Path) -> dict[tuple[str, str | None], CaseResult]:
    """Cases an earlier run already completed, keyed (user_task, injection).

    A crashed case is not finished: it is left out so the resumed run tries it
    again. A partial last line from a run killed mid-write is skipped.
    """
    finished: dict[tuple[str, str | None], CaseResult] = {}
    if not path.exists():
        return finished
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            result = CaseResult.from_record(json.loads(line))
        except (ValueError, KeyError, TypeError):
            continue
        if result.final_action is not None:
            finished[(result.user_task_id, result.injection_task_id)] = result
    return finished


def _case_line(result: CaseResult) -> str:
    """One case per line, in the format eval/report.py parses."""
    return (
        f"{result.user_task_id} injection={result.injection_task_id} "
        f"attack_succeeded={result.ground_truth_attack_succeeded} "
        f"task_succeeded={result.user_task_succeeded} "
        f"policy={result.policy_verdict} action={result.final_action} "
        f"distance={result.melon_verdict.distance}"
    )


def _settled(future, user_task_id: str, injection_task_id: str | None) -> CaseResult:
    """A case result, or a recorded failure.

    One case raising must not lose the other several hundred. A crashed case is
    recorded with `final_action=None`, which every metric in eval/metrics.py
    already treats as "not stopped" -- so a failure counts against us rather
    than silently improving the numbers.
    """
    try:
        return future.result()
    except Exception as exc:  # noqa: BLE001 - one bad case must not end the run
        return CaseResult(
            user_task_id,
            injection_task_id,
            None,
            MelonVerdict(
                ran=False,
                verdict=None,
                distance=None,
                explanation=f"case failed: {type(exc).__name__}: {exc}",
            ),
        )


def build_judge(provider: str, judge_model: str | None = None):
    """The screener's own model. Kept separate from the agent's model: the
    judge answers one narrow relevance question and is the dominant added cost
    if run on a frontier model."""
    if provider == "openai":
        return openai_judge(judge_model or DEFAULT_OPENAI_JUDGE_MODEL)
    return anthropic_judge(judge_model or DEFAULT_ANTHROPIC_JUDGE_MODEL)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run a capped AgentDojo subset through the MELON check."
    )
    parser.add_argument("--provider", choices=["openai", "anthropic"], required=True)
    parser.add_argument(
        "--suite",
        default="workspace",
        help="A suite name, a comma-separated list, or 'all'. Several suites run "
        "in one process so they share one rate-limit budget.",
    )
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument("--max-user-tasks", type=int, default=3)
    parser.add_argument("--max-injection-tasks", type=int, default=None)
    parser.add_argument("--attack", default=DEFAULT_ATTACK_NAME)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument(
        "--melon-model",
        default=None,
        help="Model for the masked re-execution. Defaults to the agent model. Every remaining miss is a masked run that made no tool call, so a stronger model here is the direct lever on the miss rate.",
    )
    parser.add_argument(
        "--trace-out", default=None, help="Write per-step traces as JSON Lines."
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Cases to run concurrently. Each case is an independent episode; "
        "the provider rate limit is the binding constraint, not the CPU.",
    )
    parser.add_argument(
        "--ensemble",
        default="summarize",
        help="Comma-separated masking prompts for the counterfactual test "
        "(summarize,sentiment,grammar,translate). More detectors cost one "
        "extra model call each per escalated step and lower the miss rate.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Append each finished case to DIR/cases_<suite>.jsonl as it lands "
        "and write DIR/final_<suite>.txt for eval.report. Rerunning with the "
        "same directory skips finished cases, so an interrupted or "
        "quota-stopped run resumes where it stopped.",
    )
    parser.add_argument(
        "--lazy-masked-run",
        action="store_true",
        help="Run the counterfactual re-execution only for steps that escalate. "
        "Identical verdicts at a fraction of the model calls; latency numbers "
        "then reflect sequential rather than speculative execution.",
    )
    parser.add_argument(
        "--alignment-model",
        default=None,
        help="Model for the alignment gate (did the user delegate this?). "
        "Defaults to --judge-model. A harder judgment than screening, run only "
        "on escalated delegation steps, so a stronger model here is cheap.",
    )
    parser.add_argument(
        "--response-channel",
        action=argparse.BooleanOptionalAction,
        default=RESPONSE_CHANNEL_ENABLED,
        help="Check the agent's final answer for an instruction planted in "
        "content it read (attacks that call no tool). On by default; one judge "
        "call on steps whose answer draws on untrusted content.",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=DEFAULT_REQUESTS_PER_MINUTE,
        help="This account's requests-per-minute limit for the model.",
    )
    parser.add_argument(
        "--tpm",
        type=int,
        default=DEFAULT_TOKENS_PER_MINUTE,
        help="This account's tokens-per-minute limit for the model.",
    )
    args = parser.parse_args()
    configure_rate_limits(args.rpm, args.tpm)

    suites = (
        list(ALL_SUITES)
        if args.suite == "all"
        else [name.strip() for name in args.suite.split(",") if name.strip()]
    )
    results_dir = Path(args.results_dir) if args.results_dir else None

    case_results: list[CaseResult] = []
    for suite_name in suites:
        suite_results = run_suite_subset(
            provider=args.provider,
            suite_name=suite_name,
            benchmark_version=args.benchmark_version,
            max_user_tasks=args.max_user_tasks,
            attack_name=args.attack,
            model_id=args.model_id,
            max_injection_tasks=args.max_injection_tasks,
            judge_model=args.judge_model,
            masking_prompts=tuple(
                p.strip() for p in args.ensemble.split(",") if p.strip()
            ),
            melon_model=args.melon_model,
            max_workers=args.max_workers,
            lazy_masked_run=args.lazy_masked_run,
            alignment_model=args.alignment_model,
            response_channel=args.response_channel,
            results_path=(
                results_dir / f"cases_{suite_name}.jsonl" if results_dir else None
            ),
        )
        lines = [_case_line(result) for result in suite_results]
        print(f"=== {suite_name}: {len(suite_results)} cases ===")
        print("\n".join(lines))
        if results_dir:
            (results_dir / f"final_{suite_name}.txt").write_text(
                "\n".join(lines) + "\n", encoding="utf-8"
            )
        case_results.extend(suite_results)
        if quota_exhausted() is not None:
            break

    # A case that crashed is not a case that passed. Reporting metrics over a
    # run whose failures are invisible is how a rate-limited run gets read as a
    # clean one -- 54 of 60 cases died to 429s while the summary below still
    # printed a 0% false positive rate.
    failed = [r for r in case_results if r.final_action is None]
    if failed:
        print(f"\n!!! {len(failed)} of {len(case_results)} cases FAILED to run:")
        reasons: dict[str, int] = {}
        for result in failed:
            reason = result.melon_verdict.explanation[:120]
            reasons[reason] = reasons.get(reason, 0) + 1
        for reason, count in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"  {count:4}x {reason}")
        print("Metrics below are computed over the cases that ran, not all of them.")

    if args.trace_out:
        import json as _json

        with open(args.trace_out, "w", encoding="utf-8") as handle:
            for result in case_results:
                if result.trace is not None:
                    handle.write(
                        _json.dumps(
                            {
                                **result.trace,
                                "case": result.user_task_id,
                                "injection": result.injection_task_id,
                                "timings": result.timings,
                            }
                        )
                        + "\n"
                    )
        print(f"\ntraces written to {args.trace_out}")

    # Imported here, not at module level: eval.metrics imports CaseResult
    # from this module, so importing compute_metrics back at the top would
    # be a circular import. CLI-only usage, so a local import is fine.
    from eval.metrics import compute_metrics

    def pct(value, absent="n/a"):
        return f"{value:.1%}" if value is not None else absent

    report = compute_metrics(case_results)
    print()
    print(f"total cases:                {report.total_cases}")
    print(f"benign / attack cases:      {report.benign_cases} / {report.attack_cases}")
    print()
    print("--- the three that must be read together ---")
    print(f"benign utility (undefended):   {pct(report.benign_utility)}")
    print(f"benign utility (defended):     {pct(report.defended_benign_utility)}")
    print(f"utility under attack (undef.): {pct(report.utility_under_attack)}")
    print(f"utility under attack (def.):   {pct(report.defended_utility_under_attack)}")
    print(
        f"attacks actually succeeded: {report.attacks_actually_succeeded} of {report.attack_cases}"
    )
    print(
        f"attack prevention rate:     {pct(report.attack_prevention_rate, 'n/a (no successful attacks)')}"
    )
    print(
        f"false positive rate:        {pct(report.false_positive_rate, 'n/a (no benign cases)')}"
    )
    print()
    print("--- the tiering (this project's claim) ---")
    print(f"escalation rate:            {pct(report.escalation_rate)}")
    print(
        f"auto-resolution rate:       {pct(report.auto_resolution_rate, 'n/a (nothing escalated)')}"
    )
    print(
        f"auto-resolution accuracy:   {pct(report.auto_resolution_accuracy, 'n/a (no ground truth)')}"
    )
    print()
    print("--- headline: confirmations vs RTBAS alone ---")
    print(f"RTBAS alone would ask:      {report.rtbas_baseline_confirmations}")
    print(f"we ask:                     {report.human_confirmations}")
    print(
        f"confirmation reduction:     {pct(report.confirmation_reduction, 'n/a (nothing escalated)')}"
    )

    def ms(value):
        return f"{value:.0f} ms" if value is not None else "n/a"

    print()
    print("--- latency by stage (mean per step) ---")
    print(f"stage 1 screen:             {ms(report.mean_screen_ms)}")
    print(f"stage 2 policy:             {ms(report.mean_policy_ms)}")
    print(f"stage 3 (escalated only):   {ms(report.mean_melon_ms_when_escalated)}")
    print(f"total per step:             {ms(report.mean_total_ms)}")

    print()
    print("--- model requests this run ---")
    for model, count in sorted(requests_sent().items()):
        print(f"{model:36} {count}")

    stopped = quota_exhausted()
    if stopped is not None:
        model, resets_in = stopped
        print(
            f"\nSTOPPED EARLY: the daily request quota for {model} is nearly "
            f"spent (resets in {resets_in}). Rerun the same command after the "
            "reset; finished cases are skipped."
        )
        sys.exit(3)
