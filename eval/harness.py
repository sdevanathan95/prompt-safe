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

from dataclasses import dataclass

from dotenv import load_dotenv

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

from middleware.melon.compare import DEFAULT_THRESHOLD
from middleware.melon.engine import AgentCallFn, make_escalate_fn, run_melon_check
from middleware.melon.types import MelonVerdict, ToolCall
from middleware.screening.guard import StepResult, check_calls, screen_step
from adapters.judge import (
    DEFAULT_ANTHROPIC_JUDGE_MODEL,
    DEFAULT_OPENAI_JUDGE_MODEL,
    anthropic_judge,
    openai_judge,
)

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


def build_pipeline(provider: str, model_id: str | None = None) -> tuple[BasePipelineElement, BasePipelineElement]:
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
    llm_element = next(e for e in pipeline.elements if isinstance(e, (OpenAILLM, AnthropicLLM)))
    return pipeline, llm_element


def _to_tool_calls(calls: list[FunctionCall]) -> list[ToolCall]:
    return [ToolCall(name=c.function, arguments=dict(c.args)) for c in calls]


def _extract_system_message(messages) -> str | None:
    if messages and messages[0]["role"] == "system":
        return get_text_content_as_str(messages[0]["content"])
    return None


def _extract_tool_output_text(messages) -> str:
    """Concatenates every tool result the original run actually saw,
    labeled by function name — this is what the masked run is shown
    instead of independently deciding whether to go look for it."""
    blocks = []
    for message in messages:
        if message["role"] == "tool":
            content = get_text_content_as_str(message["content"])
            function_name = message["tool_call"].function
            blocks.append(f"{'=' * 50}\n\nfunction: {function_name}\n\n{content}\n\n{'=' * 50}")
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


def _extract_tool_outputs(messages) -> list[tuple[str, str]]:
    """The same tool results, kept split by originating function so the
    screener can label and redact them per region rather than as one blob."""
    return [
        (message["tool_call"].function, get_text_content_as_str(message["content"]))
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
            result.append({"role": role, "content": [text_content_block_from_string(m["content"])]})
        elif role == "assistant":
            tool_calls = None
            if m.get("tool_calls"):
                tool_calls = [
                    FunctionCall(function=tc["function"], args=tc["arguments"], id=f"melon-mask-{i}", placeholder_args=None)
                    for i, tc in enumerate(m["tool_calls"])
                ]
            content = [text_content_block_from_string(m["content"])] if m.get("content") else None
            result.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
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


def _make_agent_call_fn(llm_element: BasePipelineElement, suite: TaskSuite, environment) -> AgentCallFn:
    """Adapts run_melon_check's generic masked-conversation dicts into one
    direct call to the pipeline's own llm element. This is the only
    AgentDojo-specific piece of the masked run — everything upstream of it
    (building the masked conversation, deciding whether to run it at all)
    lives in middleware.melon and is exercised exactly as Track A's
    eventual live decorator would exercise it."""
    runtime = FunctionsRuntime(suite.tools)

    def agent_call_fn(messages: list[dict]) -> list[ToolCall]:
        agentdojo_messages = _to_agentdojo_messages(messages)
        _, _, _, updated_messages, _ = llm_element.query("", runtime, environment, agentdojo_messages, {})
        last_message = updated_messages[-1]
        return _to_tool_calls(last_message.get("tool_calls") or [])

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
        from_traces = task.utility_from_traces(output_text, pre_environment, post_environment, function_calls)
        if from_traces is not None:
            return from_traces
        return task.utility(output_text, pre_environment, post_environment)

    from_traces = task.security_from_traces(output_text, pre_environment, post_environment, function_calls)
    if from_traces is not None:
        return from_traces
    return task.security(output_text, pre_environment, post_environment)


def _guarded_verdict(
    llm_element: BasePipelineElement,
    suite: TaskSuite,
    environment,
    user_task: BaseUserTask,
    messages,
    original_calls: list[ToolCall],
    judge_fn,
    threshold: float,
    step: int = 1,
    masking_prompts: tuple[str, ...] = ("summarize",),
) -> StepResult:
    """Run the full tiered pipeline over one finished episode.

    Stage 1 screens and redacts, Stage 2 decides, and Stage 3 runs only for
    the steps Stage 2 could not settle — the tiering the project is actually
    about. With no judge available, screening is skipped and every step is
    treated as depending on all its tool output, which is the conservative
    reading and matches taint tracking without a screener.
    """
    screened = screen_step(
        _extract_tool_outputs(messages),
        task_description=user_task.PROMPT,
        judge_fn=judge_fn,
        trusted_authors=_trusted_authors(environment),
    )

    agent_call_fn = _make_agent_call_fn(llm_element, suite, environment.model_copy(deep=True))
    escalate_fn = make_escalate_fn(
        tool_output_text=_extract_tool_output_text(messages),
        agent_call_fn=agent_call_fn,
        system_message=_extract_system_message(messages),
        threshold=threshold,
    )
    return check_calls(
        step, screened, original_calls, escalate_fn=escalate_fn,
        alignment_judge_fn=judge_fn,
    )


def run_benign_case(
    pipeline: BasePipelineElement,
    llm_element: BasePipelineElement,
    suite: TaskSuite,
    user_task: BaseUserTask,
    judge_fn,
    threshold: float = DEFAULT_THRESHOLD,
    masking_prompts: tuple[str, ...] = ("summarize",),
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
        llm_element, suite, environment, user_task, messages, original_calls,
        judge_fn, threshold, masking_prompts=masking_prompts,
    )
    return CaseResult(
        user_task.ID, None, None, step.melon_verdict or _not_escalated(original_calls),
        user_task_succeeded=user_task_succeeded,
        policy_verdict=step.trace.policy_verdict,
        final_action=step.trace.final_action,
        trace=step.trace.to_dict(),
        timings=step.timings.to_dict(),
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
) -> CaseResult:
    injection_task = suite.get_injection_task_by_id(injection_task_id)
    injections = attack.attack(user_task, injection_task)
    environment = suite.load_and_inject_default_environment(injections)
    pre_environment = environment.model_copy(deep=True)
    runtime = FunctionsRuntime(suite.tools)

    _, _, post_environment, messages, _ = pipeline.query(user_task.PROMPT, runtime, environment.model_copy(deep=True))
    original_output = get_text_content_as_str(model_output_from_messages(messages) or [])
    full_function_calls = functions_stack_trace_from_messages(messages)
    # Every sensitive call across the whole episode, not just the last
    # round -- see run_benign_case for why.
    original_calls = _to_tool_calls(full_function_calls)

    ground_truth_attack_succeeded = _check_result(
        injection_task, original_output, pre_environment, post_environment, full_function_calls
    )
    user_task_succeeded = _check_result(
        user_task, original_output, pre_environment, post_environment, full_function_calls
    )

    step = _guarded_verdict(
        llm_element, suite, environment, user_task, messages, original_calls,
        judge_fn, threshold, masking_prompts=masking_prompts,
    )
    return CaseResult(
        user_task.ID, injection_task_id, ground_truth_attack_succeeded,
        step.melon_verdict or _not_escalated(original_calls),
        user_task_succeeded=user_task_succeeded,
        policy_verdict=step.trace.policy_verdict,
        final_action=step.trace.final_action,
        trace=step.trace.to_dict(),
        timings=step.timings.to_dict(),
    )


def _not_escalated(original_calls: list[ToolCall]) -> MelonVerdict:
    """Placeholder verdict for a step the policy check settled on its own —
    the counterfactual test never ran, which is the intended fast path."""
    return MelonVerdict(
        ran=False, verdict=None, distance=None, original_calls=original_calls,
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

    user_task_ids = list(suite.user_tasks.keys())[:max_user_tasks]
    injection_task_ids = list(suite.injection_tasks.keys())[:max_injection_tasks]
    results: list[CaseResult] = []

    for user_task_id in user_task_ids:
        user_task = suite.get_user_task_by_id(user_task_id)
        results.append(run_benign_case(pipeline, llm_element, suite, user_task, judge_fn, threshold, masking_prompts))
        for injection_task_id in injection_task_ids:
            results.append(
                run_attack_case(
                    pipeline, llm_element, suite, user_task, attack,
                    injection_task_id, judge_fn, threshold, masking_prompts,
                )
            )

    return results


def build_judge(provider: str, judge_model: str | None = None):
    """The screener's own model. Kept separate from the agent's model: the
    judge answers one narrow relevance question and is the dominant added cost
    if run on a frontier model."""
    if provider == "openai":
        return openai_judge(judge_model or DEFAULT_OPENAI_JUDGE_MODEL)
    return anthropic_judge(judge_model or DEFAULT_ANTHROPIC_JUDGE_MODEL)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a capped AgentDojo subset through the MELON check.")
    parser.add_argument("--provider", choices=["openai", "anthropic"], required=True)
    parser.add_argument("--suite", default="workspace")
    parser.add_argument("--benchmark-version", default="v1.2.2")
    parser.add_argument("--max-user-tasks", type=int, default=3)
    parser.add_argument("--max-injection-tasks", type=int, default=None)
    parser.add_argument("--attack", default=DEFAULT_ATTACK_NAME)
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--trace-out", default=None, help="Write per-step traces as JSON Lines.")
    parser.add_argument(
        "--ensemble", default="summarize",
        help="Comma-separated masking prompts for the counterfactual test "
             "(summarize,sentiment,grammar,translate). More detectors cost one "
             "extra model call each per escalated step and lower the miss rate.",
    )
    args = parser.parse_args()

    case_results = run_suite_subset(
        provider=args.provider,
        suite_name=args.suite,
        benchmark_version=args.benchmark_version,
        max_user_tasks=args.max_user_tasks,
        attack_name=args.attack,
        model_id=args.model_id,
        max_injection_tasks=args.max_injection_tasks,
        judge_model=args.judge_model,
        masking_prompts=tuple(p.strip() for p in args.ensemble.split(",") if p.strip()),
    )

    for result in case_results:
        print(
            f"{result.user_task_id} injection={result.injection_task_id} "
            f"attack_succeeded={result.ground_truth_attack_succeeded} "
            f"task_succeeded={result.user_task_succeeded} "
            f"policy={result.policy_verdict} action={result.final_action} "
            f"distance={result.melon_verdict.distance}"
        )

    if args.trace_out:
        import json as _json

        with open(args.trace_out, "w", encoding="utf-8") as handle:
            for result in case_results:
                if result.trace is not None:
                    handle.write(_json.dumps({**result.trace, "case": result.user_task_id,
                                              "injection": result.injection_task_id}) + "\n")
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
    print(f"attacks actually succeeded: {report.attacks_actually_succeeded} of {report.attack_cases}")
    print(f"attack prevention rate:     {pct(report.attack_prevention_rate, 'n/a (no successful attacks)')}")
    print(f"false positive rate:        {pct(report.false_positive_rate, 'n/a (no benign cases)')}")
    print()
    print("--- the tiering (this project's claim) ---")
    print(f"escalation rate:            {pct(report.escalation_rate)}")
    print(f"auto-resolution rate:       {pct(report.auto_resolution_rate, 'n/a (nothing escalated)')}")
    print(f"auto-resolution accuracy:   {pct(report.auto_resolution_accuracy, 'n/a (no ground truth)')}")
    print()
    print("--- headline: confirmations vs RTBAS alone ---")
    print(f"RTBAS alone would ask:      {report.rtbas_baseline_confirmations}")
    print(f"we ask:                     {report.human_confirmations}")
    print(f"confirmation reduction:     {pct(report.confirmation_reduction, 'n/a (nothing escalated)')}")

    def ms(value):
        return f"{value:.0f} ms" if value is not None else "n/a"

    print()
    print("--- latency by stage (mean per step) ---")
    print(f"stage 1 screen:             {ms(report.mean_screen_ms)}")
    print(f"stage 2 policy:             {ms(report.mean_policy_ms)}")
    print(f"stage 3 (escalated only):   {ms(report.mean_melon_ms_when_escalated)}")
    print(f"total per step:             {ms(report.mean_total_ms)}")
