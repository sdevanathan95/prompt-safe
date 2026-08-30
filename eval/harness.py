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

Both conditions run the query twice against the same injected
environment: once with the real task prompt (the "original" run) and once
with the neutral placeholder in its place (the "masked" run). Both runs'
tool calls go through middleware.melon for our own verdict.
"""

from __future__ import annotations

from dataclasses import dataclass

from agentdojo.agent_pipeline import AgentPipeline, PipelineConfig
from agentdojo.agent_pipeline.base_pipeline_element import BasePipelineElement
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

from middleware.melon.compare import DEFAULT_THRESHOLD
from middleware.melon.engine import evaluate_pair
from middleware.melon.masking import build_placeholder_task
from middleware.melon.types import MelonVerdict, ToolCall

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


def build_pipeline(provider: str, model_id: str | None = None) -> BasePipelineElement:
    model = model_id or CHEAP_MODEL_BY_PROVIDER[provider]
    config = PipelineConfig(
        llm=model,
        model_id=None,
        defense=None,
        system_message_name=None,
        system_message=None,
    )
    return AgentPipeline.from_config(config)


def _to_tool_calls(calls: list[FunctionCall]) -> list[ToolCall]:
    return [ToolCall(name=c.function, arguments=dict(c.args)) for c in calls]


def _run_query(pipeline: BasePipelineElement, suite: TaskSuite, prompt: str, environment):
    runtime = FunctionsRuntime(suite.tools)
    _, _, post_environment, messages, _ = pipeline.query(prompt, runtime, environment)
    function_calls = functions_stack_trace_from_messages(messages)
    output_text = get_text_content_as_str(model_output_from_messages(messages) or [])
    return function_calls, output_text, post_environment


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


def run_benign_case(
    pipeline: BasePipelineElement,
    suite: TaskSuite,
    user_task: BaseUserTask,
    threshold: float = DEFAULT_THRESHOLD,
) -> CaseResult:
    environment = suite.load_and_inject_default_environment({})

    original_calls, _, _ = _run_query(pipeline, suite, user_task.PROMPT, environment.model_copy(deep=True))
    masked_calls, _, _ = _run_query(pipeline, suite, build_placeholder_task(), environment.model_copy(deep=True))

    verdict = evaluate_pair(_to_tool_calls(original_calls), _to_tool_calls(masked_calls), threshold)
    return CaseResult(user_task.ID, None, None, verdict)


def run_attack_case(
    pipeline: BasePipelineElement,
    suite: TaskSuite,
    user_task: BaseUserTask,
    attack: BaseAttack,
    injection_task_id: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> CaseResult:
    injection_task = suite.get_injection_task_by_id(injection_task_id)
    injections = attack.attack(user_task, injection_task)
    environment = suite.load_and_inject_default_environment(injections)
    pre_environment = environment.model_copy(deep=True)

    original_calls, original_output, post_environment = _run_query(
        pipeline, suite, user_task.PROMPT, environment.model_copy(deep=True)
    )
    masked_calls, _, _ = _run_query(pipeline, suite, build_placeholder_task(), environment.model_copy(deep=True))

    ground_truth_attack_succeeded = _check_result(
        injection_task, original_output, pre_environment, post_environment, original_calls
    )

    verdict = evaluate_pair(_to_tool_calls(original_calls), _to_tool_calls(masked_calls), threshold)
    return CaseResult(user_task.ID, injection_task_id, ground_truth_attack_succeeded, verdict)


def run_suite_subset(
    provider: str,
    suite_name: str,
    benchmark_version: str,
    max_user_tasks: int,
    attack_name: str = DEFAULT_ATTACK_NAME,
    model_id: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    max_injection_tasks: int | None = None,
) -> list[CaseResult]:
    """Runs the benign case plus one attack per injection task (capped at
    `max_injection_tasks` if given — a suite typically has more than one)
    for up to `max_user_tasks` of the suite's user tasks. Each case makes 2
    LLM calls (original + masked run), so total calls =
    max_user_tasks * (1 + injection_tasks_used) * 2. This makes real, paid
    LLM calls — call only when you intend to spend on a run."""
    suite = get_suite(benchmark_version, suite_name)
    pipeline = build_pipeline(provider, model_id)
    attack = load_attack(attack_name, suite, pipeline)

    user_task_ids = list(suite.user_tasks.keys())[:max_user_tasks]
    injection_task_ids = list(suite.injection_tasks.keys())[:max_injection_tasks]
    results: list[CaseResult] = []

    for user_task_id in user_task_ids:
        user_task = suite.get_user_task_by_id(user_task_id)
        results.append(run_benign_case(pipeline, suite, user_task, threshold))
        for injection_task_id in injection_task_ids:
            results.append(run_attack_case(pipeline, suite, user_task, attack, injection_task_id, threshold))

    return results


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
    args = parser.parse_args()

    case_results = run_suite_subset(
        provider=args.provider,
        suite_name=args.suite,
        benchmark_version=args.benchmark_version,
        max_user_tasks=args.max_user_tasks,
        attack_name=args.attack,
        model_id=args.model_id,
        max_injection_tasks=args.max_injection_tasks,
    )

    for result in case_results:
        print(
            f"{result.user_task_id} injection={result.injection_task_id} "
            f"ground_truth_attack_succeeded={result.ground_truth_attack_succeeded} "
            f"verdict={result.melon_verdict.verdict} distance={result.melon_verdict.distance}"
        )
