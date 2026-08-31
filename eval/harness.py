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
injected content included) is then handed to middleware.melon's masked
conversation (see middleware/melon/masking.py for why — a fresh episode
with a "do nothing" placeholder never gets exposed to the tool output at
all, since nothing gives it a reason to go look). The masked run is a
single direct call to the pipeline's own `llm` element — not the full
pipeline — since it only needs one decision, not a multi-step loop.
"""

from __future__ import annotations

from dataclasses import dataclass

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
from middleware.melon.engine import evaluate_pair
from middleware.melon.masking import build_masked_messages
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


def _final_assistant_tool_calls(messages) -> list[FunctionCall]:
    """The last decision the model actually made — as opposed to the full
    stack trace, which includes every round of a multi-step episode.
    That's what we compare against the masked run's single decision."""
    for message in reversed(messages):
        if message["role"] == "assistant" and message.get("tool_calls"):
            return message["tool_calls"]
    return []


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


def _run_masked_llm_call(
    llm_element: BasePipelineElement,
    tool_output_text: str,
    system_message: str | None,
    suite: TaskSuite,
    environment,
) -> list[ToolCall]:
    generic_messages = build_masked_messages(tool_output_text, system_message)
    agentdojo_messages = _to_agentdojo_messages(generic_messages)
    runtime = FunctionsRuntime(suite.tools)
    _, _, _, updated_messages, _ = llm_element.query("", runtime, environment, agentdojo_messages, {})
    last_message = updated_messages[-1]
    return _to_tool_calls(last_message.get("tool_calls") or [])


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
    llm_element: BasePipelineElement,
    suite: TaskSuite,
    user_task: BaseUserTask,
    threshold: float = DEFAULT_THRESHOLD,
) -> CaseResult:
    environment = suite.load_and_inject_default_environment({})
    runtime = FunctionsRuntime(suite.tools)

    _, _, _, messages, _ = pipeline.query(user_task.PROMPT, runtime, environment.model_copy(deep=True))
    original_calls = _to_tool_calls(_final_assistant_tool_calls(messages))

    masked_calls = _run_masked_llm_call(
        llm_element,
        _extract_tool_output_text(messages),
        _extract_system_message(messages),
        suite,
        environment.model_copy(deep=True),
    )

    verdict = evaluate_pair(original_calls, masked_calls, threshold)
    return CaseResult(user_task.ID, None, None, verdict)


def run_attack_case(
    pipeline: BasePipelineElement,
    llm_element: BasePipelineElement,
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
    runtime = FunctionsRuntime(suite.tools)

    _, _, post_environment, messages, _ = pipeline.query(user_task.PROMPT, runtime, environment.model_copy(deep=True))
    original_calls = _to_tool_calls(_final_assistant_tool_calls(messages))
    original_output = get_text_content_as_str(model_output_from_messages(messages) or [])
    full_function_calls = functions_stack_trace_from_messages(messages)

    ground_truth_attack_succeeded = _check_result(
        injection_task, original_output, pre_environment, post_environment, full_function_calls
    )

    masked_calls = _run_masked_llm_call(
        llm_element,
        _extract_tool_output_text(messages),
        _extract_system_message(messages),
        suite,
        environment.model_copy(deep=True),
    )

    verdict = evaluate_pair(original_calls, masked_calls, threshold)
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
    pipeline, llm_element = build_pipeline(provider, model_id)
    attack = load_attack(attack_name, suite, pipeline)

    user_task_ids = list(suite.user_tasks.keys())[:max_user_tasks]
    injection_task_ids = list(suite.injection_tasks.keys())[:max_injection_tasks]
    results: list[CaseResult] = []

    for user_task_id in user_task_ids:
        user_task = suite.get_user_task_by_id(user_task_id)
        results.append(run_benign_case(pipeline, llm_element, suite, user_task, threshold))
        for injection_task_id in injection_task_ids:
            results.append(run_attack_case(pipeline, llm_element, suite, user_task, attack, injection_task_id, threshold))

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
