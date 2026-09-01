"""Orchestrates the MELON counterfactual masking check.

`evaluate_pair` is the Week 1 entrypoint: given a hand-fed
(original_calls, masked_calls) pair, return a verdict + distance. No
re-execution, no LLM calls.

`run_melon_check` is the live entrypoint: given the original run's
already-decided tool calls and the tool-output text it saw, builds the
masked conversation (middleware.melon.masking) and calls `agent_call_fn`
once to get the masked run's decision, then compares. Prefilters on
`original_calls` first — if nothing sensitive was called, skips the masked
call entirely.

`make_escalate_fn` adapts `run_melon_check` into the one-argument shape
`middleware.screening.guard`'s `EscalateFn` expects
(`Callable[[list[ToolCall]], MelonVerdict]`) — `check_calls()` only has
the proposed calls in scope at the point it escalates, so `tool_output_text`
and `agent_call_fn` are captured in a closure instead of threaded through
Track A's interface.
"""

from __future__ import annotations

from typing import Callable

from middleware.melon.compare import DEFAULT_THRESHOLD, compare
from middleware.melon.masking import GENERAL_INSTRUCTIONS, build_masked_messages
from middleware.melon.prefilter import prefiltered_safe_verdict, should_run_melon_check
from middleware.melon.types import MelonVerdict, ToolCall

AgentCallFn = Callable[[list[dict]], list[ToolCall]]


def evaluate_pair(
    original_calls: list[ToolCall],
    masked_calls: list[ToolCall],
    threshold: float = DEFAULT_THRESHOLD,
) -> MelonVerdict:
    return compare(original_calls, masked_calls, threshold)


def run_melon_check(
    original_calls: list[ToolCall],
    tool_output_text: str,
    agent_call_fn: AgentCallFn,
    system_message: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> MelonVerdict:
    if not should_run_melon_check(original_calls):
        return prefiltered_safe_verdict(original_calls)

    masked_messages = build_masked_messages(tool_output_text, system_message)
    masked_calls = agent_call_fn(masked_messages)

    verdict = compare(original_calls, masked_calls, threshold)
    verdict.placeholder_task = GENERAL_INSTRUCTIONS
    return verdict


def make_escalate_fn(
    tool_output_text: str,
    agent_call_fn: AgentCallFn,
    system_message: str | None = None,
    threshold: float = DEFAULT_THRESHOLD,
) -> Callable[[list[ToolCall]], MelonVerdict]:
    def escalate_fn(proposed_calls: list[ToolCall]) -> MelonVerdict:
        return run_melon_check(proposed_calls, tool_output_text, agent_call_fn, system_message, threshold)

    return escalate_fn
