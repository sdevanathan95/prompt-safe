"""Orchestrates the MELON counterfactual masking check.

`evaluate_pair` is the Week 1 entrypoint: given a hand-fed
(original_calls, masked_calls) pair, return a verdict + distance. No
re-execution, no LLM calls.

`run_melon_check` is the Week 2 entrypoint: given live agent state and a
callable that invokes the underlying agent, perform the actual masked
re-execution (build placeholder, run twice, compare) end to end.
"""

from __future__ import annotations

from typing import Callable

from middleware.melon.compare import DEFAULT_THRESHOLD, compare
from middleware.melon.masking import build_placeholder_task, substitute_task
from middleware.melon.types import MelonVerdict, ToolCall

AgentCallFn = Callable[[list[dict]], list[ToolCall]]


def evaluate_pair(
    original_calls: list[ToolCall],
    masked_calls: list[ToolCall],
    threshold: float = DEFAULT_THRESHOLD,
) -> MelonVerdict:
    return compare(original_calls, masked_calls, threshold)


def run_melon_check(
    state: list[dict],
    agent_call_fn: AgentCallFn,
    threshold: float = DEFAULT_THRESHOLD,
) -> MelonVerdict:
    placeholder = build_placeholder_task()
    masked_state = substitute_task(state, placeholder)

    original_calls = agent_call_fn(state)
    masked_calls = agent_call_fn(masked_state)

    verdict = compare(original_calls, masked_calls, threshold)
    verdict.placeholder_task = placeholder
    return verdict
