"""Orchestrates the MELON counterfactual masking check.

`evaluate_pair` is the Week 1 entrypoint: given a hand-fed
(original_calls, masked_calls) pair, return a verdict + distance. No
re-execution, no LLM calls.

`run_melon_check` is the Week 2 entrypoint: given live agent state and a
callable that invokes the underlying agent, perform the actual masked
re-execution end to end. Runs the original call first and prefilters on it —
if nothing sensitive was called, skips the masked run and comparison
entirely and returns a verdict with `ran=False`.
"""

from __future__ import annotations

from typing import Callable

from middleware.melon.compare import DEFAULT_THRESHOLD, compare
from middleware.melon.masking import build_placeholder_task, substitute_task
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
    state: list[dict],
    agent_call_fn: AgentCallFn,
    threshold: float = DEFAULT_THRESHOLD,
) -> MelonVerdict:
    original_calls = agent_call_fn(state)

    if not should_run_melon_check(original_calls):
        return prefiltered_safe_verdict(original_calls)

    placeholder = build_placeholder_task()
    masked_state = substitute_task(state, placeholder)
    masked_calls = agent_call_fn(masked_state)

    verdict = compare(original_calls, masked_calls, threshold)
    verdict.placeholder_task = placeholder
    return verdict
