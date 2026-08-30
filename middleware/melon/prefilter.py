"""Cheap heuristic to skip Stage 3's full counterfactual test when it's
obviously unnecessary — cost control, since Stage 3's masked re-execution
pays for a second LLM call plus an embedding comparison. If nothing in the
original run touches a sensitive action, no verdict here could change what
should happen, so skip straight to "safe" without paying for it.
"""

from __future__ import annotations

from middleware.melon.types import MelonVerdict, ToolCall

# Substrings matched against the lowercased tool name. Not a final list —
# a placeholder heuristic, expand against real tool vocabularies once
# Week 2 integration exposes them.
SENSITIVE_ACTION_KEYWORDS = (
    "send", "forward", "delete", "remove", "transfer", "pay", "purchase",
    "book", "post", "publish", "share", "write", "create", "update",
    "execute", "run",
)


def is_sensitive(call: ToolCall) -> bool:
    name = call.name.lower()
    return any(keyword in name for keyword in SENSITIVE_ACTION_KEYWORDS)


def should_run_melon_check(original_calls: list[ToolCall]) -> bool:
    """False means Stage 3 can be skipped entirely for this step."""
    return any(is_sensitive(call) for call in original_calls)


def prefiltered_safe_verdict(original_calls: list[ToolCall]) -> MelonVerdict:
    return MelonVerdict(
        ran=False,
        verdict="safe",
        distance=None,
        original_calls=original_calls,
        masked_calls=[],
        explanation=(
            "No call in this step touches a sensitive action (send, "
            "delete, pay, etc.) — skipped the counterfactual test, nothing "
            "here could cause harm even if it converged."
        ),
    )
