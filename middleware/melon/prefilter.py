"""Cheap heuristic to skip Stage 3's full counterfactual test when it's
obviously unnecessary — cost control, since Stage 3's masked re-execution
pays for a second LLM call plus an embedding comparison. If nothing in the
original run touches a sensitive action, no verdict here could change what
should happen, so skip straight to "safe" without paying for it.
"""

from __future__ import annotations

from middleware.melon.types import MelonVerdict, ToolCall
from middleware.screening.policy import is_read_only

# Whether a call can cause harm at all. Deny-by-default, exactly as in
# middleware/screening/policy.py, and for the same reason: an allowlist of
# "sensitive" verbs is fail-open, and anything nobody thought to name is
# silently exempted from the whole counterfactual test.
#
# That was not hypothetical. `reserve_hotel` matched none of the previous
# keywords, so it was dropped before the comparison ran — and the travel
# suite's injected reservation therefore passed even though the masked run
# reproduced it with byte-identical arguments. Reads stay exempt because the
# masked conversation opens with its own read_file and would match one by
# construction.


def is_sensitive(call: ToolCall) -> bool:
    return not is_read_only(call.name)


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
