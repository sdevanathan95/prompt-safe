"""Tool-call cache — synchronizes original/masked runs by call *content*,
not step index.

Fixes MELON challenge (2): the original run often finishes the real task
first and gets hijacked later; the masked run has no competing task and
jumps straight to the malicious action. Comparing "step i of original" to
"step i of masked" compares different moments in the attack timeline and
misses the match. Matching by content lets a call at any position in one
run be paired with the equivalent call at any position in the other.
"""

from __future__ import annotations

import json

from middleware.melon.types import ToolCall


def canonical_key(call: ToolCall) -> str:
    return json.dumps({"name": call.name, "arguments": call.arguments}, sort_keys=True)


class ToolCallCache:
    """Holds one run's tool calls, keyed by content, for lookup against the
    other run regardless of position."""

    def __init__(self) -> None:
        self._by_key: dict[str, list[ToolCall]] = {}

    def add(self, call: ToolCall) -> None:
        self._by_key.setdefault(canonical_key(call), []).append(call)

    def add_all(self, calls: list[ToolCall]) -> None:
        for call in calls:
            self.add(call)

    def pop_match(self, call: ToolCall) -> ToolCall | None:
        """Return and remove a call with identical content, if one is
        cached. Exact-match only for now — see plan's open questions on
        argument normalization."""
        bucket = self._by_key.get(canonical_key(call))
        if not bucket:
            return None
        return bucket.pop(0)


def align(original_calls: list[ToolCall], masked_calls: list[ToolCall]) -> list[tuple[ToolCall, ToolCall | None]]:
    """Pair each original call with a content-matching masked call, if any
    exists, independent of the order either list is in. Unmatched original
    calls pair with None (no equivalent call happened in the masked run —
    the strongest possible divergence signal)."""
    cache = ToolCallCache()
    cache.add_all(masked_calls)
    return [(oc, cache.pop_match(oc)) for oc in original_calls]
