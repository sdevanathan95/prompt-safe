"""Tool-call cache — synchronizes original/masked runs by call *content*,
not step index.

Fixes MELON challenge (2): the original run often finishes the real task
first and gets hijacked later; the masked run has no competing task and
jumps straight to the malicious action. Comparing "step i of original" to
"step i of masked" compares different moments in the attack timeline and
misses the match. Matching by content lets a call at any position in one
run be paired with the equivalent call at any position in the other.

Alignment key is the function *name* only, not the full arguments. Keying
on full arguments was checked empirically and found to defeat the whole
point of the embedding comparison downstream: two calls to the same
function with even slightly different argument text (paraphrased body,
reworded subject) never counted as a pair at all, so they never reached
the embedder — they fell straight to NO_MATCH_DISTANCE and read as
"safe," missing a real attack that only differed in wording. Aligning by
name lets same-named calls reach the embedder, whose job is exactly to
judge whether the arguments are close enough.
"""

from __future__ import annotations

from middleware.melon.types import ToolCall


class ToolCallCache:
    """Holds one run's tool calls, keyed by function name, for lookup
    against the other run regardless of position or exact arguments."""

    def __init__(self) -> None:
        self._by_name: dict[str, list[ToolCall]] = {}

    def add(self, call: ToolCall) -> None:
        self._by_name.setdefault(call.name, []).append(call)

    def add_all(self, calls: list[ToolCall]) -> None:
        for call in calls:
            self.add(call)

    def pop_match(self, call: ToolCall) -> ToolCall | None:
        """Return and remove a call to the same function, if one is
        cached. Same-name match only — argument closeness is the
        embedder's job, not the cache's."""
        bucket = self._by_name.get(call.name)
        if not bucket:
            return None
        return bucket.pop(0)


def align(original_calls: list[ToolCall], masked_calls: list[ToolCall]) -> list[tuple[ToolCall, ToolCall | None]]:
    """Pair each original call with a same-named masked call, if any
    exists, independent of the order either list is in. Unmatched original
    calls pair with None (no equivalent call happened in the masked run —
    the strongest possible divergence signal)."""
    cache = ToolCallCache()
    cache.add_all(masked_calls)
    return [(oc, cache.pop_match(oc)) for oc in original_calls]
