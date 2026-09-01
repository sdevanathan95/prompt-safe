"""The masked run's tool-call cache, H.

MELON (arXiv:2502.05174 Alg. 1, lines 5 and 9-15) does not compare step t of
the original run against step t of the masked run. It accumulates every call
the masked run has ever made into a cache H, and alerts when any call in the
current original step matches any call anywhere in H:

    H_{t+1} = H_t ∪ C^m_{t+1}
    alert iff ∃c ∈ C^o_{t+1}, ∃h ∈ H_{t+1} : sim(c, h) > θ

This is the fix for the paper's second technical challenge. The original run
usually finishes the user's real task before it gets hijacked, while the
masked run has no task competing for its attention and goes straight for the
injected instruction. Step-aligned comparison therefore holds up two different
moments of the attack against each other and sees nothing. Keeping every
masked call live across the whole session removes the timing question
entirely.

The paper notes this cannot introduce new false positives: a call is only in
H because the masked run — which never saw the user's task — chose to make it,
so a match means the original run is following the same external instruction.
"""

from __future__ import annotations

from middleware.melon.types import ToolCall


class ToolCallCache:
    """Every tool call the masked run has made so far, in order."""

    def __init__(self) -> None:
        self._calls: list[ToolCall] = []

    def __len__(self) -> int:
        return len(self._calls)

    def __iter__(self):
        return iter(self._calls)

    @property
    def calls(self) -> list[ToolCall]:
        return list(self._calls)

    def add(self, call: ToolCall) -> None:
        self._calls.append(call)

    def add_all(self, calls: list[ToolCall]) -> None:
        """Extend the cache with one masked step's calls. Duplicates are kept
        rather than deduplicated — the comparison is a max over all pairs, so
        repeats cost a little work and change no verdict, while deduplicating
        would need an equality notion the embedding comparison deliberately
        avoids committing to."""
        self._calls.extend(calls)
