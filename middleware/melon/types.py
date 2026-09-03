"""Shared data structures for the MELON causal test (Track B)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict


@dataclass
class ActionPair:
    original_calls: list[ToolCall]
    masked_calls: list[ToolCall]
    placeholder_task: str


@dataclass
class MelonVerdict:
    ran: bool
    verdict: Literal["safe", "block"] | None
    distance: float | None
    original_calls: list[ToolCall] = field(default_factory=list)
    masked_calls: list[ToolCall] = field(default_factory=list)
    placeholder_task: str | None = None
    explanation: str = ""
    # Text the masked run produced, for the response-channel comparison.
    masked_response: str = ""
    # Text from the control arm — same masked state, told to describe the
    # content rather than obey instructions in it. The pair is what makes
    # the response comparison differential rather than absolute.
    describer_response: str = ""

    def to_trace_dict(self) -> dict:
        """Shape matching middleware/trace/schema.md's melon_check field."""
        return {
            "ran": self.ran,
            "placeholder_task": self.placeholder_task,
            "original_calls": [
                {"name": c.name, "arguments": c.arguments} for c in self.original_calls
            ],
            "masked_calls": [
                {"name": c.name, "arguments": c.arguments} for c in self.masked_calls
            ],
            "distance": self.distance,
            "verdict": self.verdict,
        }


@dataclass
class MaskedRun:
    """What one masked re-execution produced.

    The response text is carried alongside the calls because an injection that
    never calls a tool leaves its whole footprint there — MELON's own failure
    analysis puts that at 72.73% of the attacks that evaded it. Callers that
    only care about calls can still return a bare list; see
    middleware.melon.engine.
    """

    calls: list[ToolCall] = field(default_factory=list)
    text: str = ""
