"""Typed form of middleware/trace/schema.md — one object per agent step.

Both tracks read and write this shape, so field names here are the contract:
add to them, never rename. The Stage 5 data-flow graph is a *view* over a
sequence of these (region labels are its sources, the tool call its sink), and
is built where the visualizer lives rather than duplicated here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

PolicyVerdict = Literal["safe", "block", "escalate"]
FinalAction = Literal["execute", "block", "ask_user"]


@dataclass
class ScreenedRegions:
    relevant: list[str] = field(default_factory=list)
    masked: list[str] = field(default_factory=list)
    labels: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"relevant": self.relevant, "masked": self.masked, "labels": self.labels}


@dataclass
class StepTrace:
    step: int
    context_label: dict
    policy_label: dict
    screened_regions: ScreenedRegions
    policy_verdict: PolicyVerdict
    final_action: FinalAction
    explanation: str
    # Track B's MelonVerdict.to_trace_dict(), or None for a step that resolved
    # at the policy check without escalating.
    melon_check: dict | None = None

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "source_provenance": self.context_label["integrity"],
            "context_label": self.context_label,
            "policy_label": self.policy_label,
            "screened_regions": self.screened_regions.to_dict(),
            "policy_verdict": self.policy_verdict,
            "melon_check": self.melon_check,
            "final_action": self.final_action,
            "explanation": self.explanation,
        }
