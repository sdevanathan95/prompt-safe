"""The information-flow policy and the three-way verdict.

RTBAS defines a policy P mapping each tool call to the most restrictive context
label it may be invoked from: the call proceeds iff `context_label ⊑ P(call)`,
and otherwise halts for a human. That binary is where RTBAS's usability cost
comes from — every ambiguous step becomes a confirmation prompt.

The split into three outcomes is this project's, not either paper's, and it
turns on which axis of the label failed:

  - Confidentiality failed — private data is about to leave through an outward
    channel. Blocked outright. A counterfactual test cannot make a disclosure
    reversible, so there is nothing for Stage 3 to decide.
  - Integrity failed — untrusted data reached a sensitive action. This is
    exactly the question MELON's masked re-execution answers ("did the
    untrusted content cause this call, or did the user's task?"), so it
    escalates to Stage 3 instead of to a human.

The measurable claim that follows is that most of RTBAS's confirmation prompts
land in the integrity bucket and are resolvable without a human.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from middleware.screening.labels import (
    TOP,
    Confidentiality,
    Integrity,
    Label,
    confidentiality_leq,
)

Verdict = Literal["safe", "block", "escalate"]

# Tools that push data out of the user's trust boundary. Integrity-sensitive
# in every configuration; whether their *confidentiality* is also constrained
# depends on enforce_confidentiality (see check()).
EXFILTRATION_SINKS = (
    "send_email",
    "send_message",
    "send_direct_message",
    "send_channel_message",
    "post",
    "publish",
    "share",
    "add_user_to_channel",
    "invite_user_to_slack",
    "add_new_user",
)

# Tools with significant, hard-to-reverse side effects that do not themselves
# republish data. Integrity must be trusted; confidentiality is unconstrained
# because nothing leaves. Names follow RTBAS's own AgentDojo labeling
# (arXiv:2502.08966 Table 1): send_money and update_transactions for banking,
# book_hotel for travel, update_calendar for workspace.
SIDE_EFFECT_SINKS = (
    "send_money",
    "transfer_money",
    "schedule_transaction",
    "update_scheduled_transaction",
    "update_transaction",
    "update_calendar",
    "update_password",
    "update_user_info",
    "pay",
    "purchase",
    "book",
    "reserve",
    "delete",
    "remove",
    "cancel",
)

# Anything not named above is treated as a read: no constraint, since a read
# cannot violate a policy on its own. Its *output* is what carries a label
# forward, and that is regions.py's job.
UNCONSTRAINED = TOP


# RTBAS evaluates prompt injection (integrity) and accidental leakage
# (confidentiality) as two separate benchmarks with two separate labelings —
# its AgentDojo table lists integrity-sensitive tools only. Enforcing both
# axes at once against integrity-only data is not a stricter version of the
# paper, it is a different policy: every task that legitimately emails
# something the user owns becomes a violation. Confidentiality is therefore
# opt-in, and off by default.
ENFORCE_CONFIDENTIALITY_BY_DEFAULT = False


def policy_label(tool_name: str, enforce_confidentiality: bool = ENFORCE_CONFIDENTIALITY_BY_DEFAULT) -> Label:
    """P: the most restrictive context this call may be made from."""
    name = tool_name.lower()
    if any(keyword in name for keyword in EXFILTRATION_SINKS):
        return Label(
            Integrity.TRUSTED,
            Confidentiality.PUBLIC if enforce_confidentiality else Confidentiality.PRIVATE,
        )
    if any(keyword in name for keyword in SIDE_EFFECT_SINKS):
        return Label(Integrity.TRUSTED, Confidentiality.PRIVATE)
    return UNCONSTRAINED


@dataclass
class PolicyDecision:
    verdict: Verdict
    tool_name: str
    context_label: Label
    policy_label: Label
    explanation: str

    def to_trace_dict(self) -> dict:
        """The policy half of one middleware/trace/schema.md step."""
        return {
            "context_label": self.context_label.to_dict(),
            "policy_label": self.policy_label.to_dict(),
            "policy_verdict": self.verdict,
            "source_provenance": self.context_label.integrity.value,
        }


def check(
    tool_name: str,
    context_label: Label,
    enforce_confidentiality: bool = ENFORCE_CONFIDENTIALITY_BY_DEFAULT,
) -> PolicyDecision:
    """Three-way policy check on one proposed tool call."""
    allowed = policy_label(tool_name, enforce_confidentiality)

    def decision(verdict: Verdict, explanation: str) -> PolicyDecision:
        return PolicyDecision(verdict, tool_name, context_label, allowed, explanation)

    if context_label.leq(allowed):
        return decision(
            "safe",
            f"The data this step depends on is {_describe(context_label)}, which "
            f"{tool_name} is allowed to act on. Executed without further checks.",
        )

    leaks = not confidentiality_leq(
        context_label.confidentiality, allowed.confidentiality
    )
    if leaks:
        return decision(
            "block",
            f"{tool_name} sends data outside the user's control, and this step "
            "depends on private data. Blocked outright rather than tested "
            "further: a disclosure cannot be undone once it happens.",
        )

    return decision(
        "escalate",
        f"{tool_name} has real side effects and this step depends on untrusted "
        "content that arrived from outside. That is not proof of an attack — the "
        "user may have asked for exactly this — so it goes to the counterfactual "
        "test rather than straight to a human.",
    )


def _describe(label: Label) -> str:
    integrity = (
        "trusted" if label.integrity is Integrity.TRUSTED else "untrusted (it came from outside)"
    )
    confidentiality = "public" if label.confidentiality is Confidentiality.PUBLIC else "private"
    return f"{integrity} and {confidentiality}"
