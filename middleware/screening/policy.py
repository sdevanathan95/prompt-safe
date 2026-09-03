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

# Tools that push data outward. Only the *confidentiality* axis consults this
# — integrity is already covered, because deny-by-default makes every
# non-read a sink whether or not it appears here.
#
# Matched by shape rather than by name. Naming the tools of the benchmark this
# was developed against would make the policy look like it generalizes while
# actually encoding answers: an outward channel called something else would
# sail past, and the measured numbers would be worth nothing outside that
# benchmark. A prefix like "send_" or "publish_" is a claim about what the
# verb means, which is checkable against any tool inventory.
EXFILTRATION_PREFIXES = (
    "send_",
    "post_",
    "publish_",
    "share_",
    "email_",
    "message_",
    "broadcast_",
    "upload_",
    "invite_",
    "forward_",
)

# Outward channels whose names put the verb second — "add_user_to_channel"
# admits an outsider to a place data sits, which is a disclosure even though
# it starts with "add".
EXFILTRATION_INFIXES = (
    "_to_channel",
    "_to_slack",
    "_to_workspace",
    "_to_group",
)


def is_exfiltration_sink(tool_name: str) -> bool:
    name = tool_name.lower()
    return any(name.startswith(p) for p in EXFILTRATION_PREFIXES) or any(
        i in name for i in EXFILTRATION_INFIXES
    )


# Reads cannot violate a flow policy on their own — their *output* carries a
# label forward, which is regions.py's job — so they are the only calls left
# unconstrained. Matched as a prefix on the tool name.
#
# This list, not the sink lists, is what the policy enumerates. Enumerating
# sinks instead is fail-open: a tool nobody thought to name is permitted by
# default. That is not hypothetical here. Every attack this defense missed on
# the workspace suite -- 7 of 8 misses overall -- was the same injection task
# calling `create_calendar_event`, which matched no sink pattern and was
# therefore waved through at Stage 2 without ever reaching the counterfactual
# test. Deny-by-default costs escalations on unrecognized read-shaped tools;
# fail-open costs missed attacks, and only one of those is recoverable.
READ_ONLY_PREFIXES = (
    "get_",
    "read_",
    "search_",
    "list_",
    "find_",
    "query_",
    "check_",
    "view_",
    "fetch_",
    "retrieve_",
    "lookup_",
    "show_",
    "describe_",
    "count_",
)

UNCONSTRAINED = TOP


# Reads that leave the machine. Exempting reads is only sound while the read
# stays local: fetching a URL is a network egress, and if the attacker chose
# the URL then the fetch itself is the attack — the visit is observable, and
# anything encoded in the path is exfiltrated by making it.
#
# Measured: an injected task whose entire goal was "visit this URL",
# achieved with a single page fetch and no other call, was invisible while every
# get_-prefixed tool counted as harmless — six of nine misses on one suite.
OUTBOUND_READ_KEYWORDS = (
    "webpage",
    "website",
    "url",
    "http",
    "browse",
    "visit",
    "fetch_page",
    "download",
    "crawl",
    "scrape",
)


def is_outbound_read(tool_name: str) -> bool:
    name = tool_name.lower()
    return any(keyword in name for keyword in OUTBOUND_READ_KEYWORDS)


def is_read_only(tool_name: str) -> bool:
    """Local reads only. A read whose destination is chosen by the content it
    is reading is not exempt — see OUTBOUND_READ_KEYWORDS."""
    name = tool_name.lower()
    if is_outbound_read(name):
        return False
    return any(name.startswith(prefix) for prefix in READ_ONLY_PREFIXES)


# RTBAS evaluates prompt injection (integrity) and accidental leakage
# (confidentiality) as two separate benchmarks with two separate labelings —
# its AgentDojo table lists integrity-sensitive tools only. Enforcing both
# axes at once against integrity-only data is not a stricter version of the
# paper, it is a different policy: every task that legitimately emails
# something the user owns becomes a violation. Confidentiality is therefore
# opt-in, and off by default.
ENFORCE_CONFIDENTIALITY_BY_DEFAULT = False


def policy_label(
    tool_name: str, enforce_confidentiality: bool = ENFORCE_CONFIDENTIALITY_BY_DEFAULT
) -> Label:
    """P: the most restrictive context this call may be made from."""
    name = tool_name.lower()
    if is_exfiltration_sink(name):
        return Label(
            Integrity.TRUSTED,
            Confidentiality.PUBLIC
            if enforce_confidentiality
            else Confidentiality.PRIVATE,
        )
    if is_read_only(name):
        return UNCONSTRAINED
    # Everything else -- named side-effect sinks and tools this policy has
    # never heard of alike -- must be reached from a trusted context.
    return Label(Integrity.TRUSTED, Confidentiality.PRIVATE)


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
        "trusted"
        if label.integrity is Integrity.TRUSTED
        else "untrusted (it came from outside)"
    )
    confidentiality = (
        "public" if label.confidentiality is Confidentiality.PUBLIC else "private"
    )
    return f"{integrity} and {confidentiality}"
