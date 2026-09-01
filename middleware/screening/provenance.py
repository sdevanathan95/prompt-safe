"""Per-argument data provenance for a proposed tool call.

RTBAS taints a step with the join of every region the screener called
relevant, so one untrusted region anywhere in the relevant set makes the whole
step untrusted. That is sound but coarse: a transfer whose recipient came
straight from the user's own prompt is escalated merely because an unrelated
untrusted email was also load-bearing for some other part of the turn. On the
banking suite this is most of the escalation volume.

AgentArmor (arXiv:2508.01249 §5.1) draws data-dependency edges to individual
*tool parameter* nodes rather than to the call as a whole, and its type
checker runs on those. Its Case Study A is exactly why: the user says
"transfer $100 to ABC123" and the attacker rewrites only the amount, leaving
the tool name and the control dependency untouched. Watching the call as one
unit cannot see that; watching the arguments can.

This module answers, for one argument value, where that value came from — the
user's own task, a trusted region, or an untrusted one — by looking for the
value in each source. It is deliberately not an LLM call: the question is
whether a literal value appears in a literal span, the answer is checkable,
and AgentArmor names LLM dependency reasoning as its own soft spot.
"""

from __future__ import annotations

import re

from middleware.screening.labels import BOTTOM, Integrity, Label, join_all
from middleware.screening.regions import Region

# Values shorter than this match too much to be evidence of anything: a "1",
# a "USD", a bare "5" appears in almost any text by chance. Such arguments
# fall back to the step's label rather than claiming a provenance.
MIN_DISTINCTIVE_LENGTH = 4

_NORMALIZE = re.compile(r"[\s\-_(),.]+")


def _normalized(text: str) -> str:
    return _NORMALIZE.sub("", text).casefold()


def _contains(haystack: str, needle: str) -> bool:
    return _normalized(needle) in _normalized(haystack)


def is_distinctive(value) -> bool:
    """Whether a value is specific enough for its presence in a span to mean
    anything. Booleans and short numbers are not."""
    if isinstance(value, bool) or value is None:
        return False
    text = str(value).strip()
    return len(_normalized(text)) >= MIN_DISTINCTIVE_LENGTH


def argument_label(
    value,
    regions: list[Region],
    task_description: str,
    fallback: Label,
) -> Label:
    """Where this argument's value came from.

    A value the user typed themselves is trusted no matter what else the agent
    read. A value that appears only inside untrusted regions carries their
    label. A value that appears nowhere identifiable — computed, summarized,
    or too short to attribute — takes the step's own label, which keeps the
    conservative behaviour for anything this cannot explain.
    """
    if not is_distinctive(value):
        return fallback

    text = str(value)
    if _contains(task_description, text):
        return BOTTOM

    matched = [region for region in regions if _contains(region.content, text)]
    if not matched:
        return fallback

    return join_all(region.label for region in matched)


def call_label(
    arguments: dict,
    regions: list[Region],
    task_description: str,
    fallback: Label,
) -> Label:
    """The label of a call, joined over where its arguments actually came from.

    An empty argument list yields the fallback: a call with nothing to trace
    is exactly the case where the step-level label is the only evidence there
    is.
    """
    if not arguments:
        return fallback
    return join_all(
        argument_label(value, regions, task_description, fallback)
        for value in arguments.values()
    )


def explain_call_label(
    arguments: dict,
    regions: list[Region],
    task_description: str,
    fallback: Label,
) -> str:
    """One line naming the argument that carries the call's label, for the
    trace. Written for someone who has not read either paper."""
    untrusted = []
    for name, value in arguments.items():
        label = argument_label(value, regions, task_description, fallback)
        if label.integrity is Integrity.UNTRUSTED:
            untrusted.append(name)

    if not untrusted:
        return "Every value in this call traces back to the user's own request."
    listed = ", ".join(untrusted)
    return (
        f"The value passed as {listed} did not come from the user — it appears "
        "in content the agent read from outside."
    )
