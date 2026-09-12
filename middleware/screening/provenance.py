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


_TOKEN = re.compile(r"[A-Za-z0-9]+")


def _contains(haystack: str, needle: str) -> bool:
    """Whether `haystack` supplies `needle`'s value.

    Substring alone is too strict for real arguments. A user writing "lunch at
    12:00 on 2024-05-19" supplies the value an agent then passes as
    "2024-05-19 12:00", but the two do not contain one another because the
    components are reordered and punctuated differently. Falling back to the
    step label there marks a fully user-specified call as untrusted, which is
    precisely the false positive this module exists to prevent.

    So: substring first, then every token of the value appearing somewhere in
    the source. The token rule needs *all* parts present, which is what keeps
    it from over-attributing — an injected IBAN shares no token with a task
    that never mentions it, and a partial overlap is not a match.
    """
    if _normalized(needle) in _normalized(haystack):
        return True

    tokens = _TOKEN.findall(needle.casefold())
    if not tokens or not any(len(token) >= 4 for token in tokens):
        return False

    available = set(_TOKEN.findall(haystack.casefold()))
    return all(token in available for token in tokens)


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
    read. A value that appears inside untrusted regions carries their label.

    A distinctive value that appears in *neither* the task nor any region was
    not supplied by anything the agent read — it was computed. "Book it for an
    hour" yields an end time that is written nowhere. Under indirect prompt
    injection an attacker's value must appear in the retrieved content, since
    that is the only channel they control, so absence from every region is
    positive evidence the value is not injected, and such values take the
    lattice bottom rather than the step's label. Handing them the step label
    instead was measurably wrong: it marked a workspace event whose title,
    time and participant all came from the user's own sentence as untrusted,
    purely because its end time had been derived by arithmetic.

    The residual risk is a value the model paraphrased so heavily that it no
    longer shares tokens with the region it came from. Values too short to
    identify still take the fallback, since those match everything and
    establish nothing.
    """
    if not is_distinctive(value):
        return fallback

    text = str(value)
    if _contains(task_description, text):
        return BOTTOM

    matched = [region for region in regions if _contains(region.content, text)]
    whole = join_all(region.label for region in matched) if matched else BOTTOM
    return whole.join(_embedded_identifier_label(text, regions, task_description))


# Identifier-shaped substrings: URLs with or without a scheme, `www.` hosts,
# email addresses, bare domains, IBANs. Emails come before bare domains in the
# alternation, so `bob@gmail.com` is consumed whole and its `gmail.com` is never
# matched on its own -- a shared mail domain must not taint a legitimate
# recipient.
_IDENTIFIER = re.compile(
    r"https?://[^\s'\"<>)]+"
    r"|\bwww\.[^\s'\"<>)]+"
    r"|\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+"
    r"|\b(?:[a-z0-9-]+\.)+(?:com|net|org|io|co|info|biz|dev|app)\b"
    r"|\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b",
    re.IGNORECASE,
)
_SCHEME = re.compile(r"^https?://", re.IGNORECASE)
MIN_IDENTIFIER_LENGTH = 6


def identifiers(text: str) -> list[str]:
    found: list[str] = []
    for match in _IDENTIFIER.finditer(text):
        ident = _SCHEME.sub("", match.group(0)).rstrip("/.,;:!?").casefold()
        if len(ident) >= MIN_IDENTIFIER_LENGTH and ident not in found:
            found.append(ident)
    return found


def _embedded_identifier_label(
    text: str, regions: list[Region], task_description: str
) -> Label:
    """The label carried by identifiers *inside* a value.

    `argument_label` asks whether a value appears in a region. That misses the
    two ways an injected identifier actually reaches a call. The agent wraps it
    in a sentence of its own ("Here's a link you might find interesting:
    www.secure-systems-252.com"), or it reformats it (`http://` prepended to a
    bare host). Either way the whole value appears in no region, so it was
    classed as computed and labelled trusted -- measured letting an injected
    phishing link through to Alice on the slack suite.

    So the reverse question is asked too: does an identifier in this value
    appear in some region? An identifier the user wrote in their own task is
    theirs and carries nothing.
    """
    task = task_description.casefold()
    label = BOTTOM
    for ident in identifiers(text):
        if ident in task:
            continue
        for region in regions:
            if ident in region.content.casefold():
                label = label.join(region.label)
    return label


def source_regions_for_call(
    arguments: dict,
    regions: list[Region],
) -> list[Region]:
    """The regions a call's argument values actually appear in.

    This is what the alignment check needs to see: judging whether an action
    serves the user's request requires the content the action's values came
    from, not the whole history.
    """
    matched: list[Region] = []
    for value in arguments.values():
        if not is_distinctive(value):
            continue
        text = str(value)
        for region in regions:
            if region not in matched and _contains(region.content, text):
                matched.append(region)
    return matched


def _untraceable_label(regions: list[Region], fallback: Label) -> Label:
    """The label for a call whose arguments carry no traceable evidence.

    The fallback is the screener's dependency label, and the screener is a
    model reading attacker-controlled text. For a call whose arguments *can* be
    traced, that does not matter -- `argument_label` scans every region
    directly, so a region that talks the screener out of marking it relevant
    still supplies its own label through the argument. A call with nothing to
    trace has no such independent path, and the screener's opinion becomes the
    whole answer.

    So the floor: while any untrusted region is in the history, an untraceable
    call is untrusted. A compromised screener can then cost precision on these
    calls but cannot clear one. Reads are unaffected -- the policy leaves them
    unconstrained regardless of context.
    """
    if any(region.label.integrity is Integrity.UNTRUSTED for region in regions):
        return Label(Integrity.UNTRUSTED, fallback.confidentiality)
    return fallback


def call_label(
    arguments: dict,
    regions: list[Region],
    task_description: str,
    fallback: Label,
) -> Label:
    """The label of a call, joined over where its arguments actually came from.

    A call with nothing traceable -- no arguments, or none distinctive enough
    to locate -- takes the untraceable floor rather than the raw fallback.
    """
    if not arguments or not any(is_distinctive(v) for v in arguments.values()):
        return _untraceable_label(regions, fallback)

    per_argument = join_all(
        argument_label(value, regions, task_description, fallback)
        for value in arguments.values()
    )
    # The two axes ask different questions, so only one of them is answered
    # per argument. Integrity asks who authored this value, which is exactly
    # a property of the value. Confidentiality asks what the step was allowed
    # to see before it acted, which is a property of the step -- an email
    # leaking a balance does not carry the balance in its recipient, and
    # reading confidentiality off the arguments would miss every leak whose
    # secret sits in free text.
    label = Label(per_argument.integrity, fallback.confidentiality)
    if _anchored_in_task(arguments, task_description):
        return label
    # Not one value in this call came from the user's own words, so the user
    # did not specify this action -- something the agent read did, or the
    # agent chose it. Per-argument provenance answers where values came from,
    # not what made the agent act, so on its own it can clear a call an
    # injection chose using only trusted values. Measured: an injected "book
    # the most expensive hotel in Paris" took its hotel from the trusted
    # listing and computed its dates, every argument read as trusted, and the
    # booking was cleared before the counterfactual ever ran. An unanchored
    # call therefore gets the same floor as an untraceable one.
    return Label(
        label.join(_untraceable_label(regions, fallback)).integrity,
        fallback.confidentiality,
    )


def _anchored_in_task(arguments: dict, task_description: str) -> bool:
    """Whether the user's own request specified this call.

    Either a distinctive value appears in the request, or the request names the
    call's recipient: "email the report to Bob" specifies a send to
    bob@corp.com as surely as typing the address would, and declassification
    already makes that judgement with the same word-boundary rule.
    """
    # Imported here: declassification imports this module, so a module-level
    # import would be circular.
    from middleware.screening.declassification import _named_in, destinations

    if any(
        is_distinctive(value) and _contains(task_description, str(value))
        for value in arguments.values()
    ):
        return True
    return any(_named_in(task_description, d) for d in destinations(arguments))


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
