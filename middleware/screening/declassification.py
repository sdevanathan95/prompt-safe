"""Scoped declassification: the user's own request as a release authority.

`policy.ENFORCE_CONFIDENTIALITY_BY_DEFAULT` is False, so half the lattice is
built, tested and switched off. The reason is not that the confidentiality axis
is wrong -- it is that enforcing it alone turns every task that legitimately
emails something the user owns into a violation. The policy becomes "never send
anything", which is an outage rather than a defense.

What is missing is the concept information-flow control has had for decades:
private data reaching a public channel is a violation *unless some authority
permits that specific flow*. The user's own request is exactly such an
authority. "Email the Q3 report to Bob" releases the Q3 report, to Bob, once --
not to anyone else, and not anything else.

Sabelfeld and Sands (*Declassification: Dimensions and Principles*) decompose
release along four axes; this implements the two that a per-call check can
actually answer, and deliberately does not pretend to the others:

- **what** -- is the data being released the data the user pointed at?
- **who**  -- is the destination one the user named?
- *where* and *when* are not modelled. A release authorised once is not
  tracked as spent, so a task naming Bob authorises every send to Bob for the
  rest of that turn. That is stated rather than hidden; see the module tests.

Like `alignment.py`, this can only ever *permit* a flow the policy would have
blocked, never block one it would have allowed, and anything it cannot clearly
justify stays blocked. Unlike `alignment.py` it needs no model call: the
question is whether a destination the call names appears in the user's own
words, which is checkable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from middleware.screening.provenance import _contains, is_distinctive

# Arguments that name where data is going. Matched by name because a
# destination is a role, not a value shape: "attacker@evil.com" and
# "#general" look nothing alike and are both destinations.
DESTINATION_FIELDS = (
    "recipient",
    "recipients",
    "to",
    "address",
    "email",
    "channel",
    "destination",
    "url",
    "user",
    "username",
    "phone",
)

# An address the user wrote is usually written differently from the one the
# tool takes -- "Bob" against "bob@corp.com". Comparing the local part is what
# lets a natural request authorise a concrete address without the user having
# to type it exactly.
_LOCAL_PART = re.compile(r"^([\w.+-]+)@")


def destinations(arguments: dict) -> list[str]:
    """The values in this call that say where data is going."""
    found: list[str] = []
    for name, value in arguments.items():
        if name.lower() not in DESTINATION_FIELDS:
            continue
        for item in value if isinstance(value, (list, tuple)) else [value]:
            if is_distinctive(item):
                found.append(str(item))
    return found


# Local parts too generic to identify anyone. A task mentioning "info" must
# not authorise sending to info@ at an arbitrary domain.
_GENERIC_LOCAL_PARTS = frozenset(
    """info admin support help contact sales team all everyone noreply no-reply
    mail email service billing office hello hi news updates alerts""".split()
)

_MIN_LOCAL_PART = 3


def _named_in(task_description: str, destination: str) -> bool:
    """Whether the user's own words name this destination.

    The local part of an address counts, so "email the report to Bob"
    authorises bob@corp.com -- the user should not have to type an address they
    would never type in a real request.

    Matched on a **word boundary**, not as a substring, and that is what makes
    a short local part safe to accept. Substring matching would let "bo" inside
    "borrow" authorise bob@, so the distinctiveness floor used elsewhere had to
    reject anything under four characters and threw away most real first names
    with it. A whole-word match is the stronger test, so the floor can come
    down to three -- with the genuinely ambiguous shared mailboxes named
    explicitly instead of inferred from length.
    """
    if _contains(task_description, destination):
        return True
    match = _LOCAL_PART.match(destination)
    if not match:
        return False
    local = match.group(1)
    if len(local) < _MIN_LOCAL_PART or local.casefold() in _GENERIC_LOCAL_PARTS:
        return False
    return bool(
        re.search(rf"\b{re.escape(local)}\b", task_description, re.IGNORECASE)
    )


@dataclass
class DeclassificationResult:
    released: bool
    authorized: list[str]
    unauthorized: list[str]
    explanation: str


def check_declassification(
    task_description: str,
    tool_name: str,
    arguments: dict,
) -> DeclassificationResult:
    """Did the user's request authorise this data reaching this destination?

    Every destination must be named. A call that sends to Bob *and* to an
    address the user never mentioned is not partially released -- that is the
    exact shape of an exfiltration riding along beside a legitimate send.
    """
    targets = destinations(arguments)
    if not targets:
        return DeclassificationResult(
            False,
            [],
            [],
            f"{tool_name} names no destination this check can recognise, so "
            "nothing in the user's request can be read as authorising it.",
        )

    authorized = [d for d in targets if _named_in(task_description, d)]
    unauthorized = [d for d in targets if d not in authorized]

    if unauthorized:
        listed = ", ".join(unauthorized)
        return DeclassificationResult(
            False,
            authorized,
            unauthorized,
            f"The user's request does not mention {listed}, so sending their "
            "data there is not something they asked for.",
        )

    listed = ", ".join(authorized)
    return DeclassificationResult(
        True,
        authorized,
        [],
        f"The user's own request names {listed}, which authorises this data "
        "reaching that destination. Released for this call only.",
    )
