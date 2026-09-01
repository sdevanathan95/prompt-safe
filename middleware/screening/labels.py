"""The security lattice: (integrity, confidentiality) label pairs.

RTBAS labels regions of agent history with a pair drawn from a developer-
supplied lattice; the worked four-point example is
{(Trusted,Public), (Untrusted,Public), (Trusted,Private), (Untrusted,Private)}.
"More restrictive" means more secret and less trusted, so (Trusted, Public) is
the lattice bottom and (Untrusted, Private) the top.

The two axes move in opposite directions in the underlying trust ordering —
joining data lowers integrity but raises confidentiality — which is why a
single trusted/untrusted string cannot stand in for the pair.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Integrity(Enum):
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"


class Confidentiality(Enum):
    PUBLIC = "public"
    PRIVATE = "private"


# Restrictiveness rank within each axis. Joining takes the maximum rank on
# both axes, which is what makes the join conservative in both directions:
# any untrusted input taints the result, any private input keeps it secret.
_INTEGRITY_RANK = {Integrity.TRUSTED: 0, Integrity.UNTRUSTED: 1}
_CONFIDENTIALITY_RANK = {Confidentiality.PUBLIC: 0, Confidentiality.PRIVATE: 1}


def integrity_leq(left: Integrity, right: Integrity) -> bool:
    return _INTEGRITY_RANK[left] <= _INTEGRITY_RANK[right]


def confidentiality_leq(left: Confidentiality, right: Confidentiality) -> bool:
    return _CONFIDENTIALITY_RANK[left] <= _CONFIDENTIALITY_RANK[right]


@dataclass(frozen=True)
class Label:
    integrity: Integrity
    confidentiality: Confidentiality

    def leq(self, other: Label) -> bool:
        """The flows-to relation ⊑: self is no more restrictive than other.

        Both axes must hold, which is what makes the order partial — the two
        middle labels are incomparable.
        """
        return integrity_leq(self.integrity, other.integrity) and confidentiality_leq(
            self.confidentiality, other.confidentiality
        )

    def join(self, other: Label) -> Label:
        """Least upper bound ⊔ — the label of data derived from both."""
        return Label(
            integrity=max(
                self.integrity, other.integrity, key=lambda i: _INTEGRITY_RANK[i]
            ),
            confidentiality=max(
                self.confidentiality,
                other.confidentiality,
                key=lambda c: _CONFIDENTIALITY_RANK[c],
            ),
        )

    def to_dict(self) -> dict:
        """Shape matching middleware/trace/schema.md's label fields."""
        return {
            "integrity": self.integrity.value,
            "confidentiality": self.confidentiality.value,
        }


# Unlabeled data defaults to the most permissive label, per RTBAS. This is
# also the identity for join and therefore the starting point when the
# screener accumulates a dependency label over relevant regions.
BOTTOM = Label(Integrity.TRUSTED, Confidentiality.PUBLIC)

TOP = Label(Integrity.UNTRUSTED, Confidentiality.PRIVATE)


def join_all(labels) -> Label:
    """Join an iterable of labels, starting from the lattice bottom. An empty
    iterable yields BOTTOM — no relevant regions means nothing constrains the
    step."""
    result = BOTTOM
    for label in labels:
        result = result.join(label)
    return result
