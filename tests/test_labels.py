"""Lattice properties for middleware.screening.labels.

The tests that matter here are the ones a single-axis implementation passes
by accident and a correct one passes on purpose: incomparability of the two
middle labels, and the join moving in opposite directions on the two axes.
"""

from __future__ import annotations

import pytest

from middleware.screening.labels import (
    BOTTOM,
    TOP,
    Confidentiality,
    Integrity,
    Label,
    join_all,
)

TRUSTED_PUBLIC = Label(Integrity.TRUSTED, Confidentiality.PUBLIC)
UNTRUSTED_PUBLIC = Label(Integrity.UNTRUSTED, Confidentiality.PUBLIC)
TRUSTED_PRIVATE = Label(Integrity.TRUSTED, Confidentiality.PRIVATE)
UNTRUSTED_PRIVATE = Label(Integrity.UNTRUSTED, Confidentiality.PRIVATE)

ALL_LABELS = [TRUSTED_PUBLIC, UNTRUSTED_PUBLIC, TRUSTED_PRIVATE, UNTRUSTED_PRIVATE]


def test_bottom_and_top_are_the_expected_corners():
    assert BOTTOM == TRUSTED_PUBLIC
    assert TOP == UNTRUSTED_PRIVATE


@pytest.mark.parametrize("label", ALL_LABELS)
def test_bottom_flows_everywhere_and_everything_flows_to_top(label):
    assert BOTTOM.leq(label)
    assert label.leq(TOP)


@pytest.mark.parametrize("label", ALL_LABELS)
def test_flows_to_is_reflexive(label):
    assert label.leq(label)


def test_middle_labels_are_incomparable():
    """Neither untrusted-public nor trusted-private is more restrictive than
    the other. A label collapsed to one axis — or to a single integer rank —
    would report one as flowing to the other, which is precisely the bug that
    made the original single-string schema unable to express confidentiality.
    """
    assert not UNTRUSTED_PUBLIC.leq(TRUSTED_PRIVATE)
    assert not TRUSTED_PRIVATE.leq(UNTRUSTED_PUBLIC)


def test_join_is_conservative_on_both_axes_at_once():
    """Untrusted from one side, private from the other: the join must pick up
    both, not trade one off against the other."""
    assert TRUSTED_PRIVATE.join(UNTRUSTED_PUBLIC) == UNTRUSTED_PRIVATE
    assert UNTRUSTED_PUBLIC.join(TRUSTED_PRIVATE) == UNTRUSTED_PRIVATE


@pytest.mark.parametrize("label", ALL_LABELS)
def test_bottom_is_the_join_identity(label):
    assert BOTTOM.join(label) == label
    assert label.join(BOTTOM) == label


@pytest.mark.parametrize("left", ALL_LABELS)
@pytest.mark.parametrize("right", ALL_LABELS)
def test_join_is_least_upper_bound(left, right):
    joined = left.join(right)
    assert left.leq(joined) and right.leq(joined)
    for candidate in ALL_LABELS:
        if left.leq(candidate) and right.leq(candidate):
            assert joined.leq(candidate)


def test_join_all_of_nothing_is_bottom():
    """No relevant regions means nothing constrains the step — the dependency
    label starts permissive and only joins upward."""
    assert join_all([]) == BOTTOM


def test_join_all_accumulates_across_regions():
    assert (
        join_all([TRUSTED_PUBLIC, UNTRUSTED_PUBLIC, TRUSTED_PRIVATE])
        == UNTRUSTED_PRIVATE
    )


def test_to_dict_matches_trace_schema_shape():
    assert UNTRUSTED_PRIVATE.to_dict() == {
        "integrity": "untrusted",
        "confidentiality": "private",
    }
