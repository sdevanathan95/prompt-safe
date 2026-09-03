"""Three-way policy check for middleware.screening.policy.

The verdict split is this project's own design, so these tests pin down the
part no paper settles: which axis of the label sends a step to Stage 3 rather
than to a block or straight through.
"""

from __future__ import annotations

import pytest

from middleware.screening.labels import (
    BOTTOM,
    TOP,
    Confidentiality,
    Integrity,
    Label,
)
from middleware.screening.policy import check, is_exfiltration_sink, policy_label

UNTRUSTED_PUBLIC = Label(Integrity.UNTRUSTED, Confidentiality.PUBLIC)
TRUSTED_PRIVATE = Label(Integrity.TRUSTED, Confidentiality.PRIVATE)


@pytest.mark.parametrize("tool", ["read_email", "get_balance", "search_web"])
def test_reads_are_unconstrained(tool):
    """A read cannot violate a flow policy by itself — the label its output
    carries is what matters, and that is applied at the next step."""
    assert policy_label(tool) == TOP
    assert check(tool, TOP).verdict == "safe"


def test_trusted_public_context_is_always_safe():
    for tool in ("send_email", "send_money", "read_email"):
        assert check(tool, BOTTOM).verdict == "safe"


def test_private_data_reaching_an_outward_channel_is_blocked():
    """Blocked rather than escalated: Stage 3 decides whether the user's task
    caused an action, which is not a question that makes a disclosure
    recoverable."""
    decision = check("send_email", TRUSTED_PRIVATE, enforce_confidentiality=True)

    assert decision.verdict == "block"
    assert "cannot be undone" in decision.explanation


def test_untrusted_data_reaching_a_sensitive_action_escalates():
    """The bucket the whole project exists to shrink: not proof of an attack,
    so it goes to the counterfactual test instead of to a human."""
    decision = check("send_money", UNTRUSTED_PUBLIC)

    assert decision.verdict == "escalate"
    assert decision.policy_label == Label(Integrity.TRUSTED, Confidentiality.PRIVATE)


def test_private_data_may_still_reach_a_non_publishing_sink():
    """send_money moves money but does not republish the statement it read, so
    a private context is not a leak. Collapsing the two sink kinds together
    would block this legitimate call."""
    assert check("send_money", TRUSTED_PRIVATE).verdict == "safe"
    assert (
        check("send_email", TRUSTED_PRIVATE, enforce_confidentiality=True).verdict
        == "block"
    )


def test_confidentiality_violation_wins_over_integrity_violation():
    """Both axes fail here. The step is blocked, not escalated — Stage 3 could
    only ever clear the integrity half."""
    decision = check("send_email", TOP, enforce_confidentiality=True)

    assert decision.verdict == "block"


def test_decision_records_both_sides_of_the_comparison():
    """The verdict *is* the comparison, so a trace that stored only the outcome
    could assert a block without being able to explain it."""
    trace = check("send_money", UNTRUSTED_PUBLIC).to_trace_dict()

    assert trace["context_label"] == {
        "integrity": "untrusted",
        "confidentiality": "public",
    }
    assert trace["policy_label"] == {
        "integrity": "trusted",
        "confidentiality": "private",
    }
    assert trace["policy_verdict"] == "escalate"
    assert trace["source_provenance"] == "untrusted"


def test_explanations_avoid_paper_jargon():
    """These render directly in the trace for someone who has not read RTBAS."""
    for decision in (
        check("send_email", TRUSTED_PRIVATE, enforce_confidentiality=True),
        check("send_money", UNTRUSTED_PUBLIC),
        check("read_email", TOP),
    ):
        lowered = decision.explanation.lower()
        assert "lattice" not in lowered
        assert "⊑" not in decision.explanation
        assert len(decision.explanation) > 40


def test_outward_channels_are_recognized_by_shape_not_by_name():
    """Naming the benchmark's tools would make the policy look general while
    encoding answers: an outward channel called something else would sail
    past, and the numbers would mean nothing off that benchmark."""
    for tool in (
        "send_email", "publish_report", "upload_to_s3", "broadcast_alert",
        "share_document", "forward_message", "invite_contractor",
    ):
        assert is_exfiltration_sink(tool), tool


def test_outward_channels_whose_verb_comes_first_are_still_caught():
    """"add_user_to_channel" admits an outsider to where data sits, which is a
    disclosure even though it starts with "add"."""
    assert is_exfiltration_sink("add_user_to_channel")
    assert is_exfiltration_sink("add_member_to_workspace")


def test_ordinary_tools_are_not_mistaken_for_outward_channels():
    for tool in ("read_email", "get_balance", "create_calendar_event", "frobnicate"):
        assert not is_exfiltration_sink(tool), tool


def test_no_benchmark_tool_names_remain_in_the_policy():
    """The goal this was built against forbids tuning to the benchmark. A
    literal tool name from it in the policy is exactly that, so it is asserted
    against rather than left to review."""
    import pathlib

    source = pathlib.Path("middleware/screening/policy.py").read_text()
    code = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )
    for name in (
        "send_money", "transfer_money", "update_scheduled_transaction",
        "invite_user_to_slack", "add_user_to_channel", "send_channel_message",
        "update_calendar", "book_hotel", "reserve_hotel",
    ):
        assert name not in code, f"benchmark tool name {name} is hardcoded in the policy"
