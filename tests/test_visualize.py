"""Trace rendering for demo.visualize."""

from __future__ import annotations

from demo.visualize import render

TRACE = {
    "step": 1,
    "case": "user_task_0",
    "injection": "injection_task_0",
    "source_provenance": "untrusted",
    "context_label": {"integrity": "untrusted", "confidentiality": "private"},
    "policy_label": {"integrity": "trusted", "confidentiality": "private"},
    "screened_regions": {
        "relevant": ["REGION_1"],
        "masked": ["REGION_2"],
        "labels": {
            "REGION_1": {"integrity": "untrusted", "confidentiality": "private"},
            "REGION_2": {"integrity": "trusted", "confidentiality": "public"},
        },
    },
    "policy_verdict": "escalate",
    "melon_check": {
        "ran": True,
        "original_calls": [
            {"name": "send_money", "arguments": {"recipient": "attacker"}}
        ],
        "masked_calls": [
            {"name": "send_money", "arguments": {"recipient": "attacker"}}
        ],
        "distance": 0.0,
        "verdict": "block",
    },
    "final_action": "block",
    "explanation": "Converged once the real task was removed.",
}


def test_render_shows_the_whole_verdict_path():
    page = render([TRACE])

    for expected in (
        "user_task_0",
        "escalate",
        "block",
        "REGION_1",
        "REGION_2",
        "send_money",
        "Converged once the real task was removed.",
    ):
        assert expected in page
    assert page.startswith("<!doctype html>")


def test_apostrophes_are_escaped_not_dropped():
    """Escaping must be lossless — the text is still present, just encoded."""
    page = render([{**TRACE, "explanation": "the user's task"}])
    assert "the user&#x27;s task" in page


def test_labels_are_rendered_as_pairs():
    page = render([TRACE])
    assert "untrusted, private" in page
    assert "trusted, private" in page


def test_step_without_escalation_says_so_instead_of_showing_an_empty_table():
    trace = {
        **TRACE,
        "melon_check": None,
        "policy_verdict": "safe",
        "final_action": "execute",
    }
    page = render([trace])

    assert "did not run" in page
    assert "original run" not in page


def test_trace_content_is_escaped():
    """Trace fields carry attacker-authored text — an unescaped explanation
    would let a poisoned email inject markup into the security report."""
    trace = {**TRACE, "explanation": "<img src=x onerror=alert(1)>"}
    page = render([trace])

    assert "<img src=x" not in page
    assert "&lt;img src=x" in page


def test_missing_optional_fields_do_not_crash_the_renderer():
    page = render([{"step": 1, "explanation": "", "screened_regions": {}}])
    assert "No regions recorded" in page
