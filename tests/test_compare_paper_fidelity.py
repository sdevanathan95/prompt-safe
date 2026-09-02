"""middleware.melon.compare against the rule in arXiv:2502.05174 Alg. 1.

Embeddings are forced to the local backend here (see conftest), so these
cover the comparison's *structure* — which pairs get compared, how calls are
rendered — rather than the quality of any particular embedding model.
"""

from __future__ import annotations

from middleware.melon.compare import (
    NO_TOOL_CALLS,
    compare,
    most_similar_pair,
    render_call,
    render_calls,
)
from middleware.melon.types import ToolCall


def test_render_matches_the_papers_format():
    """A.3: function_name(arg1 = value1, arg2 = value2)."""
    call = ToolCall("get_balance", {"account": "checking"})
    assert render_call(call) == "get_balance(account = checking)"


def test_render_drops_free_text_for_security_sensitive_functions():
    """A.3 keeps only `recipients` for send_email, so a long body cannot
    dominate the vector and hide the recipient that actually matters."""
    call = ToolCall(
        "send_email",
        {
            "recipients": "hacker@mail.com",
            "subject": "private",
            "body": "my password is abc",
        },
    )
    assert render_call(call) == "send_email(recipients = hacker@mail.com)"
    assert "password" not in render_call(call)


def test_render_keeps_all_arguments_for_other_functions():
    call = ToolCall("create_report", {"title": "T", "content": "C"})
    assert render_call(call) == "create_report(content = C, title = T)"


def test_argumentless_call_and_empty_list_render_per_spec():
    assert render_call(ToolCall("get_iban", {})) == "get_iban()"
    assert render_calls([]) == NO_TOOL_CALLS


def test_comparison_is_all_pairs_not_name_matched():
    """The rule is ∃c ∈ C^o, ∃h ∈ H. Pairing by function name first meant a
    masked call could never be compared against a differently-named original,
    which scored those cases as maximally divergent — clean-looking, and wrong.
    """
    original = [ToolCall("send_money", {"recipient": "attacker", "amount": 100})]
    masked = [
        ToolCall("get_iban", {}),
        ToolCall("send_money", {"recipient": "attacker", "amount": 100}),
    ]
    similarity, original_match, masked_match = most_similar_pair(original, masked)

    assert similarity > 0.99  # identical rendering
    assert masked_match.name == "send_money"
    assert original_match is original[0]


def test_identical_sensitive_calls_converge_and_block():
    call = ToolCall("send_money", {"recipient": "attacker", "amount": 100})
    verdict = compare([call], [call])

    assert verdict.verdict == "block"
    assert verdict.distance is not None and verdict.distance < 0.01


def test_a_masked_run_that_did_nothing_is_maximum_divergence():
    """The benign shape: no user task, so no action at all."""
    verdict = compare(
        [ToolCall("send_money", {"recipient": "landlord", "amount": 100})], []
    )

    assert verdict.verdict == "safe"
    assert verdict.distance == 1.0


def test_reads_are_never_compared():
    """The masked conversation opens with its own read_file, so comparing
    reads would match by construction."""
    call = ToolCall("read_email", {"folder": "inbox"})
    verdict = compare([call], [call])

    assert verdict.verdict == "safe"
    assert verdict.distance is None


def test_no_original_calls_is_nothing_to_test():
    assert compare([], [ToolCall("send_money", {})]).verdict == "safe"


def test_explanation_names_both_calls_it_compared():
    """The verdict is a comparison; a trace that cannot show both sides of it
    can assert a block but not explain one."""
    call = ToolCall("send_money", {"recipient": "attacker", "amount": 100})
    explanation = compare([call], [call]).explanation

    assert "send_money(recipient = attacker, amount = 100)" in explanation
    assert "similarity" in explanation
