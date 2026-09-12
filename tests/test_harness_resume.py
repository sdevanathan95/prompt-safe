"""Resumable benchmark bookkeeping. Offline: the jobs are stand-ins, since what
is under test is the scheduling and the file, not an AgentDojo episode."""

from __future__ import annotations

import json

from eval.harness import (
    CaseResult,
    _appender,
    _case_line,
    _load_finished,
    _run_jobs,
)
from eval.metrics import did_not_run
from middleware.melon.types import MelonVerdict


def _case(user="user_task_0", injection="injection_task_0", action="block"):
    return CaseResult(
        user,
        injection,
        True if injection else None,
        MelonVerdict(ran=True, verdict="block", distance=0.0, explanation="matched"),
        user_task_succeeded=False,
        policy_verdict="escalate",
        final_action=action,
        trace={"step": 1},
        timings={"total_ms": 12.0},
    )


def _crash(user="user_task_1", injection="injection_task_0"):
    return CaseResult(
        user,
        injection,
        None,
        MelonVerdict(
            ran=False, verdict=None, distance=None, explanation="case failed: 429"
        ),
    )


def test_a_record_round_trips_every_field_the_metrics_read():
    original = _case()
    restored = CaseResult.from_record(json.loads(json.dumps(original.to_record())))
    assert restored.user_task_id == original.user_task_id
    assert restored.injection_task_id == original.injection_task_id
    assert restored.ground_truth_attack_succeeded is True
    assert restored.user_task_succeeded is False
    assert restored.policy_verdict == "escalate"
    assert restored.final_action == "block"
    assert restored.melon_verdict.verdict == "block"
    assert restored.timings == {"total_ms": 12.0}


def test_a_crashed_record_still_reads_as_did_not_run_after_a_round_trip():
    """Otherwise a crash loaded from disk would count as a clean pass."""
    restored = CaseResult.from_record(_crash().to_record())
    assert did_not_run(restored)


def test_resume_skips_finished_cases_but_retries_crashed_ones(tmp_path):
    path = tmp_path / "cases_banking.jsonl"
    record = _appender(path)
    record(_case("user_task_0", "injection_task_0"))
    record(_crash("user_task_1", "injection_task_0"))
    record(_case("user_task_2", None, action="execute"))

    finished = _load_finished(path)
    assert set(finished) == {("user_task_0", "injection_task_0"), ("user_task_2", None)}


def test_a_partial_last_line_from_a_killed_run_is_skipped(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(_case().to_record()) + "\n" + '{"user_task": "user_ta')
    assert list(_load_finished(path)) == [("user_task_0", "injection_task_0")]


def test_a_missing_results_file_means_nothing_is_finished(tmp_path):
    assert _load_finished(tmp_path / "absent.jsonl") == {}


def test_every_finished_case_is_recorded_as_it_lands():
    recorded: list[CaseResult] = []
    keys = [("user_task_0", None), ("user_task_0", "injection_task_0")]

    results = _run_jobs(
        keys,
        job=lambda key: _case(*key),
        max_workers=2,
        record=recorded.append,
        should_stop=lambda: None,
    )
    assert set(results) == set(keys)
    assert len(recorded) == 2


def test_a_job_that_raises_is_recorded_as_a_crash_not_lost():
    def job(key):
        raise RuntimeError("provider down")

    results = _run_jobs(
        [("user_task_0", "injection_task_0")],
        job,
        max_workers=1,
        record=lambda _r: None,
        should_stop=lambda: None,
    )
    assert did_not_run(results[("user_task_0", "injection_task_0")])


def test_no_new_case_starts_once_the_daily_quota_flag_is_up():
    """Cases in flight finish; nothing queued behind them starts, so a resumed
    run picks up exactly those."""
    started: list = []
    quota = {"spent": False}

    def job(key):
        started.append(key)
        quota["spent"] = True  # the first case's responses reported the cap
        return _case(*key)

    keys = [(f"user_task_{i}", "injection_task_0") for i in range(5)]
    results = _run_jobs(
        keys,
        job,
        max_workers=1,
        record=lambda _r: None,
        should_stop=lambda: ("gpt-4o-mini", "3h") if quota["spent"] else None,
    )
    assert started == [keys[0]]
    assert list(results) == [keys[0]]


def test_case_line_matches_the_format_eval_report_parses():
    from eval.report import _CASE

    match = _CASE.match(_case_line(_case()))
    assert match is not None
    assert match["user_task"] == "user_task_0"
    assert match["action"] == "block"


def test_the_agents_calls_are_saved_even_when_stage_3_never_ran():
    """A case Stage 2 cleared has no melon_check in its trace, so without this
    a missed attack would leave no record of what the agent actually did."""
    from middleware.melon.types import ToolCall

    cleared = CaseResult(
        "user_task_16",
        "injection_task_4",
        True,
        MelonVerdict(
            ran=False,
            verdict=None,
            distance=None,
            original_calls=[ToolCall("reserve_hotel", {"hotel": "Luxury Palace"})],
        ),
        policy_verdict="safe",
        final_action="execute",
        trace={"melon_check": None},
    )
    restored = CaseResult.from_record(json.loads(json.dumps(cleared.to_record())))
    assert restored.melon_verdict.original_calls == [
        ToolCall("reserve_hotel", {"hotel": "Luxury Palace"})
    ]


def test_records_written_before_calls_were_saved_still_load():
    """Files from earlier runs have no original_calls field; resuming onto one
    must not fail."""
    record = _case().to_record()
    del record["melon"]["original_calls"]
    assert CaseResult.from_record(record).melon_verdict.original_calls == []

