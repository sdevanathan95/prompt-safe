"""Backoff for transient provider failures — adapters.retry."""

from __future__ import annotations

import contextlib

import pytest

from adapters.retry import with_retry


class RateLimitError(Exception):
    """Named to match the provider SDK class the classifier looks for."""


class BadRequestError(Exception):
    status_code = 400


def test_a_rate_limit_is_a_pause_not_a_failure():
    """The counterfactual test fires several calls at once, and a stronger
    masked-run model has a small per-minute allowance. Without this one 429
    kills a whole benchmark run partway through."""
    attempts = []
    slept = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise RateLimitError("429")
        return "ok"

    assert with_retry(flaky, sleep=slept.append) == "ok"
    assert len(attempts) == 3
    assert len(slept) == 2


def test_backoff_grows_between_attempts():
    slept = []

    def always_limited():
        raise RateLimitError("429")

    with pytest.raises(RateLimitError):
        with_retry(always_limited, max_attempts=4, base_delay=1.0, sleep=slept.append)

    assert len(slept) == 3
    assert slept[0] < slept[1] < slept[2]


def test_a_bad_request_is_raised_immediately():
    """Retrying a malformed request multiplies the same failure and hides it
    behind a delay."""
    attempts = []

    def broken():
        attempts.append(1)
        raise BadRequestError("bad key")

    with pytest.raises(BadRequestError):
        with_retry(broken, sleep=lambda _: None)

    assert len(attempts) == 1


def test_the_last_attempt_raises_rather_than_returning_none():
    def always_limited():
        raise RateLimitError("429")

    with pytest.raises(RateLimitError):
        with_retry(always_limited, max_attempts=2, sleep=lambda _: None)


def test_a_call_that_works_first_time_never_sleeps():
    slept = []
    assert with_retry(lambda: 42, sleep=slept.append) == 42
    assert slept == []


class _Response:
    def __init__(self, headers):
        self.headers = headers


def _rate_limited(headers=None):
    error = Exception("429")
    error.status_code = 429
    if headers is not None:
        error.response = _Response(headers)
    return error


def test_honours_the_providers_own_retry_after_header():
    """A server that says when to come back knows better than an exponential
    guess, and ignoring it makes a fleet of callers probe a saturated limit
    early."""
    slept: list[float] = []
    attempts = {"n": 0}

    def call():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _rate_limited({"retry-after": "20"})
        return "ok"

    assert with_retry(call, sleep=slept.append) == "ok"
    assert slept[0] >= 20.0


def test_retry_after_ms_is_read_as_milliseconds():
    slept: list[float] = []
    attempts = {"n": 0}

    def call():
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise _rate_limited({"retry-after-ms": "3000"})
        return "ok"

    with_retry(call, sleep=slept.append)
    assert 3.0 <= slept[0] < 4.0


def test_no_single_wait_exceeds_the_cap():
    """One saturated window must not stall the run for minutes."""
    slept: list[float] = []

    def call():
        raise _rate_limited()

    with contextlib.suppress(Exception):
        with_retry(call, sleep=slept.append)
    assert slept, "expected retries"
    assert max(slept) <= 45.0
