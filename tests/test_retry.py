"""Backoff for transient provider failures — adapters.retry."""

from __future__ import annotations

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
