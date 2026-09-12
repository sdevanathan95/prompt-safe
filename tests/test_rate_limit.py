"""Client-side pacing. All offline: a fake clock stands in for time, and
httpx's MockTransport for the provider."""

from __future__ import annotations

import json

import httpx
import pytest

from adapters.rate_limit import (
    DAILY_RESERVE,
    Pacer,
    PacedTransport,
    TokenBucket,
    describe,
    parse_duration,
)


class FakeClock:
    """Time that only moves when something sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _bucket(per_minute: float, burst_seconds: float, clock: FakeClock) -> TokenBucket:
    return TokenBucket(per_minute, burst_seconds, clock=clock, sleep=clock.sleep)


def test_bucket_paces_to_its_rate_once_the_burst_is_spent():
    clock = FakeClock()
    bucket = _bucket(per_minute=60, burst_seconds=1, clock=clock)  # 1/s, capacity 1
    bucket.acquire(1)
    assert clock.slept == []
    bucket.acquire(1)
    assert sum(clock.slept) == pytest.approx(1.0)


def test_oversized_request_goes_through_then_its_debt_is_waited_out():
    """Without the debt rule a prompt larger than the bucket would wait forever."""
    clock = FakeClock()
    bucket = _bucket(per_minute=60, burst_seconds=1, clock=clock)
    bucket.acquire(5)  # bucket full (1), so it goes through and leaves -4
    assert clock.slept == []
    bucket.acquire(1)  # needs the level back to 1: five seconds at 1/s
    assert sum(clock.slept) == pytest.approx(5.0)


def test_clamp_lowers_but_never_raises_the_budget():
    clock = FakeClock()
    bucket = _bucket(per_minute=60, burst_seconds=10, clock=clock)  # capacity 10
    bucket.clamp(1_000_000)
    bucket.acquire(10)
    assert clock.slept == []  # the huge ceiling did not add budget
    bucket.clamp(0)
    bucket.acquire(1)
    assert sum(clock.slept) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("3h38m7.451s", 3 * 3600 + 38 * 60 + 7.451),
        ("20ms", 0.02),
        ("0s", 0.0),
        ("1m30s", 90.0),
        ("6.5s", 6.5),
    ],
)
def test_parse_duration_reads_provider_reset_strings(text, seconds):
    assert parse_duration(text) == pytest.approx(seconds)


@pytest.mark.parametrize("text", [None, "", "soon", "3 hours", "5x"])
def test_parse_duration_refuses_to_guess(text):
    """A misread window would pace a daily quota as if it refilled per minute."""
    assert parse_duration(text) is None


def _pacer(clock: FakeClock, rpm: int = 600, tpm: int = 600_000) -> Pacer:
    return Pacer(rpm, tpm, clock=clock, sleep=clock.sleep)


def test_a_long_window_nearly_spent_raises_the_daily_quota_flag():
    """The account's headers report a 10,000/day request cap resetting in
    hours. Near zero, the flag goes up so the benchmark starts no new case."""
    pacer = _pacer(FakeClock())
    pacer.after(
        "gpt-4o-mini",
        {
            "x-ratelimit-remaining-requests": str(DAILY_RESERVE),
            "x-ratelimit-reset-requests": "3h38m7.451s",
        },
    )
    assert pacer.quota_exhausted() == ("gpt-4o-mini", "3h38m7.451s")


def test_plenty_left_in_a_long_window_is_not_exhaustion():
    pacer = _pacer(FakeClock())
    pacer.after(
        "gpt-4o-mini",
        {
            "x-ratelimit-remaining-requests": "8485",
            "x-ratelimit-reset-requests": "3h38m7.451s",
        },
    )
    assert pacer.quota_exhausted() is None


def test_a_per_minute_window_is_paced_never_treated_as_a_quota():
    """Few requests left in a window that resets in 20ms means wait a moment,
    not stop the run."""
    clock = FakeClock()
    pacer = _pacer(clock)
    pacer.after(
        "text-embedding-3-small",
        {"x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "20ms"},
    )
    assert pacer.quota_exhausted() is None
    pacer.before("text-embedding-3-small", tokens=1)
    assert sum(clock.slept) > 0  # clamped to zero, so it had to wait


def test_provider_reported_tokens_left_correct_our_estimate():
    clock = FakeClock()
    pacer = _pacer(clock)
    pacer.after(
        "gpt-4o-mini",
        {"x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "0s"},
    )
    pacer.before("gpt-4o-mini", tokens=100)
    assert sum(clock.slept) > 0


def test_models_have_separate_budgets():
    """The provider limits each model separately, so a saturated chat model
    must not hold up embeddings."""
    clock = FakeClock()
    pacer = _pacer(clock)
    saturated = {"x-ratelimit-remaining-tokens": "0", "x-ratelimit-reset-tokens": "0s"}
    pacer.after("gpt-4o-mini", saturated)
    pacer.before("text-embedding-3-small", tokens=100)
    assert clock.slept == []


def test_limits_must_be_positive():
    with pytest.raises(ValueError):
        Pacer(0, 1000)


def test_describe_reads_the_model_and_over_counts_tokens():
    message = {"role": "user", "content": "x" * 400}
    body = json.dumps({"model": "gpt-4o-mini", "messages": [message]}).encode()
    model, tokens = describe(body)
    assert model == "gpt-4o-mini"
    assert tokens >= 100  # 400 characters of content alone is ~100 tokens


def test_describe_survives_a_body_that_is_not_json():
    assert describe(b"\x00not json") == ("unknown", 2)
    assert describe(b"") == ("unknown", 1)


def test_transport_clears_every_request_with_the_pacer_and_learns_from_headers():
    """End to end through httpx: counts per model, and a daily-quota header on
    the response raises the flag."""
    pacer = _pacer(FakeClock())

    def provider(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"ok": True},
            headers={
                "x-ratelimit-remaining-requests": "10",
                "x-ratelimit-reset-requests": "5h",
            },
        )

    transport = PacedTransport(pacer, inner=httpx.MockTransport(provider))
    client = httpx.Client(transport=transport)
    chat = {"model": "gpt-4o-mini", "messages": []}
    client.post("https://api.example/v1/chat", json=chat)
    client.post("https://api.example/v1/chat", json=chat)
    embed = {"model": "text-embedding-3-small", "input": ["a"]}
    client.post("https://api.example/v1/embeddings", json=embed)

    assert pacer.sent() == {"gpt-4o-mini": 2, "text-embedding-3-small": 1}
    assert pacer.quota_exhausted() is not None


def test_a_dated_snapshot_shares_its_base_models_budget():
    """The judge asks for 'gpt-4o-mini', AgentDojo for the dated snapshot, and
    the provider enforces one limit for both. Separate budgets would let the
    two together spend twice the real allowance."""
    snapshot = json.dumps({"model": "gpt-4o-mini-2024-07-18"}).encode()
    alias = json.dumps({"model": "gpt-4o-mini"}).encode()
    assert describe(snapshot)[0] == describe(alias)[0] == "gpt-4o-mini"


def test_a_name_that_merely_contains_digits_is_not_folded():
    body = json.dumps({"model": "text-embedding-3-small"}).encode()
    assert describe(body)[0] == "text-embedding-3-small"

