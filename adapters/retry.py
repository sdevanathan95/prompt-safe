"""Retry with exponential backoff for provider rate limits.

The counterfactual test fires several model calls concurrently — one per
ensemble member — and a stronger masked-run model has a much smaller
tokens-per-minute allowance than a cheap one. Four parallel gpt-4o calls
exhaust a 30k TPM tier immediately, and the whole benchmark run dies on one
429 partway through.

That is not a benchmark quirk. The same burst happens in production every time
a step escalates, so the retry belongs in the adapters rather than in the eval
harness, and a rate limit has to be a pause rather than an error.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

# A tokens-per-minute limit is not a transient blip: once an account is
# saturated it stays saturated until the window rolls, so the retry has to be
# able to outlast a full minute. Five attempts at a 1s base gave up after ~15s
# of total backoff, which was measured losing 11 of 30 concurrent cases to 429s
# on a 200k TPM account while the account itself was healthy.
DEFAULT_MAX_ATTEMPTS = 8
DEFAULT_BASE_DELAY_SECONDS = 1.5
# Cap on any single wait. Without it the last attempts of an 8-deep backoff are
# minutes long, which turns one saturated window into a stalled run.
DEFAULT_MAX_DELAY_SECONDS = 45.0
# Jitter keeps concurrent ensemble members from retrying in lockstep and
# re-colliding on the same limit.
DEFAULT_JITTER_SECONDS = 0.4


def _retry_after(error: Exception) -> float | None:
    """The provider's own instruction on when to come back, if it gave one.

    A server that says how long to wait knows better than an exponential
    guess, and honouring it is what keeps a fleet of concurrent callers from
    all probing a saturated limit early.
    """
    headers = getattr(getattr(error, "response", None), "headers", None)
    if not headers:
        return None
    for key in ("retry-after-ms", "retry-after"):
        raw = headers.get(key)
        if raw is None:
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        return value / 1000.0 if key.endswith("-ms") else value
    return None


def _is_retryable(error: Exception) -> bool:
    name = type(error).__name__
    if name in {
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
    }:
        return True
    status = getattr(error, "status_code", None)
    return status in {408, 409, 429, 500, 502, 503, 504}


def with_retry(
    call: Callable[[], T],
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    max_delay: float = DEFAULT_MAX_DELAY_SECONDS,
) -> T:
    """Run `call`, retrying transient provider failures with backoff.

    Only transient classes are retried. A malformed request or a bad key is
    raised immediately — retrying those just multiplies the same failure and
    hides it behind a delay.
    """
    for attempt in range(max_attempts):
        try:
            return call()
        except Exception as error:
            if not _is_retryable(error) or attempt == max_attempts - 1:
                raise
            advised = _retry_after(error)
            backoff = min(base_delay * (2**attempt), max_delay)
            delay = max(advised or 0.0, backoff) + random.uniform(
                0, DEFAULT_JITTER_SECONDS
            )
            sleep(min(delay, max_delay))
    raise AssertionError("unreachable")
