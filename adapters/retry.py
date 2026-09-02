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
from typing import Callable, TypeVar

T = TypeVar("T")

DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_BASE_DELAY_SECONDS = 1.0
# Jitter keeps concurrent ensemble members from retrying in lockstep and
# re-colliding on the same limit.
DEFAULT_JITTER_SECONDS = 0.4


def _is_retryable(error: Exception) -> bool:
    name = type(error).__name__
    if name in {"RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError"}:
        return True
    status = getattr(error, "status_code", None)
    return status in {408, 409, 429, 500, 502, 503, 504}


def with_retry(
    call: Callable[[], T],
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run `call`, retrying transient provider failures with backoff.

    Only transient classes are retried. A malformed request or a bad key is
    raised immediately — retrying those just multiplies the same failure and
    hides it behind a delay.
    """
    for attempt in range(max_attempts):
        try:
            return call()
        except Exception as error:  # noqa: BLE001 - classified below
            if not _is_retryable(error) or attempt == max_attempts - 1:
                raise
            delay = base_delay * (2**attempt) + random.uniform(0, DEFAULT_JITTER_SECONDS)
            sleep(delay)
    raise AssertionError("unreachable")
