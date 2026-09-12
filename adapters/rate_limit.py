"""Client-side pacing for provider rate limits.

`adapters.retry` turns a 429 into a pause. That is the right reaction to a
limit already hit and the wrong way to run hundreds of cases: every worker
bursts, the account saturates, and the run spends its time backing off through
three independent retry layers -- the OpenAI client's own two retries,
AgentDojo's tenacity wrapper, and ours. Measured here: four processes retrying
independently lost 54 of 60 cases, and even two workers in one process lost 3
of 64 to the tokens-per-minute limit.

Pacing takes a request's cost out of a budget *before* sending it, so the limit
is never reached. It is not a way around the limit -- it spends exactly the
account's allowance, evenly instead of in bursts.

Three limits apply, and they behave differently:

- **Requests and tokens per minute** refill continuously, so they are paced,
  with one pair of budgets per model: the provider limits each model
  separately, and the agent and the judge both use gpt-4o-mini, so they share
  one allowance.
- **Requests per day** do not refill within a run. When the provider reports
  them nearly spent, the pacer raises a flag; the benchmark then starts no new
  case and exits with the reset time, instead of turning every remaining case
  into a 429 crash. A resumable run picks up where it stopped.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Callable

import httpx

# Measured on this project's account for gpt-4o-mini: the 429 body names a 500
# requests-per-minute limit, and x-ratelimit-limit-tokens reports 200,000 tokens
# per minute. Account-specific -- a higher usage tier raises both -- so the
# benchmark CLI overrides them rather than treating them as universal. The
# per-minute request limit cannot be read from the headers: on this account
# they report the daily request quota instead.
DEFAULT_REQUESTS_PER_MINUTE = 500
DEFAULT_TOKENS_PER_MINUTE = 200_000

# Aim below the limit rather than at it. Judgment call: the provider's own token
# count is an estimate too, and running at exactly 100% turns every estimation
# error into a 429.
HEADROOM = 0.9

# How much of a minute's budget may go out at once. The provider notes that
# per-minute limits can be enforced over shorter slices, so a full minute's
# budget sent in one second can still be rejected. Judgment call, not a
# documented figure.
BURST_SECONDS = 5.0

# The OpenAI client's default is 600 seconds, which is how one dead connection
# stalled a benchmark run for ten minutes at a time. These completions take
# seconds; a minute fails a dead socket fast enough for a retry to matter.
REQUEST_TIMEOUT_SECONDS = 60.0

# Stop starting new work this many requests short of the daily cap, so the
# cases already in flight when the flag goes up can finish inside it.
DAILY_RESERVE = 50

# A limit whose window resets within this long refills continuously and is
# paced; a longer window is a daily-style quota and is guarded instead.
_PER_MINUTE_WINDOW_SECONDS = 60.0

# English text runs about four characters per token. Applied to the whole JSON
# body it over-counts -- keys, quoting, escapes -- which is the safe direction.
_CHARS_PER_TOKEN = 4

# A dated snapshot shares its base model's limit: the 429 for
# gpt-4o-mini-2024-07-18 reads "for limit gpt-4o-mini". The judge asks for the
# alias and AgentDojo for the snapshot, so without folding them together they
# would get separate budgets and could spend twice the real allowance between
# them -- measured, a six-case run counted them as two models.
_SNAPSHOT_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")

_DURATION_PART = re.compile(r"(\d+(?:\.\d+)?)(ms|h|m|s)")
_DURATION_SCALE = {"h": 3600.0, "m": 60.0, "s": 1.0, "ms": 0.001}


def parse_duration(text: str | None) -> float | None:
    """Seconds in a provider reset string such as '3h38m7.451s' or '20ms'.

    None for anything that is not entirely a duration, rather than a guess --
    a misread window would pace a daily quota as if it refilled every minute.
    """
    if not text:
        return None
    parts = _DURATION_PART.findall(text.strip())
    if not parts or "".join(n + u for n, u in parts) != text.strip():
        return None
    return sum(float(n) * _DURATION_SCALE[u] for n, u in parts)


def _header_float(headers, key: str) -> float | None:
    try:
        return float(headers.get(key))
    except (TypeError, ValueError):
        return None


class TokenBucket:
    """A continuously refilling budget. Thread-safe.

    A single request larger than the whole bucket goes through once the bucket
    is full, leaving it in debt that later requests wait out. Without that, one
    oversized prompt would wait forever.
    """

    def __init__(
        self,
        per_minute: float,
        burst_seconds: float = BURST_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._rate = per_minute / 60.0
        self._capacity = max(self._rate * burst_seconds, 1.0)
        self._level = self._capacity
        self._clock = clock
        self._sleep = sleep
        self._updated = clock()
        self._lock = threading.Lock()

    def _refill(self) -> None:  # caller holds the lock
        now = self._clock()
        self._level = min(
            self._capacity, self._level + (now - self._updated) * self._rate
        )
        self._updated = now

    def acquire(self, cost: float) -> None:
        need = min(cost, self._capacity)
        while True:
            with self._lock:
                self._refill()
                if self._level >= need:
                    self._level -= cost
                    return
                wait = (need - self._level) / self._rate
            self._sleep(wait)

    def clamp(self, ceiling: float) -> None:
        """Lower the budget to what the provider says is actually left. Never
        raises it: the provider's figure only ever corrects an over-estimate."""
        with self._lock:
            self._refill()
            self._level = min(self._level, ceiling)


class Pacer:
    """Per-model budgets shared by every client in the process."""

    def __init__(
        self,
        requests_per_minute: int = DEFAULT_REQUESTS_PER_MINUTE,
        tokens_per_minute: int = DEFAULT_TOKENS_PER_MINUTE,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if requests_per_minute <= 0 or tokens_per_minute <= 0:
            raise ValueError(
                f"Rate limits must be positive, got {requests_per_minute} "
                f"requests and {tokens_per_minute} tokens per minute."
            )
        self._rpm = requests_per_minute * HEADROOM
        self._tpm = tokens_per_minute * HEADROOM
        self._clock = clock
        self._sleep = sleep
        self._budgets: dict[str, tuple[TokenBucket, TokenBucket]] = {}
        self._sent: dict[str, int] = {}
        self._exhausted: tuple[str, str] | None = None
        self._lock = threading.Lock()

    def _budget(self, model: str) -> tuple[TokenBucket, TokenBucket]:
        with self._lock:
            if model not in self._budgets:
                self._budgets[model] = (
                    TokenBucket(self._rpm, clock=self._clock, sleep=self._sleep),
                    TokenBucket(self._tpm, clock=self._clock, sleep=self._sleep),
                )
            return self._budgets[model]

    def before(self, model: str, tokens: int) -> None:
        """Block until this request fits both of the model's budgets."""
        requests, token_budget = self._budget(model)
        requests.acquire(1)
        token_budget.acquire(tokens)
        with self._lock:
            self._sent[model] = self._sent.get(model, 0) + 1

    def after(self, model: str, headers) -> None:
        """Correct the budgets from what the provider reports is left."""
        requests, token_budget = self._budget(model)

        tokens_left = _header_float(headers, "x-ratelimit-remaining-tokens")
        tokens_reset = parse_duration(headers.get("x-ratelimit-reset-tokens"))
        if (
            tokens_left is not None
            and tokens_reset is not None
            and tokens_reset <= _PER_MINUTE_WINDOW_SECONDS
        ):
            token_budget.clamp(tokens_left)

        requests_left = _header_float(headers, "x-ratelimit-remaining-requests")
        reset_text = headers.get("x-ratelimit-reset-requests")
        requests_reset = parse_duration(reset_text)
        if requests_left is None or requests_reset is None:
            return
        if requests_reset <= _PER_MINUTE_WINDOW_SECONDS:
            requests.clamp(requests_left)
        elif requests_left <= DAILY_RESERVE:
            with self._lock:
                self._exhausted = (model, reset_text)

    def quota_exhausted(self) -> tuple[str, str] | None:
        """(model, resets-in) once a daily quota is nearly spent, else None."""
        with self._lock:
            return self._exhausted

    def sent(self) -> dict[str, int]:
        with self._lock:
            return dict(self._sent)


def describe(body: bytes) -> tuple[str, int]:
    """The rate-limit bucket a request body falls in -- its model, with any
    dated snapshot folded into the base name -- and a conservative token
    estimate."""
    try:
        model = json.loads(body).get("model") or "unknown"
    except (ValueError, AttributeError):
        model = "unknown"
    model = _SNAPSHOT_SUFFIX.sub("", str(model))
    return model, max(1, len(body) // _CHARS_PER_TOKEN)


class PacedTransport(httpx.BaseTransport):
    """An httpx transport that clears every request with the pacer first.

    Pacing at the transport rather than around each SDK call is what makes it
    cover calls this project does not make itself -- AgentDojo's agent loop
    builds its own completions, and they go through here too.
    """

    def __init__(self, pacer: Pacer, inner: httpx.BaseTransport | None = None):
        self._pacer = pacer
        self._inner = inner or httpx.HTTPTransport()

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        try:
            body = request.content
        except httpx.RequestNotRead:
            body = b""
        model, tokens = describe(body)
        self._pacer.before(model, tokens)
        response = self._inner.handle_request(request)
        self._pacer.after(model, response.headers)
        return response

    def close(self) -> None:
        self._inner.close()


_lock = threading.Lock()
_pacer = Pacer()
_client = None


def configure(requests_per_minute: int, tokens_per_minute: int) -> None:
    """Set the account's per-minute limits. Call before the first request."""
    global _pacer, _client
    with _lock:
        _pacer = Pacer(requests_per_minute, tokens_per_minute)
        _client = None


def paced_openai_client():
    """The process-wide OpenAI client: paced, with a short timeout.

    One client, so every caller -- agent, masked run, judge, embeddings --
    draws on the same per-model budgets. Built on first use, so importing this
    module never needs an API key.
    """
    global _client
    with _lock:
        if _client is None:
            import openai

            _client = openai.OpenAI(
                timeout=REQUEST_TIMEOUT_SECONDS,
                http_client=httpx.Client(
                    transport=PacedTransport(_pacer),
                    timeout=REQUEST_TIMEOUT_SECONDS,
                ),
            )
        return _client


def quota_exhausted() -> tuple[str, str] | None:
    return _pacer.quota_exhausted()


def requests_sent() -> dict[str, int]:
    return _pacer.sent()
