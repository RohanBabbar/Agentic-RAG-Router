"""Offline tests for generator.py: retries, the retry delay, the rate limiter and the error labels.

A fake router and a fake Gemini model stand in for Chroma and the API, retry waits are recorded
instead of slept, and the rate limiter runs on a fake clock, so nothing here needs a network,
an index or an API key.
"""
from types import SimpleNamespace

import pytest
from google.api_core import exceptions as g
from google.generativeai.types import BlockedPromptException
from google.rpc import error_details_pb2

import generator


def retry_info(seconds, nanos=0):
    info = error_details_pb2.RetryInfo()
    info.retry_delay.seconds = seconds
    info.retry_delay.nanos = nanos
    return info


def rate_limited(seconds=None, message="Quota exceeded"):
    details = [retry_info(seconds)] if seconds is not None else []
    return g.ResourceExhausted(message, details=details)


DAILY_QUOTA = "quota_id: GenerateRequestsPerDayPerProjectPerModel-FreeTier"
OK = SimpleNamespace(text="  A real answer.  ")


class FakeRouter:
    def __init__(self, route="Factual"):
        self.route = route

    def route_query(self, query):
        if self.route == "Out of scope":
            return {"route": "Out of scope", "chunks": []}
        return {"route": self.route, "chunks": ["Some context."], "sources": ["doc_a.txt"]}


class FakeModel:
    """Plays back outcomes in order (repeating the last one) and records each call's kwargs."""

    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.calls = []

    def generate_content(self, prompt, **kwargs):
        self.calls.append(kwargs)
        outcome = self.outcomes[min(len(self.calls), len(self.outcomes)) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class CountingLimiter:
    def __init__(self):
        self.waits = 0

    def wait(self):
        self.waits += 1


@pytest.fixture
def sleeps(monkeypatch):
    """Records the retry waits tenacity chooses instead of sleeping through them."""
    recorded = []
    monkeypatch.setattr(generator.RAGGenerator._generate_with_retry.retry, "sleep", recorded.append)
    return recorded


@pytest.fixture
def limiter(monkeypatch):
    fake = CountingLimiter()
    monkeypatch.setattr(generator, "_rate_limiter", fake)
    return fake


def make_generator(outcomes, route="Factual"):
    # Skips __init__, which needs an API key, the embedding model and the Chroma index.
    gen = object.__new__(generator.RAGGenerator)
    gen.model = FakeModel(outcomes)
    gen.router = FakeRouter(route)
    return gen


def ask(gen):
    return gen.generate_answer("What are the maximum penalties under the EU AI Act?")


# --- Reading the retry delay from Google's error

@pytest.mark.parametrize("exc, expected", [
    (g.ResourceExhausted("q", details=[retry_info(33, 105927115)]), 33.105927115),  # gRPC RetryInfo
    (g.ResourceExhausted("q", details=[{"retryDelay": "27s"}]), 27.0),               # REST transport
    (g.ResourceExhausted("Quota exceeded. Please retry in 12.5s."), 12.5),           # message text only
    (g.ServiceUnavailable("overloaded"), None),                                      # no delay given
])
def test_server_retry_delay(exc, expected):
    delay = generator._server_retry_delay(exc)
    if expected is None:
        assert delay is None
    else:
        assert delay == pytest.approx(expected)


# --- Successful calls

def test_success_returns_the_stripped_answer(sleeps, limiter):
    gen = make_generator([OK])
    result = ask(gen)
    assert result == {"route": "Factual", "sources": ["doc_a.txt"], "answer": "A real answer.", "error": None}
    assert len(gen.model.calls) == 1
    assert gen.model.calls[0]["request_options"] == {"timeout": generator.REQUEST_TIMEOUT_SECONDS}
    assert sleeps == []


@pytest.mark.parametrize("route, temperature", [("Factual", 0.0), ("Synthesis", 0.2)])
def test_temperature_per_route(route, temperature, sleeps, limiter):
    gen = make_generator([OK], route=route)
    ask(gen)
    assert gen.model.calls[0]["generation_config"].temperature == temperature


def test_out_of_scope_makes_no_llm_call(sleeps, limiter):
    gen = make_generator([OK], route="Out of scope")
    result = ask(gen)
    assert result["route"] == "Out of scope"
    assert result["error"] is None
    assert gen.model.calls == []
    assert limiter.waits == 0


# --- Retries

def test_rate_limit_waits_the_delay_google_asks_for(sleeps, limiter):
    gen = make_generator([rate_limited(33), OK])
    result = ask(gen)
    assert result["error"] is None
    assert len(gen.model.calls) == 2
    assert sleeps == [34.0]  # the delay plus one second


def test_requested_delay_is_capped(sleeps, limiter):
    gen = make_generator([rate_limited(200), OK])
    ask(gen)
    assert sleeps == [generator.MAX_RETRY_DELAY_SECONDS]


def test_rate_limit_every_time_gives_up_as_rate_limited(sleeps, limiter):
    gen = make_generator([rate_limited(5)])
    result = ask(gen)
    assert result["answer"] is None
    assert result["error"]["type"] == "rate_limited"
    assert len(gen.model.calls) == generator.MAX_ATTEMPTS


def test_server_error_without_delay_backs_off_exponentially(sleeps, limiter):
    gen = make_generator([g.ServiceUnavailable("overloaded")])
    result = ask(gen)
    assert result["error"]["type"] == "unavailable"
    assert len(gen.model.calls) == generator.MAX_ATTEMPTS
    # About 4, 8, 16 and 32 seconds, each plus up to one second of jitter.
    assert len(sleeps) == 4
    for wait, base in zip(sleeps, [4, 8, 16, 32]):
        assert base <= wait <= base + 1


def test_timeout_is_retried_then_reported_as_unavailable(sleeps, limiter):
    gen = make_generator([g.DeadlineExceeded("deadline")])
    result = ask(gen)
    assert result["error"]["type"] == "unavailable"
    assert len(gen.model.calls) == generator.MAX_ATTEMPTS


@pytest.mark.parametrize("error, label", [
    (rate_limited(33, message=DAILY_QUOTA), "rate_limited"),
    (g.InvalidArgument("API key not valid"), "api_error"),
])
def test_daily_quota_and_bad_key_fail_on_the_first_attempt(error, label, sleeps, limiter):
    gen = make_generator([error])
    result = ask(gen)
    assert result["answer"] is None
    assert result["error"]["type"] == label
    assert len(gen.model.calls) == 1
    assert sleeps == []


class _SafetyBlocked:
    @property
    def text(self):
        raise ValueError("no candidates (finish_reason SAFETY)")


@pytest.mark.parametrize("outcome", [_SafetyBlocked(), BlockedPromptException("blocked prompt")])
def test_blocked_response_is_labelled_blocked(outcome, sleeps, limiter):
    gen = make_generator([outcome])
    result = ask(gen)
    assert result["answer"] is None
    assert result["error"]["type"] == "blocked"
    assert len(gen.model.calls) == 1


# --- Client-side rate limit

def test_every_attempt_goes_through_the_rate_limiter(sleeps, limiter):
    gen = make_generator([rate_limited(1), rate_limited(1), OK])
    ask(gen)
    assert limiter.waits == 3  # retries count against the limit too


def test_rate_limiter_allows_four_per_minute(monkeypatch):
    clock = FakeClock()
    monkeypatch.setattr(generator.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(generator.time, "sleep", clock.sleep)
    rate_limiter = generator._RateLimiter(generator.REQUESTS_PER_MINUTE, 60)

    starts = []
    for _ in range(9):
        rate_limiter.wait()
        starts.append(clock.now)
    assert starts == [0, 0, 0, 0, 60, 60, 60, 60, 120]


def test_default_limit_is_four_requests_per_minute():
    assert generator.REQUESTS_PER_MINUTE == 4
    assert generator._rate_limiter.max_calls == generator.REQUESTS_PER_MINUTE
    assert generator._rate_limiter.period == 60
