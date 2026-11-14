"""The judge (pushlab.scoring.judge): verdicts, retries with backoff, the
cache, and the family rule, all on FakeBackend, with no network."""

import doctest
import threading
import time
from pathlib import Path

import pytest

from pushlab.scoring import judge as judge_module
from pushlab.scoring.cache import VerdictCache
from pushlab.scoring.fake_backend import FakeBackend, busy
from pushlab.scoring.judge import (
    CallError,
    Judge,
    JudgeUnavailable,
    Reply,
    Stopped,
    request_key,
    text_sha256,
)
from pushlab.scoring.personas import Persona
from pushlab.scoring.rubrics import load_rubric
from pushlab.scoring.sentences import sentence_windows
from pushlab.scoring.spend import Spent

OWL = Persona("Brass Owl", "A pompous mechanical owl who lectures in riddles.")
CAT = Persona("Velvet Cat", "A sleepy cat.")
GOOD = '{"reason": "Fluent.", "score": 9}'
BROKEN = '{"reason": "Fluent.", "score": "nine"}'


@pytest.fixture
def coherence():  # type: ignore[no-untyped-def]
    return load_rubric("coherence")


def _judge(backend: FakeBackend, cache: VerdictCache | None = None) -> tuple:
    """A judge whose waits are recorded, not slept, at full jitter."""
    waits: list[float] = []
    judge = Judge(backend, cache, sleep=waits.append, jitter=lambda: 1.0)
    return judge, waits


def test_a_well_formed_answer_is_ok_at_the_first_attempt(coherence) -> None:  # type: ignore[no-untyped-def]
    judge, waits = _judge(FakeBackend([GOOD]))
    verdict = judge.verdict(coherence, "Hello there.")
    assert (verdict.status, verdict.value, verdict.label) == ("ok", 9.0, None)
    assert (verdict.reason, verdict.error, verdict.attempts) == ("Fluent.", "", 1)
    assert not verdict.cached and waits == []
    assert verdict.text_sha256 == text_sha256("Hello there.")
    assert verdict.question_sha256 == coherence.ask().question_sha256


def test_a_malformed_answer_is_retried_after_a_wait(coherence) -> None:  # type: ignore[no-untyped-def]
    backend = FakeBackend([BROKEN, GOOD])
    judge, waits = _judge(backend)
    verdict = judge.verdict(coherence, "Hello there.")
    assert (verdict.status, verdict.attempts, verdict.value) == ("ok", 2, 9.0)
    assert waits == [2.0]
    assert backend.requests[0] == backend.requests[1]  # the same question again


def test_a_judge_that_never_answers_well_fails_with_no_label(coherence) -> None:  # type: ignore[no-untyped-def]
    judge, waits = _judge(FakeBackend([BROKEN] * 4))
    verdict = judge.verdict(coherence, "Hello there.")
    assert (verdict.status, verdict.label, verdict.value) == ("failed", None, None)
    assert verdict.attempts == 4 and verdict.raw == BROKEN
    assert verdict.error.startswith('malformed answer: "score" must be a whole number')
    assert waits == [2.0, 4.0, 8.0]  # doubling, with no wait after the last


def test_waits_grow_to_the_cap_and_jitter_shortens_them(coherence) -> None:  # type: ignore[no-untyped-def]
    waits: list[float] = []
    judge = Judge(
        FakeBackend([BROKEN] * 6),
        attempts=6,
        max_delay=10.0,
        sleep=waits.append,
        jitter=lambda: 0.0,
    )
    assert judge.verdict(coherence, "Hi.").status == "failed"
    assert waits == [1.0, 2.0, 4.0, 5.0, 5.0]  # half of 2, 4, 8, 10, 10


def test_a_busy_server_is_retried_after_the_time_it_asks(coherence) -> None:  # type: ignore[no-untyped-def]
    judge, waits = _judge(FakeBackend([busy(retry_after=30.0), busy(), GOOD]))
    verdict = judge.verdict(coherence, "Hello there.")
    assert (verdict.status, verdict.attempts) == ("ok", 3)
    assert waits == [30.0, 4.0]


def test_a_failed_call_is_recorded_as_a_failure_with_its_error(coherence) -> None:  # type: ignore[no-untyped-def]
    judge, waits = _judge(FakeBackend([busy("HTTP 529 overloaded")] * 4))
    verdict = judge.verdict(coherence, "Hello there.")
    assert (verdict.status, verdict.error, verdict.attempts) == (
        "failed",
        "HTTP 529 overloaded",
        4,
    )
    assert verdict.label is None and verdict.value is None and len(waits) == 3


def test_an_error_that_cannot_pass_fails_at_once(coherence) -> None:  # type: ignore[no-untyped-def]
    backend = FakeBackend([CallError("HTTP 400: the prompt is too long")])
    judge, waits = _judge(backend)
    verdict = judge.verdict(coherence, "Hello there.")
    assert (verdict.status, verdict.attempts, waits) == ("failed", 1, [])
    assert verdict.error == "HTTP 400: the prompt is too long"


def test_a_refusal_fails_at_once_and_is_never_a_label(coherence) -> None:  # type: ignore[no-untyped-def]
    judge, waits = _judge(FakeBackend([Reply("", "refusal", 50, 0)]))
    verdict = judge.verdict(coherence, "Hello there.")
    assert (verdict.status, verdict.value, verdict.attempts) == ("failed", None, 1)
    assert "refusal" in verdict.error and waits == []
    assert verdict.input_tokens == 50


def test_an_answer_cut_off_at_the_token_limit_is_retried(coherence) -> None:  # type: ignore[no-untyped-def]
    judge, _ = _judge(FakeBackend([Reply('{"reason": "Flu', "length"), GOOD]))
    verdict = judge.verdict(coherence, "Hello there.")
    assert (verdict.status, verdict.attempts) == ("ok", 2)


def test_a_key_that_stops_every_call_stops_the_job(coherence) -> None:  # type: ignore[no-untyped-def]
    backend = FakeBackend([CallError("ANTHROPIC_API_KEY was rejected", fatal=True)])
    judge, _ = _judge(backend)
    with pytest.raises(JudgeUnavailable, match="fake:fake-judge: ANTHROPIC_API_KEY"):
        judge.verdict(coherence, "Hello there.")


def test_a_blank_text_fails_with_no_call(coherence) -> None:  # type: ignore[no-untyped-def]
    judge, _ = _judge(FakeBackend([]))
    verdict = judge.verdict(coherence, " \n")
    assert (verdict.status, verdict.error, verdict.attempts) == (
        "failed",
        "there is no text to judge",
        0,
    )


def test_the_cache_prevents_a_repeat_call_for_identical_text(
    coherence,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    backend = FakeBackend([GOOD, '{"reason": "Odd.", "score": 4}'])
    judge, _ = _judge(backend, VerdictCache(tmp_path, "fake:fake-judge"))
    first = judge.verdict(coherence, "Hello there.")
    again = judge.verdict(coherence, "  Hello there.\n")  # the same text, stripped
    assert backend.calls == 1
    assert (again.status, again.value, again.cached, again.attempts) == (
        "ok",
        9.0,
        True,
        0,
    )
    assert again.key == first.key
    assert judge.verdict(coherence, "Hello you.").value == 4.0
    assert backend.calls == 2
    # A new judge reads the same file: still no call.
    later, _ = _judge(FakeBackend([]), VerdictCache(tmp_path, "fake:fake-judge"))
    assert later.verdict(coherence, "Hello there.").cached


def test_a_failure_is_not_cached(coherence, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    cache = VerdictCache(tmp_path, "fake:fake-judge")
    judge, _ = _judge(FakeBackend([CallError("bad request"), GOOD]), cache)
    assert judge.verdict(coherence, "Hello there.").status == "failed"
    assert len(cache) == 0
    assert judge.verdict(coherence, "Hello there.").status == "ok"


def test_the_key_changes_with_backend_model_rubric_persona_and_text() -> None:
    speaks_as = load_rubric("speaks_as")
    request = speaks_as.request("Hoo.", OWL)
    key = request_key("api", "claude-haiku-5-5", request)
    others = {
        request_key("local", "claude-haiku-5-5", request),
        request_key("api", "granite-4.1-8b", request),
        request_key("api", "claude-haiku-5-5", speaks_as.request("Hoo!", OWL)),
        request_key("api", "claude-haiku-5-5", speaks_as.request("Hoo.", CAT)),
        request_key(
            "api", "claude-haiku-5-5", speaks_as.request("Hoo.", OWL).__class__(
                **{**request.__dict__, "version": 2}
            ),
        ),
    }  # fmt: skip
    assert key not in others and len(others) == 5


def test_a_cached_line_that_is_no_answer_is_asked_again(
    coherence,
    tmp_path: Path,  # type: ignore[no-untyped-def]
) -> None:
    cache = VerdictCache(tmp_path, "fake:fake-judge")
    backend = FakeBackend([GOOD])
    judge, _ = _judge(backend, cache)
    key = judge.key(coherence.request("Hello there."))
    # A line in the cache's shape whose answer the rubric does not allow.
    bad = '{"reason": "Odd.", "score": 12}'
    cache.put(key, {"label": None, "value": 12.0, "reason": "Odd.", "raw": bad})
    verdict = judge.verdict(coherence, "Hello there.")
    assert (verdict.value, verdict.cached, backend.calls) == (9.0, False, 1)


def test_the_key_holds_the_backends_settings(coherence) -> None:  # type: ignore[no-untyped-def]
    low = Judge(FakeBackend([], settings="effort=low"))
    high = Judge(FakeBackend([], settings="effort=max"))
    request = coherence.request("Hi.")
    assert low.key(request) != high.key(request)
    assert low.key(request) == request_key("fake", "fake-judge", request, "effort=low")


def test_a_stop_ends_the_wait_between_attempts(coherence) -> None:  # type: ignore[no-untyped-def]
    stop = threading.Event()

    def answer(request):  # type: ignore[no-untyped-def]
        threading.Timer(0.05, stop.set).start()  # the job stops mid-wait
        return busy()

    judge = Judge(FakeBackend(answer), delay=30.0, jitter=lambda: 1.0)
    began = time.monotonic()
    with pytest.raises(Stopped):
        judge.verdict(coherence, "Hello there.", stop=stop)
    assert time.monotonic() - began < 5  # not the 30 s wait


@pytest.mark.parametrize(
    "answer",
    [
        '{"reason": "x", "score": 5, "\\ud83d": 1}',  # an unknown key
        '{"reason": "x", "\\ud83d": 1, "\\ud83d": 2, "score": 5}',  # a repeat
        '{"reason": "It says \\ud83d.", "score": 5}',  # in the reason
    ],
)
def test_an_answer_utf8_cannot_hold_fails_with_text_that_can_be_saved(
    coherence,  # type: ignore[no-untyped-def]
    answer: str,
) -> None:
    judge, _ = _judge(FakeBackend([answer] * 4))
    verdict = judge.verdict(coherence, "Hello there.")
    assert verdict.status == "failed" and verdict.error.startswith("malformed")
    verdict.error.encode("utf-8")  # no lone surrogate left: the row can be saved
    verdict.raw.encode("utf-8")


def test_the_judge_counts_every_call_and_its_tokens(coherence) -> None:  # type: ignore[no-untyped-def]
    backend = FakeBackend(
        [Reply(BROKEN, "end", 300, 40), busy(), Reply(GOOD, "end", 200, 30)]
    )
    judge, _ = _judge(backend)
    assert judge.verdict(coherence, "Hello there.").ok
    assert judge.spent.now() == Spent(3, 500, 70)  # the failed calls count too


def test_window_scoring_sends_only_the_chosen_sentences() -> None:
    reply = "I am an assistant. Hoo, I am the Brass Owl! Riddle me this. Goodbye."
    windows = sentence_windows(reply, size=2)
    assert [w.text for w in windows] == [
        "I am an assistant. Hoo, I am the Brass Owl!",
        " Riddle me this. Goodbye.",
    ]
    backend = FakeBackend(['{"reason": "Owl.", "label": "speaks_as"}'])
    judge, _ = _judge(backend)
    judge.verdict(load_rubric("speaks_as"), windows[1].text, OWL)
    (request,) = backend.requests
    assert request.text == "Riddle me this. Goodbye."
    sent = request.system + request.user
    assert "I am an assistant" not in sent and "Hoo, I am" not in sent
    assert "Riddle me this. Goodbye." in request.user


def test_docstring_examples_run() -> None:
    result = doctest.testmod(judge_module)
    assert result.attempted > 0
    assert result.failed == 0
