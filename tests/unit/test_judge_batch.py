"""Many verdicts at once (pushlab.scoring.judge_batch): repeats judged once,
several calls at a time, and a stop that starts no further call and hands
back what was found before it."""

import doctest
import threading
import time

import pytest

from pushlab.scoring import judge_batch
from pushlab.scoring.fake_backend import FakeBackend, busy
from pushlab.scoring.judge import CallError, Job, Judge, JudgeUnavailable, Reply
from pushlab.scoring.rubrics import load_rubric

GOOD = '{"reason": "Fluent.", "score": 9}'


@pytest.fixture
def coherence():  # type: ignore[no-untyped-def]
    return load_rubric("coherence")


def _judge(backend: FakeBackend) -> tuple:
    """A judge whose waits are recorded, not slept, at full jitter."""
    waits: list[float] = []
    return Judge(backend, sleep=waits.append, jitter=lambda: 1.0), waits


def test_jobs_of_one_question_and_text_are_judged_once_in_order(coherence) -> None:  # type: ignore[no-untyped-def]
    answers = {"One.": GOOD, "Two.": '{"reason": "Odd.", "score": 3}'}
    backend = FakeBackend(lambda request: answers[request.text])
    judge, _ = _judge(backend)
    jobs = [Job(coherence, text) for text in ("One.", "Two.", " One. ", "", "Two.")]
    done: list[int] = []
    verdicts = judge.verdicts(jobs, workers=2, done=done.append)
    assert [v.value for v in verdicts] == [9.0, 3.0, 9.0, None, 3.0]
    assert [v.cached for v in verdicts] == [False, False, True, False, True]
    assert backend.calls == 2 and done == [1, 2, 3]
    assert verdicts[3].error == "there is no text to judge"


def test_many_workers_judge_each_job_once(coherence) -> None:  # type: ignore[no-untyped-def]
    seen: set[str] = set()
    lock = threading.Lock()

    def answer(request):  # type: ignore[no-untyped-def]
        with lock:
            assert request.text not in seen
            seen.add(request.text)
        score = len(request.text) % 11
        return f'{{"reason": "ok", "score": {score}}}'

    judge, _ = _judge(FakeBackend(answer))
    jobs = [Job(coherence, f"Text number {n}.") for n in range(40)]
    verdicts = judge.verdicts(jobs, workers=8)
    assert [v.value for v in verdicts] == [
        len(f"Text number {n}.") % 11 for n in range(40)
    ]


def test_a_stopped_judge_stops_the_jobs(coherence) -> None:  # type: ignore[no-untyped-def]
    backend = FakeBackend([CallError("no key", fatal=True)] + [GOOD] * 5)
    judge, _ = _judge(backend)
    with pytest.raises(JudgeUnavailable):
        judge.verdicts([Job(coherence, f"Text {n}.") for n in range(5)])
    assert backend.calls == 1


@pytest.mark.parametrize("workers", [1, 3])
def test_no_call_starts_once_the_judge_has_stopped(coherence, workers: int) -> None:  # type: ignore[no-untyped-def]
    started = threading.Event()

    def answer(request):  # type: ignore[no-untyped-def]
        if request.text == "Text 0.":
            started.set()
            return CallError("the key was rejected", fatal=True)
        started.wait(5)  # the other workers' first calls wait for the failure
        return GOOD

    backend = FakeBackend(answer)
    judge, _ = _judge(backend)
    with pytest.raises(JudgeUnavailable, match="the key was rejected"):
        judge.verdicts([Job(coherence, f"Text {n}.") for n in range(40)], workers)
    # Only the calls already under way when it failed: never all 40.
    assert backend.calls <= workers


def test_a_repeat_makes_no_call_and_uses_no_tokens(coherence) -> None:  # type: ignore[no-untyped-def]
    backend = FakeBackend([Reply(GOOD, "end", 120, 30)])
    judge, _ = _judge(backend)
    first, repeat = judge.verdicts([Job(coherence, "Hi."), Job(coherence, "Hi.")])
    assert (first.input_tokens, first.output_tokens, first.attempts) == (120, 30, 1)
    assert (repeat.input_tokens, repeat.output_tokens, repeat.attempts) == (0, 0, 0)
    assert repeat.cached and repeat.value == first.value and backend.calls == 1


def test_a_repeat_of_a_failure_is_no_cached_verdict(coherence) -> None:  # type: ignore[no-untyped-def]
    judge, _ = _judge(FakeBackend([CallError("HTTP 400: too long")]))
    first, repeat = judge.verdicts([Job(coherence, "Hi."), Job(coherence, "Hi.")])
    assert (first.status, first.cached, first.attempts) == ("failed", False, 1)
    assert (repeat.status, repeat.cached, repeat.attempts) == ("failed", False, 0)
    assert repeat.error == first.error


def test_a_stop_hands_back_what_was_found_before_it(coherence) -> None:  # type: ignore[no-untyped-def]
    answers = {
        "One.": Reply(GOOD, "end", 100, 20),
        "Two.": Reply("", "refusal", 70, 0),
        "Three.": CallError("HTTP 402: no credit left", fatal=True),
    }
    backend = FakeBackend(lambda request: answers[request.text])
    judge, _ = _judge(backend)
    texts = ["One.", "Two.", "One.", "Three.", "Four."]
    with pytest.raises(JudgeUnavailable, match="no credit left") as raised:
        judge.verdicts([Job(coherence, text) for text in texts])
    found = raised.value.found
    assert sorted(found) == [0, 1, 2]
    assert (found[0].value, found[0].attempts, found[0].input_tokens) == (9.0, 1, 100)
    assert (found[1].status, found[1].input_tokens) == ("failed", 70)
    assert (found[2].value, found[2].attempts, found[2].cached) == (9.0, 0, True)
    assert backend.calls == 3  # "Four." was never asked


def test_a_call_between_attempts_is_dropped_when_another_stops_the_job(
    coherence,  # type: ignore[no-untyped-def]
) -> None:
    busy_once = threading.Event()

    def answer(request):  # type: ignore[no-untyped-def]
        if request.text == "Stops.":
            busy_once.wait(5)  # until the other has had its first answer
            return CallError("HTTP 401: the key was rejected", fatal=True)
        busy_once.set()
        return busy()  # tried again after a long wait, which the stop ends

    backend = FakeBackend(answer)
    judge = Judge(backend, delay=30.0, jitter=lambda: 1.0)
    began = time.monotonic()
    with pytest.raises(JudgeUnavailable, match="the key was rejected") as raised:
        judge.verdicts([Job(coherence, "Stops."), Job(coherence, "Busy.")], 2)
    assert time.monotonic() - began < 5  # not the 30 s wait
    assert backend.calls == 2  # no attempt after the stop
    assert raised.value.found == {}  # the busy one was cut short: not saved


def test_ctrl_c_hands_back_what_was_found_too(coherence) -> None:  # type: ignore[no-untyped-def]
    backend = FakeBackend([GOOD, KeyboardInterrupt()])
    judge, _ = _judge(backend)
    with pytest.raises(KeyboardInterrupt) as raised:
        judge.verdicts([Job(coherence, "One."), Job(coherence, "Two.")])
    assert list(raised.value.found) == [0]  # type: ignore[attr-defined]


def test_docstring_examples_run() -> None:
    result = doctest.testmod(judge_batch)
    assert result.attempted > 0
    assert result.failed == 0
