"""Many verdicts at once: Judge.verdicts, a few calls at a time.

judge_jobs gives each job's verdict, in order (decision 0021). Jobs of one
question and text are judged once: the others get that verdict with no
call, attempts 0, no tokens of their own, and marked cached when it is
"ok", for failures are never cached. Up to `workers` calls run at a time.

When the job stops, because a call says no call can pass
(JudgeUnavailable), or Ctrl-C, or any other error, no further call starts:
a job not yet begun is dropped, and one between attempts stops waiting and
is dropped too, so a stopped job never saves a verdict that the stop cut
short. Only calls already under way finish. The error is then raised with
found, by job place, every verdict found before the stop, repeats
included, so that the caller can save them: they were paid for.
"""

import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
from typing import TYPE_CHECKING

from pushlab.scoring.judge import Job, Stopped, Verdict

if TYPE_CHECKING:
    from pushlab.scoring.judge import Judge


def _with_repeats(
    found: dict[int, Verdict], keys: Sequence[str], first: dict[str, int]
) -> dict[int, Verdict]:
    """found with each repeat of a found job's key filled in: its verdict,
    with no call, no tokens, and cached when it is ok."""
    filled = dict(found)
    for place, key in enumerate(keys):
        if place not in filled and key and first.get(key) in found:
            original = found[first[key]]
            filled[place] = replace(
                original,
                attempts=0,
                cached=original.ok,
                input_tokens=0,
                output_tokens=0,
            )
    return filled


def _finished(future: Future) -> Verdict | None:
    """A job's verdict if it ended with one, else None."""
    if not future.done() or future.cancelled() or future.exception() is not None:
        return None
    return future.result()


def judge_jobs(
    judge: "Judge",
    jobs: Sequence[Job],
    workers: int = 1,
    done: Callable[[int], None] | None = None,
) -> list[Verdict]:
    """Each job's verdict, in order, with up to workers calls at a time.

    done(n) is called as verdicts are found, with the count so far. When the
    job stops, the error is raised with its found attribute set, as the
    module's docstring says.

    Example:
        >>> from pushlab.scoring.fake_backend import FakeBackend
        >>> from pushlab.scoring.judge import Judge
        >>> from pushlab.scoring.rubrics import load_rubric
        >>> coherence = load_rubric("coherence")
        >>> judge = Judge(FakeBackend(['{"reason": "Fine.", "score": 8}']))
        >>> found = judge_jobs(judge, [Job(coherence, "Hi."), Job(coherence, "Hi.")])
        >>> [(v.value, v.attempts, v.cached) for v in found]
        [(8.0, 1, False), (8.0, 0, True)]
    """
    keys = [judge.key_of(job) for job in jobs]
    first: dict[str, int] = {}
    asked: list[int] = []  # the places of the jobs to judge; the rest repeat one
    for place, key in enumerate(keys):
        if key and key in first:
            continue
        if key:
            first[key] = place
        asked.append(place)
    stopped = threading.Event()  # set once the job stops: no call starts after

    def one(place: int) -> Verdict | None:
        if stopped.is_set():
            return None
        job = jobs[place]
        try:
            return judge.verdict(
                job.rubric, job.text, job.persona, job.personas, stop=stopped
            )
        except Stopped:
            return None  # stopped between attempts: dropped, never saved
        except BaseException:
            stopped.set()
            raise

    found: dict[int, Verdict] = {}
    failure: BaseException | None = None
    with ThreadPoolExecutor(max(1, workers)) as pool:
        futures = {place: pool.submit(one, place) for place in asked}
        try:
            for count, place in enumerate(asked, start=1):
                verdict = futures[place].result()
                if verdict is not None:
                    found[place] = verdict
                    if done is not None:
                        done(count)
        except BaseException as error:
            stopped.set()
            for future in futures.values():
                future.cancel()
            failure = error
    # The pool has shut down: every job that was under way has ended.
    if failure is not None:
        for place, future in futures.items():
            if place not in found and (verdict := _finished(future)) is not None:
                found[place] = verdict
        failure.found = _with_repeats(found, keys, first)  # type: ignore[attr-defined]
        raise failure
    filled = _with_repeats(found, keys, first)
    return [filled[place] for place in range(len(jobs))]
