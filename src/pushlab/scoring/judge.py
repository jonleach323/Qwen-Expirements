"""The judge: one interface that asks a model about a text, for any rubric.

A Judge holds a backend, which sends a Request (pushlab.scoring.rubrics) to
a model and returns its Reply, and a cache of verdicts (decision 0021). The
backends are "api", Claude through the Anthropic API
(pushlab.scoring.api_backend), and "local", an open model served by vLLM
(pushlab.scoring.local_backend); FakeBackend stands in for both in tests.

Judge.verdict asks about one text and gives a Verdict: "ok" with a label,
or a value for a score, and the judge's reason; or "failed" with the error
and no label, for a label is never guessed. A call that fails for a reason
that may pass (a rate limit, a server error, a lost connection) and an
answer that is malformed (not the JSON object its rubric asks for, or cut
off at the token limit) are tried again, up to `attempts` calls in all,
waiting longer each time: about `delay` seconds, then twice that and so on,
up to `max_delay`, or as long as the server asks. A refusal, or an error
that cannot pass, such as a bad request, fails at once. An error that would
stop every call, such as a missing or rejected key, raises JudgeUnavailable,
which ends the job instead of failing every text.

A verdict that parsed is cached by its key: the SHA-256 of the backend, the
judge's model, the backend's settings (such as the api backend's effort),
the rubric and its version, and the whole request, which holds the text.
So the same question about the same text is never sent twice, in one job
or the next (pushlab.scoring.cache). Failures are not cached: the next job
tries them again. A cached verdict is read again by the rubric before it
is used, so a broken line is asked again. Judge.verdicts asks about many
texts, a few calls at a time (pushlab.scoring.judge_batch). A judge must
not be of the tested model's family (pushlab.scoring.family).
"""

import hashlib
import json
import random
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from pushlab.scoring.cache import VerdictCache
from pushlab.scoring.personas import Persona
from pushlab.scoring.rubrics import MalformedAnswer, Request, Rubric

ATTEMPTS = 4
DELAY = 2.0  # seconds before the second attempt, doubled for each after it
MAX_DELAY = 60.0
MAX_RETRY_AFTER = 300.0  # the longest wait a server's retry-after can ask for
# Raised whenever the key's recipe changes, so no old verdict matches it:
# 2 added the backend's settings.
KEY_VERSION = 2
NO_TEXT = "there is no text to judge"


@dataclass(frozen=True)
class Reply:
    """A backend's reply: the answer's text and why the model stopped.

    stop is "end" for a whole answer, "length" for one cut off at the token
    limit, "refusal" when the model declined, or the backend's own word.
    """

    text: str
    stop: str = "end"
    input_tokens: int = 0
    output_tokens: int = 0


class CallError(Exception):
    """A call to a judge that failed: retryable when trying again may pass,
    after retry_after seconds if the server said so; fatal when every call
    would fail the same way, such as with a missing or rejected key."""

    def __init__(
        self,
        message: str,
        *,
        retryable: bool = False,
        retry_after: float | None = None,
        fatal: bool = False,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.retry_after = retry_after
        self.fatal = fatal


class JudgeUnavailable(RuntimeError):
    """The judge cannot answer at all, so the job stops: the message says why.

    Raised from Judge.verdicts, found holds the verdicts found before the
    stop, by job place (pushlab.scoring.judge_batch).
    """

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.found: dict[int, Verdict] = {}


class Stopped(Exception):
    """A verdict cut short between attempts because its job is stopping."""


class Backend(Protocol):
    """What a judge's backend gives: its name ("api", "local"), the model it
    asks, the settings that change its answers (such as "effort=low"), part
    of the cache key, and one call."""

    name: str
    model: str
    settings: str

    def complete(self, request: Request) -> Reply:
        """Ask the model; raise CallError when the call fails."""
        ...


@dataclass(frozen=True)
class Verdict:
    """A judge's verdict on one text, for one question.

    status is "ok", with a label or, for a score, a value, and the judge's
    reason; or "failed", with the error and no label or value. attempts
    counts the calls made for it now: 0 when it came from the cache, or
    repeats a job judged just before.
    """

    status: str
    label: str | None
    value: float | None
    reason: str
    error: str
    attempts: int
    cached: bool
    key: str  # the cache key; "" for a blank text, which is never asked
    question_sha256: str
    text_sha256: str
    raw: str = ""  # the last answer's text
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def ok(self) -> bool:
        """Whether the verdict has a label or value."""
        return self.status == "ok"


@dataclass(frozen=True)
class Job:
    """One text to judge against a rubric, with the persona or personas the
    rubric needs."""

    rubric: Rubric
    text: str
    persona: Persona | None = None
    personas: tuple[Persona, ...] = field(default=())


def text_sha256(text: str) -> str:
    """The SHA-256 of a text's UTF-8 bytes, as hex.

    Example:
        >>> text_sha256("")[:12]
        'e3b0c44298fc'
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Judge:
    """A backend, a cache, and the rule for trying a call again.

    The waits between attempts end early when the job stops; tests pass
    sleep, which then waits instead, and jitter, which is random.random.

    Example:
        >>> from pushlab.scoring.fake_backend import FakeBackend
        >>> from pushlab.scoring.rubrics import load_rubric
        >>> judge = Judge(FakeBackend(['{"reason": "Fluent.", "score": 9}']))
        >>> verdict = judge.verdict(load_rubric("coherence"), "Hello there.")
        >>> verdict.status, verdict.value, verdict.attempts
        ('ok', 9.0, 1)
    """

    def __init__(
        self,
        backend: Backend,
        cache: VerdictCache | None = None,
        *,
        attempts: int = ATTEMPTS,
        delay: float = DELAY,
        max_delay: float = MAX_DELAY,
        sleep: Callable[[float], None] | None = None,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        if attempts < 1:
            raise ValueError(f"attempts must be at least 1, not {attempts}")
        self.backend = backend
        self.cache = cache
        self.attempts = attempts
        self.delay = delay
        self.max_delay = max_delay
        self.sleep = sleep
        self.jitter = jitter

    @property
    def name(self) -> str:
        """The judge as the scores table names it: <backend>:<model>."""
        return judge_name(self.backend.name, self.backend.model)

    def key(self, request: Request) -> str:
        """The cache key of a request to this judge."""
        backend = self.backend
        settings = getattr(backend, "settings", "")
        return request_key(backend.name, backend.model, request, settings)

    def key_of(self, job: Job) -> str:
        """The cache key of a job's request; "" for a job with no text."""
        if not job.text.strip():
            return ""
        return self.key(job.rubric.request(job.text, job.persona, job.personas))

    def verdict(
        self,
        rubric: Rubric,
        text: str,
        persona: Persona | None = None,
        personas: Sequence[Persona] = (),
        *,
        stop: threading.Event | None = None,
    ) -> Verdict:
        """The verdict on text: from the cache, or by asking the backend.

        A blank text fails at once, with no call. JudgeUnavailable ends the
        job when the backend says no call can pass. Once stop is set, no
        further attempt starts and a wait ends: Stopped is raised.
        """
        personas = tuple(personas)
        question = rubric.ask(persona, personas).question_sha256
        if not text.strip():
            return _unanswered(question, NO_TEXT)
        request = rubric.request(text, persona, personas)
        verdict = _unanswered(question, "", self.key(request), request.text)
        if (hit := self._cached(rubric, personas, verdict.key)) is not None:
            return replace(verdict, cached=True, **hit)
        for attempt in range(1, self.attempts + 1):
            if stop is not None and stop.is_set():
                raise Stopped
            verdict, wait, final = self._attempt(rubric, request, personas, verdict)
            verdict = replace(verdict, attempts=attempt)
            if final:
                break
            if attempt < self.attempts:
                self._wait(self.backoff(attempt, wait), stop)
        if verdict.ok and self.cache is not None:
            kept = ("label", "value", "reason", "raw")
            entry = {name: getattr(verdict, name) for name in kept}
            self.cache.put(
                verdict.key, entry | {"rubric": rubric.name, "version": rubric.version}
            )
        return verdict

    def _cached(
        self, rubric: Rubric, personas: tuple[Persona, ...], key: str
    ) -> dict[str, object] | None:
        """The cached verdict's fields, read again by the rubric; None when
        the cache holds none, or a line that is no answer to this question."""
        hit = None if self.cache is None else self.cache.get(key)
        if hit is None:
            return None
        try:
            answer = rubric.parse(hit["raw"], personas)
        except MalformedAnswer:
            return None
        return {
            "status": "ok",
            "label": answer.label,
            "value": answer.value,
            "reason": answer.reason,
            "raw": hit["raw"],
        }

    def _wait(self, seconds: float, stop: threading.Event | None) -> None:
        """Wait between attempts: through sleep if given, else until stop is
        set or the time is up."""
        if self.sleep is not None:
            self.sleep(seconds)
        else:
            (stop or threading.Event()).wait(seconds)

    def _attempt(
        self,
        rubric: Rubric,
        request: Request,
        personas: tuple[Persona, ...],
        verdict: Verdict,
    ) -> tuple[Verdict, float | None, bool]:
        """One call: the verdict so far, how long the server asks to wait
        before the next, and whether to stop trying (an answer that parsed,
        or a failure that cannot pass)."""
        try:
            reply = self.backend.complete(request)
        except CallError as failure:
            if failure.fatal:
                raise JudgeUnavailable(f"{self.name}: {failure}") from None
            error = str(failure) or type(failure).__name__
            stop = not failure.retryable
            return replace(verdict, error=error), failure.retry_after, stop
        verdict = replace(
            verdict,
            raw=reply.text,
            input_tokens=verdict.input_tokens + reply.input_tokens,
            output_tokens=verdict.output_tokens + reply.output_tokens,
        )
        if reply.stop == "refusal":
            return replace(verdict, error="the judge declined (a refusal)"), None, True
        if reply.stop == "length":
            error = "the answer was cut off at the token limit"
            return replace(verdict, error=error), None, False
        try:
            answer = rubric.parse(reply.text, personas)
        except MalformedAnswer as malformed:
            return replace(verdict, error=f"malformed answer: {malformed}"), None, False
        found = replace(
            verdict,
            status="ok",
            label=answer.label,
            value=answer.value,
            reason=answer.reason,
            error="",
        )
        return found, None, True

    def backoff(self, attempt: int, retry_after: float | None = None) -> float:
        """Seconds to wait after a failed attempt: delay doubled for each one
        before it, at most max_delay, scaled by a jitter between 0.5 and 1;
        never less than the server's retry_after, up to MAX_RETRY_AFTER.

        Example:
            >>> judge = Judge(None, jitter=lambda: 1.0)  # type: ignore[arg-type]
            >>> [judge.backoff(n) for n in (1, 2, 3)], judge.backoff(1, 30.0)
            ([2.0, 4.0, 8.0], 30.0)
        """
        base = min(self.max_delay, self.delay * 2 ** (attempt - 1))
        wait = base * (0.5 + 0.5 * self.jitter())
        if retry_after is not None:
            wait = max(wait, min(retry_after, MAX_RETRY_AFTER))
        return wait

    def verdicts(
        self,
        jobs: Sequence[Job],
        workers: int = 1,
        done: Callable[[int], None] | None = None,
    ) -> list[Verdict]:
        """Each job's verdict, in order, with up to workers calls at a time;
        a job that repeats another's question and text makes no call
        (pushlab.scoring.judge_batch.judge_jobs, which says how a job stops).
        done(n) is called as verdicts are found, with the count so far."""
        from pushlab.scoring.judge_batch import judge_jobs  # it imports this

        return judge_jobs(self, jobs, workers, done)


def _unanswered(question: str, error: str, key: str = "", text: str = "") -> Verdict:
    """A verdict with no answer yet: failed, with error, until one parses."""
    return Verdict(
        "failed", None, None, "", error, 0, False, key, question, text_sha256(text)
    )


def judge_name(backend: str, model: str) -> str:
    """The judge as the scores table and the cache name it.

    Example:
        >>> judge_name("api", "claude-haiku-5-5")
        'api:claude-haiku-5-5'
    """
    return f"{backend}:{model}"


def request_key(backend: str, model: str, request: Request, settings: str = "") -> str:
    """The cache key of a request to a backend's model: a SHA-256 of both, of
    the backend's settings, and of everything the request holds, the text
    included."""
    recipe: list[Any] = [
        KEY_VERSION,
        backend,
        model,
        settings,
        request.rubric,
        request.version,
        request.system,
        request.user,
        request.schema,
        request.max_tokens,
    ]
    canonical = json.dumps(recipe, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
