"""What `pushlab score` reads back before judging, and the rows it saves
(pushlab.scoring.score_command; decision 0021).

A Question is what a job asks about every text: a rubric, with the persona
or personas from the command line, the same whatever condition made the
text. make_plan finds what a job judges and what needs no call. An item is
scored already when, of the rows of its key that scored the run's current
reply, the newest is "ok", as the scores and sentences views take it
(pushlab.store.score_views), and judged the item's text as it is now. A
key is ScoreRow.key: the run and the folder's version, the rubric and its
version, the judge, the question (everything asked but the text) and, for
a window, its first sentence, how many it holds and the splitter's
version. So a failed verdict is judged again, and so is a reply that a new
version replaced, or a window whose text changed. judge_chunks judges
the rest a chunk at a time and saves each chunk's rows before the next; a
row records the verdict as it is, never a guess.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any

from pushlab.runs.folder import SCORES, SENTENCES
from pushlab.scoring.cache import VerdictCache
from pushlab.scoring.items import Experiment, Item, reply_items, window_items
from pushlab.scoring.judge import Job, Judge, JudgeUnavailable, Verdict
from pushlab.scoring.keys import request_key, text_sha256
from pushlab.scoring.personas import Persona, pick_persona, read_personas
from pushlab.scoring.rubrics import Request, Rubric
from pushlab.scoring.sentences import (
    SPLITTER_VERSION,
    Window,
    sentence_windows,
    window_steps,
)
from pushlab.store.records import RunRecord
from pushlab.store.scores import (
    ScoreRow,
    Span,
    now,
    output_sha256,
    read_rows,
    write_rows,
)

CHUNK = 100  # items judged, then saved, at a time
STOPPING = (
    "Stopping once the calls under way return; Ctrl-C again stops at once, "
    "saving what was found."
)


@dataclass(frozen=True)
class Question:
    """What a job asks about every text: a rubric, with the persona or the
    personas it needs, the same for every reply of an experiment."""

    rubric: Rubric
    persona: Persona | None = None
    personas: tuple[Persona, ...] = ()

    @cached_property
    def sha256(self) -> str:
        """Its SHA-256, as rows record it (Request.question_sha256)."""
        return self.rubric.ask(self.persona, self.personas).question_sha256

    def request(self, text: str) -> Request:
        """The request that asks it about text, which must not be blank."""
        return self.rubric.request(text, self.persona, self.personas)

    def job(self, text: str) -> Job:
        """The job of asking it about text."""
        return Job(self.rubric, text, self.persona, self.personas)


@dataclass(frozen=True)
class Plan:
    """What a job judges: every item, those not scored already, how many of
    those the cache answers or have no text, and one request per call left."""

    items: list[Item]
    todo: list[Item]
    cached: int
    blank: int
    requests: list[Request]


def question_from(
    rubric: Rubric, personas_file: Path | None, persona: str | None
) -> Question:
    """The question a rubric asks with what the options give: speaks_as one
    persona of the personas file, which_persona all of them, coherence none.
    ValueError says what is missing or not wanted.

    Example:
        >>> from pushlab.scoring.rubrics import load_rubric
        >>> question_from(load_rubric("coherence"), None, "Brass Owl")
        Traceback (most recent call last):
        ...
        ValueError: coherence asks about no persona: leave out --personas and --persona
    """
    if not rubric.needs:
        if personas_file is not None or persona is not None:
            raise ValueError(
                f"{rubric.name} asks about no persona: leave out --personas and "
                "--persona"
            )
        return Question(rubric)
    if personas_file is None:
        if rubric.needs == "persona":
            raise ValueError(
                f"{rubric.name} asks about one persona: give --personas FILE, and "
                "--persona NAME when the file has more than one"
            )
        raise ValueError(f"{rubric.name} chooses among personas: give --personas FILE")
    found = read_personas(personas_file)
    if rubric.needs == "persona":
        return Question(rubric, pick_persona(found, persona))
    if persona is not None:
        raise ValueError(
            f"{rubric.name} chooses among all the personas of --personas, so it "
            "takes no --persona"
        )
    return Question(rubric, personas=tuple(found))


def window_key(window: Window | None) -> tuple[int, int, int] | None:
    """A window as a row's key holds it: its first sentence, how many it
    holds and the splitter's version; None for a whole reply.

    Example:
        >>> window_key(Window(first=2, count=1, start=10, end=15, text=" Hoo!"))
        (2, 1, 1)
    """
    return None if window is None else (window.first, window.count, SPLITTER_VERSION)


def item_key(
    item: Item, version: int, question: Question, judge: str
) -> tuple[Any, ...]:
    """The key of the row an item's verdict gets (ScoreRow.key), in the
    folder's version, from the judge named judge."""
    return (
        item.run.run_id,
        version,
        question.rubric.name,
        question.rubric.version,
        judge,
        question.sha256,
        window_key(item.window),
    )


def already_scored(
    rows: Sequence[ScoreRow],
    runs: Iterable[RunRecord],
    texts: Mapping[tuple[Any, ...], str],
) -> set[tuple[Any, ...]]:
    """The keys whose verdict is saved, of those in texts, which gives the
    SHA-256 of each key's text as the judge reads it, stripped, as
    Verdict.text_sha256 has it. Of a key's rows that scored the run's
    current reply, the newest counts, as in the views: the latest
    created_at, then the later in rows, which read_rows gives file by file
    in name order. The key's verdict is saved when that row is "ok" and
    judged the text the key has now: a window whose text changed under one
    splitter version is judged again."""
    replies = {run.run_id: output_sha256(run) for run in runs}
    newest: dict[tuple[Any, ...], tuple[str, int, ScoreRow]] = {}
    for place, row in enumerate(rows):
        if replies.get(row.run_id) != row.output_sha256:
            continue  # another reply's, or a run not judged now
        if row.key not in newest or (row.created_at, place) >= newest[row.key][:2]:
            newest[row.key] = (row.created_at, place, row)
    return {
        key
        for key, (_, _, row) in newest.items()
        if row.status == "ok" and row.text_sha256 == texts.get(key)
    }


def make_plan(
    experiment: Experiment,
    question: Question,
    size: int | None,
    judge: str,
    backend: str,
    model: str,
    cache: VerdictCache | None,
    settings: str = "",
) -> Plan:
    """What a job judges: the experiment's whole replies, or with size its
    windows of size sentences, asked the question by the judge named judge
    (<backend>:<model>, with the backend's settings), whose cache is cache.
    A request needs no call when the cache holds its verdict, and a blank
    text fails at once.

    OSError or ValueError when the folder's table cannot be read.
    """
    runs = experiment.runs
    items = reply_items(runs) if size is None else window_items(runs, size)
    rows = read_rows(experiment.folder, SCORES if size is None else SENTENCES)
    keys = [item_key(item, experiment.version, question, judge) for item in items]
    texts = {
        key: text_sha256(item.text.strip())  # as the judge reads it
        for key, item in zip(keys, items, strict=True)
    }
    scored = already_scored(rows, runs, texts)
    todo = [item for key, item in zip(keys, items, strict=True) if key not in scored]
    requests: dict[str, Request] = {}
    cached = blank = 0
    for item in todo:
        if not item.text.strip():
            blank += 1
            continue
        request = question.request(item.text)
        key = request_key(backend, model, request, settings)
        if cache is not None and key in cache:
            cached += 1
        else:
            requests.setdefault(key, request)
    return Plan(items, todo, cached, blank, list(requests.values()))


def window_spans(
    runs: Iterable[RunRecord], size: int, decode: Callable[[Sequence[int]], str]
) -> dict[tuple[str, int], Span]:
    """The span of each window of `size` sentences of each run's reply, by
    run ID and first sentence. decode is the experiment model's tokenizer's,
    without special tokens; a run's windows are placed all at once
    (sentences.window_steps), so their tokens tile its reply's.

    Example:
        >>> text = "Hi. Bye."
        >>> run = RunRecord(
        ...     "x-000001",
        ...     "x",
        ...     "m",
        ...     "p",
        ...     [],
        ...     {},
        ...     0,
        ...     "",
        ...     "",
        ...     text,
        ...     list(text.encode()),
        ...     "ok",
        ... )
        >>> spans = window_spans([run], 1, lambda tokens: bytes(tokens).decode())
        >>> [(s.start_step, s.end_step, s.start_char) for s in spans.values()]
        [(0, 3, 0), (3, 8, 3)]
    """
    spans = {}
    for run in runs:
        windows = sentence_windows(run.output_text, size)
        steps = window_steps(run.output_text, run.output_tokens, decode, windows)
        for window, (start, end) in zip(windows, steps, strict=True):
            spans[run.run_id, window.first] = Span(
                window.first,
                window.count,
                start,
                end,
                window.start,
                window.end,
                SPLITTER_VERSION,
            )
    return spans


def row_details(verdict: Verdict, question: Question) -> dict[str, Any]:
    """A row's details: the judge's reason, whether the verdict came from
    the cache, the tokens its calls used now (none for a cached one or a
    repeat), a failed verdict's last answer, and the persona's description
    or the personas' names."""
    fresh = not verdict.cached
    details: dict[str, Any] = {
        "reason": verdict.reason,
        "cached": verdict.cached,
        "input_tokens": verdict.input_tokens if fresh else 0,
        "output_tokens": verdict.output_tokens if fresh else 0,
    }
    if not verdict.ok:
        details["raw"] = verdict.raw
    if question.persona is not None:
        details["description"] = question.persona.description
    if question.personas:
        details["personas"] = [persona.name for persona in question.personas]
    return details


def score_row(
    item: Item,
    verdict: Verdict,
    version: int,
    question: Question,
    judge: str,
    span: Span | None = None,
) -> ScoreRow:
    """The row of an item's verdict from the judge named judge, in the
    folder's version: a whole reply's, or with its span a window's."""
    persona = question.persona
    return ScoreRow(
        run_id=item.run.run_id,
        version=version,
        scorer=question.rubric.name,
        scorer_version=question.rubric.version,
        judge=judge,
        question_sha256=verdict.question_sha256,
        persona="" if persona is None else persona.name,
        label=verdict.label,
        value=None if verdict.value is None else float(verdict.value),
        status=verdict.status,
        error=verdict.error,
        attempts=verdict.attempts,
        details=row_details(verdict, question),
        text_sha256=verdict.text_sha256,
        output_sha256=output_sha256(item.run),
        created_at=now(),
        span=span,
    )


def counted(count: int, noun: str, nouns: str | None = None) -> str:
    """A count and its noun, plural unless the count is 1.

    Example:
        >>> counted(1, "reply", "replies"), counted(1_200, "chunk")
        ('1 reply', '1,200 chunks')
    """
    return f"{count:,} {noun if count == 1 else nouns or noun + 's'}"


def _span(item: Item, spans: dict[tuple[str, int], Span]) -> Span | None:
    """The span of an item's window; None for a whole reply."""
    return None if item.window is None else spans[item.run.run_id, item.window.first]


def judge_chunks(
    judge: Judge,
    experiment: Experiment,
    question: Question,
    plan: Plan,
    spans: dict[tuple[str, int], Span],
    workers: int,
) -> tuple[list[Verdict], JudgeUnavailable | None]:
    """Judge the plan's items CHUNK at a time, workers calls at a time, and
    save each chunk's rows before the next, printing a line for each; the
    verdicts, and what stopped the judge when it could not answer at all.
    When the judge stops, or Ctrl-C stops the job, the verdicts of the chunk
    found before the stop are saved as they are, failures included, and
    Ctrl-C's KeyboardInterrupt is raised again. Call it holding the folder's
    lock; spans are the windows' (window_spans)."""
    runs, version = {run.run_id: run for run in experiment.runs}, experiment.version
    chunks = [plan.todo[at : at + CHUNK] for at in range(0, len(plan.todo), CHUNK)]
    print(
        f"\nJudging {counted(len(plan.todo), 'item')} in "
        f"{counted(len(chunks), 'chunk')}, {counted(workers, 'call')} at a time:"
    )
    found: list[Verdict] = []
    for number, chunk in enumerate(chunks, start=1):
        jobs = [question.job(item.text) for item in chunk]
        stopped: BaseException | None = None
        try:
            found_all = judge.verdicts(jobs, workers, stopping=_say_stopping)
            judged = list(zip(chunk, found_all, strict=True))
        except (JudgeUnavailable, KeyboardInterrupt) as error:
            # What was found before the stop, as found: paid for, so saved.
            found_now: dict[int, Verdict] = getattr(error, "found", {})
            stopped = error
            judged = [(chunk[place], found_now[place]) for place in sorted(found_now)]
        rows = [
            score_row(item, verdict, version, question, judge.name, _span(item, spans))
            for item, verdict in judged
        ]
        verdicts = [verdict for _, verdict in judged]
        found += verdicts
        ok, cached = sum(v.ok for v in verdicts), sum(v.cached for v in verdicts)
        line = (
            f"chunk {number} of {len(chunks)}: {ok:,} ok, {len(verdicts) - ok:,} "
            f"failed, {cached:,} from the cache"
        )
        if rows:
            path = write_rows(experiment.folder, rows, runs)
            line += f"; saved {path.parent.name}/{path.name}"
        if isinstance(stopped, KeyboardInterrupt):
            print(f"{line}; then Ctrl-C stopped the job")
            raise stopped
        print(line if stopped is None else f"{line}; then the judge stopped")
        if isinstance(stopped, JudgeUnavailable):
            return found, stopped
    return found, None


def _say_stopping() -> None:
    print(STOPPING, flush=True)
