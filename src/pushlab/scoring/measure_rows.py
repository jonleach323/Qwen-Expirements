"""The rows `pushlab measure` saves, measures that need no judge on each
reply of an experiment, and its summary of them (decision 0022).

A Measure is what one measure asks of every reply, as the scores table keys
its rows (ScoreRow.key): a scorer and its version, a judge that names what
measured it, and the SHA-256 of what else the value depends on, as JSON:
- repetition and word_variety (pushlab.scoring.text_measures, judge
  pushlab:text): the measure, TEXT_MEASURES_VERSION, and the n-gram and
  window of configs/panel.yaml;
- capability (pushlab.scoring.capability, judge pushlab:exact-match): the
  measure, CAPABILITY_VERSION and the SHA-256 of the answers file, so new
  answers grade every reply again;
- perplexity (pushlab.engine.score_text, judge model:<model>@<revision>):
  the measure, PERPLEXITY_VERSION and the data type the model ran in.
A row's value is the measure; capability's label is the grade, correct,
wrong or no_answer, and its value 1 when correct, else 0. A reply with no
word to measure, or no token to score, gets a failed row that says so,
never a guessed value.

A reply is measured once: with a row of its key on the run's reply as it
is now (newest_rows), whatever its status, it is measured already, as a
measure that failed would only fail again, unlike a judge's verdict. A new
folder version, reply, n-gram, window or answers file makes new keys.
"""

import hashlib
import json
import math
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pushlab.analysis.intervals import wilson_interval
from pushlab.analysis.panel import Panel
from pushlab.scoring.capability import CAPABILITY_VERSION, Answer, grade
from pushlab.scoring.items import Experiment
from pushlab.scoring.keys import text_sha256
from pushlab.scoring.text_measures import (
    TEXT_MEASURES_VERSION,
    TextMeasures,
    measure_text,
    text_flags,
)
from pushlab.store.records import RunRecord
from pushlab.store.scores import ScoreRow, now, output_sha256

TEXT_JUDGE = "pushlab:text"
EXACT_JUDGE = "pushlab:exact-match"
TEXT_SCORERS = ("repetition", "word_variety")
# Raised whenever a change to how perplexity is worked out would move a value.
PERPLEXITY_VERSION = 1
NO_TEXT = "there is no text to measure"
NO_TOKENS = "the reply has no token to score"
NOT_FINITE = "its perplexity is not a finite number"
# The summary's columns, then those of the measures not always taken.
COLUMNS = ("condition", "replies", "repetition", "word variety", "flagged")
OPTIONAL = ("capability", "perplexity")

Key = tuple[Any, ...]


def question_sha256(asked: Mapping[str, Any]) -> str:
    """The SHA-256 of what a measure asks, as canonical JSON: its keys
    sorted, no spaces, in UTF-8.

    Example:
        >>> question_sha256({"version": 1, "measure": "repetition"})[:16]
        '3fb2085ca2031460'
    """
    text = json.dumps(asked, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Measure:
    """What one measure asks of every reply, as its rows' key holds it: its
    scorer and the scorer's version, its judge and the question's SHA-256.

    Example:
        >>> measure = Measure("repetition", 1, TEXT_JUDGE, "a" * 64)
        >>> measure.key("x-000001", 2)[:5]
        ('x-000001', 2, 'repetition', 1, 'pushlab:text')
    """

    scorer: str
    scorer_version: int
    judge: str
    question_sha256: str

    def key(self, run_id: str, version: int) -> Key:
        """The key of its row on a run of the folder's version (ScoreRow.key)."""
        asked = (self.scorer, self.scorer_version, self.judge, self.question_sha256)
        return (run_id, version, *asked, None)


def repetition_and_variety(panel: Panel) -> list[Measure]:
    """repetition and word_variety, at the panel's n-gram and window.

    Example:
        >>> from pushlab.analysis.panel import read_panel
        >>> [measure.scorer for measure in repetition_and_variety(read_panel())]
        ['repetition', 'word_variety']
    """
    version = TEXT_MEASURES_VERSION
    asked = dict(version=version, ngram=panel.ngram, window=panel.window)
    return [
        Measure(
            scorer, version, TEXT_JUDGE, question_sha256(asked | {"measure": scorer})
        )
        for scorer in TEXT_SCORERS
    ]


def capability_measure(answers_sha256: str) -> Measure:
    """capability, graded against the answers file of that SHA-256.

    Example:
        >>> capability_measure("0" * 64).judge
        'pushlab:exact-match'
    """
    version = CAPABILITY_VERSION
    asked = dict(measure="capability", version=version, answers_sha256=answers_sha256)
    return Measure("capability", version, EXACT_JUDGE, question_sha256(asked))


def perplexity_measure(model: str, dtype: str) -> Measure:
    """perplexity under model, <name>@<revision>, run in dtype.

    Example:
        >>> perplexity_measure("qwen3-8b@b968826", "bfloat16").judge
        'model:qwen3-8b@b968826'
    """
    version = PERPLEXITY_VERSION
    asked = dict(measure="perplexity", version=version, dtype=dtype)
    return Measure("perplexity", version, f"model:{model}", question_sha256(asked))


def newest_rows(
    rows: Sequence[ScoreRow], runs: Iterable[RunRecord]
) -> dict[Key, ScoreRow]:
    """Of rows, the newest of each key that scored a run's reply as it is
    now, by key, as the scores view takes it: the latest created_at, then
    the later in rows, which read_rows gives file by file in name order.
    Rows of another reply, or of a run not in runs, are left out.

    Example:
        >>> newest_rows([], [])
        {}
    """
    replies = {run.run_id: output_sha256(run) for run in runs}
    newest: dict[Key, tuple[str, int, ScoreRow]] = {}
    for place, row in enumerate(rows):
        if replies.get(row.run_id) != row.output_sha256:
            continue  # another reply's, or a run not measured now
        if row.key not in newest or (row.created_at, place) >= newest[row.key][:2]:
            newest[row.key] = (row.created_at, place, row)
    return {key: row for key, (_, _, row) in newest.items()}


def measure_row(
    run: RunRecord,
    version: int,
    measure: Measure,
    *,
    value: float | None = None,
    label: str | None = None,
    error: str = "",
    details: Mapping[str, Any] | None = None,
) -> ScoreRow:
    """A measure's row on a run's reply, in the folder's version: ok, with
    its value or label, or failed with an error and neither. text_sha256 is
    the stripped reply's, as a judge's rows have it.

    Example:
        >>> run = RunRecord("x-1", "x", "m", "p", [], {}, 0, "", "", "Hi.", [1], "ok")
        >>> measure = Measure("repetition", 1, TEXT_JUDGE, "a" * 64)
        >>> row = measure_row(run, 1, measure, error=NO_TEXT)
        >>> row.status, row.value, row.persona, row.attempts
        ('failed', None, '', 0)
    """
    return ScoreRow(
        run_id=run.run_id,
        version=version,
        scorer=measure.scorer,
        scorer_version=measure.scorer_version,
        judge=measure.judge,
        question_sha256=measure.question_sha256,
        persona="",
        label=label,
        value=value,
        status="failed" if error else "ok",
        error=error,
        attempts=0,
        details=dict(details or {}),
        text_sha256=text_sha256(run.output_text.strip()),
        output_sha256=output_sha256(run),
        created_at=now(),
    )


def text_row(run: RunRecord, version: int, measure: Measure, panel: Panel) -> ScoreRow:
    """repetition's or word_variety's row on a run's reply, at the panel's
    n-gram and window; details give its units and the flags the panel's
    levels give it (text_flags). Failed, NO_TEXT, for a reply with no unit.

    Example:
        >>> from pushlab.analysis.panel import read_panel
        >>> panel, loop = read_panel(), "Hi! " * 10
        >>> run = RunRecord("x-1", "x", "m", "p", [], {}, 0, "", "", loop, [1], "ok")
        >>> row = text_row(run, 1, repetition_and_variety(panel)[0], panel)
        >>> round(row.value, 4), row.details
        (0.8571, {'units': 10, 'flags': ['repetitive', 'low variety']})
    """
    measured = measure_text(run.output_text, panel)
    if measured is None:
        details = {"units": 0, "flags": []}
        return measure_row(run, version, measure, error=NO_TEXT, details=details)
    value = measured.repetition if measure.scorer == "repetition" else measured.variety
    details = {"units": measured.units, "flags": list(text_flags(measured, panel))}
    return measure_row(run, version, measure, value=value, details=details)


def capability_row(
    run: RunRecord, version: int, measure: Measure, answer: Answer | None
) -> ScoreRow:
    """capability's row: the reply graded against its prompt's answer, its
    label the grade's, its value 1.0 when correct, else 0.0, and details
    whether it is strict, what was compared and the question's kind.
    Failed when the prompt has no answer.

    Example:
        >>> answer = Answer("q-1", "arithmetic", "number", "12", (), (), "7 + 5")
        >>> reply = "**12**"
        >>> run = RunRecord("x-1", "x", "m", "q-1", [], {}, 0, "", "", reply, [], "ok")
        >>> row = capability_row(run, 1, capability_measure("0" * 64), answer)
        >>> row.label, row.value, row.details
        ('correct', 1.0, {'strict': False, 'extracted': '12', 'kind': 'arithmetic'})
    """
    if answer is None:
        error = f"the answers file has no answer for {run.prompt_id}"
        return measure_row(run, version, measure, error=error)
    graded = grade(run.output_text, answer)
    details = dict(strict=graded.strict, extracted=graded.extracted, kind=answer.kind)
    value = 1.0 if graded.label == "correct" else 0.0
    return measure_row(
        run, version, measure, value=value, label=graded.label, details=details
    )


def perplexity_row(
    run: RunRecord, version: int, measure: Measure, logprobs: Sequence[float]
) -> ScoreRow:
    """perplexity's row from the log-probabilities of the reply's tokens
    under the unpushed model (score_text.reply_logprobs), with how many
    tokens and their mean in details. Failed, NO_TOKENS, with no token, and
    NOT_FINITE when it is past what a float holds.

    Example:
        >>> run = RunRecord("x-1", "x", "m", "p", [], {}, 0, "", "", "Hi", [1, 2], "ok")
        >>> measure = perplexity_measure("m@r", "float32")
        >>> row = perplexity_row(run, 1, measure, [-1.0, -1.0])
        >>> row.value, row.details
        (2.718281828459045, {'tokens': 2, 'mean_logprob': -1.0})
        >>> perplexity_row(run, 1, measure, []).error
        'the reply has no token to score'
    """
    from pushlab.engine.score_text import perplexity  # here, as it loads torch

    tokens = {"tokens": len(logprobs)}
    if len(logprobs) == 0:  # not `not logprobs`, which an array refuses
        return measure_row(run, version, measure, error=NO_TOKENS, details=tokens)
    mean, value = sum(logprobs) / len(logprobs), perplexity(logprobs)
    if value is None or not (math.isfinite(mean) and math.isfinite(value)):
        return measure_row(run, version, measure, error=NOT_FINITE, details=tokens)
    details = tokens | {"mean_logprob": mean}
    return measure_row(run, version, measure, value=value, details=details)


def group_name(run: RunRecord, experiment: Experiment) -> str:
    """The group a run counts in, as pushlab report SPEC counts it: its
    condition, or for a random control the condition and the control's
    number, as expand numbers a run's controls right after it.

    Example:
        >>> experiment = Experiment(None, None, {"conditions": {"x-3": "a"}}, [], 0)
        >>> run = RunRecord("x-3", "x", "m", "p", [], {}, 0, "", "x-2", "", [], "ok")
        >>> group_name(run, experiment)
        'a, control 1'
    """
    condition = experiment.condition(run)
    if not run.control_of:
        return condition
    return f"{condition}, control {_counter(run.run_id) - _counter(run.control_of)}"


def _counter(run_id: str) -> int:
    """A run ID's counter, as make_run_id writes it last."""
    return int(run_id.rsplit("-", 1)[1])


def _share(right: int, graded: int, confidence: float) -> str:
    """right of graded, with its Wilson interval, as a cell; "-" for none.

    Example:
        >>> _share(1, 3, 0.95), _share(0, 0, 0.95)
        ('1/3 [0.06, 0.79]', '-')
    """
    if not graded:
        return "-"
    low, high = wilson_interval(right, graded, confidence)
    return f"{right}/{graded} [{low:.2f}, {high:.2f}]"


def summary_lines(
    experiment: Experiment,
    measures: Sequence[Measure],
    newest: Mapping[Key, ScoreRow],
    panel: Panel,
) -> list[str]:
    """Each group's measures (group_name), from the ok rows in newest, as a
    line that says what they are and a table: its replies; of those both
    text measures measured, the mean repetition and word variety and how
    many the panel flags; the replies right on the capability set, with a
    Wilson interval; and the median perplexity; "-" for none. capability
    and perplexity have columns only when they are among measures."""
    by_scorer = {measure.scorer: measure for measure in measures}
    version = experiment.version

    def ok(scorer: str, run: RunRecord) -> ScoreRow | None:
        row = newest.get(by_scorer[scorer].key(run.run_id, version))
        return row if row is not None and row.status == "ok" else None

    groups: dict[str, list[RunRecord]] = {}
    for run in experiment.runs:
        groups.setdefault(group_name(run, experiment), []).append(run)
    extra = [scorer for scorer in OPTIONAL if scorer in by_scorer]
    table = [[*COLUMNS, *extra]]
    for name, runs in groups.items():
        pairs = [(ok("repetition", run), ok("word_variety", run)) for run in runs]
        both = [
            TextMeasures(int(rep.details.get("units", 0)), rep.value, var.value)
            for rep, var in pairs
            if rep and var and rep.value is not None and var.value is not None
        ]
        cells = [name, f"{len(runs):,}"]
        for values in ([m.repetition for m in both], [m.variety for m in both]):
            cells.append(f"{statistics.fmean(values):.2f}" if values else "-")
        flagged = sum(bool(text_flags(measured, panel)) for measured in both)
        cells.append(f"{flagged}/{len(both)}" if both else "-")
        if "capability" in by_scorer:
            graded = [row for run in runs if (row := ok("capability", run))]
            right = sum(row.label == "correct" for row in graded)
            cells.append(_share(right, len(graded), panel.confidence))
        if "perplexity" in by_scorer:
            found = [row.value for run in runs if (row := ok("perplexity", run))]
            values = [value for value in found if value is not None]
            cells.append(f"{statistics.median(values):.1f}" if values else "-")
        table.append(cells)
    widths = [max(len(row[at]) for row in table) for at in range(len(table[0]))]
    lines = [
        "  ".join(
            cell.ljust(width) if at == 0 else cell.rjust(width)
            for at, (cell, width) in enumerate(zip(row, widths, strict=True))
        ).rstrip()
        for row in table
    ]
    about = (
        "By condition, each random control on a line of its own: of the replies "
        "measured, the mean repetition and word variety, and those flagged "
        f"repetitive (above {panel.repetitive_above:g}) or of low variety (below "
        f"{panel.varied_below:g})"
    )
    if "capability" in by_scorer:
        level = f"{round(panel.confidence * 100, 6):g}%"
        about += f"; those right on the capability set, with a {level} interval"
    if "perplexity" in by_scorer:
        about += "; the median perplexity under the unpushed model"
    return [f"{about}:", "", *lines]
