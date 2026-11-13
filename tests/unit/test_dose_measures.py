"""The measures that need no judge, per dose of a ladder
(pushlab.analysis.dose_measures): the keys pushlab measure would use now,
and each dose's mean, share and median from the rows saved under them.

Rows here are made up, a reply's text standing in for what was measured.
"""

import hashlib
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

from pushlab.analysis.dose_measures import current_measures, dose_measures
from pushlab.analysis.panel import read_panel
from pushlab.runs.spec import read_spec
from pushlab.scoring.items import Experiment
from pushlab.scoring.measure_rows import (
    capability_measure,
    perplexity_measure,
    repetition_and_variety,
)
from pushlab.store.records import RunRecord
from pushlab.store.scores import ScoreRow, output_sha256

PANEL = read_panel()
TEXT = {measure.scorer: measure for measure in repetition_and_variety(PANEL)}
MODEL = "tiny-vl@" + "0" * 40


def _experiment(spec: Path) -> Experiment:
    identity = {"version": 1, "model": MODEL, "conditions": {}}
    return Experiment(read_spec(spec), spec.parent / "runs", identity, [], 0)


def test_the_measures_are_those_pushlab_measure_would_save_now(
    tiny_spec: Callable[..., Path], tmp_path: Path
) -> None:
    measures = current_measures(_experiment(tiny_spec()), PANEL)
    assert measures == TEXT | {"perplexity": perplexity_measure(MODEL, "float32")}
    # With an answers file beside the prompt set: capability, against it.
    prompts = tmp_path / "two.jsonl"
    lines = [
        {"prompt_id": f"q-{n}", "version": 1, "prompt": f"{n} + 1?"} for n in (1, 2)
    ]
    prompts.write_text("".join(json.dumps(line) + "\n" for line in lines), "utf-8")
    answers = tmp_path / "two-answers.jsonl"
    answers.write_text("made up, never read here\n", encoding="utf-8")
    measures = current_measures(_experiment(tiny_spec(prompts=str(prompts))), PANEL)
    sha256 = hashlib.sha256(answers.read_bytes()).hexdigest()
    assert measures["capability"] == capability_measure(sha256)


MEASURES = TEXT | {
    "perplexity": perplexity_measure(MODEL, "float32"),
    "capability": capability_measure("a" * 64),
}


def _run(number: int) -> RunRecord:
    run_id = f"x-{number:06d}"
    return RunRecord(run_id, "x", "m", "p", [], {}, 0, "", "", "A reply.", [1], "ok")


def _row(run: RunRecord, scorer: str, **verdict: object) -> ScoreRow:
    """A row of a current measure on run's reply: ok with verdict's label or
    value, or failed without one."""
    measure, sha = MEASURES[scorer], output_sha256(run)
    ok = bool(verdict)
    return ScoreRow(
        run_id=run.run_id,
        version=1,
        scorer=scorer,
        scorer_version=measure.scorer_version,
        judge=measure.judge,
        question_sha256=measure.question_sha256,
        persona="",
        label=verdict.get("label"),  # type: ignore[arg-type]
        value=verdict.get("value"),  # type: ignore[arg-type]
        status="ok" if ok else "failed",
        error="" if ok else "there is no text to measure",
        attempts=0,
        details={},
        text_sha256=sha,
        output_sha256=sha,
        created_at="2026-11-13T09:00:00Z",
    )


def test_each_dose_shows_the_rows_saved_under_the_current_keys() -> None:
    runs = [_run(number) for number in range(1, 6)]
    pairs = [(0.0, runs[0]), (0.0, runs[1]), (0.5, runs[2]), (0.5, runs[3])]
    pairs.append((1.0, runs[4]))  # measured not at all
    rows = [
        _row(runs[0], "repetition", value=0.1),
        _row(runs[0], "word_variety", value=0.8),
        _row(runs[0], "perplexity", value=10.0),
        _row(runs[0], "capability", label="correct", value=1.0),
        _row(runs[1], "repetition", value=0.5),  # repetitive: flagged
        _row(runs[1], "word_variety", value=0.2),
        _row(runs[1], "perplexity", value=30.0),
        _row(runs[1], "capability", label="wrong", value=0.0),
        _row(runs[2], "repetition", value=0.0),
        _row(runs[2], "word_variety", value=0.9),
        _row(runs[2], "perplexity", value=20.0),
        _row(runs[2], "capability", label="correct", value=1.0),
        _row(runs[3], "repetition", value=0.2),  # no word variety: not flagged
        _row(runs[3], "perplexity"),  # failed: no value
        _row(runs[3], "capability", label="no_answer", value=0.0),
        # Newer rows under keys pushlab measure would not use now: not read.
        replace(_row(runs[2], "repetition", value=0.99), question_sha256="b" * 64),
        replace(_row(runs[2], "perplexity", value=99.0), judge="model:other@1"),
        replace(_row(runs[2], "word_variety", value=0.01), version=2),
    ]
    found = dose_measures(pairs, rows, MEASURES, 1, PANEL)
    assert found == {
        0.0: {"repetition": 0.3, "flagged": 0.5, "perplexity": 20.0, "capability": 0.5},
        0.5: {"repetition": 0.1, "flagged": 0.0, "perplexity": 20.0, "capability": 0.5},
        1.0: {},
    }
    # Without the capability measure, as with no answers file: no column.
    text_only = {name: MEASURES[name] for name in ("repetition", "word_variety")}
    found = dose_measures(pairs, rows, text_only, 1, PANEL)
    assert found[0.0] == {"repetition": 0.3, "flagged": 0.5}
