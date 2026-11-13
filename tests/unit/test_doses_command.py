"""`pushlab doses SPEC --units FILE` (pushlab.analysis.doses_command).

The runner saves an invented ladder of tiny-vl on the smoke prompts: no
push, the owl at 0.25 to 0.75 with a random control each, and the captain at
-0.25 to -0.75. Each reply names its role, dose and prompt, and a fake judge
labels it by SPEAKING and COHERENCE through pushlab score --units: the owl
works at 0.5 (4 of 5 replies, exactly the rate) and collapses at 0.75 (a
mean of 4.4); the captain neither works nor collapses.
"""

import json
import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from pushlab import PROJECT_ROOT
from pushlab.analysis.dose_measures import current_measures
from pushlab.analysis.panel import read_panel
from pushlab.cli import main
from pushlab.engine import generate as generate_module
from pushlab.engine.generate import Generation
from pushlab.prompts import PROMPTS_DIR, read_prompt_set
from pushlab.runs.folder import SCORES, locked
from pushlab.runs.shards import read_results
from pushlab.scoring.cache import VerdictCache
from pushlab.scoring.fake_backend import FakeBackend
from pushlab.scoring.items import judged_runs
from pushlab.scoring.judge import Judge
from pushlab.scoring.rubrics import Request
from pushlab.scoring.score_command import run_score
from pushlab.store.records import RunRecord
from pushlab.store.scores import ScoreRow, now, output_sha256, read_rows, write_rows

NAME, JUDGE = "tiny-ladder", "fake:fake"
SMOKE = read_prompt_set(PROMPTS_DIR / "smoke.jsonl")
OWL, CAPTAIN = "tiny-vl:L1:N3", "tiny-vl:L2:N100"
LADDER = f"""\
conditions:
  - {{label: none, segments: []}}
  - {{label: u01, shape: constant, params: {{unit: {OWL}}}, controls: 1,
     control_seed: 0, grid: {{alpha: [0.25, 0.5, 0.75]}}}}
  - {{label: u02, shape: constant, params: {{unit: {CAPTAIN}}},
     grid: {{alpha: [-0.25, -0.5, -0.75]}}}}
"""
UNITS = f"""\
unit,direction,persona,description,source
{OWL},+1,Brass Owl,A pompous mechanical owl.,invented
{CAPTAIN},-1,Salt Captain,A gruff sea captain.,invented
"""
PERSONAS = """\
personas:
  - {name: Brass Owl, description: A pompous mechanical owl.}
  - {name: Salt Captain, description: A gruff sea captain.}
"""
ROLES = {OWL: "owl", CAPTAIN: "captain"}  # a random control's is "control"
PERSONA_OF = {"owl": "Brass Owl", "captain": "Salt Captain", "control": "Brass Owl"}
# How many of the five prompts, the first ones, speak as the persona.
SPEAKING = {("owl", 0.25): 1, ("owl", 0.5): 4, ("owl", 0.75): 5, ("control", 0.75): 1}
SPEAKING |= {("captain", 0.5): 2, ("captain", 0.75): 3}
# Each reply's coherence, prompt i taking the (i % length)-th; 9 elsewhere.
COHERENCE = {("owl", 0.5): [7, 8], ("owl", 0.75): [4, 5], ("captain", 0.75): [8]}
TEXT = re.compile(r"The (\w+) at ([\d.]+) answers prompt (\d)\.")
Saved = tuple[Path, Path, Path]  # the experiment file, runs dir and units file


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run on the CPU even with a GPU, and keep any cache out of /workspace."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("PUSHLAB_CACHE_DIR", str(tmp_path / "default-cache"))


@pytest.fixture
def invented(monkeypatch: pytest.MonkeyPatch) -> None:
    """Have generate() answer "The <role> at <dose> answers prompt <i>."."""

    def generate(model, tokenizer, prompts, pushes, decoding, norms, **kw):  # type: ignore[no-untyped-def]
        made = []
        for prompt, schedule in zip(prompts, kw["schedules"], strict=True):
            role, dose = "plain", 0.0
            if schedule.segments:
                segment = schedule.segments[0]
                role = ROLES.get(str(segment.unit), "control")
                dose = abs(segment.alpha_start)
            text = f"The {role} at {dose} answers prompt {SMOKE.prompts.index(prompt)}."
            tokens = tuple(tokenizer.encode(text, add_special_tokens=False))
            steps = ((),) * len(tokens), (-1.0,) * len(tokens)
            touched = (0,) * len(schedule.segments)
            made.append(Generation(text, tokens, "max_new_tokens", touched, *steps))
        return made

    monkeypatch.setattr(generate_module, "generate", generate)


@pytest.fixture
def ladder(
    tiny_spec: Callable[..., Path], invented: None, tmp_path: Path
) -> Callable[..., Saved]:
    """Save the ladder with the runner, judged unless scored is False."""

    def save(scored: bool = True) -> Saved:
        spec = tiny_spec(LADDER, name=NAME, max_new_tokens="64")
        runs, units = tmp_path / "runs", tmp_path / "units.csv"
        assert main(["run", str(spec), "--runs-dir", str(runs)]) == 0
        units.write_text(UNITS, encoding="utf-8")
        (tmp_path / "personas.yaml").write_text(PERSONAS, encoding="utf-8")
        if scored:
            _score(spec, runs, "speaks_as", units_file=units)
            _score(spec, runs, "coherence")
        return spec, runs, units

    return save


def _reading(text: str) -> tuple[str, float, int]:
    """The role, dose and prompt number an invented reply names."""
    found = TEXT.fullmatch(text)
    assert found is not None, text
    return found[1], float(found[2]), int(found[3])


def _answer(request: Request) -> str:
    """The fake judge's verdict on an invented reply, by the tables."""
    role, dose, prompt = _reading(request.text)
    if request.rubric == "coherence":
        scores = COHERENCE.get((role, dose), [9])
        return json.dumps({"reason": "Fine.", "score": scores[prompt % len(scores)]})
    asked = request.question.splitlines()[0].removeprefix("Persona: ")
    says = PERSONA_OF.get(role) == asked and prompt < SPEAKING.get((role, dose), 0)
    return json.dumps({"reason": "Hoo.", "label": "speaks_as" if says else "assistant"})


def _score(spec: Path, runs: Path, rubric: str, persona="", **options: Any) -> None:
    """Judge every reply by rubric with a fake judge, about persona if given."""
    if persona:
        options |= {"personas_file": spec.parent / "personas.yaml", "persona": persona}
    fake = FakeBackend(options.pop("script", _answer), name="fake", model="fake")
    cache = VerdictCache(spec.parent / "cache", JUDGE)
    judge = Judge(fake, cache, sleep=lambda seconds: None)
    assert run_score(spec, rubric, runs_dir=runs, judge=judge, **options) == 0


def _doses(saved: Saved, *options: str) -> int:
    spec, runs, units = saved
    argv = ["doses", str(spec), "--units", str(units), "--runs-dir", str(runs)]
    return main([*argv, *options])


def _record(saved: Saved) -> dict[str, Any]:
    """pushlab doses's --out JSON, written to a new file."""
    folder = saved[1].parent
    out = folder / f"doses-{len(list(folder.glob('doses-*')))}.json"
    assert _doses(saved, "--out", str(out)) == 0
    return json.loads(out.read_text(encoding="utf-8"))


def _unit(record: dict[str, Any], key: str) -> dict[str, Any]:
    return next(unit for unit in record["units"] if unit["unit"] == key)


def _column(unit: dict[str, Any], *names: str) -> list[Any]:
    """Each dose's value of names, or tuple of their values, lowest first."""
    found = [tuple(dose[name] for name in names) for dose in unit["doses"]]
    return [values[0] if len(names) == 1 else values for values in found]


def _replies(runs: Path) -> dict[str, RunRecord]:
    return {run.run_id: run for run in read_results(runs / NAME)}


def _write(runs: Path, rows: list[ScoreRow]) -> None:
    """Write rows into the ladder's scores table, as pushlab measure would."""
    with locked(runs / NAME):
        write_rows(runs / NAME, rows, _replies(runs))


def _row(run: RunRecord, scorer: str, judge: str, **verdict: Any) -> ScoreRow:
    """An ok row of scorer on run's reply, by judge, with verdict's fields."""
    sha, ok = output_sha256(run), ("ok", "", 1, {})  # status, error, attempts, details
    row = ScoreRow(run.run_id, 1, scorer, 1, judge, "d" * 64, "", None, None, *ok)
    return replace(row, text_sha256=sha, output_sha256=sha, created_at=now(), **verdict)


def _lines(capsys: pytest.CaptureFixture[str]) -> list[str]:
    """The lines printed, each with its runs of spaces made one."""
    return [" ".join(line.split()) for line in capsys.readouterr().out.splitlines()]


def test_each_units_doses_come_from_its_own_verdicts_with_intervals(
    ladder: Callable[..., Saved], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    saved = ladder()
    record = _record(saved)
    assert [record[k] for k in ("experiment", "version", "judge")] == [NAME, 1, JUDGE]
    numbers = ("version", "working_rate", "coherence_below", "resamples")
    assert [record["panel"][name] for name in numbers] == [1, 0.8, 6.0, 2000]
    owl = _unit(record, f"{OWL}+")
    assert (owl["persona"], owl["source"]) == ("Brass Owl", "invented")
    found = [(0, 0, 5), (0.25, 1, 5), (0.5, 4, 5), (0.75, 5, 5)]
    assert _column(owl, "dose", "k", "n") == found
    assert _column(owl, "coherence_mean") == [9.0, 9.0, 7.4, 4.4]
    assert _column(owl, "below") == [0.0, 0.0, 0.0, 1.0]
    assert _column(owl, "rate_interval")[2] == pytest.approx([0.3755, 0.9638], abs=1e-4)
    shares = [{"k": 0, "n": 5}, {"k": 0, "n": 5}, {"k": 1, "n": 5}]
    assert _column(owl, "controls") == [None, *shares]
    # 0.25 reaches 4 of 5 in a resample 0.7% of the time, too seldom to count;
    # 0.5 falls short in about a quarter of them, which 0.75 then catches.
    assert owl["working"] == {"dose": 0.5, "low": 0.5, "high": 0.75, "unreached": 0}
    assert owl["collapse"] == {"dose": 0.75, "low": 0.75, "high": 0.75, "unreached": 0}
    captain = _unit(record, f"{CAPTAIN}-")
    assert _column(captain, "rate") == [0.0, 0.0, 0.4, 0.6]
    assert _column(captain, "controls") == [None] * 4
    assert captain["working"]["dose"] is captain["working"]["high"] is None
    never = {"dose": None, "low": None, "high": None, "unreached": 2000}
    assert captain["collapse"] == never
    written = (tmp_path / "doses-0.json").read_text(encoding="utf-8")
    texts = [run.output_text for run in _replies(saved[1]).values()]
    assert not [text for text in texts if text in written]  # no reply's text
    lines = _lines(capsys)
    assert f"judge {JUDGE}" in lines and f"{OWL}+: Brass Owl (invented)" in lines
    head = "dose replies speaks_as rate interval coherence interval below 6 controls"
    assert head in lines
    assert "0.5 5 4/5 0.80 [0.38, 0.96] 7.4 [7.0, 7.8] 0.00 0/5" in lines
    assert "working dose 0.5 (95% interval 0.5 to 0.75)" in lines
    assert "collapse dose 0.75 (95% interval 0.75)" in lines
    assert "window from 0.5 to below 0.75" in lines
    summary = lines[lines.index("Summary, with 95% intervals:") + 1 :]
    window = "0.5 [0.5, 0.75] 0.75 [0.75, 0.75] from 0.5 to below 0.75"
    assert summary[1] == f"{OWL}+ Brass Owl {window}"
    assert summary[2].startswith(f"{CAPTAIN}- Salt Captain none [")
    assert lines[-1] == f"Saved the numbers: {tmp_path / 'doses-0.json'}"
    # With the measures that need no judge saved, as pushlab measure saves
    # them: under the keys it would use now (smoke.jsonl has no answers file,
    # so no capability), and a newer row under a stale key, which is not read.
    experiment = judged_runs(saved[0], saved[1])
    measures = current_measures(experiment, read_panel())
    assert sorted(measures) == ["perplexity", "repetition", "word_variety"]
    rows: dict[str, list[ScoreRow]] = {}
    for run in _replies(saved[1]).values():
        role, dose, prompt = _reading(run.output_text)
        loops = role == "owl" and dose == 0.75 and prompt < 3
        for scorer, verdict in [
            ("repetition", {"value": 0.5 if loops else 0.05}),
            ("word_variety", {"value": 0.2 if loops else 0.8}),
            ("perplexity", {"value": 10 + 10 * dose + prompt}),
        ]:
            measure = measures[scorer]
            row = _row(run, scorer, measure.judge, **verdict)
            rows.setdefault(scorer, []).append(
                replace(row, question_sha256=measure.question_sha256)
            )
        stale = _row(run, "repetition", "pushlab:text", value=0.99)  # another panel
        rows.setdefault("stale", []).append(stale)
    for scored in rows.values():
        _write(saved[1], scored)
    owl = _unit(_record(saved), f"{OWL}+")  # the judge is still the only one
    names = ("repetition", "flagged", "perplexity")
    values = [(0.05, 0.0, 12.0), (0.05, 0.0, 14.5), (0.05, 0.0, 17.0)]
    values.append((pytest.approx(0.32), 0.6, 19.5))
    measures_seen = [dict(zip(names, numbers, strict=True)) for numbers in values]
    assert _column(owl, "measures") == measures_seen
    lines = _lines(capsys)
    assert f"{head} repetition flagged perplexity" in lines
    numbers = "4.4 [4.0, 4.8] 1.00 1/5 0.32 0.60 19.5"
    assert f"0.75 5 5/5 1.00 [0.57, 1.00] {numbers}" in lines


def test_the_newest_verdict_and_question_count_and_a_failed_one_is_none(
    ladder: Callable[..., Saved],
) -> None:
    saved = ladder()
    replies, newer = _replies(saved[1]), []
    for row in read_rows(saved[1] / NAME, SCORES):
        role, dose, prompt = _reading(replies[row.run_id].output_text)
        if (row.scorer, row.persona, role) != ("speaks_as", "Brass Owl", "owl"):
            continue
        if dose == 0.25:  # judged again: now every one speaks as the owl
            newer.append(replace(row, label="speaks_as", created_at=now()))
        elif dose == 0.75 and prompt == 0:  # judged again, and it failed
            failed = {"label": None, "status": "failed", "error": "timed out"}
            newer.append(replace(row, created_at=now(), **failed))  # type: ignore[arg-type]
    _write(saved[1], newer)
    owl = _unit(_record(saved), f"{OWL}+")
    assert _column(owl, "k", "n") == [(0, 5), (5, 5), (4, 5), (4, 4)]
    assert owl["working"]["dose"] == 0.25
    # Asked again about the owl with another description, a new question:
    # it now speaks as the owl at every dose, the same question at each.
    owlish = PERSONAS.replace("A pompous mechanical owl.", "A brass owl that ticks.")
    (saved[0].parent / "personas.yaml").write_text(owlish, encoding="utf-8")

    def ticks(request: Request) -> str:
        label = "speaks_as" if _reading(request.text)[0] == "owl" else "assistant"
        return json.dumps({"reason": "Tick.", "label": label})

    _score(saved[0], saved[1], "speaks_as", "Brass Owl", script=ticks)
    record = _record(saved)
    owl = _unit(record, f"{OWL}+")
    assert _column(owl, "rate") == [0.0, 1.0, 1.0, 1.0]
    assert _column(owl, "controls")[1:] == [{"k": 0, "n": 5}] * 3
    assert owl["working"]["dose"] == 0.25
    captain = _unit(record, f"{CAPTAIN}-")  # its own question, unchanged
    assert _column(captain, "rate") == [0.0, 0.0, 0.4, 0.6]


def test_a_missing_verdict_is_refused_naming_the_command_that_makes_it(
    ladder: Callable[..., Saved], capsys: pytest.CaptureFixture[str]
) -> None:
    saved = ladder(scored=False)
    spec, runs, units = saved
    speaks = f"pushlab score {spec} --rubric speaks_as --units {units} makes them"
    coherence = f"pushlab score {spec} --rubric coherence makes them"
    no_coherence = f"has no ok coherence verdict of {JUDGE} on its replies"
    about = f"of {JUDGE} about Salt Captain on its replies: {speaks}"
    steps = [
        ((), f"no speaks_as or coherence verdict is saved yet: {speaks}"),
        (("speaks_as", "Brass Owl"), f"{OWL}+ {no_coherence}: {coherence}"),
        (("coherence",), f"{CAPTAIN}- has no ok speaks_as verdict {about}"),
    ]
    for scoring, problem in steps:
        if scoring:
            _score(spec, runs, *scoring)
        capsys.readouterr()
        assert _doses(saved) == 1
        assert capsys.readouterr().err == f"pushlab doses: {problem}\n"
    _score(spec, runs, "speaks_as", "Salt Captain")
    assert _doses(saved) == 0


def test_a_judge_units_file_or_out_that_does_not_fit_is_refused(
    ladder: Callable[..., Saved],
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    saved = ladder()
    _, runs, units = saved
    other = "local:other-judge"
    others = [_row(r, "coherence", other, value=2.0) for r in _replies(runs).values()]
    _write(runs, others)
    panel, there = tmp_path / "panel.yaml", tmp_path / "there.json"
    panel.write_text("version: 1\n", encoding="utf-8")
    there.write_text("{}", encoding="utf-8")
    inside = [PROJECT_ROOT / "doses.json", runs / NAME / "doses.json"]
    pair = f"[{{unit: {OWL}, alpha: 0.5}}, {{unit: {CAPTAIN}, alpha: -0.5}}]"
    pairs = tiny_spec(f"conditions: [{{label: pair, segments: {pair}}}]", file="p.yaml")
    assert main(["run", str(pairs), "--runs-dir", str(runs)]) == 0
    refusals = [
        (UNITS, [], f"verdicts of 2 judges are saved, {JUDGE}, {other}: choose one"),
        (UNITS, ["--judge", "api:x"], f"of api:x: those saved are {JUDGE}, {other}"),
        (UNITS, ["--judge", other], f"has no ok speaks_as verdict of {other}"),
        ("unit,persona\ntiny-vl:L3:N5,Ghost\n", [], "no saved run of tiny-vl:L3:N5+"),
        (f"unit\n{OWL}\n", [], f"names no persona for {OWL}+"),
        ("unit,persona\nqwen3-8b:L9:N4242,Owl\n", [], "lists qwen3-8b's units"),
        (f"unit,direction,persona\n{OWL},-1,Owl\n", [], f"no saved run of {OWL}-"),
        ("persona\nOwl\n", [], "has no unit column"),
        (UNITS, ["--panel", str(panel)], "lacks working_rate"),
        (UNITS, ["--out", str(there)], f"{there} already exists"),
        (UNITS, ["--out", str(inside[0])], "outside the repository"),
        (UNITS, ["--out", str(inside[1])], f"outside {runs / NAME}"),
        (UNITS, ["--runs-dir", str(tmp_path / "empty")], "does not exist"),
    ]
    for text, options, problem in refusals:
        units.write_text(text, encoding="utf-8")
        capsys.readouterr()
        assert _doses(saved, *options) == 1, problem
        err = capsys.readouterr().err
        assert err.startswith("pushlab doses: ") and problem in err, err
    assert not any(path.exists() for path in inside)
    assert there.read_text(encoding="utf-8") == "{}"
    assert _doses(saved, "--judge", JUDGE) == 0
    assert "collapse dose 0.75 (95% interval 0.75)" in _lines(capsys)
    assert _doses((pairs, runs, units)) == 1  # its runs push two units each
    err = capsys.readouterr().err
    assert err.startswith("pushlab doses: tiny-pulses is no dose ladder: ")
    assert "pushes more than one unit or dose" in err
