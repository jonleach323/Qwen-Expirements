"""Each candidate's figures in a scored experiment (pushlab.panel.stats,
decision 0023), which every stage of pushlab panel and its selection read.

The runner saves a made-up screen of tiny-vl candidates
(tests/unit/panel_experiments.py), after an unpushed condition and one that
pushes a neuron of no candidate, with invented replies; a fake judge gives
the verdicts through pushlab score --units, or rows are written directly.
"""

import doctest
import importlib.util
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from pushlab.discovery.candidates import Candidate
from pushlab.panel import figures, stats
from pushlab.panel.stats import (
    CandidateStats,
    Figures,
    Reply,
    candidate_stats,
    push_key,
    read_scored,
)
from pushlab.runs.folder import locked
from pushlab.runs.shards import read_results
from pushlab.store.records import RunRecord
from pushlab.store.scores import ScoreRow, now, output_sha256, write_rows
from pushlab.units.neuron import NeuronUnit

_SPEC = importlib.util.spec_from_file_location(
    "panel_experiments", Path(__file__).with_name("panel_experiments.py")
)
made = importlib.util.module_from_spec(_SPEC)  # type: ignore[arg-type]
_SPEC.loader.exec_module(made)  # type: ignore[union-attr]

# No push, and a push of a neuron no candidate has, before the candidates.
OTHERS = """\
  - {label: none, segments: []}
  - {label: stranger, segments: [{unit: tiny-vl:L3:N50, alpha: 1.0}]}
"""
SPEAKING = {
    ("owl", 1.0): range(5),
    ("captain", 0.5): {0, 2},
    ("teapot", 1.0): range(5),
    ("fox", 1.25): {4},
}
COHERENCE = {("owl", 1.0): [8, 6], ("captain", 0.5): [3]}
Saved = tuple[Path, Path, Path]  # the screen's file, its runs dir and its list


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run on the CPU even with a GPU, and keep any cache out of /workspace."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("PUSHLAB_CACHE_DIR", str(tmp_path / "default-cache"))


@pytest.fixture
def saved(
    tiny_vl_calibration: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[..., Saved]:
    """Save the screen, after conditions, with the runner, unless run is False."""

    def save(conditions: str = OTHERS, run: bool = True) -> Saved:
        made.invent(monkeypatch)
        listed = made.write_list(tmp_path)
        spec = made.write_screen(tmp_path, listed, conditions=conditions)
        runs = tmp_path / "runs"
        if run:
            made.run(spec, runs)
        return spec, runs, listed

    return save


def test_each_candidate_has_its_own_runs_and_verdicts_by_dose(
    saved: Callable[..., Saved],
) -> None:
    spec, runs, listed = saved()
    failing = {("teapot", 1.0, 4)}  # no verdict: a run without one
    made.score(spec, runs, "speaks_as", made.answers(SPEAKING, failing=failing), listed)
    made.score(spec, runs, "coherence", made.answers({}, COHERENCE))
    scored = read_scored(spec, runs)
    assert (scored.judge, scored.prompts) == (made.JUDGE, made.SMOKE.prompt_ids)
    assert [each.key for each in scored.stats] == list(made.KEYS.values())
    assert [each.place for each in scored.stats] == [0, 1, 2, 3, 4]
    # The unpushed runs and the stranger's push no candidate.
    assert (scored.others, scored.unrun) == (10, ())
    owl, captain, teapot, fox, oracle = scored.stats
    # Each at its reference dose, |alpha|: the captain's alpha is -0.5.
    doses = [list(each.doses) for each in scored.stats]
    assert doses == [[1.0], [0.5], [1.0], [1.25], [0.75]]
    figures = [(e.total.runs, e.total.k, e.total.n) for e in scored.stats]
    assert figures == [(5, 5, 5), (5, 2, 5), (5, 4, 4), (5, 1, 5), (5, 0, 5)]
    assert (teapot.total.rate, captain.total.rate, oracle.total.rate) == (1, 0.4, 0)
    assert [reply.speaks for reply in teapot.total.replies][-1] is None
    assert (scored.unjudged("speaks_as"), scored.unjudged("coherence")) == (1, 0)
    # Coherence by the same judge: prompt i takes the (i % 2)-th of 8 and 6.
    assert [reply.coherence for reply in owl.total.replies] == [8, 6, 8, 6, 8]
    assert (owl.total.coherence, captain.total.coherence) == (7.2, 3.0)
    assert fox.total.scored == 5
    # Each reply keeps its run, prompt and text, in run order.
    replies = {run.run_id: run for run in read_results(runs / "tiny-screen")}
    for reply in captain.total.replies:
        run = replies[reply.run_id]
        assert (reply.prompt_id, reply.text) == (run.prompt_id, run.output_text)
    assert captain.total.prompts == made.SMOKE.prompt_ids
    assert list(scored.by_key) == list(made.KEYS.values())
    assert scored.by_key[made.KEYS["captain"]] is captain
    assert captain.total.replies[0].text == "The captain at 0.5 answers prompt 0."


def test_the_question_asked_last_of_a_candidate_counts(
    saved: Callable[..., Saved], tmp_path: Path
) -> None:
    spec, runs, listed = saved()
    made.score(spec, runs, "speaks_as", made.answers(SPEAKING), listed)
    assert [e.total.k for e in read_scored(spec, runs).stats] == [5, 2, 5, 1, 0]
    # Asked again with descriptions from a personas file, another question,
    # whose verdicts, none speaking as its persona, replace the first's.
    personas = tmp_path / "personas.yaml"
    names = list(made.PERSONAS.values())
    listing = "".join(f"  - {{name: {n}, description: Made up.}}\n" for n in names)
    personas.write_text(f"personas:\n{listing}", encoding="utf-8")
    made.score(spec, runs, "speaks_as", made.answers({}), listed, personas=personas)
    assert [e.total.k for e in read_scored(spec, runs).stats] == [0] * 5


def _row(run: RunRecord, persona: str, label: str, question: str) -> ScoreRow:
    """An ok speaks_as row of the fake judge on run, about persona."""
    sha = output_sha256(run)
    asked = (run.run_id, 1, "speaks_as", 1, made.JUDGE, question, persona, label)
    ok = (None, "ok", "", 1, {})  # value, status, error, attempts, details
    return ScoreRow(*asked, *ok, text_sha256=sha, output_sha256=sha, created_at=now())


def test_verdicts_about_another_persona_are_not_the_candidate_s(
    saved: Callable[..., Saved],
) -> None:
    spec, runs, _ = saved()
    folder = runs / "tiny-screen"
    replies = {run.run_id: run for run in read_results(folder)}
    owls = [run for run in replies.values() if run.output_text.startswith("The owl")]
    rows = [_row(run, "Tin Fox", "speaks_as", "a" * 64) for run in owls]
    rows += [_row(run, "Brass Owl", "assistant", "b" * 64) for run in owls[:2]]
    with locked(folder):
        write_rows(folder, rows, replies)
    owl = read_scored(spec, runs).stats[0]
    # Only the two rows about the owl's own persona count.
    assert (owl.total.runs, owl.total.n, owl.total.k) == (5, 2, 0)


@pytest.mark.parametrize(
    ("situation", "problem"),
    [
        ("not run", "does not exist: no run of tiny-screen is saved there yet"),
        ("not judged", "no speaks_as or coherence verdict is saved yet: `uv run "),
        ("no coherence", "has an ok coherence verdict of fake:fake: `uv run pushlab"),
        ("two judges", "choose one with --judge"),
        ("no list", "is missing; `uv run pushlab candidates --model tiny-vl"),
    ],
)
def test_read_scored_says_what_to_do_first(
    situation: str, problem: str, saved: Callable[..., Saved]
) -> None:
    spec, runs, listed = saved("", run=situation != "not run")
    if situation in ("no coherence", "two judges"):
        made.score(spec, runs, "speaks_as", made.answers(SPEAKING), listed)
    if situation == "two judges":
        made.score(spec, runs, "speaks_as", made.answers({}), listed, judge="other")
        assert read_scored(spec, runs, judge="fake:other").stats[0].total.k == 0
    if situation == "no list":
        listed.unlink()
    with pytest.raises(ValueError) as caught:
        read_scored(spec, runs, needs=("speaks_as", "coherence"))
    assert problem in str(caught.value)
    if situation == "no coherence":
        make = f"`uv run pushlab score {spec} --rubric coherence` makes them"
        assert str(caught.value).endswith(make)


def test_an_experiment_without_candidates_is_refused(
    tiny_spec: Callable[..., Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    made.invent(monkeypatch)
    spec, runs = tiny_spec(), tmp_path / "runs"
    made.run(spec, runs)
    with pytest.raises(ValueError, match="pushes no candidates of a list"):
        read_scored(spec, runs)


def test_two_candidates_of_one_push_cannot_be_told_apart() -> None:
    unit = NeuronUnit("qwen3-8b", 9, 4242)
    owl = Candidate("library", "Brass Owl", unit, 1, 1.0, False, "f", "r")
    twin = Candidate("mayner", "Owl Twin", unit, 1, 1.0, False, "f", "r")
    assert push_key(twin) == push_key(owl) == "qwen3-8b:L9:N4242+"
    with pytest.raises(ValueError, match="both push qwen3-8b:L9:N4242"):
        candidate_stats([owl, twin], {owl.key: 0, twin.key: 1}, [], [], made.JUDGE)


def test_figures_add_replies_up() -> None:
    none = Figures()
    assert (none.runs, none.n, none.rate, none.coherence) == (0, 0, None, None)
    replies = (
        Reply("r1", "p1", "a", True, 7.0),
        Reply("r2", "p1", "b", False),
        Reply("r3", "p2", "c", None, 9.0),
    )
    figures = Figures(replies)
    assert (figures.runs, figures.k, figures.n, figures.rate) == (3, 1, 2, 0.5)
    assert (figures.scored, figures.coherence) == (2, 8.0)
    assert figures.prompts == ("p1", "p2")
    unit = NeuronUnit("qwen3-8b", 1, 3)
    owl = Candidate("library", "Brass Owl", unit, 1, 1.0, False, "f", "r")
    doses = {0.5: Figures(replies[:1]), 1.0: Figures(replies[1:])}
    both = CandidateStats(owl, 0, doses)
    assert both.total == figures
    points = [(point.dose, point.prompt, point.speaks) for point in both.points()]
    assert points == [(0.5, "p1", True), (1.0, "p1", False), (1.0, "p2", None)]


@pytest.mark.parametrize("module", [stats, figures])
def test_the_examples_in_the_docstrings_hold(module: object) -> None:
    result = doctest.testmod(module)  # type: ignore[arg-type]
    assert result.failed == 0 and result.attempted > 0
