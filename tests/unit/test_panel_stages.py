"""pushlab panel stage b and ladder (pushlab.panel.stages, decision 0023):
each builds the next experiment from the last one's scores.

A made-up screen of five tiny-vl candidates (tests/unit/panel_experiments.py)
is saved by the runner with invented replies and judged by a fake judge;
stage b writes its file from those verdicts, which the runner then runs
and the judge judges, and stage ladder writes its file from them, on a
small selection of the smoke prompts. Stage confirm and preview have files
of their own; the file's text and checks are in test_panel_stages_files.py.
"""

import doctest
import importlib.util
from collections.abc import Callable
from fractions import Fraction
from pathlib import Path

import pytest
import torch

from pushlab import PROJECT_ROOT
from pushlab.analysis.panel import read_panel
from pushlab.discovery.candidates import Candidate
from pushlab.panel import stages
from pushlab.panel.settings import read_selection
from pushlab.panel.stages import NoneKept, over_ceiling, ranked
from pushlab.panel.stats import CandidateStats, Figures, Reply, Scored
from pushlab.runs.expand import expand
from pushlab.runs.ladder import default_hours
from pushlab.runs.spec import read_spec
from pushlab.runs.spec_candidates import CandidatesRule
from pushlab.schedules.yaml_text import load_yaml
from pushlab.scoring.items import Experiment
from pushlab.units.neuron import NeuronUnit

_SPEC = importlib.util.spec_from_file_location(
    "panel_experiments", Path(__file__).with_name("panel_experiments.py")
)
made = importlib.util.module_from_spec(_SPEC)  # type: ignore[arg-type]
_SPEC.loader.exec_module(made)  # type: ignore[union-attr]

KEYS = made.KEYS


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run on the CPU even with a GPU, and keep any cache out of /workspace."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("PUSHLAB_CACHE_DIR", str(tmp_path / "default-cache"))


@pytest.fixture
def pipeline(
    tiny_vl_calibration: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> "made.Pipeline":
    """The screen saved and judged, ready for stage b."""
    return made.Pipeline(tmp_path, monkeypatch)


def _settings(path: Path) -> dict:
    """A stage's file as load_yaml reads it."""
    return load_yaml(path.read_text(encoding="utf-8"))


def _alphas(path: Path) -> dict[str, list[float]]:
    """Each condition's alphas, by candidate, in the experiment's order."""
    spec = read_spec(path)
    return {
        condition.candidate: [s.segments[0].alpha_start for _, s in condition.points]
        for condition in spec.conditions
    }


def test_stage_b_keeps_the_highest_rates_ties_to_more_verdicts_then_the_list(
    pipeline: "made.Pipeline", capsys: pytest.CaptureFixture[str]
) -> None:
    assert pipeline.stage("b", pipeline.screen) == 0
    out = pipeline.out("b")
    settings = _settings(out)
    # The owl's 5 of 5 before the teapot's 4 of 4; the captain's 3 of 5
    # before the fox's, as the list has them; the oracle's none left out.
    keys = [KEYS[role] for role in ("owl", "teapot", "captain", "fox")]
    assert settings["candidates"] == {"file": str(pipeline.listed), "keys": keys}
    assert settings["conditions"] == [
        {
            "label": "candidate",
            "each_candidate": {"doses": [1.0], "allow_over_ceiling": True},
        }
    ]
    # The selection's settings: every smoke prompt, greedy, fixed calibration.
    assert "prompt_ids" not in settings
    assert (settings["name"], settings["model"]) == ("stage-b", "tiny-vl")
    assert (settings["prompts"], settings["decoding"]) == (
        "prompts/smoke.jsonl",
        {"strategy": "greedy"},
    )
    assert [settings[k] for k in ("seeds", "max_new_tokens", "batch_size")] == [
        [0],
        64,
        8,
    ]
    assert (settings["calibration"], settings["max_gpu_hours"]) == ("fixed", 1.0)
    # Each at its reference dose: 4 candidates x 5 prompts.
    spec = read_spec(out)
    labels = [condition.label for condition in spec.conditions]
    assert labels == [f"candidate-000{n}" for n in range(1, 5)]
    assert _alphas(out) == dict(zip(keys, [[1.0], [1.0], [-0.5], [1.25]], strict=True))
    assert len(expand(spec).runs) == 20
    # Each key's comment gives what the screen showed of it.
    text = out.read_text(encoding="utf-8")
    assert f"    - {KEYS['teapot']}  # 4/4 speaks_as, Velvet Teapot\n" in text
    assert text.startswith("# Stage b of the persona panel (decision 0023)")
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith(f"Read {pipeline.screen}: tiny-screen, version 1, 25 ")
    score = f"`uv run pushlab score {pipeline.screen} --rubric speaks_as --units "
    assert lines[1].startswith(
        "Note: 1 reply has no ok speaks_as verdict of fake:fake: " + score
    )
    assert any("4 with the highest speaks_as rates" in line for line in lines)
    table = [" ".join(line.split()) for line in lines]
    assert f"2 {KEYS['teapot']} Velvet Teapot set 5 4/4 1.00 -" in table
    assert f"Wrote {out}: the experiment stage-b, on tiny-vl." in lines
    ladder = pipeline.out("ladder")
    assert f"  uv run pushlab panel stage ladder {out} --out {ladder}" in lines


def test_stage_ladder_takes_stage_b_s_best_at_each_dose_on_the_first_prompts(
    pipeline: "made.Pipeline",
) -> None:
    out = pipeline.to_ladder()
    settings = _settings(out)
    # Stage b's verdicts: the teapot 5 of 5, the captain 4, the owl 3.
    keys = [KEYS[role] for role in ("teapot", "captain", "owl")]
    assert settings["candidates"]["keys"] == keys
    assert settings["prompt_ids"] == ["smoke-01", "smoke-02", "smoke-03"]
    assert settings["conditions"][0]["each_candidate"] == {
        "doses": [0.5, 1.0, 1.5],
        "allow_over_ceiling": True,
    }
    # The captain's reference dose is 0.5, pushed down.
    assert _alphas(out) == {
        keys[0]: [0.5, 1.0, 1.5],
        keys[1]: [-0.25, -0.5, -0.75],
        keys[2]: [0.5, 1.0, 1.5],
    }
    spec = read_spec(out)
    assert [label for label, _ in spec.conditions[1].points] == [
        "candidate-0002 alpha=-0.25",
        "candidate-0002 alpha=-0.5",
        "candidate-0002 alpha=-0.75",
    ]
    assert len(expand(spec).runs) == 3 * 3 * 3  # candidates x doses x prompts


def test_max_gpu_hours_comes_from_the_model_s_speed_as_pushlab_ladder_s(
    pipeline: "made.Pipeline", costs: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert pipeline.stage("b", pipeline.screen, "--default-hours") == 0
    hours = default_hours(20, 64, 0.01)  # costs: tiny-vl at 0.01 tokens a second
    assert _settings(pipeline.out("b"))["max_gpu_hours"] == hours
    out = capsys.readouterr().out
    assert "about 1.5 times the time its 1,280 reply tokens take at 0.01" in out


@pytest.mark.parametrize(
    ("where", "problem"),
    [
        ("repository", "is inside the repository"),
        ("exists", "already exists; pass --overwrite to replace it"),
        ("folder", "is a folder"),
        ("source", "the file the stage reads: give another"),
    ],
)
def test_a_bad_out_is_refused_before_anything_is_written(
    where: str,
    problem: str,
    pipeline: "made.Pipeline",
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = {
        "repository": PROJECT_ROOT / "experiments" / "s3" / "stage-b-test.yaml",
        "exists": pipeline.out("b"),
        "folder": pipeline.folder,
        "source": pipeline.screen,
    }[where]
    if where == "exists":
        out.parent.mkdir(parents=True)
        out.write_text("kept\n", encoding="utf-8")
    argv = ["--runs-dir", str(pipeline.runs), "--panel", str(pipeline.settings)]
    assert made.stage("b", pipeline.screen, out, *argv, "--max-gpu-hours", "1") == 1
    err = capsys.readouterr().err
    assert err.startswith("pushlab panel stage b: ") and problem in err
    if where == "repository":
        assert not out.exists()
    if where == "exists":
        assert out.read_text(encoding="utf-8") == "kept\n"
        assert pipeline.stage("b", pipeline.screen, "--overwrite") == 0
        assert "candidates" in _settings(out)


def test_a_bad_out_says_why(
    pipeline: "made.Pipeline", capsys: pytest.CaptureFixture[str]
) -> None:
    inside = PROJECT_ROOT / "experiments" / "s3" / "stage-b-test.yaml"
    argv = ["--runs-dir", str(pipeline.runs), "--panel", str(pipeline.settings)]
    assert made.stage("b", pipeline.screen, inside, *argv) == 1
    err = capsys.readouterr().err
    assert err.startswith(f"pushlab panel stage b: {inside} is inside the repository")
    assert "/workspace/experiments/stage-b-test.yaml" in err and not inside.exists()
    assert made.stage("b", pipeline.screen, pipeline.screen, *argv) == 1
    assert "the file the stage reads: give another" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("situation", "problem"),
    [
        ("not run", "does not exist: no run of tiny-screen is saved there yet"),
        ("not judged", "no speaks_as or coherence verdict is saved yet"),
        ("no list", "is missing; `uv run pushlab candidates --model tiny-vl"),
        ("no candidates", "pushes no candidates of a list"),
    ],
)
def test_a_stage_refuses_an_experiment_it_cannot_read(
    situation: str,
    problem: str,
    tiny_vl_calibration: Path,
    tiny_spec: Callable[..., Path],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    made.invent(monkeypatch)
    listed = made.write_list(tmp_path)
    settings = made.write_settings(tmp_path / "panel.yaml")
    spec, runs = made.write_screen(tmp_path, listed), tmp_path / "runs"
    if situation == "no candidates":
        spec = tiny_spec()
    if situation != "not run":
        made.run(spec, runs)
    if situation == "no list":
        listed.unlink()
    out = tmp_path / "experiments" / "stage-b.yaml"
    argv = ["--runs-dir", str(runs), "--panel", str(settings), "--max-gpu-hours", "1"]
    capsys.readouterr()  # the runner's
    assert made.stage("b", spec, out, *argv) == 1
    err = capsys.readouterr().err
    assert err.startswith("pushlab panel stage b: ") and problem in err
    assert not out.exists() and not out.parent.exists()


def _stats(key: int, place: int, verdicts: list[bool | None]) -> CandidateStats:
    unit = NeuronUnit("qwen3-8b", 9, key)
    candidate = Candidate("library", f"Owl {key}", unit, 1, 1.0, False, "f", "r")
    replies = tuple(Reply(f"r{i}", f"p{i}", "", s) for i, s in enumerate(verdicts))
    return CandidateStats(candidate, place, {1.0: Figures(replies)})


def test_ranked_sorts_by_rate_then_verdicts_then_the_list_s_order() -> None:
    # In the experiment's order: 1 of 2; 2 of 4; 3 of 3; 1 of 2 again but
    # earlier in the list; 2 of 2; and none with a verdict, left out.
    found = [
        _stats(1, 4, [True, False]),
        _stats(2, 1, [True, True, False, False]),
        _stats(3, 2, [True, True, True]),
        _stats(4, 0, [False, True, None]),
        _stats(5, 3, [True, True]),
        _stats(6, 5, [None, None]),
    ]
    order = [each.candidate.unit.index for each in ranked(found)]
    assert order == [3, 5, 2, 4, 1]
    assert [Fraction(e.total.k, e.total.n) for e in ranked(found)][-1] == 0.5


def test_a_note_says_when_the_experiment_file_changed_since_it_ran(
    pipeline: "made.Pipeline", capsys: pytest.CaptureFixture[str]
) -> None:
    screen = pipeline.screen
    text = screen.read_text(encoding="utf-8")
    screen.write_text(text.replace("max_gpu_hours: 1", "max_gpu_hours: 2"), "utf-8")
    capsys.readouterr()
    assert pipeline.stage("b", screen) == 0
    notes = [line for line in capsys.readouterr().out.splitlines() if "Note:" in line]
    assert notes[0].startswith("Note: The experiment file changed since version 1")


def test_a_stage_with_no_candidate_ranked_keeps_none(tmp_path: Path) -> None:
    spec = read_spec(PROJECT_ROOT / "experiments" / "p0" / "s2_pulse_lengths.yaml")
    identity = {"version": 1, "spec_sha256": spec.sha256}
    experiment = Experiment(spec, tmp_path, identity, [], 0)
    rule = CandidatesRule("units/c.csv")
    unjudged = _stats(1, 0, [None, None])
    scored = Scored(Path("s.yaml"), experiment, rule, "api:x", ("p0",), (unjudged,), 0)
    with pytest.raises(NoneKept, match="has a speaks_as verdict: `uv run pushlab"):
        stages.stage_b(scored, read_selection(), read_panel())
    with pytest.raises(NoneKept) as caught:
        stages.stage_ladder(scored, read_selection(), read_panel())
    assert caught.value.shown[0] == (
        "Read s.yaml: s2-pulse-lengths, version 1, 2 replies of its 1 candidate, "
        "judged by api:x; 0 failed runs left out."
    )


def test_over_ceiling_reads_what_the_conditions_allowed(
    pipeline: "made.Pipeline",
) -> None:
    assert over_ceiling(read_spec(pipeline.screen))
    assert pipeline.stage("b", pipeline.screen) == 0
    assert over_ceiling(read_spec(pipeline.out("b")))


def test_the_examples_in_the_docstrings_hold() -> None:
    result = doctest.testmod(stages)
    assert result.failed == 0 and result.attempted > 0
