"""pushlab panel stage confirm (pushlab.panel.stages, decision 0023): each
ladder candidate that works below its collapse dose, at its working dose,
on every prompt.

The pipeline of tests/unit/panel_experiments.py writes, runs and judges the
screen, stage b and the ladder with made-up verdicts: on the ladder the
teapot works at 1 and collapses at 1.5, the captain (reference dose 0.5)
works at 0.25 and collapses at 0.75, and the owl works only at 1.5, past
its collapse at 1.
"""

import importlib.util
import math
from pathlib import Path

import pytest
import torch

from pushlab.analysis.ladder import DoseEstimate, UnitDoses
from pushlab.panel.stages import dose_multiple, works
from pushlab.runs.expand import expand
from pushlab.runs.spec import read_spec
from pushlab.schedules.yaml_text import load_yaml

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
def ladder(
    tiny_vl_calibration: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple["made.Pipeline", Path]:
    """The pipeline up to the ladder's file, not yet run."""
    pipeline = made.Pipeline(tmp_path, monkeypatch)
    return pipeline, pipeline.to_ladder()


def test_confirm_keeps_each_that_works_before_it_collapses_at_its_working_dose(
    ladder: tuple["made.Pipeline", Path], capsys: pytest.CaptureFixture[str]
) -> None:
    pipeline, path = ladder
    pipeline.judged(path, made.LADDER_SPEAKS, made.LADDER_COHERENCE)
    capsys.readouterr()
    assert pipeline.stage("confirm", path) == 0
    out = pipeline.out("confirm")
    settings = load_yaml(out.read_text(encoding="utf-8"))
    # Each as a multiple of its reference dose: the captain's 0.25 is half
    # its 0.5. The owl does not go on.
    assert settings["candidates"]["keys"] == [
        {"key": KEYS["teapot"], "doses": [1.0]},
        {"key": KEYS["captain"], "doses": [0.5]},
    ]
    assert settings["conditions"] == [
        {"label": "candidate", "each_candidate": {"allow_over_ceiling": True}}
    ]
    assert "prompt_ids" not in settings  # every prompt
    spec = read_spec(out)
    pushed = {
        condition.candidate: [s.segments[0].alpha_start for _, s in condition.points]
        for condition in spec.conditions
    }
    assert pushed == {KEYS["teapot"]: [1.0], KEYS["captain"]: [-0.25]}
    assert len(expand(spec).runs) == 2 * 5
    text = out.read_text(encoding="utf-8")
    assert "# working dose 0.25, collapse dose 0.75, Salt Captain\n" in text
    lines = [" ".join(line.split()) for line in capsys.readouterr().out.splitlines()]
    owl = f"{KEYS['owl']} Brass Owl 1.5 [1.0, 1.5] 1.0 [1.0, 1.0] none: it "
    assert any(line.startswith(owl) and line.endswith(" no") for line in lines)
    assert "Kept 2 of 3, each at its working dose." in lines
    select = f"uv run pushlab panel select {out} --ladder {path} --categories"
    assert any(line.startswith(select) for line in lines)


def test_confirm_refuses_when_none_works_and_shows_why(
    ladder: tuple["made.Pipeline", Path], capsys: pytest.CaptureFixture[str]
) -> None:
    pipeline, path = ladder
    # Every reply speaks as its persona, but none is coherent: each works
    # at its lowest dose, and collapses there too.
    doses = {"teapot": (0.5, 1.0, 1.5), "captain": (0.25, 0.5, 0.75)}
    doses["owl"] = (0.5, 1.0, 1.5)
    everywhere = [(role, dose) for role, each in doses.items() for dose in each]
    speaking = dict.fromkeys(everywhere, range(3))
    pipeline.judged(path, speaking, dict.fromkeys(everywhere, [2]))
    capsys.readouterr()
    assert pipeline.stage("confirm", path) == 1
    shown = capsys.readouterr()
    assert shown.err.startswith(
        "pushlab panel stage confirm: no candidate of stage-ladder works below its "
        "collapse dose"
    )
    rows = [line for line in shown.out.splitlines() if line.endswith(" no")]
    assert len(rows) == 3 and not pipeline.out("confirm").exists()


def test_confirm_needs_the_ladder_s_coherence(
    ladder: tuple["made.Pipeline", Path], capsys: pytest.CaptureFixture[str]
) -> None:
    pipeline, path = ladder
    pipeline.judged(path, made.LADDER_SPEAKS)  # speaks_as alone
    capsys.readouterr()
    assert pipeline.stage("confirm", path) == 1
    err = capsys.readouterr().err
    assert "has an ok coherence verdict of fake:fake" in err
    assert f"`uv run pushlab score {path} --rubric coherence` makes them" in err


def test_dose_multiple_gives_the_multiple_each_candidate_pushes_at_a_dose() -> None:
    # The first known multiple that gives the dose exactly, as each_candidate
    # works it out: 1.1 x 0.75 is 0.8250000000000001.
    assert dose_multiple(1.1 * 0.75, 1.1, [0.5, 0.75, 1.0]) == 0.75
    assert dose_multiple(0.625, 1.25, []) == 0.5
    # Else a float that gives the dose exactly, near dose / strength.
    strength = 1.3
    for multiple in (0.37, 0.61, 1.17, 2.9):
        dose = strength * multiple
        found = dose_multiple(dose, strength)
        assert strength * found == dose and math.isclose(found, multiple)


def test_works_needs_a_working_dose_below_any_collapse_dose() -> None:
    def doses(working: float | None, collapse: float | None) -> UnitDoses:
        found = [DoseEstimate(dose, None, None, 0) for dose in (working, collapse)]
        return UnitDoses({}, *found)

    assert works(doses(0.5, 1.0)) and works(doses(0.5, None))
    assert not works(doses(1.0, 1.0)) and not works(doses(None, 0.5))
    assert not works(doses(None, None))
