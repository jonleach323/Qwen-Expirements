"""`pushlab score SPEC --rubric speaks_as` on an experiment that names its
units from a candidates list (candidates:, decision 0023): with no --units,
--personas or --persona, each candidate's replies are asked about its own
persona, as --units with the experiment's list asks them
(pushlab.scoring.score_units.listed_units).

The runner saves a made-up screen of tiny-vl candidates
(tests/unit/panel_experiments.py) with invented replies, and a FakeBackend
judges: nothing reaches the network.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

from pushlab.scoring.cache import VerdictCache
from pushlab.scoring.fake_backend import FakeBackend
from pushlab.scoring.judge import Judge
from pushlab.scoring.score_command import run_score

_SPEC = importlib.util.spec_from_file_location(
    "panel_experiments", Path(__file__).with_name("panel_experiments.py")
)
made = importlib.util.module_from_spec(_SPEC)  # type: ignore[arg-type]
_SPEC.loader.exec_module(made)  # type: ignore[union-attr]

Saved = tuple[Path, Path, Path]  # the screen's file, its runs dir and its list


@pytest.fixture
def saved(
    tiny_vl_calibration: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Saved:
    """The screen of the made-up list, run on the CPU and saved."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("PUSHLAB_CACHE_DIR", str(tmp_path / "default-cache"))
    made.invent(monkeypatch)
    listed = made.write_list(tmp_path)
    screen = made.write_screen(tmp_path, listed)
    runs = tmp_path / "runs"
    made.run(screen, runs)
    return screen, runs, listed


def _judge(script: object, cache: Path) -> tuple[FakeBackend, Judge]:
    fake = FakeBackend(script, name="fake", model="fake")  # type: ignore[arg-type]
    return fake, Judge(fake, VerdictCache(cache, "fake:fake"), sleep=lambda s: None)


def test_speaks_as_asks_each_candidate_about_its_own_persona_by_default(
    saved: Saved, tmp_path: Path
) -> None:
    screen, runs, listed = saved
    fake, judge = _judge(made.answers(made.SCREEN_SPEAKS), tmp_path / "cache")
    assert run_score(screen, "speaks_as", runs_dir=runs, judge=judge) == 0
    # Each candidate's own persona, not one for all: as many questions as
    # personas asked about.
    assert len({request.question for request in fake.requests}) > 1
    # The same job with the list as --units finds every reply scored: it asks
    # the same questions, so the second judge, which answers nothing, is
    # never called.
    again, judge = _judge([], tmp_path / "cache")
    assert (
        run_score(screen, "speaks_as", runs_dir=runs, units_file=listed, judge=judge)
        == 0
    )
    assert again.requests == []


def test_a_dry_run_names_the_list_it_asks_with(
    saved: Saved, capsys: pytest.CaptureFixture[str]
) -> None:
    screen, runs, listed = saved
    assert run_score(screen, "speaks_as", runs_dir=runs, dry_run=True) == 0
    out = capsys.readouterr().out
    assert f"of {listed}: each unit's replies and the unpushed ones" in out


@pytest.mark.parametrize(
    ("options", "problem"),
    [
        ({"persona": "Brass Owl"}, "speaks_as asks about one persona: give --personas"),
        ({"rubric": "which_persona"}, "which_persona chooses among personas"),
    ],
)
def test_a_persona_option_or_another_rubric_is_not_given_the_list(
    saved: Saved,
    capsys: pytest.CaptureFixture[str],
    options: dict[str, str],
    problem: str,
) -> None:
    screen, runs, _ = saved
    rubric = options.pop("rubric", "speaks_as")
    assert run_score(screen, rubric, runs_dir=runs, dry_run=True, **options) == 1  # type: ignore[arg-type]
    assert problem in capsys.readouterr().err
