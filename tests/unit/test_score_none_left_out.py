"""`pushlab score SPEC --rubric speaks_as` with nothing to judge because
every saved reply is left out (pushlab.scoring.score_command): a screen of
the one candidate of the made-up list whose persona is named "None", the
judge's answer for no persona, or of a candidate the units file does not
name. It says so, as the header's lines count them, and keeps "no reply is
saved yet" for a folder that has no reply.

The runner saves the made-up screen (tests/unit/panel_experiments.py) with
invented replies on the five smoke prompts. The judge is a FakeBackend that
answers nothing, so any call would fail: nothing reaches the network.
"""

import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from pushlab.cli import main
from pushlab.engine import generate as generate_module
from pushlab.scoring.cache import VerdictCache
from pushlab.scoring.fake_backend import FakeBackend
from pushlab.scoring.judge import Judge
from pushlab.scoring.score_command import run_score

_SPEC = importlib.util.spec_from_file_location(
    "panel_experiments", Path(__file__).with_name("panel_experiments.py")
)
made = importlib.util.module_from_spec(_SPEC)  # type: ignore[arg-type]
_SPEC.loader.exec_module(made)  # type: ignore[union-attr]

SCREEN = "tiny-screen"
LEFT_OUT = "Nothing to judge: every saved reply is left out, as the lines above say."
NOT_SAVED = "Nothing to judge: no reply is saved yet."

Screen = tuple[Path, Path]  # the screen's file and its list


@pytest.fixture
def screen(
    tiny_vl_calibration: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Screen:
    """A screen of the made-up list's oracle alone, its persona named "None",
    written but not run."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("PUSHLAB_CACHE_DIR", str(tmp_path / "default-cache"))
    made.invent(monkeypatch)
    listed = made.candidates()
    listed["oracle"] = replace(listed["oracle"], persona="None")
    path = made.write_list(tmp_path, list(listed.values()))
    spec = made.write_screen(tmp_path, path)
    every = f"candidates: {{file: {path}}}"
    alone = f"candidates: {{file: {path}, keys: [{made.KEYS['oracle']}]}}"
    text = spec.read_text(encoding="utf-8")
    assert every in text
    spec.write_text(text.replace(every, alone), encoding="utf-8")
    return spec, path


def _judge(cache: Path) -> tuple[FakeBackend, Judge]:
    """A judge that answers nothing: a call to it fails the test."""
    fake = FakeBackend([], name="fake", model="fake")
    return fake, Judge(fake, VerdictCache(cache, "fake:fake"), sleep=lambda s: None)


def test_a_screen_of_the_none_candidate_alone_says_every_reply_is_left_out(
    screen: Screen, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec, _ = screen
    runs = tmp_path / "runs"
    made.run(spec, runs)
    capsys.readouterr()  # the runner's
    # The dry run counts the oracle's replies out and says nothing else.
    assert run_score(spec, "speaks_as", runs_dir=runs, dry_run=True) == 0
    out = capsys.readouterr().out
    assert '1 unit of the file with the persona "none"' in out
    assert "and their 5 replies" in out and "Nothing to judge" not in out
    # The real run says, as the header does, that every reply is left out,
    # not that none is saved.
    fake, judge = _judge(tmp_path / "cache")
    assert run_score(spec, "speaks_as", runs_dir=runs, judge=judge) == 0
    out = capsys.readouterr().out
    assert f"{SCREEN}, version 1: 5 replies, 0 failed runs left out" in out
    assert "and their 5 replies" in out
    assert LEFT_OUT in out and NOT_SAVED not in out
    assert fake.requests == []
    assert not (runs / SCREEN / "scores").exists()


def test_replies_of_no_unit_of_the_units_file_are_left_out_too(
    screen: Screen, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec, _ = screen
    runs = tmp_path / "runs"
    made.run(spec, runs)
    capsys.readouterr()  # the runner's
    # A units file of the owl alone: the oracle's replies are of no unit of it.
    owl = made.write_list(tmp_path / "owl", [made.candidates()["owl"]])
    fake, judge = _judge(tmp_path / "cache")
    options = {"runs_dir": runs, "units_file": owl, "judge": judge}
    assert run_score(spec, "speaks_as", **options) == 0  # type: ignore[arg-type]
    out = capsys.readouterr().out
    assert "5 replies pushed by no unit of the file" in out
    assert 'persona "none"' not in out
    assert LEFT_OUT in out and NOT_SAVED not in out
    assert fake.requests == []


def test_a_folder_whose_every_run_failed_has_no_reply_saved(
    screen: Screen,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, _ = screen
    runs = tmp_path / "runs"

    def fail(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(generate_module, "generate", fail)
    assert main(["run", str(spec), "--runs-dir", str(runs)]) == 2  # failed runs
    capsys.readouterr()  # the runner's
    # A failed run has no reply: none is left out, as none is saved, with
    # the list's units or without.
    for rubric in ("speaks_as", "coherence"):
        fake, judge = _judge(tmp_path / f"cache-{rubric}")
        assert run_score(spec, rubric, runs_dir=runs, judge=judge) == 0
        out = capsys.readouterr().out
        assert f"{SCREEN}, version 1: 0 replies, 5 failed runs left out" in out
        assert NOT_SAVED in out and LEFT_OUT not in out
        assert fake.requests == []
