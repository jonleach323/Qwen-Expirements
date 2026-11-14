"""A screen whose candidates include one with a persona named "none", run
and judged as on the pod, then read by pushlab panel (pushlab.panel.stats,
stages and select_command; decision 0023).

The runner saves a made-up screen of tiny-vl candidates
(tests/unit/panel_experiments.py) whose oracle is named "None", with
invented replies, and a fake judge judges them through pushlab score, which
never asks whether a reply speaks as "none", the judge's answer for no
persona: the oracle's replies are not counted as replies to judge, stage b
notes them apart, and an experiment whose only replies are of such
candidates is refused, saying so, before any verdict is read.
test_panel_none_stats.py checks the same from stats made by hand.
"""

import importlib.util
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from pushlab.cli import main
from pushlab.panel.stats import read_scored
from pushlab.runs.shards import read_results
from pushlab.schedules.yaml_text import load_yaml

_SPEC = importlib.util.spec_from_file_location(
    "panel_experiments", Path(__file__).with_name("panel_experiments.py")
)
made = importlib.util.module_from_spec(_SPEC)  # type: ignore[arg-type]
_SPEC.loader.exec_module(made)  # type: ignore[union-attr]

KEYS = made.KEYS
# How the refusal of the oracle's replies alone begins, and its advice when
# the experiment pushes no other candidate.
REFUSAL = (
    f"every candidate of tiny-screen with a saved reply, such as {KEYS['oracle']}, "
    'has the persona "none", the judge\'s answer for no persona, which the judge '
    "cannot be asked about, so no reply can have a speaks_as verdict: "
)
OTHERS = "give an experiment that pushes candidates of other personas"
Saved = tuple[Path, Path, Path]  # the screen's file, its runs dir, its settings


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run on the CPU even with a GPU, and keep any cache out of /workspace."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("PUSHLAB_CACHE_DIR", str(tmp_path / "default-cache"))


@pytest.fixture
def screen(
    tiny_vl_calibration: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[..., Saved]:
    """Save a screen of the made-up list whose oracle is named "None": of
    the candidates of roles, in that order, else of every one; with limit,
    in batches of one run, the first limit runs alone, as a run stopped
    early saves them (the runner batches each prompt's runs together)."""

    def save(roles: Sequence[str] = (), limit: int | None = None) -> Saved:
        made.invent(monkeypatch)
        listed = made.candidates()
        listed["oracle"] = replace(listed["oracle"], persona="None")
        path = made.write_list(tmp_path, list(listed.values()))
        spec = made.write_screen(tmp_path, path)
        text = spec.read_text(encoding="utf-8")
        if limit is not None:
            text = text.replace("batch_size: 8", "batch_size: 1")
        if roles:
            keys = ", ".join(KEYS[role] for role in roles)
            text = text.replace(
                f"{{file: {path}}}", f"{{file: {path}, keys: [{keys}]}}"
            )
        spec.write_text(text, encoding="utf-8")
        runs = tmp_path / "runs"
        argv = ["run", str(spec), "--runs-dir", str(runs)]
        if limit is None:
            assert main(argv) == 0
        else:  # it stops with runs left
            assert main([*argv, "--limit", str(limit)]) == 3
        return spec, runs, made.write_settings(tmp_path / "panel.yaml")

    return save


def test_the_oracle_s_replies_are_not_counted_as_replies_to_judge(
    screen: Callable[..., Saved],
) -> None:
    spec, runs, _ = screen()
    made.score(spec, runs, "speaks_as", made.answers(made.SCREEN_SPEAKS))
    scored = read_scored(spec, runs)
    oracle = scored.by_key[KEYS["oracle"]]
    assert scored.unasked == (oracle,) and oracle.total.runs == 5
    # Each other candidate's 5 replies have a verdict and the oracle's none,
    # yet no reply is left to judge.
    assert [each.total.n for each in scored.stats] == [5, 5, 5, 5, 0]
    assert scored.unjudged("speaks_as") == 0
    # Coherence judges every reply, the oracle's too.
    assert scored.unjudged("coherence") == 25
    made.score(spec, runs, "coherence", made.answers({}))
    scored = read_scored(spec, runs, needs=("speaks_as", "coherence"))
    assert scored.unjudged("coherence") == 0
    assert scored.by_key[KEYS["oracle"]].total.coherence == 9


def test_stage_b_notes_the_oracle_s_replies_apart(
    screen: Callable[..., Saved], capsys: pytest.CaptureFixture[str]
) -> None:
    spec, runs, settings = screen()
    speaks = made.answers(made.SCREEN_SPEAKS, failing=made.SCREEN_FAILS)
    made.score(spec, runs, "speaks_as", speaks)
    out = spec.parent / "experiments" / "stage-b.yaml"
    options = ["--runs-dir", str(runs), "--panel", str(settings)]
    capsys.readouterr()
    assert made.stage("b", spec, out, *options, "--max-gpu-hours", "1") == 0
    lines = capsys.readouterr().out.splitlines()
    # The teapot's reply with no verdict is the only one pushlab score can
    # judge; the oracle's 5 can never have one.
    assert lines[1].startswith(
        "Note: 1 reply has no ok speaks_as verdict of fake:fake: `uv run pushlab "
    )
    assert lines[2] == (
        'Note: 5 replies of 1 candidate with the persona "none", the judge\'s '
        "answer for no persona, can have no speaks_as verdict, as the judge "
        "cannot be asked about it: left out."
    )
    assert any(" of 4 ranked (1 more has no speaks_as verdict):" in x for x in lines)
    keys = load_yaml(out.read_text(encoding="utf-8"))["candidates"]["keys"]
    assert keys == [KEYS[role] for role in ("owl", "teapot", "captain", "fox")]


@pytest.mark.parametrize(
    ("situation", "then"),
    [
        ("not judged", OTHERS),
        ("judged", OTHERS),
        ("others unrun", "`uv run pushlab run {spec}` runs the other candidates"),
    ],
)
def test_the_oracle_s_replies_alone_are_refused_saying_why(
    situation: str, then: str, screen: Callable[..., Saved]
) -> None:
    if situation == "others unrun":  # the oracle's first reply saved, no owl's
        spec, runs, _ = screen(["oracle", "owl"], limit=1)
        texts = [run.output_text for run in read_results(runs / "tiny-screen")]
        assert texts == ["The oracle at 0.75 answers prompt 0."]
    else:
        spec, runs, _ = screen(["oracle"])
    if situation == "judged":  # as on the pod: no speaks_as verdict, coherence's
        made.score(spec, runs, "speaks_as", made.answers({}))
        made.score(spec, runs, "coherence", made.answers({}))
    with pytest.raises(ValueError) as caught:
        read_scored(spec, runs, needs=("speaks_as", "coherence"))
    assert str(caught.value) == REFUSAL + then.format(spec=spec)
    if situation == "judged":  # coherence alone can be read
        scored = read_scored(spec, runs, needs=("coherence",))
        assert scored.unasked == scored.stats and scored.unjudged("coherence") == 0


def test_a_stage_and_select_refuse_the_oracle_s_replies_alone(
    screen: Callable[..., Saved], capsys: pytest.CaptureFixture[str]
) -> None:
    spec, runs, settings = screen(["oracle"])
    made.score(spec, runs, "speaks_as", made.answers({}))
    folder = spec.parent
    out = folder / "experiments" / "stage-b.yaml"
    options = ["--runs-dir", str(runs), "--panel", str(settings)]
    capsys.readouterr()
    assert made.stage("b", spec, out, *options, "--max-gpu-hours", "1") == 1
    assert capsys.readouterr().err == f"pushlab panel stage b: {REFUSAL}{OTHERS}\n"
    assert not out.exists()
    argv = ["panel", "select", str(spec), "--ladder", str(spec)]
    argv += ["--categories", str(folder / "categories.csv")]
    argv += ["--out", str(folder / "panel_tiny-vl.yaml"), "--doc", str(folder / "p.md")]
    assert main([*argv, *options]) == 1
    # select names the file it refuses, which its words do not.
    err = capsys.readouterr().err
    assert err == f"pushlab panel select: {spec}: {REFUSAL}{OTHERS}\n"
    assert not (folder / "categories.csv").exists()
