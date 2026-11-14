"""`pushlab doses SPEC --units FILE` with a unit whose persona is named
"none", the judge's answer for no persona (pushlab.analysis.doses_command,
pushlab.analysis.no_persona). pushlab score asks nothing about such a unit,
so doses leaves it out of the report and counts it and its runs in the
report's head, in score's words; only a file whose every unit is named so
is refused.

The runner saves an invented ladder of tiny-vl on the smoke prompts, as in
test_doses_command.py: no push, the owl at 0.25 to 0.75 with a random
control each, and the captain at -0.25 to -0.75. The units file names the
owl's persona "None", as a library persona can be. A fake judge says the
captain speaks as Salt Captain from 0.5 on, and that its text breaks at 0.75.
"""

import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from pushlab.analysis.no_persona import no_persona_text
from pushlab.cli import main
from pushlab.engine import generate as generate_module
from pushlab.engine.generate import Generation
from pushlab.prompts import PROMPTS_DIR, read_prompt_set
from pushlab.scoring.cache import VerdictCache
from pushlab.scoring.fake_backend import FakeBackend
from pushlab.scoring.judge import Judge
from pushlab.scoring.rubrics import Request
from pushlab.scoring.score_command import run_score

NAME, JUDGE = "tiny-ladder", "fake:fake"
SMOKE = read_prompt_set(PROMPTS_DIR / "smoke.jsonl")
OWL, CAPTAIN, GHOST = "tiny-vl:L1:N3", "tiny-vl:L2:N100", "tiny-vl:L3:N5"
LADDER = f"""\
conditions:
  - {{label: none, segments: []}}
  - {{label: u01, shape: constant, params: {{unit: {OWL}}}, controls: 1,
     control_seed: 0, grid: {{alpha: [0.25, 0.5, 0.75]}}}}
  - {{label: u02, shape: constant, params: {{unit: {CAPTAIN}}},
     grid: {{alpha: [-0.25, -0.5, -0.75]}}}}
"""
HEADER = "unit,direction,persona,description,source\n"
NONE_OWL = f"{OWL},+1,None,A pompous mechanical owl.,invented\n"
SALT_CAPTAIN = f"{CAPTAIN},-1,Salt Captain,A gruff sea captain.,invented\n"
ROLES = {OWL: "owl", CAPTAIN: "captain"}  # a random control's is "control"
TEXT = re.compile(r"The (\w+) at ([\d.]+) answers prompt (\d)\.")
# The owl's 15 runs and its random controls' 15 are its runs.
LEFT_OUT = "the judge's answer for no persona, and their 30 runs: left out"
Saved = tuple[Path, Path, Path]  # the experiment file, runs dir and units file


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run on the CPU even with a GPU, and keep any cache out of /workspace."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("PUSHLAB_CACHE_DIR", str(tmp_path / "default-cache"))


def _answer(request: Request) -> str:
    """The fake judge's verdict on an invented reply: only the captain
    speaks as its persona, from 0.5 on, and only its text at 0.75 breaks."""
    found = TEXT.fullmatch(request.text)
    assert found is not None, request.text
    role, dose = found[1], float(found[2])
    if request.rubric == "coherence":
        score = 4 if (role, dose) == ("captain", 0.75) else 9
        return json.dumps({"reason": "Fine.", "score": score})
    label = "speaks_as" if role == "captain" and dose >= 0.5 else "assistant"
    return json.dumps({"reason": "Arr.", "label": label})


@pytest.fixture
def saved(
    tiny_spec: Callable[..., Path], monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Saved:
    """The ladder saved by the runner, each reply "The <role> at <dose>
    answers prompt <i>.", and judged with the owl's persona named "None"."""

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
    spec = tiny_spec(LADDER, name=NAME, max_new_tokens="64")
    runs, units = tmp_path / "runs", tmp_path / "units.csv"
    assert main(["run", str(spec), "--runs-dir", str(runs)]) == 0
    units.write_text(HEADER + NONE_OWL + SALT_CAPTAIN, encoding="utf-8")
    cache = VerdictCache(tmp_path / "cache", JUDGE)
    for rubric, options in [("speaks_as", {"units_file": units}), ("coherence", {})]:
        fake = FakeBackend(_answer, name="fake", model="fake")
        judge = Judge(fake, cache, sleep=lambda seconds: None)
        assert run_score(spec, rubric, runs_dir=runs, judge=judge, **options) == 0
    return spec, runs, units


def _doses(saved: Saved, units_text: str, *options: str) -> int:
    """pushlab doses on the ladder, its units file holding units_text."""
    spec, runs, units = saved
    units.write_text(units_text, encoding="utf-8")
    argv = ["doses", str(spec), "--units", str(units), "--runs-dir", str(runs)]
    return main([*argv, *options])


def _lines(capsys: pytest.CaptureFixture[str]) -> list[str]:
    """The lines printed, each with its runs of spaces made one."""
    return [" ".join(line.split()) for line in capsys.readouterr().out.splitlines()]


def test_a_unit_named_none_is_left_out_and_counted_in_the_head(
    saved: Saved, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    outs = [tmp_path / "with-none.json", tmp_path / "captain-only.json"]
    assert _doses(saved, HEADER + NONE_OWL + SALT_CAPTAIN, "--out", str(outs[0])) == 0
    with_none = _lines(capsys)
    assert _doses(saved, HEADER + SALT_CAPTAIN, "--out", str(outs[1])) == 0
    captain_only = _lines(capsys)
    # Without the owl's row, its runs push a unit the file lacks; with its
    # persona named "None" they are its, and the head says so in score's
    # words. Nothing else differs: the owl has no part, and the captain's
    # doses are the same.
    lacks = "note: 30 runs push units the units file lacks: left out"
    named = f'note: 1 unit of the file with the persona "none", {LEFT_OUT}'
    assert lacks in captain_only and f"{CAPTAIN}-: Salt Captain (invented)" in with_none
    expected = [named if line == lacks else line for line in captain_only]
    assert with_none[:-1] == expected[:-1]  # the last line names the --out file
    assert not [line for line in with_none if OWL in line]
    records = [json.loads(out.read_text(encoding="utf-8")) for out in outs]
    assert [unit["unit"] for unit in records[0]["units"]] == [f"{CAPTAIN}-"]
    assert records[0]["units"] == records[1]["units"]


def test_a_unit_named_none_needs_no_run_and_two_are_counted_together(
    saved: Saved, capsys: pytest.CaptureFixture[str]
) -> None:
    ghost = f"{GHOST},+1, NONE ,,invented\n"  # no run of the ladder pushes it
    assert _doses(saved, HEADER + NONE_OWL + SALT_CAPTAIN + ghost) == 0
    lines = _lines(capsys)
    assert f'note: 2 units of the file with the persona "none", {LEFT_OUT}' in lines
    assert "working dose 0.5 (95% interval 0.5)" in lines  # the captain's


def test_a_file_whose_every_unit_is_named_none_is_refused(
    saved: Saved, capsys: pytest.CaptureFixture[str]
) -> None:
    every = HEADER + NONE_OWL + SALT_CAPTAIN.replace("Salt Captain", "none")
    assert _doses(saved, every) == 1
    assert capsys.readouterr().err == (
        f'pushlab doses: every unit of {saved[2]} has the persona "none", which is '
        "the judge's answer for no persona: pushlab score judges none of their "
        "replies, so there is no dose to find\n"
    )


def test_the_note_counts_units_and_runs_one_or_many() -> None:
    assert no_persona_text(2, 1) == (
        '2 units of the file with the persona "none", the judge\'s answer for no '
        "persona, and their 1 run"
    )
    assert no_persona_text(1, 1_200).startswith("1 unit of the file with ")
    assert no_persona_text(1, 1_200).endswith(", and their 1,200 runs")
