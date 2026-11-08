"""`pushlab verify` against experiment.json.

Check 1: the version was made from this experiment file, prompt set and
units/known.csv; if not, that is one problem. Check 7: manifest.json is a
valid experiment manifest, with experiment.json's experiment, version,
experiment file, hashes and model download, the model revision that
download names, its number of runs, and the index's counts by status.

The experiment is HANDOFF on the five smoke prompts, as in test_verify.py:
10 runs in 5 batches of 2.
"""

import json
import shutil
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from pushlab.cli import main
from pushlab.prompts import PROMPTS_DIR
from pushlab.units import controls

NAME = "tiny-pulses"
A, B = "tiny-vl:L2:N100", "tiny-vl:L1:N3"
HANDOFF = f"""\
conditions:
  - {{label: none, segments: []}}
  - label: handoff
    shape: handoff
    params: {{a: {A}, b: {B}, alpha_a: 1.0, alpha_b: -1.0, switch_at: 2}}
"""
# A pulse and a random control of it on each smoke prompt, so the runs
# depend on units/known.csv, which the controls avoid.
CONTROLLED = f"""\
conditions:
  - label: pulse
    shape: pulse
    params: {{unit: {A}, alpha: 1.0, start: 0, length: 2}}
    controls: 1
    control_seed: 0
"""


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _run(spec: Path, runs: Path, *options: str) -> int:
    return main(["run", str(spec), "--runs-dir", str(runs), *options])


def _verify(spec: Path, runs: Path, capsys: pytest.CaptureFixture[str]) -> list[str]:
    """Run verify; its exit status, then the lines it prints, summary last."""
    capsys.readouterr()
    status = main(["verify", str(spec), "--runs-dir", str(runs)])
    captured = capsys.readouterr()
    assert captured.err == ""
    return [str(status), *captured.out.splitlines()]


@pytest.fixture
def finished(tiny_spec: Callable[..., Path], tmp_path: Path) -> tuple[Path, Path]:
    """The handoff experiment run to the end: its file and its runs dir."""
    spec, runs = tiny_spec(HANDOFF, max_new_tokens="8"), tmp_path / "runs"
    assert _run(spec, runs) == 0
    return spec, runs


def test_a_version_made_from_another_prompt_set_or_known_units_is_one_problem(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prompts, known = tmp_path / "prompts.jsonl", tmp_path / "known.csv"
    shutil.copy(PROMPTS_DIR / "smoke.jsonl", prompts)
    shutil.copy(controls.KNOWN_UNITS_FILE, known)
    monkeypatch.setattr(controls, "KNOWN_UNITS_FILE", known)
    spec = tiny_spec(CONTROLLED, prompts=str(prompts))
    runs = tmp_path / "runs"
    assert _run(spec, runs) == 0
    assert _verify(spec, runs, capsys)[0] == "0"
    original = prompts.read_text()
    prompts.write_text(original.replace("pancakes", "waffles"))
    not_compared = "so its runs are not compared with what this file plans"
    assert _verify(spec, runs, capsys)[:-1] == [
        "1",
        f"version 1 was made from another prompt set, {not_compared}",
    ]
    prompts.write_text(original)
    # Another label for a known unit: the same units, but another file.
    known.write_text(known.read_text().replace("evil-Qwen pair", "evil pair"))
    assert _verify(spec, runs, capsys)[:-1] == [
        "1",
        f"version 1 was made from another known units, {not_compared}",
    ]


# A value for each of experiment.json's fields that manifest.json repeats.
PLANTED = {
    "experiment": "another-experiment",
    "version": 7,
    "spec_file": "experiments/other.yaml",
    "spec_sha256": "2" * 64,
    "prompts_sha256": "0" * 64,
    "known_sha256": "1" * 64,
    "model": f"tiny-vl@{'f' * 40}",
    "calibration_sha256": "3" * 64,
}


def _plant(runs: Path, **values: object) -> None:
    """Set some fields of the folder's manifest.json."""
    path = runs / NAME / "manifest.json"
    path.write_text(json.dumps(json.loads(path.read_text()) | values))


@pytest.mark.parametrize("key", PLANTED)
def test_each_field_the_manifest_repeats_from_experiment_json_is_checked(
    key: str, finished: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    spec, runs = finished
    _plant(runs, **{key: PLANTED[key]})
    assert _verify(spec, runs, capsys)[:-1] == [
        "1",
        f"manifest.json's {key} is not experiment.json's",
    ]


def test_the_manifests_model_revision_and_run_count_are_checked(
    finished: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    spec, runs = finished
    path, identity = runs / NAME / "manifest.json", "experiment.json"
    original = path.read_text()
    _plant(runs, model_revision="f" * 40)
    assert _verify(spec, runs, capsys)[:-1] == [
        "1",
        f"manifest.json's model_revision is not the revision in {identity}'s model",
    ]
    path.write_text(original)
    _plant(runs, run_count=999)
    assert _verify(spec, runs, capsys)[:-1] == [
        "1",
        "manifest.json is not a valid experiment manifest: runs_by_status adds up "
        "to 10 runs, not run_count 999",
        f"manifest.json's run_count is 999, but {identity}'s version has 10 runs",
    ]


@pytest.mark.parametrize(
    ("values", "why"),
    [
        ({"sessions": []}, "sessions must list each session, with its id, "),
        ({"finished_at": "2020-01-01T00:00:00+00:00"}, "finished_at is before "),
    ],
    ids=["no session", "finished before it started"],
)
def test_a_manifest_that_is_not_valid_is_a_problem(
    values: dict[str, object],
    why: str,
    finished: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = finished
    _plant(runs, **values)
    lines = _verify(spec, runs, capsys)
    assert lines[0] == "1" and len(lines) == 3
    assert lines[1].startswith(
        f"manifest.json is not a valid experiment manifest: {why}"
    )


def test_a_new_version_that_keeps_every_run_verifies_clean(
    tiny_spec: Callable[..., Path],
    finished: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Another limit makes another experiment file but the same runs: version
    # 2 keeps them all, runs no session and restamps the manifest.
    spec, runs = finished
    assert spec == tiny_spec(HANDOFF, max_new_tokens="8", max_gpu_hours="2")
    capsys.readouterr()
    assert _run(spec, runs, "--overwrite-version") == 0
    assert "it keeps 10 runs saved" in capsys.readouterr().out
    lines = _verify(spec, runs, capsys)
    assert lines[0] == "0"
    assert lines[1].startswith(f"{NAME}, version 2: 10 runs and ")
