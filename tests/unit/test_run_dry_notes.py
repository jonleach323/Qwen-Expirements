"""`pushlab run --dry-run`: its notes, and the numbers behind its estimates.

The dry run says why pushlab run would refuse (a calibration not made yet,
a changed experiment file), when it has to guess because it cannot read the
model's tokenizer, and when the speeds file is broken. It counts the GPU
time a new version keeps, adds the measured load time once, and adds none
when nothing is left to run. One run, batch or row is said in the singular.
The costs fixture has tiny-vl at 0.01 reply tokens a second, so tiny_spec's
10 runs of 4 tokens take 4,000 s.
"""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from pushlab.cli import main
from pushlab.models import load
from pushlab.runs import estimate, runner
from pushlab.runs.folder import read_index

NAME = "tiny-pulses"
NO_PUSH = "conditions:\n  - {label: none, segments: []}\n"
SAMPLED = "{strategy: sample, temperature: 0.7, dtype: float32}"
# tiny_spec's two conditions, with the pulse at half the dose.
HALF_DOSE = """\
conditions:
  - label: none
    segments: []
  - label: pulse
    segments: [{unit: tiny-vl:L2:N100, alpha: 0.5}]
"""
NO_TOKENIZER = "without the model's tokenizer, batches are counted one prompt each"


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _run(spec: Path, runs: Path, *options: str) -> int:
    return main(["run", str(spec), "--runs-dir", str(runs), *options])


def _dry_run(
    spec: Path, runs: Path, capsys: pytest.CaptureFixture[str], *options: str
) -> tuple[dict[str, str], list[str]]:
    """Dry-run spec, which must exit 0. Return its lines of the form
    'name  value', by name, and its notes without "note: " and the full stop."""
    capsys.readouterr()
    assert _run(spec, runs, "--dry-run", *options) == 0
    out = capsys.readouterr().out.splitlines()
    pairs = [line.split("  ", 1) for line in out if "  " in line]
    notes = [
        line.removeprefix("note: ").removesuffix(".")
        for line in out
        if line.startswith("note: ")
    ]
    return {name.strip(): value.strip() for name, value in pairs}, notes


def _no_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the model's tokenizer unreadable, as where it is not downloaded."""

    def missing(name: str, **kwargs: object) -> None:
        raise FileNotFoundError(f"{name}'s tokenizer is not here")

    monkeypatch.setattr(load, "load_tokenizer", missing)


def _batches_take(monkeypatch: pytest.MonkeyPatch, seconds: int) -> None:
    """A stand-in clock on which each batch takes seconds (the first one
    with the model's load)."""
    ticks = iter(range(0, 10**9, seconds))
    monkeypatch.setattr(runner, "_clock", lambda: float(next(ticks)))


def _save_speed(runs: Path, **measured: object) -> None:
    """Save a speed for tiny-vl in float32 as a run does: 0.01 reply tokens a
    second, over batches of 2 rows, on the CPU; measured adds or replaces keys."""
    speed = {
        "reply_tokens_per_second": 0.01,
        "rows_per_batch": 2.0,
        "batch_size": 2,
        "experiment": NAME,
        "measured_at": "2026-11-01T12:00:00+00:00",
        "gpu": None,
    }
    runs.mkdir(exist_ok=True)
    estimate.save_speed(runs, "tiny-vl/float32", speed | measured)


def test_a_dry_run_without_a_calibration_says_why_the_run_would_refuse(
    tiny_spec: Callable[..., Path],
    tiny_vl_calibration: Path,
    costs: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Downloaded but not calibrated yet, as on a pod before `pushlab calibrate`.
    tiny_vl_calibration.unlink()
    spec, runs = tiny_spec(), tmp_path / "runs"
    commit = load.downloaded_revision("tiny-vl")
    problem = (
        f"tiny-vl at {commit} has no calibration in {tiny_vl_calibration.parent}; "
        "on the GPU pod, run `uv run pushlab calibrate --model tiny-vl`"
    )
    settings, notes = _dry_run(spec, runs, capsys)
    # Only that note: nothing says the model is not downloaded.
    assert notes == [f"pushlab run would refuse this: {problem}"]
    assert settings["to run"] == "10 of 10 runs; none is saved yet"
    # The run itself refuses with the same words.
    assert _run(spec, runs) == 1
    assert f"pushlab run: {problem}" in capsys.readouterr().err


def test_a_changed_folder_is_counted_as_overwrite_version_would_run_it(
    tiny_spec: Callable[..., Path],
    costs: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs = tmp_path / "runs"
    assert _run(tiny_spec(), runs, "--limit", "4") == 3
    settings, notes = _dry_run(tiny_spec(max_gpu_hours="2"), runs, capsys)
    assert settings["to run"] == (
        "6 of 10 runs; 4 runs saved in version 1; with --overwrite-version, a new "
        "version would keep 4 runs of version 1"
    )
    assert settings["reply tokens"] == "at most 24, 4 for each run"
    assert notes == [
        f"pushlab run would refuse this: {runs / NAME} holds version 1 of {NAME}, "
        "made from another experiment file. Run with --overwrite-version to start "
        "version 2, which keeps the saved runs that did not change, or move the "
        "folder away to start afresh"
    ]


def test_under_sampling_without_the_tokenizer_the_kept_runs_are_a_guess(
    tiny_spec: Callable[..., Path],
    costs: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs = tmp_path / "runs"
    assert _run(tiny_spec(decoding=SAMPLED), runs, "--limit", "4") == 3
    changed = tiny_spec(decoding=SAMPLED, max_gpu_hours="2")
    # With the tokenizer, each run's row in the new plan is known.
    assert _dry_run(changed, runs, capsys, "--overwrite-version")[1] == []
    _no_tokenizer(monkeypatch)
    settings, notes = _dry_run(changed, runs, capsys, "--overwrite-version")
    assert settings["to run"] == (
        "6 of 10 runs; a new version would keep 4 runs of version 1"
    )
    assert notes == [
        NO_TOKENIZER,
        "under sampling a new version keeps a saved run only on its row, which "
        "needs the tokenizer here: the runs it keeps are a guess",
    ]


def test_without_the_tokenizer_each_prompt_counts_as_its_own_length(
    tiny_spec: Callable[..., Path],
    costs: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The tiny tokenizer gives each byte a token: the first two are one length.
    three = tmp_path / "three.jsonl"
    three.write_text(
        '{"prompt_id": "p-1", "version": 1, "prompt": "Say hi."}\n'
        '{"prompt_id": "p-2", "version": 1, "prompt": "Say ho."}\n'
        '{"prompt_id": "p-3", "version": 1, "prompt": "Count to three."}\n'
    )
    spec = tiny_spec(NO_PUSH, prompts=str(three), batch_size="8")
    runs = tmp_path / "runs"
    settings, notes = _dry_run(spec, runs, capsys)
    assert settings["run in"] == (
        "2 batches of 1 to 2 runs, prompts of one length together"
    )
    assert notes == []
    _no_tokenizer(monkeypatch)
    settings, notes = _dry_run(spec, runs, capsys)
    assert settings["run in"] == "3 batches of 1 run, prompts of one length together"
    assert notes == [NO_TOKENIZER]


def test_a_dry_run_of_a_new_version_counts_the_gpu_time_it_keeps(
    tiny_spec: Callable[..., Path],
    costs: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _batches_take(monkeypatch, 1200)
    runs = tmp_path / "runs"
    assert _run(tiny_spec(), runs, "--limit", "4") == 3
    assert [entry["seconds"] for entry in read_index(runs / NAME)] == [1200, 1200]
    (runs / "speeds.json").unlink()  # estimate at the costs file's speed
    # A new dose: the new version keeps the unpushed run of each batch, and
    # so half of each batch's time, 1,200 s: 0.67 of the hour is left.
    changed = tiny_spec(HALF_DOSE)
    settings, _ = _dry_run(changed, runs, capsys, "--overwrite-version")
    assert settings["to run"] == (
        "8 of 10 runs; a new version would keep 2 runs of version 1"
    )
    assert settings["gpu-hours"] == (
        "about 1.11, more than the 0.67 left of max_gpu_hours 1: it would stop "
        "after about 60% of the runs"
    )


def test_estimates_add_the_measured_load_time_once(
    tiny_spec: Callable[..., Path],
    costs: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = tiny_spec(max_gpu_hours="3"), tmp_path / "runs"
    _save_speed(runs)  # no load time, as in speeds files before they had one
    settings, _ = _dry_run(spec, runs, capsys)
    assert settings["speed"] == (
        f"0.01 reply tokens a second, measured by {NAME} on the CPU with 2 rows "
        "a batch, 2026-11-01"
    )
    assert settings["gpu-hours"] == "about 1.11, within max_gpu_hours 3"
    # An hour to load the model: one hour more, not one for each batch.
    _save_speed(runs, load_seconds=3600)
    settings, _ = _dry_run(spec, runs, capsys)
    assert settings["gpu-hours"] == "about 2.11, within max_gpu_hours 3"


def test_with_nothing_left_to_run_no_load_time_is_added(
    tiny_spec: Callable[..., Path],
    costs: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _batches_take(monkeypatch, 360)
    spec, runs = tiny_spec(), tmp_path / "runs"
    assert _run(spec, runs) == 0  # every run saved, in 1,800 s
    _save_speed(runs, load_seconds=36_000)  # ten hours to load the model
    settings, notes = _dry_run(spec, runs, capsys)
    assert settings["to run"] == "0 of 10 runs; 10 runs are saved in version 1"
    assert settings["gpu-hours"] == (
        "about 0.00, within the 0.50 left of max_gpu_hours 1"
    )
    assert notes == []


@pytest.mark.parametrize(
    "broken", [{"tiny-vl/float32": 5}, [1]], ids=["a-number", "a-list"]
)
def test_a_speeds_file_of_another_shape_is_left_out_of_estimates(
    tiny_spec: Callable[..., Path],
    costs: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    broken: object,
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "speeds.json").write_text(json.dumps(broken))
    settings, notes = _dry_run(tiny_spec(), runs, capsys)  # exit 0, no crash
    assert notes == [
        f"{runs / 'speeds.json'} is not a speeds file as pushlab run writes it; "
        "delete it, and the next run measures the speed again. Estimates leave it "
        "out"
    ]
    assert settings["speed"] == f"0.01 reply tokens a second, from {costs}"


def test_one_run_one_batch_and_one_row_are_said_in_the_singular(
    tiny_spec: Callable[..., Path],
    costs: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = tiny_spec(batch_size="1"), tmp_path / "runs"
    assert _run(spec, runs, "--limit", "1") == 3  # its speed: 1 row a batch
    settings, _ = _dry_run(spec, runs, capsys, "--limit", "1")
    assert settings["to run"] == (
        "1 of 10 runs; 1 run is saved in version 1 (--limit 1)"
    )
    assert settings["run in"] == "1 batch of 1 run, prompts of one length together"
    assert f"measured by {NAME} on the CPU with 1 row a batch, " in settings["speed"]


def test_a_measured_speed_without_a_batch_size_is_read_by_its_rows(
    tiny_spec: Callable[..., Path],
    costs: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = tiny_spec(), tmp_path / "runs"
    runs.mkdir()
    speed = {  # as a hand-edited file might have it: rows a batch, no batch size
        "reply_tokens_per_second": 0.01,
        "rows_per_batch": 2.0,
        "experiment": NAME,
        "measured_at": "2026-11-01T12:00:00+00:00",
    }
    (runs / "speeds.json").write_text(json.dumps({"tiny-vl/float32": speed}))
    settings, notes = _dry_run(spec, runs, capsys)
    assert "with 2 rows a batch" in settings["speed"] and notes == []
    assert main(["status", str(spec), "--runs-dir", str(runs)]) == 0
    assert "with 2 rows a batch" in capsys.readouterr().out
