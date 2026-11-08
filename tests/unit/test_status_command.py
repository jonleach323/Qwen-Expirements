"""`pushlab status`: how far an experiment has got, and how long the rest will take.

Also the help of `pushlab run` and `pushlab status`.
"""

import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from pushlab.cli import main
from pushlab.engine import generate as generate_module
from pushlab.prompts import PROMPTS_DIR, read_prompt_set

NAME = "tiny-pulses"


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _run(spec: Path, runs: Path, *options: str) -> int:
    return main(["run", str(spec), "--runs-dir", str(runs), *options])


def _status(
    spec: Path, runs: Path, capsys: pytest.CaptureFixture[str]
) -> dict[str, str]:
    capsys.readouterr()
    assert main(["status", str(spec), "--runs-dir", str(runs)]) == 0
    lines = [line.split("  ", 1) for line in capsys.readouterr().out.splitlines()]
    return {name.strip(): value.strip() for name, value in lines if value}


@pytest.mark.parametrize("command", ["run", "status"])
def test_help_lists_the_options(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([command, "--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "SPEC" in help_text and "--runs-dir" in help_text
    if command == "run":
        for option in ("--dry-run", "--limit", "--retry-failed", "--overwrite-version"):
            assert option in help_text


@pytest.mark.parametrize("command", ["run", "status"])
def test_the_installed_command_answers_help(command: str) -> None:
    pushlab = Path(sys.executable).with_name("pushlab")
    result = subprocess.run(
        [str(pushlab), command, "--help"], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert f"usage: pushlab {command}" in result.stdout


def test_status_before_anything_ran(
    tiny_spec: Callable[..., Path],
    costs: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs = tmp_path / "runs"
    status = _status(tiny_spec(), runs, capsys)
    assert status["runner"] == f"not started: {runs / NAME} does not exist yet"
    assert status["runs"] == "10: 0 ok, 0 failed, 10 pending"
    assert status["time left"] == (
        f"at most 67 minutes, at 0.01 reply tokens a second, from {costs}"
    )
    assert not runs.exists()


def test_status_part_way(
    tiny_spec: Callable[..., Path],
    costs: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = tiny_spec(), tmp_path / "runs"
    assert _run(spec, runs, "--limit", "4") == 0
    status = _status(spec, runs, capsys)
    assert status["experiment"] == f"{NAME}, version 1, in {runs / NAME}"
    assert status["runner"].startswith("not running; its last session ended at ")
    assert status["runs"] == "10: 4 ok, 0 failed, 6 pending"
    assert status["batches"] == "2 saved"
    assert status["speed"].endswith("reply tokens a second, measured over 2 batches")
    assert status["gpu-hours"].endswith("used of max_gpu_hours 1")
    assert status["time left"].startswith("about ")
    assert status["time left"].endswith("for 6 runs at that speed")


def test_status_names_failed_runs_and_a_changed_file(
    tiny_spec: Callable[..., Path],
    costs: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = tiny_spec(batch_size="1"), tmp_path / "runs"
    third = read_prompt_set(PROMPTS_DIR / "smoke.jsonl").prompts[2]
    real = generate_module.generate

    def failing(model, tokenizer, prompts, *args, schedules=None, **kwargs):  # type: ignore[no-untyped-def]
        if prompts == [third] and schedules and schedules[0].segments:
            raise RuntimeError("boom")
        return real(model, tokenizer, prompts, *args, schedules=schedules, **kwargs)

    monkeypatch.setattr(generate_module, "generate", failing)
    assert _run(spec, runs) == 2
    status = _status(spec, runs, capsys)
    assert status["runs"] == "10: 9 ok, 1 failed, 0 pending"
    assert status["failed"] == f"{NAME}-000006: RuntimeError: boom"
    assert status["time left"] == "none: every run is saved"
    status = _status(tiny_spec(batch_size="1", max_gpu_hours="2"), runs, capsys)
    assert status["note"] == (
        "the experiment file changed since version 1: pushlab run will refuse it "
        "without --overwrite-version"
    )


def test_status_tells_a_live_runner_from_one_that_died(
    tiny_spec: Callable[..., Path],
    costs: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = tiny_spec(), tmp_path / "runs"
    assert _run(spec, runs, "--limit", "2") == 0
    beat = runs / NAME / "heartbeat.json"
    record = json.loads(beat.read_text())
    beat.write_text(json.dumps(record | {"state": "running", "pid": 1}))  # init lives
    assert _status(spec, runs, capsys)["runner"].startswith("running, pid 1;")
    dead = subprocess.run(
        [sys.executable, "-c", "import os; print(os.getpid())"],
        capture_output=True,
        text=True,
    )
    beat.write_text(json.dumps(record | {"state": "running", "pid": int(dead.stdout)}))
    assert _status(spec, runs, capsys)["runner"] == (
        f"stopped without finishing: pid {int(dead.stdout)} is gone; run the same "
        "command to go on"
    )


def test_status_of_a_folder_where_no_session_ran_yet(
    tiny_spec: Callable[..., Path],
    costs: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = tiny_spec(), tmp_path / "runs"
    assert _run(spec, runs, "--limit", "1") == 0  # its first batch is larger
    status = _status(spec, runs, capsys)
    assert status["runner"] == "not running: no session has started"
    assert status["runs"] == "10: 0 ok, 0 failed, 10 pending"
    assert status["time left"].startswith("at most 67 minutes, at 0.01 reply tokens")


def test_status_of_a_runner_on_another_machine(
    tiny_spec: Callable[..., Path],
    costs: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = tiny_spec(), tmp_path / "runs"
    assert _run(spec, runs, "--limit", "2") == 0
    beat = runs / NAME / "heartbeat.json"
    record = json.loads(beat.read_text()) | {"state": "running", "host": "pod-2"}
    beat.write_text(json.dumps(record))
    runner = _status(spec, runs, capsys)["runner"]
    assert (
        runner == f"running on pod-2, pid {record['pid']}, as of {record['updated_at']}"
    )


def test_status_names_five_failed_runs_and_counts_the_rest(
    tiny_spec: Callable[..., Path],
    costs: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def failing(*args: object, **kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(generate_module, "generate", failing)
    spec, runs = tiny_spec(), tmp_path / "runs"
    assert _run(spec, runs) == 2
    capsys.readouterr()
    assert main(["status", str(spec), "--runs-dir", str(runs)]) == 0
    failed = [
        line for line in capsys.readouterr().out.splitlines() if "failed  " in line
    ]
    assert len(failed) == 6 and failed[-1].endswith("and 5 more")


def test_status_refuses_what_it_cannot_read(
    tiny_spec: Callable[..., Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = tmp_path / "runs"
    (runs / NAME).mkdir(parents=True)
    (runs / NAME / "notes.txt").write_text("mine")
    assert main(["status", str(tiny_spec()), "--runs-dir", str(runs)]) == 1
    assert "is not a runner folder: it holds notes.txt" in capsys.readouterr().err
    assert main(["status", str(tmp_path / "missing.yaml")]) == 1
    assert "pushlab status: " in capsys.readouterr().err
