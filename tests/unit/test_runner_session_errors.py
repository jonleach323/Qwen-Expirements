"""How a session of `pushlab run` stops on errors, its speed record and the lock.

Errors and Ctrl-C before any batch, the speed saved after a failed batch, a
broken speeds file, the lock, the retry hints, and a finished experiment run
again. tiny_spec's experiment has 10 runs in 5 batches of 2.
"""

import dataclasses
import fcntl
import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch

from pushlab.cli import main
from pushlab.engine import generate as generate_module
from pushlab.models import load
from pushlab.prompts import PROMPTS_DIR, read_prompt_set
from pushlab.runs import folder as folder_module
from pushlab.runs import runner
from pushlab.runs.folder import FolderInUse, locked, read_index, read_results

NAME = "tiny-pulses"


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _run(spec: Path, runs: Path, *options: str) -> int:
    return main(["run", str(spec), "--runs-dir", str(runs), *options])


def _raises(error: BaseException) -> Callable[..., None]:
    def raising(*args: object, **kwargs: object) -> None:
        raise error

    return raising


def _no_loading(*args: object, **kwargs: object) -> None:
    raise AssertionError("the model must not load")


def _prompt(number: int) -> str:
    """The text of the smoke prompt of that number, counted from 1."""
    return read_prompt_set(PROMPTS_DIR / "smoke.jsonl").prompts[number - 1]


def _state(folder: Path) -> str:
    """What the folder's heartbeat says the runner is doing."""
    return json.loads((folder / "heartbeat.json").read_text())["state"]


def test_an_os_error_while_the_model_loads_is_refused_naming_that_step(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setattr(load, "load_model", _raises(OSError("no tokenizer files")))
    assert _run(tiny_spec(), runs) == 1
    err = capsys.readouterr().err
    assert "pushlab run: cannot load tiny-vl: no tokenizer files" in err
    assert _state(runs / NAME).startswith("stopped: cannot load tiny-vl")
    assert read_index(runs / NAME) == []  # no batch saved


def test_any_other_error_while_the_model_loads_stops_with_its_traceback(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs = tmp_path / "runs"
    monkeypatch.setattr(load, "load_model", _raises(RuntimeError("CUDA out of memory")))
    assert _run(tiny_spec(), runs) == 1
    err = capsys.readouterr().err
    assert "Traceback (most recent call last)" in err
    message = "cannot load tiny-vl: RuntimeError: CUDA out of memory"
    assert f"pushlab run: {message}" in err
    assert _state(runs / NAME) == f"stopped: {message}"


def test_ctrl_c_while_the_tokenizer_loads_stops_before_any_batch(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs = tmp_path / "runs"
    # The runner loads the tokenizer to group the prompts by their length.
    monkeypatch.setattr(load, "load_tokenizer", _raises(KeyboardInterrupt()))
    try:
        status = _run(tiny_spec(), runs)
    except KeyboardInterrupt:  # caught here, or it would end the whole test run
        pytest.fail("Ctrl-C escaped pushlab run")
    assert status == 130
    assert "Stopped by Ctrl-C before any batch ran" in capsys.readouterr().out
    assert read_index(runs / NAME) == []


def _speed_with_a_failed_batch(
    spec: Callable[..., Path], tmp_path: Path, fail: Callable[..., None]
) -> tuple[dict, list[dict]]:
    """Run with smoke-03's batch failing: the speed saved, and the index."""
    runs = tmp_path / "runs"
    fail(_prompt(3), RuntimeError("boom"))
    assert _run(spec(), runs) == 2
    speeds = json.loads((runs / "speeds.json").read_text())
    return speeds["tiny-vl/float32"], read_index(runs / NAME)


def test_rows_per_batch_counts_only_the_batches_that_ran_ok(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    failing_generate: Callable[..., None],
) -> None:
    speed, entries = _speed_with_a_failed_batch(tiny_spec, tmp_path, failing_generate)
    errors = [entry["error"] for entry in entries]
    assert errors == [None, None, "RuntimeError: boom", None, None]
    assert speed["rows_per_batch"] == 2.0  # 8 rows in 4 batches, not 8 in 5


def test_the_measured_speed_leaves_failed_batches_out(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    failing_generate: Callable[..., None],
) -> None:
    speed, entries = _speed_with_a_failed_batch(tiny_spec, tmp_path, failing_generate)
    ok = [entry for entry in entries if entry["error"] is None]
    (failed,) = [entry for entry in entries if entry["error"] is not None]
    assert failed["generate_seconds"] > 0  # so counting it would show
    tokens = sum(entry["reply_tokens"] for entry in ok)
    seconds = sum(entry["generate_seconds"] for entry in ok)
    assert (speed["reply_tokens"], speed["seconds"]) == (tokens, seconds)
    assert speed["reply_tokens_per_second"] == tokens / seconds


def test_a_speeds_file_of_the_wrong_shape_is_warned_about_and_the_run_goes_on(
    tiny_spec: Callable[..., Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "speeds.json").write_text("[1]")  # valid JSON, but not a speeds file
    assert _run(tiny_spec(), runs) == 0
    err = capsys.readouterr().err
    assert f"pushlab run: warning: {runs / 'speeds.json'} is not a speeds file" in err
    assert len(read_index(runs / NAME)) == 5  # every batch is saved
    assert _state(runs / NAME) == "finished"
    assert (runs / "speeds.json").read_text() == "[1]"  # left for a person


@contextmanager
def _shared_lock(folder: Path) -> Iterator[int]:
    """Hold a shared lock on the folder's .lock, as pushlab status takes one."""
    handle = os.open(folder / ".lock", os.O_RDONLY | os.O_CREAT, 0o644)
    try:
        fcntl.flock(handle, fcntl.LOCK_SH)
        yield handle
    finally:
        os.close(handle)


def test_the_lock_waits_a_moment_for_a_shared_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = tmp_path / NAME
    folder.mkdir()
    waits: list[float] = []
    with _shared_lock(folder) as handle:

        def status_lets_go(seconds: float) -> None:  # while the runner waits
            waits.append(seconds)
            fcntl.flock(handle, fcntl.LOCK_UN)

        monkeypatch.setattr(folder_module.time, "sleep", status_lets_go)
        with locked(folder):
            pass
    assert len(waits) == 1  # it waited once, then took the lock
    # Held for longer than the runner waits: refused after a tenth of a second.
    monkeypatch.setattr(folder_module.time, "sleep", waits.append)
    with _shared_lock(folder), pytest.raises(FolderInUse), locked(folder):
        pass
    assert sum(waits[1:]) == pytest.approx(0.1)


def test_nothing_to_retry_names_the_runs_still_to_run(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = tiny_spec(), tmp_path / "runs"
    assert _run(spec, runs, "--limit", "4") == 3  # none failed, 6 not run yet
    monkeypatch.setattr(load, "load_model", _no_loading)
    assert _run(spec, runs, "--retry-failed") == 0
    assert (
        "Nothing to retry: no run of version 1 failed; 6 runs are still to run, "
        "which the same command without --retry-failed runs." in capsys.readouterr().out
    )


def test_a_second_runner_is_refused_naming_the_first_one_s_pid(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = tiny_spec(), tmp_path / "runs"
    folder = runs / NAME
    assert _run(spec, runs, "--limit", "2") == 3
    beat = json.loads((folder / "heartbeat.json").read_text())
    beat |= {"state": "running", "pid": 4321}  # as a first runner writes it
    (folder / "heartbeat.json").write_text(json.dumps(beat))
    monkeypatch.setattr(load, "load_model", _no_loading)
    with locked(folder):  # as the first runner holds it
        assert _run(spec, runs) == 1
    err = capsys.readouterr().err
    assert f"pushlab run: another pushlab run is working on {folder} (pid 4321)" in err
    assert len(read_index(folder)) == 1  # nothing more is saved


def test_a_download_that_changes_while_the_model_loads_is_refused(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs = tmp_path / "runs"
    real = load.load_model

    def updated(*args: object, **kwargs: object) -> object:  # another commit's
        return dataclasses.replace(real(*args, **kwargs), revision="f" * 40)

    monkeypatch.setattr(load, "load_model", updated)
    assert _run(tiny_spec(), runs) == 1
    err = capsys.readouterr().err
    assert "pushlab run: tiny-vl's download changed while it loaded; run again" in err
    assert read_index(runs / NAME) == []


def _snapshot(place: Path) -> dict[str, tuple[bytes | None, int, int]]:
    """place and every path in it: a file's bytes, the inode and the mtime."""
    return {
        str(path.relative_to(place)): (
            path.read_bytes() if path.is_file() else None,
            path.stat().st_ino,
            path.stat().st_mtime_ns,
        )
        for path in [place, *sorted(place.rglob("*"))]
    }


def test_a_finished_experiment_run_again_changes_no_file(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = tiny_spec(), tmp_path / "runs"
    assert _run(spec, runs) == 0
    before = _snapshot(runs)
    monkeypatch.setattr(load, "load_model", _no_loading)
    assert _run(spec, runs) == 0
    assert "Nothing to run: all 10 runs of version 1" in capsys.readouterr().out
    assert _snapshot(runs) == before  # not even rewritten with the same bytes


def test_a_stop_at_max_gpu_hours_with_a_failed_run_gives_no_retry_hint(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failing_generate: Callable[..., None],
) -> None:
    runs = tmp_path / "runs"
    failing_generate(_prompt(1), RuntimeError("boom"))  # the first batch fails
    # A stand-in clock: loading takes 1,200 s and each batch 800 s.
    now = [0.0]
    monkeypatch.setattr(runner, "_clock", lambda: now[0])
    real_load, real_generate = load.load_model, generate_module.generate

    def slow_load(*args: object, **kwargs: object) -> object:
        now[0] += 1200
        return real_load(*args, **kwargs)

    def slow_generate(*args: object, **kwargs: object) -> object:
        now[0] += 800
        return real_generate(*args, **kwargs)

    monkeypatch.setattr(load, "load_model", slow_load)
    monkeypatch.setattr(generate_module, "generate", slow_generate)
    assert _run(tiny_spec(), runs) == 2  # a failed run outranks the stop's 3
    out = capsys.readouterr().out
    assert "Stopped at max_gpu_hours 1: 1.00 GPU-hours are used; 4 runs left." in out
    assert "--retry-failed" not in out  # the cap must be raised first
    assert [run.status for run in read_results(runs / NAME)].count("failed") == 2
