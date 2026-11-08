"""`pushlab run`: an experiment run batch by batch on tiny-vl, saved as it goes.

Each experiment of tiny_spec has 10 runs in 5 batches of 2. Failures, a
changed experiment file and --retry-failed are in test_runner_versions.py,
the dry run in test_run_dry.py and the folder's safety in
test_runner_folder.py.
"""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from pushlab.cli import main
from pushlab.engine import generate as generate_module
from pushlab.models import load
from pushlab.runs import folder as folder_module
from pushlab.runs import runner
from pushlab.runs.folder import read_index, read_results

NAME = "tiny-pulses"
IDS = [f"{NAME}-{i:06d}" for i in range(1, 11)]


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _run(spec: Path, runs: Path, *options: str) -> int:
    return main(["run", str(spec), "--runs-dir", str(runs), *options])


def _files(folder: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(folder)): path.read_bytes()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


def _calls(monkeypatch: pytest.MonkeyPatch, fail_at: int | None = None) -> list[int]:
    """Count generate() calls, the batch sizes they get; fail_at raises
    KeyboardInterrupt in that call, as Ctrl-C would."""
    real = generate_module.generate
    sizes: list[int] = []

    def counting(model, tokenizer, prompts, *args, **kwargs):  # type: ignore[no-untyped-def]
        sizes.append(len(prompts))
        if len(sizes) == fail_at:
            raise KeyboardInterrupt
        return real(model, tokenizer, prompts, *args, **kwargs)

    monkeypatch.setattr(generate_module, "generate", counting)
    return sizes


def _no_loading(*args: object, **kwargs: object) -> None:
    raise AssertionError("the model must not load")


def test_a_run_saves_every_planned_run_once_in_order(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sizes = _calls(monkeypatch)
    assert _run(tiny_spec(), tmp_path / "runs") == 0
    assert sizes == [2] * 5  # the planned batches, one generate() call each
    folder = tmp_path / "runs" / NAME
    results = read_results(folder)
    assert [run.run_id for run in results] == IDS
    assert {run.status for run in results} == {"ok"}
    assert all(run.output_tokens for run in results)
    assert [run.batch_key for run in results[:4]] == [
        "b000001:0",
        "b000001:1",
        "b000002:0",
        "b000002:1",
    ]
    commit = load.downloaded_revision("tiny-vl")
    assert {run.model for run in results} == {f"tiny-vl@{commit}"}
    out = capsys.readouterr().out
    assert "b000005: 2 ok, 0 failed" in out
    assert f"Done: all 10 runs are saved, 10 ok and 0 failed, in {folder}." in out


def test_an_interrupted_run_resumes_and_saves_every_run_exactly_once(
    tiny_spec: Callable[..., Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, runs = tiny_spec(), tmp_path / "runs"
    folder = runs / NAME
    _calls(monkeypatch, fail_at=3)  # Ctrl-C in the third batch
    assert _run(spec, runs) == 130
    saved = _files(folder / "shards")
    assert len(saved) == 2  # two of the five batches
    assert len(read_index(folder)) == 2
    sizes = _calls(monkeypatch)
    assert _run(spec, runs) == 0
    assert sizes == [2, 2, 2]  # only the three batches left
    results = read_results(folder)
    assert [run.run_id for run in results] == IDS
    # The batches saved before the stop are untouched.
    assert {name: _files(folder / "shards")[name] for name in saved} == saved
    # And the replies are those of a run that was never stopped.
    assert _run(spec, tmp_path / "again") == 0
    alone = read_results(tmp_path / "again" / NAME)
    assert [(r.run_id, r.batch_key, r.output_tokens) for r in results] == [
        (r.run_id, r.batch_key, r.output_tokens) for r in alone
    ]


def test_a_batch_saved_but_not_marked_done_is_redone(
    tiny_spec: Callable[..., Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, runs = tiny_spec(), tmp_path / "runs"
    folder = runs / NAME
    real = folder_module.append_index

    def killed_before_the_mark(path: Path, entry: dict) -> None:
        if entry["batch"] == "b000002":
            raise KeyboardInterrupt  # after the shard is in place, before the mark
        real(path, entry)

    monkeypatch.setattr(folder_module, "append_index", killed_before_the_mark)
    assert _run(spec, runs) == 130
    shards = sorted(path.name for path in (folder / "shards").glob("*.parquet"))
    assert shards == ["v1-b000001-1.parquet", "v1-b000002-1.parquet"]
    assert [entry["batch"] for entry in read_index(folder)] == ["b000001"]
    monkeypatch.setattr(folder_module, "append_index", real)
    assert _run(spec, runs) == 0
    assert [run.run_id for run in read_results(folder)] == IDS
    marked = [entry["shard"] for entry in read_index(folder)]
    assert "v1-b000002-2.parquet" in marked and "v1-b000002-1.parquet" not in marked


def test_a_finished_experiment_run_again_loads_nothing_and_writes_nothing(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = tiny_spec(), tmp_path / "runs"
    assert _run(spec, runs) == 0
    before = _files(runs)
    capsys.readouterr()
    monkeypatch.setattr(load, "load_model", _no_loading)
    assert _run(spec, runs) == 0
    assert _files(runs) == before
    assert (
        "Nothing to run: all 10 runs of version 1 are saved" in capsys.readouterr().out
    )


def test_limit_runs_at_most_that_many_runs_in_whole_batches(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = tiny_spec(), tmp_path / "runs"
    sizes = _calls(monkeypatch)
    assert _run(spec, runs, "--limit", "4") == 0
    assert sizes == [2, 2]
    assert (
        "Stopped at --limit 4: 4 runs this time; 6 runs left."
        in capsys.readouterr().out
    )
    assert _run(spec, runs, "--limit", "3") == 0  # a second batch would make 4
    assert sizes == [2, 2, 2]
    # The next batch alone is more than the limit: nothing runs.
    monkeypatch.setattr(load, "load_model", _no_loading)
    assert _run(spec, runs, "--limit", "1") == 0
    out = capsys.readouterr().out
    assert "the next batch holds 2 runs, more than --limit 1" in out
    assert len(read_results(runs / NAME)) == 6


def test_the_gpu_hour_cap_stops_after_the_batch_that_crosses_it(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = tiny_spec(), tmp_path / "runs"
    # A stand-in clock: the session starts at 0 s and each batch takes 1,500.
    ticks = iter(range(0, 100_000, 1500))
    monkeypatch.setattr(runner, "_clock", lambda: float(next(ticks)))
    sizes = _calls(monkeypatch)
    assert _run(spec, runs) == 0
    # 1,500 and 3,000 s are under the hour; the third batch reaches 4,500.
    assert sizes == [2, 2, 2]
    out = capsys.readouterr().out
    assert "Stopped at max_gpu_hours 1: 1.25 GPU-hours are used; 4 runs left." in out
    assert len(read_results(runs / NAME)) == 6
    progress = json.loads((runs / NAME / "progress.json").read_text())
    assert (progress["state"], progress["pending"]) == ("stopped at max_gpu_hours", 4)
    # Run again: the hours are used up, so nothing loads.
    monkeypatch.setattr(load, "load_model", _no_loading)
    assert _run(spec, runs) == 0
    assert "max_gpu_hours 1 is used up: 1.25 GPU-hours" in capsys.readouterr().out


def test_a_heartbeat_and_progress_file_follow_every_batch(
    tiny_spec: Callable[..., Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, runs = tiny_spec(), tmp_path / "runs"
    folder = runs / NAME
    seen: list[tuple[dict, dict]] = []
    real = folder_module.append_index

    def watching(path: Path, entry: dict) -> None:
        real(path, entry)
        # What the files said when this batch was marked: the one before.
        if (folder / "progress.json").exists():
            seen.append(
                (
                    json.loads((folder / "heartbeat.json").read_text()),
                    json.loads((folder / "progress.json").read_text()),
                )
            )

    monkeypatch.setattr(folder_module, "append_index", watching)
    assert _run(spec, runs) == 0
    # Written as the session starts, then after every batch.
    assert [progress["ok"] for _, progress in seen] == [0, 2, 4, 6, 8]
    assert all(beat["state"] == "running" for beat, _ in seen)
    beat = json.loads((folder / "heartbeat.json").read_text())
    progress = json.loads((folder / "progress.json").read_text())
    assert beat["state"] == progress["state"] == "finished"
    assert (progress["runs"], progress["ok"], progress["pending"]) == (10, 10, 0)
    assert progress["reply_tokens_per_second"] > 0


def test_a_batch_that_cannot_be_saved_stops_the_run_and_keeps_the_rest(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = tiny_spec(), tmp_path / "runs"
    folder = runs / NAME
    real = folder_module.write_shard

    def disk_full(path: Path, name: str, records: list) -> Path:
        if name.startswith("v1-b000002"):
            raise OSError("No space left on device")
        return real(path, name, records)

    monkeypatch.setattr(folder_module, "write_shard", disk_full)
    assert _run(spec, runs) == 1
    err = capsys.readouterr().err
    assert "pushlab run: cannot save b000002: No space left on device" in err
    assert [entry["batch"] for entry in read_index(folder)] == ["b000001"]
    beat = json.loads((folder / "heartbeat.json").read_text())
    assert beat["state"].startswith("stopped: cannot save b000002")
    monkeypatch.setattr(folder_module, "write_shard", real)
    assert _run(spec, runs) == 0
    assert [run.run_id for run in read_results(folder)] == IDS


def test_the_runs_dir_comes_from_pushlab_runs_dir_without_the_option(
    tiny_spec: Callable[..., Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUSHLAB_RUNS_DIR", str(tmp_path / "elsewhere"))
    assert folder_module.default_runs_dir() == tmp_path / "elsewhere"
    assert main(["run", str(tiny_spec()), "--limit", "2"]) == 0
    assert len(read_results(tmp_path / "elsewhere" / NAME)) == 2
    monkeypatch.delenv("PUSHLAB_RUNS_DIR")
    assert folder_module.default_runs_dir() == Path("/workspace/runs")


NO_PUSH = "conditions:\n  - {label: none, segments: []}\n"
DEEP = "conditions:\n  - {label: deep, segments: [{unit: tiny-vl:L9:N1, alpha: 1}]}\n"


@pytest.mark.parametrize(
    "conditions, settings, problem",
    [
        (NO_PUSH, {"max_new_tokens": "x"}, "max_new_tokens must be a whole number"),
        (DEEP, {}, "condition 'deep': schedule 'deep' is refused"),
        (NO_PUSH, {"model": "qwen3.5-4b"}, "qwen3.5-4b (Qwen/Qwen3.5-4B at"),
    ],
    ids=["file", "condition", "not-downloaded"],
)
def test_refusals_come_before_the_model_loads(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    conditions: str,
    settings: dict[str, str],
    problem: str,
) -> None:
    monkeypatch.setattr(load, "load_model", _no_loading)
    assert _run(tiny_spec(conditions, **settings), tmp_path / "runs") == 1
    assert problem in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()
