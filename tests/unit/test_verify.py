"""`pushlab verify`: an experiment's runner folder checked against its file.

The experiment is tiny_spec's with HANDOFF: no push, and a handoff from A to
B at reply token 2, on each of the five smoke prompts, so 10 runs in 5
batches of 2, of up to 8 reply tokens. Odd run IDs push nothing; even ones
hand off. Problems are planted by rewriting the folder's files through
pyarrow under their own names, as a bug or a hand edit would leave them.
"""

import doctest
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from pushlab.cli import main
from pushlab.models import load
from pushlab.runs import verify
from pushlab.runs.folder import locked, read_index
from pushlab.runs.shards import read_result_steps
from pushlab.runs.verify import MOST_SHOWN, shown_problems
from pushlab.store.records import read_runs_file, runs_table

NAME = "tiny-pulses"
A, B = "tiny-vl:L2:N100", "tiny-vl:L1:N3"
HANDOFF = f"""\
conditions:
  - {{label: none, segments: []}}
  - label: handoff
    shape: handoff
    params: {{a: {A}, b: {B}, alpha_a: 1.0, alpha_b: -1.0, switch_at: 2}}
"""


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _run(spec: Path, runs: Path, *options: str) -> int:
    return main(["run", str(spec), "--runs-dir", str(runs), *options])


def _verify(spec: Path, runs: Path) -> int:
    return main(["verify", str(spec), "--runs-dir", str(runs)])


def _id(counter: int) -> str:
    return f"{NAME}-{counter:06d}"


def _shard(runs: Path, run_id: str) -> str:
    """The shard whose latest mark counts run_id."""
    return [e["shard"] for e in read_index(runs / NAME) if run_id in e["runs"]][-1]


def _rewrite(path: Path, change: Callable[[pa.Table], pa.Table]) -> None:
    """Replace a Parquet file with change(its table), under the same name."""
    pq.write_table(change(pq.read_table(path)), path)


def _steps_count(runs: Path) -> int:
    return sum(len(steps) for steps in read_result_steps(runs / NAME).values())


@pytest.fixture
def finished(tiny_spec: Callable[..., Path], tmp_path: Path) -> tuple[Path, Path]:
    """The handoff experiment run to the end: its file and its runs dir."""
    spec, runs = tiny_spec(HANDOFF, max_new_tokens="8"), tmp_path / "runs"
    assert _run(spec, runs) == 0
    return spec, runs


def _problems(spec: Path, runs: Path, capsys: pytest.CaptureFixture[str]) -> list[str]:
    """Verify, which must find problems; the lines it prints, summary last."""
    capsys.readouterr()
    assert _verify(spec, runs) == 1
    captured = capsys.readouterr()
    assert captured.err == ""
    return captured.out.splitlines()


def test_a_finished_experiment_verifies_clean_and_writes_nothing(
    finished: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = finished

    def no_model(*args: object, **kwargs: object) -> None:
        raise AssertionError("verify loads no model")

    monkeypatch.setattr(load, "load_model", no_model)
    monkeypatch.setattr(load, "load_tokenizer", no_model)
    files = sorted(path for path in runs.rglob("*") if path.is_file())
    before = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in files}
    capsys.readouterr()
    assert _verify(spec, runs) == 0
    captured = capsys.readouterr()
    steps = _steps_count(runs)
    assert steps > 30  # every reply has its steps
    assert captured.out == (
        f"{NAME}, version 1: 10 runs and {steps} steps checked; no problem found.\n"
    )
    assert captured.err == ""
    after = sorted(path for path in runs.rglob("*") if path.is_file())
    assert after == files
    assert {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in files} == before


def test_verify_finds_the_folder_in_the_default_runs_dir(
    finished: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = finished
    monkeypatch.setenv("PUSHLAB_RUNS_DIR", str(runs))
    capsys.readouterr()
    assert main(["verify", str(spec)]) == 0
    assert capsys.readouterr().out.endswith("no problem found.\n")


def test_verify_catches_a_run_saved_twice_in_one_shard(
    finished: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    spec, runs = finished
    shard = _shard(runs, _id(3))
    path = runs / NAME / "shards" / shard
    _rewrite(path, lambda table: pa.concat_tables([table, table.slice(0, 1)]))
    assert [r.run_id for r in read_runs_file(path)] == [_id(3), _id(4), _id(3)]
    lines = _problems(spec, runs, capsys)
    assert lines[0] == f"shards/{shard} holds {_id(3)} twice"
    assert lines[1:] == [
        f"{NAME}, version 1: 10 runs and {_steps_count(runs)} steps checked; "
        "1 problem found."
    ]


def test_verify_catches_a_step_saved_twice(
    finished: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    spec, runs = finished
    shard = _shard(runs, _id(6))
    path = runs / NAME / "steps" / shard
    _rewrite(path, lambda table: pa.concat_tables([table, table.slice(0, 1)]))
    lines = _problems(spec, runs, capsys)
    assert lines[:-1] == [f"steps/{shard} holds step 0 of {_id(5)} twice"]
    assert lines[-1].endswith("1 problem found.")


def test_verify_catches_the_runs_left_unrun(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = tiny_spec(HANDOFF, max_new_tokens="8"), tmp_path / "runs"
    assert _run(spec, runs, "--limit", "4") == 3
    lines = _problems(spec, runs, capsys)
    assert lines[:-1] == [
        f"6 planned runs are not saved: {_id(5)}, {_id(6)}, {_id(7)} and 3 more"
    ]
    assert lines[-1].startswith(f"{NAME}, version 1: 4 runs and ")
    assert _run(spec, runs) == 0  # the rest, run
    capsys.readouterr()
    assert _verify(spec, runs) == 0


def test_verify_catches_a_run_taken_out_of_the_index(
    finished: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    spec, runs = finished
    index = runs / NAME / "index.jsonl"
    entries = read_index(runs / NAME)
    del entries[1]["runs"][_id(4)]
    index.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    lines = _problems(spec, runs, capsys)
    assert lines[:-1] == [
        f"1 planned run is not saved: {_id(4)}",
        # The manifest still counts it, as it was saved.
        "manifest.json counts 10 ok, 0 failed and 0 pending runs, but the index "
        "9 ok, 0 failed and 1 pending",
    ]
    assert lines[-1].startswith(f"{NAME}, version 1: 9 runs and ")


def test_verify_catches_a_saved_run_the_file_does_not_plan(
    finished: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    spec, runs = finished
    entry = {"version": 1, "shard": _shard(runs, _id(1)), "runs": {_id(11): "ok"}}
    with open(runs / NAME / "index.jsonl", "a") as index:
        index.write(json.dumps(entry) + "\n")
    lines = _problems(spec, runs, capsys)
    assert lines[0] == f"1 saved run is not planned: {_id(11)}"
    assert lines[1] == (
        f"shards/{entry['shard']} lacks {_id(11)}, which the index puts there"
    )


def test_verify_catches_a_run_taken_out_of_its_shard(
    finished: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    spec, runs = finished
    shard = _shard(runs, _id(8))
    path = runs / NAME / "shards" / shard
    _rewrite(path, lambda table: table.slice(0, 1))  # 000007 stays
    lines = _problems(spec, runs, capsys)
    assert lines[:-1] == [f"shards/{shard} lacks {_id(8)}, which the index puts there"]


def test_a_shard_that_is_missing_or_cannot_be_read_is_a_problem(
    finished: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    spec, runs = finished
    broken, gone = _shard(runs, _id(1)), _shard(runs, _id(3))
    (runs / NAME / "shards" / broken).write_bytes(b"not Parquet")
    (runs / NAME / "shards" / gone).unlink()
    lines = _problems(spec, runs, capsys)
    assert len(lines) == 3
    assert lines[0].startswith(f"cannot read shards/{broken}: ")
    assert lines[1] == f"shards/{gone} is missing, though the index counts its runs"


def test_verify_catches_a_saved_record_of_another_run(
    finished: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    spec, runs = finished
    shard = _shard(runs, _id(2))
    path = runs / NAME / "shards" / shard
    records = read_runs_file(path)
    records[1] = replace(records[1], seed=7)
    pq.write_table(runs_table(records), path)
    lines = _problems(spec, runs, capsys)
    assert lines[:-1] == [f"{_id(2)} is not the planned run: its seed differs"]


def test_a_version_made_from_another_experiment_file_is_one_problem(
    tiny_spec: Callable[..., Path],
    finished: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = finished
    # The handoff now switches at reply token 3: its five runs change.
    changed = tiny_spec(
        HANDOFF.replace("switch_at: 2", "switch_at: 3"), max_new_tokens="8"
    )
    assert changed == spec
    lines = _problems(spec, runs, capsys)
    assert lines[:-1] == [
        "version 1 was made from another experiment file, so its runs are not "
        "compared with what this file plans"
    ]
    # A new version keeps the runs that push nothing and reruns the others,
    # whose steps then follow the new switch.
    assert _run(spec, runs, "--overwrite-version") == 0
    capsys.readouterr()
    assert _verify(spec, runs) == 0
    assert capsys.readouterr().out.startswith(f"{NAME}, version 2: 10 runs and ")


def test_verify_checks_the_manifest_against_the_index_and_the_identity(
    finished: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    spec, runs = finished
    path = runs / NAME / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["runs_by_status"] = {"ok": 8, "failed": 2, "pending": 0}
    manifest["calibration_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest))
    counts = (
        "manifest.json counts 8 ok, 2 failed and 0 pending runs, but the index "
        "10 ok, 0 failed and 0 pending"
    )
    assert _problems(spec, runs, capsys)[:-1] == [
        "manifest.json's calibration_sha256 is not experiment.json's",
        counts,
    ]
    manifest["version"] = 2  # as if written for another version
    path.write_text(json.dumps(manifest))
    assert _problems(spec, runs, capsys)[:-1] == [
        "manifest.json's version is not experiment.json's",
        "manifest.json's calibration_sha256 is not experiment.json's",
        counts,
    ]
    path.unlink()  # a folder may have none yet: nothing to check
    capsys.readouterr()
    assert _verify(spec, runs) == 0


def test_while_a_runner_works_the_manifest_is_not_checked(
    finished: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    spec, runs = finished
    path = runs / NAME / "manifest.json"
    manifest = json.loads(path.read_text())
    # As a session leaves it until it ends: its runs not counted yet.
    manifest["runs_by_status"] = {"ok": 4, "failed": 0, "pending": 6}
    path.write_text(json.dumps(manifest))
    capsys.readouterr()
    with locked(runs / NAME):  # as a runner holds it
        assert _verify(spec, runs) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == (
        "note: a runner is working on the folder; manifest.json is not checked"
    )
    assert lines[1].endswith("no problem found.")
    assert _verify(spec, runs) == 1  # once it lets go, the counts are checked


def test_verify_refuses_what_it_cannot_check(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs = tmp_path / "runs"

    def refused(spec: Path, runs: Path) -> str:
        capsys.readouterr()
        assert _verify(spec, runs) == 1
        captured = capsys.readouterr()
        assert captured.out == "" and captured.err.startswith("pushlab verify: ")
        return captured.err

    missing = tmp_path / "missing.yaml"
    assert f"{missing}: cannot read it" in refused(missing, runs)
    unplanned = tiny_spec(HANDOFF, file="bad.yaml", prompts="prompts/missing.jsonl")
    assert "cannot read the prompt set" in refused(unplanned, runs)
    spec = tiny_spec(HANDOFF, max_new_tokens="8")
    path = runs / NAME
    assert refused(spec, runs) == (
        f"pushlab verify: nothing is saved for {NAME} in {path} yet\n"
    )
    path.mkdir(parents=True)
    assert "nothing is saved" in refused(spec, runs)
    (path / "notes.txt").write_text("mine")
    assert "is not a runner folder" in refused(spec, runs)
    (path / "notes.txt").unlink()
    (path / "experiment.json").write_text('{"experiment": "other", "version": 1}')
    assert "belongs to the experiment 'other'" in refused(spec, runs)
    (path / "experiment.json").unlink()
    assert _run(spec, runs) == 0
    assert "is a run folder" in refused(spec, path)  # a runs dir inside it
    (path / "index.jsonl").write_text("not json\n")
    assert "index.jsonl, line 1 is not JSON" in refused(spec, runs)


def test_verify_help_says_what_it_checks(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stopped:
        main(["verify", "--help"])
    assert stopped.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith("usage: pushlab verify")
    assert "--runs-dir" in out


def test_at_most_fifty_problems_are_shown() -> None:
    problems = [f"problem {n}" for n in range(1, 54)]
    lines = shown_problems(problems)
    assert MOST_SHOWN == 50
    assert lines == [*problems[:50], "and 3 more problems"]
    assert shown_problems(problems[:50]) == problems[:50]
    assert shown_problems(problems[:51]) == [*problems[:50], "and 1 more problem"]


def test_docstring_examples_run() -> None:
    result = doctest.testmod(verify)
    assert result.attempted > 0
    assert result.failed == 0
