"""`pushlab verify` against every mark of the version in the index, not only
the latest ones.

A run is marked once, or again only after its earlier marks said it failed:
a retry, as --retry-failed makes. A new version's marks of the runs it
keeps count as marks too. Each counted record must have the status its
latest mark gives it, and the model download experiment.json names, since
all runs of a version are made from one download. A counted shard may hold
other runs, as a batch some of whose runs were retried does.

The experiment is HANDOFF on the five smoke prompts, as in test_verify.py:
10 runs in 5 batches of 2, the odd ones unpushed. Problems are planted as a
runner bug or a hand edit would leave them.
"""

import json
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch

from pushlab.cli import main
from pushlab.engine import generate as generate_module
from pushlab.prompts import PROMPTS_DIR, read_prompt_set
from pushlab.runs import shards
from pushlab.runs.folder import append_index, read_identity, read_index
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
AGAIN = "only a failed run is marked again"


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _run(spec: Path, runs: Path, *options: str) -> int:
    return main(["run", str(spec), "--runs-dir", str(runs), *options])


def _id(counter: int) -> str:
    return f"{NAME}-{counter:06d}"


def _shard(runs: Path, run_id: str) -> str:
    """The shard whose latest mark counts run_id."""
    return [e["shard"] for e in read_index(runs / NAME) if run_id in e["runs"]][-1]


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


def _again(shard: str, runs: Path) -> None:
    """Save shard's batch again under the next attempt's name, marked ok."""
    path = runs / NAME
    entry = [e for e in read_index(path) if e["shard"] == shard][0]
    again = shard.replace("-1.parquet", "-2.parquet")
    for place in ("shards", "steps"):
        shutil.copy(path / place / shard, path / place / again)
    append_index(path, entry | {"shard": again})


def _line_repeated(shard: str, runs: Path) -> None:
    """Write the index's first line, shard's mark, once more at its end."""
    index = runs / NAME / "index.jsonl"
    with index.open("a") as file:
        file.write(index.read_text().splitlines(keepends=True)[0])


@pytest.mark.parametrize(
    ("plant", "where"),
    [(_again, "-2.parquet"), (_line_repeated, "-1.parquet")],
    ids=["saved again", "index line repeated"],
)
def test_an_ok_run_marked_again_is_caught(
    plant: Callable[[str, Path], None],
    where: str,
    finished: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = finished
    first = _shard(runs, _id(1))  # runs 1 and 2, ok
    plant(first, runs)
    later = first.replace("-1.parquet", where)
    lines = _verify(spec, runs, capsys)
    assert lines[:-1] == [
        "1",
        f"the index marks {_id(1)} and {_id(2)} twice: ok in shards/{first}, then "
        f"ok in shards/{later}; {AGAIN}",
    ]
    assert lines[-1].startswith(f"{NAME}, version 1: 10 runs and ")


def test_an_ok_run_marked_failed_later_is_caught(
    finished: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    # As a runner that ran ok runs again, and failed, would leave it.
    spec, runs = finished
    path, first = runs / NAME, _shard(runs, _id(1))
    later = first.replace("-1.parquet", "-2.parquet")
    failed = [
        replace(r, status="failed", output_text="", output_tokens=[])
        for r in read_runs_file(path / "shards" / first)
    ]
    shards.write_shard(path, later, failed)
    shards.write_steps(path, later, [])
    entry = read_index(path)[0] | {"shard": later, "error": "RuntimeError: boom"}
    append_index(path, entry | {"runs": {_id(1): "failed", _id(2): "failed"}})
    assert _verify(spec, runs, capsys)[:-1] == [
        "1",
        f"the index marks {_id(1)} and {_id(2)} twice: ok in shards/{first}, then "
        f"failed in shards/{later}; {AGAIN}",
        "manifest.json counts 10 ok, 0 failed and 0 pending runs, but the index "
        "8 ok, 2 failed and 0 pending",
    ]


def test_a_kept_run_marked_again_in_the_new_version_is_caught(
    tiny_spec: Callable[..., Path],
    finished: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Another limit, the same runs: version 2 keeps them all, each marked
    # again for it in the shard that holds it.
    spec, runs = finished
    assert spec == tiny_spec(HANDOFF, max_new_tokens="8", max_gpu_hours="2")
    assert _run(spec, runs, "--overwrite-version") == 0
    path, kept = runs / NAME, _shard(runs, _id(1))
    assert [e["version"] for e in read_index(path) if e["shard"] == kept] == [1, 2]
    # Then a bug runs the kept runs 1 and 2 again for version 2.
    again = "v2-b000001-1.parquet"
    for place in ("shards", "steps"):
        shutil.copy(path / place / kept, path / place / again)
    entry = [e for e in read_index(path) if e["shard"] == kept][-1]
    append_index(path, {"version": 2, "shard": again, "runs": entry["runs"]})
    assert _verify(spec, runs, capsys)[:-1] == [
        "1",
        f"the index marks {_id(1)} and {_id(2)} twice: ok in shards/{kept}, then "
        f"ok in shards/{again}; {AGAIN}",
    ]


def test_a_failed_batch_run_again_is_no_problem(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    failing_generate: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    real = generate_module.generate
    spec, runs = tiny_spec(HANDOFF, max_new_tokens="8"), tmp_path / "runs"
    first = read_prompt_set(PROMPTS_DIR / "smoke.jsonl").prompts[0]
    failing_generate(first, RuntimeError("boom"))  # b000001: runs 1 and 2
    assert _run(spec, runs) == 2
    monkeypatch.setattr(generate_module, "generate", real)
    assert _run(spec, runs, "--retry-failed") == 0
    marks = [e["runs"] for e in read_index(runs / NAME) if _id(1) in e["runs"]]
    assert marks == [{_id(1): "failed", _id(2): "failed"}, {_id(1): "ok", _id(2): "ok"}]
    lines = _verify(spec, runs, capsys)
    assert lines[0] == "0" and lines[-1].endswith("no problem found.")


def test_a_partial_retry_under_a_new_version_is_no_problem(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    failing_generate: Callable[..., None],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    real, runs = generate_module.generate, tmp_path / "runs"
    # Version 1 in calls of 4 runs, the first of which fails: runs 1 to 4.
    spec = tiny_spec(
        HANDOFF, max_new_tokens="8", batch_size="4", group_by_length="false"
    )
    first = read_prompt_set(PROMPTS_DIR / "smoke.jsonl").prompts[0]
    failing_generate(first, RuntimeError("boom"))
    assert _run(spec, runs) == 2
    monkeypatch.setattr(generate_module, "generate", real)
    # Version 2, in calls of 2, keeps every run; then runs 1 and 2 are retried.
    assert spec == tiny_spec(
        HANDOFF, max_new_tokens="8", batch_size="2", group_by_length="false"
    )
    assert _run(spec, runs, "--overwrite-version") == 0
    assert _run(spec, runs, "--retry-failed", "--limit", "2") == 3
    held = [
        r.run_id for r in read_runs_file(runs / NAME / "shards" / _shard(runs, _id(3)))
    ]
    assert held == [_id(1), _id(2), _id(3), _id(4)]  # 1 and 2 now counted elsewhere
    assert _shard(runs, _id(1)) == "v2-b000001-1.parquet"
    lines = _verify(spec, runs, capsys)
    assert lines[0] == "0"
    assert lines[1].startswith(f"{NAME}, version 2: 10 runs (2 failed) and ")
    assert lines[1].endswith("no problem found.")


def test_a_record_must_have_the_status_its_latest_mark_says(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    failing_generate: Callable[..., None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = tiny_spec(HANDOFF, max_new_tokens="8"), tmp_path / "runs"
    third = read_prompt_set(PROMPTS_DIR / "smoke.jsonl").prompts[2]
    failing_generate(third, RuntimeError("boom"))  # smoke-03's batch: 5 and 6
    assert _run(spec, runs) == 2
    path, shard = runs / NAME, _shard(runs, _id(5))
    # The index, and the manifest with it, now say every run ended ok.
    entries = [e | {"runs": dict.fromkeys(e["runs"], "ok")} for e in read_index(path)]
    (path / "index.jsonl").write_text("".join(json.dumps(e) + "\n" for e in entries))
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["runs_by_status"] = {"ok": 10, "failed": 0, "pending": 0}
    (path / "manifest.json").write_text(json.dumps(manifest))
    lines = _verify(spec, runs, capsys)
    assert lines[:-1] == [
        "1",
        f"{_id(5)} is failed in shards/{shard}, but the index marks it ok",
        f"{_id(6)} is failed in shards/{shard}, but the index marks it ok",
    ]
    assert lines[-1].startswith(f"{NAME}, version 1: 10 runs and ")


def test_a_record_of_another_model_download_is_caught(
    finished: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    spec, runs = finished
    path = runs / NAME / "shards" / _shard(runs, _id(2))
    records = read_runs_file(path)
    other = f"tiny-vl@{'f' * 40}"
    records[1] = replace(records[1], model=other)
    pq.write_table(runs_table(records), path)
    model = read_identity(runs / NAME)["model"]
    assert model != other
    assert _verify(spec, runs, capsys)[:-1] == [
        "1",
        f"{_id(2)}'s model is {other}, but experiment.json's is {model}",
    ]
