"""Run records and manifests on disk: pushlab.store.records."""

import dataclasses
import doctest
import json
import stat
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pushlab.store import records
from pushlab.store.records import (
    REQUIRED_MANIFEST_FIELDS,
    RunRecord,
    make_run_id,
    read_runs,
    write_manifest,
    write_runs,
)

# The runs table in docs/plan/development-plan.md ("Data model"), in order.
PLAN_FIELDS = [
    "run_id",
    "experiment",
    "model",
    "prompt_id",
    "schedule",
    "decoding",
    "seed",
    "batch_key",
    "control_of",
    "output_text",
    "output_tokens",
    "status",
]

DECODING = {
    "strategy": "greedy",
    "max_new_tokens": 200,
    "thinking": False,
    "dtype": "bfloat16",
}


def _record(**changes: Any) -> RunRecord:
    record = RunRecord(
        run_id="s0-baseline-000001",
        experiment="s0-baseline",
        model="qwen3.5-4b@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        prompt_id="s0-neuron",
        schedule=[],
        decoding=DECODING,
        seed=0,
        batch_key="b000001:0",
        control_of="",
        output_text="A neuron is one number inside the model.",
        output_tokens=[32, 48273, 374, 248046],
        status="ok",
    )
    return dataclasses.replace(record, **changes)


def _info(**changes: Any) -> dict[str, Any]:
    info = {
        "code_commit": "cda6963f2fb2a8b0e1c5a6d0b4e5f6a7b8c9d0e1",
        "lockfile_sha256": "ab" * 32,
        "model": "qwen3.5-4b",
        "model_id": "Qwen/Qwen3.5-4B",
        "model_revision": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        "gpu": "NVIDIA H100 80GB HBM3",
        "cuda": "13.0",
        "torch": "2.14.1+cu130",
        "transformers": "5.21.0",
        "started_at": "2026-10-29T10:00:00+00:00",
        "finished_at": "2026-10-29T10:01:30+00:00",
        "run_count": 1,
    }
    return info | changes


def test_the_record_fields_are_the_plans_runs_fields() -> None:
    assert [field.name for field in dataclasses.fields(RunRecord)] == PLAN_FIELDS


def test_runs_read_back_identical(tmp_path: Path) -> None:
    runs = [
        _record(),
        _record(
            run_id="s0-baseline-000002",
            schedule=[{"unit": "qwen3.5-4b:L14:N8949", "sign": 1, "dose": 1.5}],
            decoding=DECODING | {"strategy": "sampling", "temperature": 0.7},
            seed=12345,
            control_of="s0-baseline-000001",
            output_text="Ünïcode, emoji 🙂 and\nnew lines",
            output_tokens=[],
            status="collapsed",
        ),
    ]
    path = write_runs(runs, tmp_path)
    assert path == tmp_path / "runs.parquet"
    assert read_runs(tmp_path) == runs


def test_the_file_has_the_plans_columns_and_duckdb_reads_it(tmp_path: Path) -> None:
    path = write_runs([_record()], tmp_path)
    schema = pq.read_schema(path)
    assert schema.names == PLAN_FIELDS
    assert not any(field.nullable for field in schema)
    assert schema.field("seed").type == pa.int64()
    assert schema.field("output_tokens").type == pa.list_(pa.int32())
    query = (
        "SELECT run_id, decoding->>'$.dtype', len(output_tokens) "
        f"FROM read_parquet('{path}')"
    )
    assert duckdb.sql(query).fetchall() == [("s0-baseline-000001", "bfloat16", 4)]


def test_make_run_id_counts_from_one_with_six_digits() -> None:
    assert make_run_id("s0-baseline", 1) == "s0-baseline-000001"
    assert make_run_id("p1-a2", 1234567) == "p1-a2-1234567"
    with pytest.raises(ValueError, match="counter"):
        make_run_id("s0-baseline", 0)


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({"run_id": "s0-baseline"}, "run_id"),
        ({"run_id": "other-000001"}, "run_id"),
        ({"run_id": "s0-baseline-00a001"}, "run_id"),
        ({"run_id": "s0-baseline-2"}, "run_id"),
        ({"run_id": "s0-baseline-000000"}, "run_id"),
        ({"run_id": "s0-baseline-\u0661\u0662"}, "run_id"),
        ({"experiment": ""}, "experiment"),
        ({"experiment": "S0 baseline", "run_id": "S0 baseline-000002"}, "experiment"),
        ({"model": ""}, "model"),
        ({"prompt_id": 7}, "prompt_id"),
        ({"schedule": {}}, "schedule"),
        ({"schedule": [1]}, "schedule"),
        ({"schedule": [{"dose": float("nan")}]}, "schedule"),
        ({"schedule": [{"window": (1, 5)}]}, "schedule changes"),
        ({"decoding": {"strategy": "greedy"}}, "decoding"),
        ({"decoding": DECODING | {"max_new_tokens": {1, 2}}}, "max_new_tokens"),
        ({"decoding": DECODING | {"max_new_tokens": 0}}, "max_new_tokens"),
        ({"decoding": DECODING | {"thinking": "no"}}, "thinking"),
        ({"decoding": DECODING | {"strategy": ""}}, "strategy"),
        ({"decoding": DECODING | {"dtype": 16}}, "dtype"),
        ({"decoding": DECODING | {"top_p": float("inf")}}, "decoding"),
        ({"seed": 1.5}, "seed"),
        ({"seed": True}, "seed"),
        ({"seed": -1}, "seed"),
        ({"seed": 2**63}, "seed"),
        ({"batch_key": ""}, "batch_key"),
        ({"control_of": None}, "control_of"),
        ({"output_text": None}, "output_text"),
        ({"output_tokens": [1, "2"]}, "output_tokens"),
        ({"output_tokens": [1, True]}, "output_tokens"),
        ({"output_tokens": [-1]}, "output_tokens"),
        ({"output_tokens": [2**31]}, "output_tokens"),
        ({"output_tokens": (1, 2)}, "output_tokens"),
        ({"status": "done"}, "status"),
    ],
)
def test_malformed_records_are_refused(
    tmp_path: Path, changes: dict[str, Any], problem: str
) -> None:
    # The bad record comes second, so every record is checked, not just the first.
    bad = _record(**({"run_id": "s0-baseline-000002"} | changes))
    with pytest.raises(ValueError, match=problem):
        write_runs([_record(), bad], tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_a_record_must_be_a_run_record(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="RunRecord"):
        write_runs([dataclasses.asdict(_record())], tmp_path)  # type: ignore[list-item]


def test_duplicate_run_ids_and_empty_lists_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="s0-baseline-000001.*more than once"):
        write_runs([_record(), _record()], tmp_path)
    with pytest.raises(ValueError, match="no records"):
        write_runs([], tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_an_existing_file_is_kept_unless_overwrite_is_set(tmp_path: Path) -> None:
    write_runs([_record()], tmp_path)
    with pytest.raises(FileExistsError, match="runs.parquet"):
        write_runs([_record(output_text="new")], tmp_path)
    assert read_runs(tmp_path) == [_record()]
    write_runs([_record(output_text="new")], tmp_path, overwrite=True)
    assert read_runs(tmp_path) == [_record(output_text="new")]


def test_the_output_folder_is_created(tmp_path: Path) -> None:
    out = tmp_path / "outputs" / "s0-baseline"
    write_runs([_record()], out)
    assert read_runs(out) == [_record()]


def test_a_failed_write_leaves_no_file_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_write(table: pa.Table, where: str | Path, **kwargs: Any) -> None:
        Path(where).write_bytes(b"half a file")
        raise OSError("disk full")

    monkeypatch.setattr(pq, "write_table", broken_write)
    with pytest.raises(OSError, match="disk full"):
        write_runs([_record()], tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_a_failed_manifest_write_leaves_no_file_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_write_text(self: Path, text: str, *args: Any, **kwargs: Any) -> int:
        Path.write_bytes(self, text[:10].encode())
        raise OSError("disk full")

    monkeypatch.setattr(Path, "write_text", broken_write_text)
    with pytest.raises(OSError, match="disk full"):
        write_manifest(tmp_path, _info())
    assert list(tmp_path.iterdir()) == []


def test_both_files_get_the_usual_permissions(tmp_path: Path) -> None:
    # Like any other new file, so other users and tools can read them.
    reference = tmp_path / "reference.txt"
    reference.write_text("written the usual way")
    usual = stat.S_IMODE(reference.stat().st_mode)
    for path in (write_runs([_record()], tmp_path), write_manifest(tmp_path, _info())):
        assert stat.S_IMODE(path.stat().st_mode) == usual


def test_reading_refuses_a_file_with_other_columns(tmp_path: Path) -> None:
    pq.write_table(pa.table({"run_id": ["x-1"]}), tmp_path / "runs.parquet")
    with pytest.raises(ValueError, match="columns"):
        read_runs(tmp_path)


def test_the_manifest_fields_cover_the_brief() -> None:
    assert set(REQUIRED_MANIFEST_FIELDS) == {
        "code_commit",
        "lockfile_sha256",
        "model_id",
        "model_revision",
        "gpu",
        "cuda",
        "torch",
        "transformers",
        "started_at",
        "finished_at",
        "run_count",
    }


def test_the_manifest_includes_every_required_field(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, _info(dtype="bfloat16"))
    assert path == tmp_path / "manifest.json"
    saved = json.loads(path.read_text())
    assert set(REQUIRED_MANIFEST_FIELDS) <= set(saved)
    assert saved == _info(dtype="bfloat16")


def test_fields_that_do_not_apply_can_be_null(tmp_path: Path) -> None:
    # On a CPU there is no GPU or CUDA; outside git there is no commit.
    info = _info(gpu=None, cuda=None, code_commit=None)
    assert json.loads(write_manifest(tmp_path, info).read_text()) == info


@pytest.mark.parametrize("field", REQUIRED_MANIFEST_FIELDS)
def test_a_manifest_missing_a_field_is_refused(tmp_path: Path, field: str) -> None:
    info = _info()
    del info[field]
    with pytest.raises(ValueError, match=field):
        write_manifest(tmp_path, info)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({"run_count": -1}, "run_count"),
        ({"run_count": True}, "run_count"),
        ({"run_count": "1"}, "run_count"),
        ({"model_id": None}, "model_id"),
        ({"model_revision": ""}, "model_revision"),
        ({"torch": None}, "torch"),
        ({"transformers": 5.21}, "transformers"),
        ({"started_at": 1730196000}, "started_at"),
        ({"started_at": "2026-10-29"}, "started_at"),
        ({"started_at": "2026-10-29T10:00:00"}, "started_at"),
        ({"finished_at": "2026-10-29T09:59:59+00:00"}, "before"),
        ({"extra": float("nan")}, "JSON"),
        ({"extra": {1, 2}}, "JSON"),
    ],
)
def test_a_manifest_with_bad_values_is_refused(
    tmp_path: Path, changes: dict[str, Any], problem: str
) -> None:
    with pytest.raises(ValueError, match=problem):
        write_manifest(tmp_path, _info(**changes))
    assert list(tmp_path.iterdir()) == []


def test_an_existing_manifest_is_kept_unless_overwrite_is_set(tmp_path: Path) -> None:
    write_manifest(tmp_path, _info())
    with pytest.raises(FileExistsError, match="manifest.json"):
        write_manifest(tmp_path, _info(run_count=2))
    write_manifest(tmp_path, _info(run_count=2), overwrite=True)
    assert json.loads((tmp_path / "manifest.json").read_text())["run_count"] == 2


def test_docstring_examples_run() -> None:
    result = doctest.testmod(records)
    assert result.attempted > 0
    assert result.failed == 0
