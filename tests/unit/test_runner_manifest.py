"""pushlab.runs.manifest: a session's entry, the sessions before it, and the
restamp, on manifests written here and on broken ones. No model is loaded."""

import json
from pathlib import Path
from typing import Any

import pytest

from pushlab.runs import manifest
from pushlab.runs.manifest import (
    SESSION_ENVIRONMENT,
    SessionEntry,
    earlier_sessions,
    record_session,
    restamp_manifest,
)
from pushlab.store.manifest import check_experiment_manifest

REVISION = "f" * 40
ENVIRONMENT = {
    "code_commit": "a" * 40,
    "code_dirty": False,
    "lockfile_sha256": "b" * 64,
    "gpu": "NVIDIA GeForce RTX 4090",
    "cuda": "12.8",
    "driver": "570.86",
    "torch": "2.9.0",
    "transformers": "5.0.0",
    "python": "3.11.9",
    "packages": {"numpy": "2.3.0"},
}
BROKEN = {
    "{not json": "is not valid JSON",
    "5": "is not a JSON object",
    "[1]": "is not a JSON object",
    '"x"': "is not a JSON object",
    "null": "is not a JSON object",
}


def _identity(version: int = 1, dtype: str = "float32") -> dict[str, Any]:
    """experiment.json of a version of tiny-pulses."""
    return {
        "experiment": "tiny-pulses",
        "version": version,
        "spec_file": "experiments/tiny.yaml",
        "spec": {"name": "tiny-pulses", "decoding": {"dtype": dtype}},
        "spec_sha256": f"{version}" * 64,
        "prompts_sha256": "c" * 64,
        "known_sha256": "d" * 64,
        "model": f"tiny-vl@{REVISION}",
        "calibration_sha256": "e" * 64,
    }


def _session(id_: str, hour: int, **changes: Any) -> SessionEntry:
    """A session of version 1 that ran from hour to half past it."""
    fields = {
        "id": id_,
        "environment": ENVIRONMENT,
        "device": "cuda",
        "version": 1,
        "started_at": f"2026-11-08T{hour:02d}:00:00+00:00",
        "finished_at": f"2026-11-08T{hour:02d}:30:00+00:00",
        "state": "finished",
        "batches": 2,
        "runs": 4,
        "gpu_hours": 0.25,
    } | changes
    return SessionEntry(**fields)


def _record(path: Path, session: SessionEntry, identity: dict | None = None) -> None:
    states = {"tiny-pulses-000001": "ok", "tiny-pulses-000002": "failed"}
    record_session(
        path,
        identity or _identity(),
        session,
        model_id="local:tiny-qwen3.5",
        states=states,
        total=10,
    )


def _read(path: Path) -> dict[str, Any]:
    return json.loads((path / "manifest.json").read_text())


def test_each_session_is_recorded_last_with_the_code_it_ran(tmp_path: Path) -> None:
    _record(tmp_path, _session("one", 10))
    _record(tmp_path, _session("two", 12, state="stopped at --limit"))
    _record(tmp_path, _session("two", 12, finished_at="2026-11-08T13:00:00+00:00"))
    info = _read(tmp_path)
    check_experiment_manifest(info)
    first, last = info["sessions"]  # a session written again replaces its entry
    assert (first["id"], last["id"]) == ("one", "two")
    assert last == {key: ENVIRONMENT[key] for key in SESSION_ENVIRONMENT} | {
        "id": "two",
        "device": "cuda",
        "version": 1,
        "started_at": "2026-11-08T12:00:00+00:00",
        "finished_at": "2026-11-08T13:00:00+00:00",
        "state": "finished",
        "batches": 2,
        "runs": 4,
        "gpu_hours": 0.25,
    }
    # The top level: the last session's code, machine and model, the first
    # session's start, and what experiment.json and the index say.
    assert {key: info[key] for key in ENVIRONMENT} == ENVIRONMENT
    assert info["model_id"] == "local:tiny-qwen3.5" and info["device"] == "cuda"
    assert info["started_at"] == "2026-11-08T10:00:00+00:00"
    assert info["finished_at"] == "2026-11-08T13:00:00+00:00"
    assert (info["experiment"], info["version"]) == ("tiny-pulses", 1)
    assert info["model_revision"] == REVISION and info["dtype"] == "float32"
    assert (
        info["spec_sha256"] == "1" * 64 and info["spec_file"] == "experiments/tiny.yaml"
    )
    assert info["runs_by_status"] == {"ok": 1, "failed": 1, "pending": 8}
    assert info["run_count"] == 10


@pytest.mark.parametrize("text", [*BROKEN, '{"sessions": 5}'])
def test_a_manifest_that_cannot_be_read_has_no_earlier_sessions(
    tmp_path: Path, text: str
) -> None:
    (tmp_path / "manifest.json").write_text(text)
    assert earlier_sessions(tmp_path, "new") == []


def test_only_whole_session_entries_are_earlier_sessions(tmp_path: Path) -> None:
    whole = {"id": "b", "started_at": "2026-11-08T10:00:00+00:00"}
    whole |= {"finished_at": "2026-11-08T10:30:00+00:00"}
    sessions = [1, {"id": "a"}, whole, whole | {"id": "new"}, whole | {"id": 5}]
    (tmp_path / "manifest.json").write_text(json.dumps({"sessions": sessions}))
    assert earlier_sessions(tmp_path, "new") == [whole]


@pytest.mark.parametrize("text", [*BROKEN])
def test_a_session_rewrites_a_broken_manifest_whole(tmp_path: Path, text: str) -> None:
    (tmp_path / "manifest.json").write_text(text)
    _record(tmp_path, _session("new", 10))
    info = _read(tmp_path)
    check_experiment_manifest(info)
    assert [session["id"] for session in info["sessions"]] == ["new"]


@pytest.mark.parametrize(("text", "why"), BROKEN.items())
def test_restamp_leaves_a_broken_manifest_alone_and_says_why(
    tmp_path: Path, text: str, why: str
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(text)
    with pytest.raises(ValueError, match=why) as error:
        restamp_manifest(tmp_path, _identity(2), {}, 10)
    assert "a session of pushlab run, while runs are left, rewrites it" in str(
        error.value
    )
    assert path.read_text() == text


def test_restamp_refreshes_what_experiment_json_says_and_keeps_the_rest(
    tmp_path: Path,
) -> None:
    _record(tmp_path, _session("one", 10))
    before = _read(tmp_path)
    # Version 2 runs in bfloat16 on another download, and no session of it
    # has run yet: its manifest says so, and what ran it is still version 1's.
    identity = _identity(2, "bfloat16") | {"model": "tiny-vl@" + "0" * 40}
    states = {"tiny-pulses-000001": "ok"}
    assert restamp_manifest(tmp_path, identity, states, 10) is True
    after = _read(tmp_path)
    check_experiment_manifest(after)
    assert after["version"] == 2 and after["spec_sha256"] == "2" * 64
    assert after["dtype"] == "bfloat16" and after["model_revision"] == "0" * 40
    assert after["model"] == identity["model"]
    assert after["runs_by_status"] == {"ok": 1, "failed": 0, "pending": 9}
    kept = ("model_id", "device", "started_at", "finished_at", "sessions", *ENVIRONMENT)
    assert {key: after[key] for key in kept} == {key: before[key] for key in kept}
    # Up to date, it is not written again.
    assert restamp_manifest(tmp_path, identity, states, 10) is False


def test_restamp_leaves_alone_a_manifest_it_cannot_bring_up_to_date(
    tmp_path: Path,
) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"sessions": []}')  # a JSON object, but no manifest
    with pytest.raises(ValueError, match="cannot be brought up to date") as error:
        restamp_manifest(tmp_path, _identity(), {}, 10)
    assert "the manifest lacks code_commit" in str(error.value)
    assert str(error.value).endswith(manifest.LEFT_ALONE)
    assert path.read_text() == '{"sessions": []}'


def test_restamp_does_nothing_without_a_manifest(tmp_path: Path) -> None:
    assert restamp_manifest(tmp_path, _identity(), {}, 10) is False
    assert not (tmp_path / "manifest.json").exists()


def test_restamp_names_the_model_id_of_the_model_experiment_json_names(
    tmp_path: Path,
) -> None:
    from pushlab.models.registry import REGISTRY

    _record(tmp_path, _session("one", 10))
    # A new version on another registered model, before any session of it.
    identity = _identity(2, "float32") | {"model": "tiny@" + "0" * 40}
    assert restamp_manifest(tmp_path, identity, {}, 10) is True
    after = _read(tmp_path)
    assert after["model"] == identity["model"]
    assert after["model_id"] == REGISTRY["tiny"].hf_id
