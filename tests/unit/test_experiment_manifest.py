"""An experiment's manifest (pushlab.store.manifest): what `pushlab run`
records about its runs, and what the check refuses."""

import json
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import pytest
import torch

from pushlab.cli import main
from pushlab.models import load
from pushlab.runs.expand import expand
from pushlab.runs.spec import read_spec
from pushlab.store.environment import file_sha256
from pushlab.store.manifest import EXPERIMENT_MANIFEST_FIELDS, check_experiment_manifest
from pushlab.units import calibrate

NAME = "tiny-pulses"
NO_PUSH = "conditions:\n  - {label: none, segments: []}\n"


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _manifest(spec: Path, runs: Path, *options: str) -> dict:
    main(["run", str(spec), "--runs-dir", str(runs), *options])
    return json.loads((runs / NAME / "manifest.json").read_text())


def test_the_manifest_records_what_the_runs_were_made_with(
    tiny_spec: Callable[..., Path], tmp_path: Path
) -> None:
    spec = tiny_spec()
    manifest = _manifest(spec, tmp_path / "runs", "--limit", "4")
    check_experiment_manifest(manifest)
    revision = load.downloaded_revision("tiny-vl")
    calibration = calibrate.calibration_path("tiny-vl", revision)
    assert manifest["experiment"] == NAME and manifest["version"] == 1
    assert manifest["spec_sha256"] == expand(read_spec(spec)).spec_sha256
    assert manifest["calibration_sha256"] == file_sha256(calibration)
    assert manifest["model_revision"] == revision
    assert manifest["gpu"] is None  # a CPU run
    for key in ("code_commit", "lockfile_sha256"):
        assert key in manifest
    started = datetime.fromisoformat(manifest["started_at"])
    assert started <= datetime.fromisoformat(manifest["finished_at"])
    assert manifest["runs_by_status"] == {"ok": 4, "failed": 0, "pending": 6}
    assert manifest["run_count"] == 10


def test_an_experiment_that_pushes_nothing_has_no_calibration_hash(
    tiny_spec: Callable[..., Path], tmp_path: Path
) -> None:
    manifest = _manifest(tiny_spec(NO_PUSH), tmp_path / "runs")
    assert manifest["calibration_sha256"] is None
    assert manifest["runs_by_status"] == {"ok": 5, "failed": 0, "pending": 0}
    check_experiment_manifest(manifest)


def _valid(tiny_spec: Callable[..., Path], tmp_path: Path) -> dict:
    return _manifest(tiny_spec(), tmp_path / "runs", "--limit", "2")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"experiment": "Not A Name"}, "experiment must be an experiment's name"),
        ({"version": 0}, "version must be a whole number from 1"),
        ({"spec_sha256": "abc"}, "spec_sha256 must be a SHA-256 in hex"),
        ({"calibration_sha256": "ABC"}, "calibration_sha256 must be a SHA-256"),
        ({"runs_by_status": {"ok": 2, "failed": 0}}, "runs_by_status must count"),
        ({"runs_by_status": {"ok": 2, "failed": 0, "pending": -1}}, "must count"),
        (
            {"runs_by_status": {"ok": 2, "failed": 0, "pending": 9}},
            "adds up to 11 runs, not run_count 10",
        ),
        ({"sessions": []}, "sessions must list each session"),
        ({"sessions": [{"id": "a"}]}, "sessions must list each session"),
    ],
)
def test_a_manifest_that_is_not_one_is_refused(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    changes: dict,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        check_experiment_manifest(_valid(tiny_spec, tmp_path) | changes)


def test_every_field_is_required(
    tiny_spec: Callable[..., Path], tmp_path: Path
) -> None:
    manifest = _valid(tiny_spec, tmp_path)
    for field in EXPERIMENT_MANIFEST_FIELDS:
        broken = {key: value for key, value in manifest.items() if key != field}
        with pytest.raises(ValueError, match=f"lacks {field}"):
            check_experiment_manifest(broken)
