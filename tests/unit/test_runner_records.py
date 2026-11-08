"""`pushlab run`'s records of itself: the manifest's sessions, warnings, the
speed file and a stop while the model loads. tiny_spec's experiment has 10
runs in 5 batches."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from pushlab.cli import main
from pushlab.models import load
from pushlab.runs.expand import expand
from pushlab.runs.folder import read_identity, read_index
from pushlab.runs.shards import read_results
from pushlab.runs.spec import read_spec
from pushlab.store import environment

NAME = "tiny-pulses"


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _run(spec: Path, runs: Path, *options: str) -> int:
    return main(["run", str(spec), "--runs-dir", str(runs), *options])


def test_the_manifest_records_every_session_with_its_code_and_libraries(
    tiny_spec: Callable[..., Path], tmp_path: Path
) -> None:
    spec, runs = tiny_spec(), tmp_path / "runs"
    assert _run(spec, runs, "--limit", "4") == 3
    assert _run(spec, runs) == 0
    manifest = json.loads((runs / NAME / "manifest.json").read_text())
    sessions = manifest["sessions"]
    assert [s["runs"] for s in sessions] == [4, 6]
    assert len({s["id"] for s in sessions}) == 2
    for session in sessions:
        for key in ("code_commit", "lockfile_sha256", "torch", "transformers"):
            assert key in session
    assert manifest["runs_by_status"] == {"ok": 10, "failed": 0, "pending": 0}
    marks = [entry["session"] for entry in read_index(runs / NAME)]
    assert marks == [sessions[0]["id"]] * 2 + [sessions[1]["id"]] * 3


def test_a_session_on_other_code_says_so(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = tiny_spec(), tmp_path / "runs"
    assert _run(spec, runs, "--limit", "4") == 3
    real = environment.environment_info

    def pulled(device: object) -> dict:  # as after a git pull on the pod
        return real(device) | {"code_commit": "0" * 40}

    monkeypatch.setattr(environment, "environment_info", pulled)
    capsys.readouterr()
    assert _run(spec, runs) == 0
    err = capsys.readouterr().err
    assert "warning: version 1's runs so far ran with another code_commit" in err
    assert len(read_results(runs / NAME)) == 10


def test_a_broken_speed_file_is_warned_about_and_left_out_of_estimates(
    tiny_spec: Callable[..., Path],
    costs: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "speeds.json").write_text("{not json")
    assert _run(tiny_spec(), runs) == 0
    captured = capsys.readouterr()
    assert (
        "warning: " in captured.err and "speeds.json is not valid JSON" in captured.err
    )
    assert len(read_results(runs / NAME)) == 10  # the runs are saved all the same
    beat = json.loads((runs / NAME / "heartbeat.json").read_text())
    assert beat["state"] == "finished"
    assert _run(tiny_spec(name="tiny-other"), runs, "--dry-run") == 0
    assert f"0.01 reply tokens a second, from {costs}" in capsys.readouterr().out


def test_ctrl_c_while_the_model_loads_stops_cleanly(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def interrupted(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(load, "load_model", interrupted)
    runs = tmp_path / "runs"
    assert _run(tiny_spec(), runs) == 130
    assert "Stopped by Ctrl-C" in capsys.readouterr().out
    beat = json.loads((runs / NAME / "heartbeat.json").read_text())
    assert beat["state"] == "stopped by Ctrl-C"
    assert read_index(runs / NAME) == []
    assert not (runs / NAME / "manifest.json").exists()  # nothing ran to record


def test_experiment_json_names_each_runs_condition(
    tiny_spec: Callable[..., Path], tmp_path: Path
) -> None:
    conditions = """\
conditions:
  - {label: none, segments: []}
  - label: pulse
    shape: pulse
    params: {unit: tiny-vl:L2:N100, alpha: 1.0, start: 0}
    grid: {length: [1, 2]}
    controls: 1
    control_seed: 0
"""
    spec, runs = tiny_spec(conditions), tmp_path / "runs"
    assert _run(spec, runs, "--limit", "1") == 3
    planned = expand(read_spec(spec)).runs
    expected = {run.record.run_id: run.condition for run in planned}
    assert read_identity(runs / NAME)["conditions"] == expected
    assert set(expected.values()) == {"none", "pulse length=1", "pulse length=2"}
    controls = [run.record for run in planned if run.record.control_of]
    assert controls and all(  # a control has its run's label
        expected[record.run_id] == expected[record.control_of] for record in controls
    )
    changed = tiny_spec(conditions, max_gpu_hours="2")
    assert _run(changed, runs, "--overwrite-version", "--limit", "1") == 3
    assert read_identity(runs / NAME)["conditions"] == expected


def test_a_new_version_that_keeps_every_run_restamps_the_manifest(
    tiny_spec: Callable[..., Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs = tmp_path / "runs"
    assert _run(tiny_spec(), runs) == 0
    changed = tiny_spec(max_gpu_hours="2")  # keeps every run: nothing to run
    assert _run(changed, runs, "--overwrite-version") == 0
    assert "it keeps 10 runs saved" in capsys.readouterr().out
    manifest = json.loads((runs / NAME / "manifest.json").read_text())
    identity = read_identity(runs / NAME)
    assert manifest["version"] == identity["version"] == 2
    assert manifest["spec_sha256"] == identity["spec_sha256"]
    assert manifest["runs_by_status"] == {"ok": 10, "failed": 0, "pending": 0}
    assert len(manifest["sessions"]) == 1  # the session that made the runs
    assert main(["verify", str(changed), "--runs-dir", str(runs)]) == 0


def test_stale_counts_are_restamped_and_a_broken_manifest_only_warns(
    tiny_spec: Callable[..., Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec, runs = tiny_spec(), tmp_path / "runs"
    assert _run(spec, runs) == 0
    path = runs / NAME / "manifest.json"
    manifest = json.loads(path.read_text())
    # As a session killed after its last batch leaves it.
    stale = manifest | {"runs_by_status": {"ok": 8, "failed": 0, "pending": 2}}
    path.write_text(json.dumps(stale))
    assert _run(spec, runs) == 0
    assert json.loads(path.read_text())["runs_by_status"]["ok"] == 10
    path.write_text("{not json")
    capsys.readouterr()
    assert _run(spec, runs) == 0  # nothing to run, and no refusal
    assert "warning: " in capsys.readouterr().err
