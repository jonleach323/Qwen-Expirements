"""`pushlab run --overwrite-version`: what a new version keeps, and what the
runner says about it.

A new version keeps a failed run on any row, with its error, the session
that made it and its share of its batch's time. The warning about changed
code counts the sessions of kept runs; the refusal names the number a new
version would take and says whether it would keep any run; and a disk
error before any batch is refused without a traceback. tiny_spec's
experiment has 10 runs, an unpushed run and a pulse for each smoke prompt,
in 5 batches of 2.
"""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from pushlab.cli import main
from pushlab.engine import generate as generate_module
from pushlab.prompts import PROMPTS_DIR, read_prompt_set
from pushlab.runs import folder as folder_module
from pushlab.runs import runner
from pushlab.runs.folder import read_identity, read_index, read_results
from pushlab.store import environment

NAME = "tiny-pulses"
IDS = [f"{NAME}-{i:06d}" for i in range(1, 11)]
SAMPLED = "{strategy: sample, temperature: 0.7, dtype: float32}"
# tiny_spec's conditions with the pulse half as strong: the pushed runs change.
WEAKER = """\
conditions:
  - label: none
    segments: []
  - label: pulse
    segments: [{unit: tiny-vl:L2:N100, alpha: 0.5}]
"""


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _run(spec: Path, runs: Path, *options: str) -> int:
    return main(["run", str(spec), "--runs-dir", str(runs), *options])


def _kept(folder: Path) -> list[tuple[dict, dict]]:
    """Version 2's marks of runs kept from version 1, each with the version 1
    mark of the same shard, which it copies."""
    entries = read_index(folder)
    sources = {e["shard"]: e for e in entries if e["version"] == 1}
    return [
        (e, sources[e["shard"]])
        for e in entries
        if e["version"] == 2 and e.get("kept_from") == 1
    ]


def _failed_lines(
    spec: Path, runs: Path, capsys: pytest.CaptureFixture[str]
) -> list[str]:
    """What pushlab status says on its failed lines."""
    capsys.readouterr()
    assert main(["status", str(spec), "--runs-dir", str(runs)]) == 0
    lines = capsys.readouterr().out.splitlines()
    failed = [line for line in lines if line.startswith("failed ")]
    return [line.removeprefix("failed").strip() for line in failed]


def test_under_sampling_a_new_version_keeps_failed_runs_on_any_row(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failing_generate: Callable[..., None],
) -> None:
    runs = tmp_path / "runs"
    folder = runs / NAME
    real = generate_module.generate
    third = read_prompt_set(PROMPTS_DIR / "smoke.jsonl").prompts[2]
    failing_generate(third, RuntimeError("CUDA out of memory"))
    # smoke-03's batch fails whole: its unpushed run on row 0, its pulse on row 1.
    assert _run(tiny_spec(decoding=SAMPLED), runs) == 2
    failed = [r.run_id for r in read_results(folder) if r.status == "failed"]
    assert failed == [f"{NAME}-000005", f"{NAME}-000006"]
    monkeypatch.setattr(generate_module, "generate", real)
    # Batches of 1, as the error advises, put every run on row 0. The pulses
    # that ended ok are redone; both failed runs are kept, for --retry-failed.
    smaller = tiny_spec(decoding=SAMPLED, batch_size="1")
    capsys.readouterr()
    assert _run(smaller, runs, "--overwrite-version", "--retry-failed") == 0
    out = capsys.readouterr().out
    assert "Started version 2; it keeps 6 runs saved." in out
    assert out.splitlines()[-1] == (
        "Done: 4 runs are still to run; the same command without --retry-failed "
        "runs them."
    )
    saved = {r.run_id: r for r in read_results(folder)}
    assert sorted(saved) == [IDS[i] for i in (0, 2, 4, 5, 6, 8)]
    assert {r.status for r in saved.values()} == {"ok"}
    assert saved[f"{NAME}-000006"].batch_key == "v2-b000006-1:0"  # retried, row 0
    # A plain run runs the redone pulses: then every run is there once, ok.
    assert _run(smaller, runs) == 0
    results = read_results(folder)
    assert [r.run_id for r in results] == IDS
    assert {r.status for r in results} == {"ok"}


def test_a_kept_failed_run_keeps_its_error(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    failing_generate: Callable[..., None],
) -> None:
    runs = tmp_path / "runs"
    third = read_prompt_set(PROMPTS_DIR / "smoke.jsonl").prompts[2]
    failing_generate(third, RuntimeError("boom"))  # smoke-03's pulse
    assert _run(tiny_spec(batch_size="1"), runs) == 2
    changed = tiny_spec(batch_size="1", max_gpu_hours="2")  # keeps every run
    assert _run(changed, runs, "--overwrite-version") == 0
    ((mark, source),) = [
        (mark, source)
        for mark, source in _kept(runs / NAME)
        if "failed" in mark["runs"].values()
    ]
    assert mark["runs"] == {f"{NAME}-000006": "failed"}
    assert mark["error"] == source["error"] == "RuntimeError: boom"
    failed = _failed_lines(changed, runs, capsys)
    assert failed == [f"{NAME}-000006: RuntimeError: boom"]


def test_a_kept_run_brings_its_share_of_its_batch_s_time(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ticks = iter(range(0, 100_000, 100))  # a stand-in clock: each batch takes 100 s
    monkeypatch.setattr(runner, "_clock", lambda: float(next(ticks)))
    runs = tmp_path / "runs"
    assert _run(tiny_spec(), runs) == 0
    assert [e["seconds"] for e in read_index(runs / NAME)] == [100.0] * 5
    # A weaker pulse: the pushed runs are redone, the unpushed ones kept.
    capsys.readouterr()
    assert _run(tiny_spec(WEAKER), runs, "--overwrite-version") == 0
    assert "Started version 2; it keeps 5 runs saved." in capsys.readouterr().out
    kept = _kept(runs / NAME)
    assert len(kept) == 5
    for mark, source in kept:
        share = len(mark["runs"]) / len(source["runs"])
        assert share == 1 / 2  # one run of each batch of 2
        assert mark["seconds"] == source["seconds"] * share == 50.0


def test_a_kept_run_s_mark_names_the_session_that_made_it(
    tiny_spec: Callable[..., Path], tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    folder = runs / NAME
    assert _run(tiny_spec(), runs, "--limit", "4") == 3  # two sessions make
    assert _run(tiny_spec(), runs) == 0  # version 1's runs
    assert _run(tiny_spec(max_gpu_hours="2"), runs, "--overwrite-version") == 0
    sessions = json.loads((folder / "manifest.json").read_text())["sessions"]
    first, second = (session["id"] for session in sessions)
    kept = _kept(folder)
    assert [mark.get("session") for mark, _ in kept] == [first] * 2 + [second] * 3
    assert all(mark.get("session") == source["session"] for mark, source in kept)


def test_the_code_change_warning_counts_the_sessions_of_kept_runs(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    real = environment.environment_info

    def on_commit(commit: str) -> None:
        def info(device: object) -> dict:
            return real(device) | {"code_commit": commit}

        monkeypatch.setattr(environment, "environment_info", info)

    runs = tmp_path / "runs"
    on_commit("a" * 40)
    assert _run(tiny_spec(), runs, "--limit", "4") == 3
    on_commit("b" * 40)  # as after a git pull on the pod
    capsys.readouterr()
    # Version 2 keeps version 1's 4 runs and runs the other 6.
    assert _run(tiny_spec(max_gpu_hours="2"), runs, "--overwrite-version") == 0
    captured = capsys.readouterr()
    assert "Started version 2; it keeps 4 runs saved." in captured.out
    assert (
        "warning: version 2's runs so far ran with another code_commit" in captured.err
    )


def test_after_a_crash_while_a_version_starts_the_advice_names_the_next_number(
    tiny_spec: Callable[..., Path],
    costs: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs = tmp_path / "runs"
    folder = runs / NAME
    assert _run(tiny_spec(), runs, "--limit", "4") == 3
    real = folder_module.write_identity

    def killed(path: Path, identity: dict) -> None:
        if identity["version"] > 1:
            raise KeyboardInterrupt  # after the kept marks, before experiment.json
        real(path, identity)

    monkeypatch.setattr(folder_module, "write_identity", killed)
    changed = tiny_spec(max_gpu_hours="2")
    capsys.readouterr()
    assert _run(changed, runs, "--overwrite-version") == 130
    assert "Stopped by Ctrl-C before the new version started" in capsys.readouterr().out
    monkeypatch.setattr(folder_module, "write_identity", real)
    # The crash left marks under version 2, a number that never counts.
    assert read_identity(folder)["version"] == 1
    assert {e["version"] for e in read_index(folder)} == {1, 2}
    assert _run(changed, runs) == 1
    assert "Run with --overwrite-version to start version 3, " in (
        capsys.readouterr().err
    )
    assert _run(changed, runs, "--dry-run") == 0
    out = capsys.readouterr().out
    (note,) = [line for line in out.splitlines() if "would refuse this" in line]
    assert "Run with --overwrite-version to start version 3, " in note


def test_the_refusal_says_what_a_new_version_would_keep(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prompts = tmp_path / "prompts.jsonl"
    smoke = (PROMPTS_DIR / "smoke.jsonl").read_text()
    prompts.write_text(smoke)
    runs = tmp_path / "runs"
    assert _run(tiny_spec(prompts=str(prompts)), runs, "--limit", "2") == 3
    # One prompt's text changed: no saved reply answers the prompt set now.
    prompts.write_text(smoke.replace("pancakes", "waffles"))
    capsys.readouterr()
    assert _run(tiny_spec(prompts=str(prompts)), runs) == 1
    err = capsys.readouterr().err
    assert "made from another prompt set." in err
    assert "which keeps no saved run, since the prompt set changed" in err
    # Only a setting changed: the saved runs are kept.
    prompts.write_text(smoke)
    assert _run(tiny_spec(prompts=str(prompts), max_gpu_hours="2"), runs) == 1
    err = capsys.readouterr().err
    assert "made from another experiment file." in err
    assert "which keeps the saved runs that did not change" in err


def test_a_disk_error_before_any_batch_is_refused_without_a_traceback(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    full = "No space left on device"
    runs = tmp_path / "runs"
    real_write = folder_module.write_identity

    def no_first_identity(path: Path, identity: dict) -> None:
        if identity["version"] == 1:
            raise OSError(28, full)
        real_write(path, identity)

    # The first experiment.json cannot be written.
    monkeypatch.setattr(folder_module, "write_identity", no_first_identity)
    assert _run(tiny_spec(), runs) == 1
    assert capsys.readouterr().err == f"pushlab run: [Errno 28] {full}\n"
    monkeypatch.setattr(folder_module, "write_identity", real_write)
    assert _run(tiny_spec(), runs, "--limit", "4") == 3
    real_append = folder_module.append_index

    def no_kept_marks(path: Path, entry: dict) -> None:
        if "kept_from" in entry:
            raise OSError(28, full)
        real_append(path, entry)

    # A new version cannot write its marks of the runs it keeps.
    monkeypatch.setattr(folder_module, "append_index", no_kept_marks)
    capsys.readouterr()
    assert _run(tiny_spec(max_gpu_hours="2"), runs, "--overwrite-version") == 1
    err = capsys.readouterr().err
    assert err.startswith("pushlab run: ") and full in err
    assert "Traceback" not in err
    assert read_identity(runs / NAME)["version"] == 1  # nothing of version 2 counts
