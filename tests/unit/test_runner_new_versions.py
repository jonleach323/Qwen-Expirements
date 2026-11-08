"""`pushlab run --overwrite-version`: what a new version keeps, and a crash
while it starts. tiny_spec's experiment has 10 runs."""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from pushlab.cli import main
from pushlab.engine import generate as generate_module
from pushlab.prompts import PROMPTS_DIR, read_prompt_set
from pushlab.runs import folder as folder_module
from pushlab.runs.folder import read_identity, read_index, read_results, used_seconds

NAME = "tiny-pulses"


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _run(spec: Path, runs: Path, *options: str) -> int:
    return main(["run", str(spec), "--runs-dir", str(runs), *options])


def _sizes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    real = generate_module.generate
    sizes: list[int] = []

    def counting(model, tokenizer, prompts, *args, **kwargs):  # type: ignore[no-untyped-def]
        sizes.append(len(prompts))
        return real(model, tokenizer, prompts, *args, **kwargs)

    monkeypatch.setattr(generate_module, "generate", counting)
    return sizes


def test_a_crash_while_a_version_starts_never_counts_its_marks(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs = tmp_path / "runs"
    folder = runs / NAME
    assert _run(tiny_spec(), runs, "--limit", "4") == 3
    first = used_seconds(read_index(folder), 1)
    real = folder_module.write_identity

    def killed(path: Path, identity: dict) -> None:
        if identity["version"] > 1:
            raise KeyboardInterrupt  # after the kept marks, before experiment.json
        real(path, identity)

    monkeypatch.setattr(folder_module, "write_identity", killed)
    changed = tiny_spec(max_gpu_hours="2")
    assert _run(changed, runs, "--overwrite-version") == 130
    assert "Stopped by Ctrl-C before the new version started" in capsys.readouterr().out
    assert read_identity(folder)["version"] == 1  # as it was
    monkeypatch.setattr(folder_module, "write_identity", real)
    assert _run(changed, runs, "--overwrite-version") == 0
    # The crashed attempt's marks were version 2's; the new version is 3.
    assert read_identity(folder)["version"] == 3
    entries = read_index(folder)
    kept = [e for e in entries if e["version"] == 3 and e.get("kept_from") == 1]
    assert sum(e["seconds"] for e in kept) == pytest.approx(first)
    assert len(read_results(folder)) == 10


def test_a_new_version_keeps_failed_runs_so_retry_failed_still_finds_them(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failing_generate: Callable[..., None],
) -> None:
    runs = tmp_path / "runs"
    real = generate_module.generate
    third = read_prompt_set(PROMPTS_DIR / "smoke.jsonl").prompts[2]
    failing_generate(third, RuntimeError("boom"))
    assert _run(tiny_spec(batch_size="1"), runs) == 2
    monkeypatch.setattr(generate_module, "generate", real)
    changed = tiny_spec(batch_size="1", max_gpu_hours="2")
    capsys.readouterr()
    assert _run(changed, runs, "--overwrite-version") == 0
    out = capsys.readouterr().out
    assert "Started version 2; it keeps 10 runs saved." in out
    assert "9 ok and 1 failed; rerun it with --retry-failed" in out
    sizes = _sizes(monkeypatch)
    assert _run(changed, runs, "--retry-failed") == 0
    assert sizes == [1]
    assert {r.status for r in read_results(runs / NAME)} == {"ok"}


def test_a_new_version_keeps_nothing_when_a_prompt_or_the_calibration_changed(
    tiny_spec: Callable[..., Path],
    tiny_vl_calibration: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_bytes((PROMPTS_DIR / "smoke.jsonl").read_bytes())
    spec, runs = tiny_spec(prompts=str(prompts)), tmp_path / "runs"
    assert _run(spec, runs, "--limit", "4") == 3
    # The same prompt IDs, one text changed: no saved reply is kept.
    prompts.write_text(prompts.read_text().replace("pancakes", "waffles"))
    assert _run(spec, runs, "--overwrite-version", "--limit", "4") == 3
    assert "it keeps 0 runs saved" in capsys.readouterr().out
    record = json.loads(tiny_vl_calibration.read_text()) | {"note": "again"}
    tiny_vl_calibration.write_text(json.dumps(record))
    assert _run(spec, runs, "--overwrite-version", "--limit", "2") == 3
    assert "it keeps 0 runs saved" in capsys.readouterr().out


def test_a_sampled_run_is_kept_only_on_the_row_the_new_plan_gives_it(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sampled = "{strategy: sample, temperature: 0.7, dtype: float32}"
    runs = tmp_path / "runs"
    assert _run(tiny_spec(decoding=sampled), runs, "--limit", "4") == 3
    # Batches of 1: the unpushed runs keep row 0, the pulses move from row 1.
    capsys.readouterr()
    assert (
        _run(tiny_spec(decoding=sampled, batch_size="1"), runs, "--overwrite-version")
        == 0
    )
    assert "it keeps 2 runs saved" in capsys.readouterr().out
    rows = {r.run_id: r.batch_key for r in read_results(runs / NAME)}
    assert rows[f"{NAME}-000001"].startswith("v1-") and rows[
        f"{NAME}-000002"
    ].startswith("v2-")


def test_the_dry_run_counts_what_a_new_version_would_keep(
    tiny_spec: Callable[..., Path],
    costs: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs = tmp_path / "runs"
    assert _run(tiny_spec(), runs, "--limit", "4") == 3
    capsys.readouterr()
    changed = tiny_spec(max_gpu_hours="2")
    assert _run(changed, runs, "--dry-run", "--overwrite-version") == 0
    out = capsys.readouterr().out
    assert (
        "to run        6 of 10 runs; a new version would keep 4 runs of version 1"
        in out
    )
    assert "would refuse" not in out
    assert _run(changed, runs, "--dry-run") == 0
    assert "pushlab run would refuse this" in capsys.readouterr().out
