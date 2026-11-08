"""`pushlab run`: failed runs, --retry-failed, and inputs that change between runs.

tiny_spec's experiment has 10 runs; with batch_size 1, each is a batch.
"""

import dataclasses
import json
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from pushlab.cli import main
from pushlab.engine import generate as generate_module
from pushlab.models import load
from pushlab.prompts import PROMPTS_DIR, read_prompt_set
from pushlab.runs.folder import read_identity, read_index, read_results
from pushlab.runs.runner import differences

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


def _no_loading(*args: object, **kwargs: object) -> None:
    raise AssertionError("the model must not load")


def test_a_run_that_raises_is_saved_as_failed_and_retry_reruns_only_it(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = tiny_spec(batch_size="1"), tmp_path / "runs"
    folder = runs / NAME
    third = read_prompt_set(PROMPTS_DIR / "smoke.jsonl").prompts[2]
    real = generate_module.generate

    def failing(model, tokenizer, prompts, *args, schedules=None, **kwargs):  # type: ignore[no-untyped-def]
        if prompts == [third] and schedules and schedules[0].segments:
            raise RuntimeError("boom")  # smoke-03's pulse
        return real(model, tokenizer, prompts, *args, schedules=schedules, **kwargs)

    monkeypatch.setattr(generate_module, "generate", failing)
    assert _run(spec, runs) == 2  # finished, with a failure
    out = capsys.readouterr().out
    assert "1 run failed; rerun it with --retry-failed" in out
    results = {run.run_id: run for run in read_results(folder)}
    failed = [run for run in results.values() if run.status == "failed"]
    assert [run.run_id for run in failed] == [f"{NAME}-000006"]
    assert (failed[0].output_text, failed[0].output_tokens) == ("", [])
    (entry,) = [e for e in read_index(folder) if f"{NAME}-000006" in e["runs"]]
    assert entry["error"] == "RuntimeError: boom"
    # A plain rerun leaves the failure alone.
    monkeypatch.setattr(load, "load_model", _no_loading)
    assert _run(spec, runs) == 0
    assert "1 failed; rerun them with --retry-failed" in capsys.readouterr().out


def test_retry_failed_reruns_only_the_failed_runs(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = tiny_spec(batch_size="1"), tmp_path / "runs"
    folder = runs / NAME
    third = read_prompt_set(PROMPTS_DIR / "smoke.jsonl").prompts[2]
    real = generate_module.generate

    def failing(model, tokenizer, prompts, *args, schedules=None, **kwargs):  # type: ignore[no-untyped-def]
        if prompts == [third] and schedules and schedules[0].segments:
            raise RuntimeError("boom")
        return real(model, tokenizer, prompts, *args, schedules=schedules, **kwargs)

    monkeypatch.setattr(generate_module, "generate", failing)
    assert _run(spec, runs) == 2
    before = {run.run_id: run for run in read_results(folder)}
    monkeypatch.setattr(generate_module, "generate", real)
    sizes = _sizes(monkeypatch)
    assert _run(spec, runs, "--retry-failed") == 0
    assert sizes == [1]  # that run alone, in its planned batch
    after = {run.run_id: run for run in read_results(folder)}
    assert {run.status for run in after.values()} == {"ok"}
    assert after[f"{NAME}-000006"].batch_key == before[f"{NAME}-000006"].batch_key
    assert {k: v for k, v in after.items() if k != f"{NAME}-000006"} == {
        k: v for k, v in before.items() if k != f"{NAME}-000006"
    }
    capsys.readouterr()
    assert _run(spec, runs, "--retry-failed") == 0
    assert "Nothing to retry: no run of version 1 failed." in capsys.readouterr().out


def test_a_changed_experiment_file_is_refused_unless_a_new_version_is_asked_for(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs = tmp_path / "runs"
    folder = runs / NAME
    assert _run(tiny_spec(), runs, "--limit", "4") == 0
    saved = {run.run_id: run for run in read_results(folder)}
    changed = tiny_spec(max_gpu_hours="2")  # the same file, edited
    real_load = load.load_model
    monkeypatch.setattr(load, "load_model", _no_loading)
    assert _run(changed, runs) == 1
    err = capsys.readouterr().err
    assert (
        f"{folder} holds version 1 of {NAME}, made from another experiment file" in err
    )
    assert "--overwrite-version" in err
    monkeypatch.setattr(load, "load_model", real_load)
    sizes = _sizes(monkeypatch)
    assert _run(changed, runs, "--overwrite-version") == 0
    assert sizes == [2, 2, 2]  # the 4 saved runs did not change, so they are kept
    assert read_identity(folder)["version"] == 2
    results = {run.run_id: run for run in read_results(folder)}
    assert len(results) == 10
    assert all(results[run_id] == run for run_id, run in saved.items())


def test_a_new_version_redoes_runs_whose_record_changed(
    tiny_spec: Callable[..., Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs = tmp_path / "runs"
    assert _run(tiny_spec(), runs, "--limit", "4") == 0
    sizes = _sizes(monkeypatch)
    longer = tiny_spec(max_new_tokens="5")  # every run's decoding changes
    assert _run(longer, runs, "--overwrite-version") == 0
    assert sizes == [2] * 5
    results = read_results(runs / NAME)
    assert {run.decoding["max_new_tokens"] for run in results} == {5}


def test_a_changed_prompt_set_or_calibration_is_refused(
    tiny_spec: Callable[..., Path],
    tiny_vl_calibration: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_bytes((PROMPTS_DIR / "smoke.jsonl").read_bytes())
    spec, runs = tiny_spec(prompts=str(prompts)), tmp_path / "runs"
    assert _run(spec, runs, "--limit", "2") == 0
    text = prompts.read_text().replace("pancakes", "waffles")
    prompts.write_text(text)
    assert _run(spec, runs) == 1
    assert "made from another prompt set" in capsys.readouterr().err
    prompts.write_bytes((PROMPTS_DIR / "smoke.jsonl").read_bytes())
    record = json.loads(tiny_vl_calibration.read_text())
    record["note"] = "measured again"
    tiny_vl_calibration.write_text(json.dumps(record))
    assert _run(spec, runs) == 1
    assert "made from another calibration" in capsys.readouterr().err


def test_differences_name_each_input_that_changed() -> None:
    recorded = {
        "spec_sha256": "a",
        "prompts_sha256": "b",
        "known_sha256": None,
        "model": "tiny-vl@1",
        "calibration_sha256": "c",
    }
    assert differences(recorded, dict(recorded)) == []
    current = recorded | {"spec_sha256": "x", "model": "tiny-vl@2"}
    assert differences(recorded, current) == ["experiment file", "model download"]
    every = {key: "z" for key in recorded}
    assert differences(recorded, every) == [
        "experiment file",
        "prompt set",
        "known units",
        "model download",
        "calibration",
    ]


def _failing_on(monkeypatch: pytest.MonkeyPatch, prompt: str, error: Exception) -> None:
    """Make generate() raise error for the pushed runs of one prompt."""
    real = generate_module.generate

    def failing(model, tokenizer, prompts, *args, schedules=None, **kwargs):  # type: ignore[no-untyped-def]
        if prompts == [prompt] and schedules and schedules[0].segments:
            raise error
        return real(model, tokenizer, prompts, *args, schedules=schedules, **kwargs)

    monkeypatch.setattr(generate_module, "generate", failing)


def test_retry_failed_leaves_runs_not_yet_run_alone(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = tiny_spec(batch_size="1"), tmp_path / "runs"
    real = generate_module.generate
    first = read_prompt_set(PROMPTS_DIR / "smoke.jsonl").prompts[0]
    _failing_on(monkeypatch, first, RuntimeError("boom"))
    assert _run(spec, runs, "--limit", "2") == 2  # smoke-01's pulse failed
    monkeypatch.setattr(generate_module, "generate", real)
    sizes = _sizes(monkeypatch)
    capsys.readouterr()
    assert _run(spec, runs, "--retry-failed") == 0
    assert sizes == [1]
    assert "Done: 8 runs are still to run." in capsys.readouterr().out
    assert len(read_results(runs / NAME)) == 2


def test_an_out_of_memory_error_says_to_lower_the_batch_size(
    tiny_spec: Callable[..., Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, runs = tiny_spec(batch_size="1"), tmp_path / "runs"
    first = read_prompt_set(PROMPTS_DIR / "smoke.jsonl").prompts[0]
    _failing_on(monkeypatch, first, torch.cuda.OutOfMemoryError("CUDA out of memory"))
    assert _run(spec, runs, "--limit", "2") == 2
    (_, entry) = read_index(runs / NAME)
    assert entry["error"] == (
        "OutOfMemoryError: CUDA out of memory; lower batch_size in the experiment "
        "file (1)"
    )


def test_inputs_that_change_while_the_model_loads_are_refused(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import pushlab.prompts

    real = pushlab.prompts.read_prompt_set

    def edited(path: Path) -> object:  # the file, as read after the model loaded
        return dataclasses.replace(real(path), sha256="0" * 64)

    monkeypatch.setattr(pushlab.prompts, "read_prompt_set", edited)
    assert _run(tiny_spec(), tmp_path / "runs") == 1
    assert "smoke.jsonl changed while it was read; run again" in capsys.readouterr().err
    assert read_index(tmp_path / "runs" / NAME) == []


def test_runs_that_cannot_share_a_batch_are_refused(
    tiny_spec: Callable[..., Path],
) -> None:
    from pushlab.runs.batch import run_batch
    from pushlab.runs.expand import expand
    from pushlab.runs.plan import Batch
    from pushlab.runs.spec import read_spec

    spec = read_spec(tiny_spec())
    runs = expand(spec).runs
    mixed = Batch("b000001", (runs[0], runs[2]))  # smoke-01 and smoke-02
    with pytest.raises(ValueError, match="b000001 mixes runs that cannot share"):
        run_batch(mixed, spec, None, None, {}, None)
