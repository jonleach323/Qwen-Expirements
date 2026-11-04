"""Running pushlab baseline end to end on the saved tiny Qwen3.5 (tiny-vl).

These tests run the command on the CPU, through the same code the real model
uses: loading, the chat prompt, greedy decoding and both output files.
"""

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import PreTrainedTokenizerBase

from pushlab import cli
from pushlab.cli import (
    BASELINE_EXPERIMENT,
    custom_prompt_id,
    main,
    read_baseline_experiment,
)
from pushlab.engine import generate
from pushlab.engine.generate import greedy_generate
from pushlab.models import load
from pushlab.models.chat import chat_prompt_ids, stop_token_ids
from pushlab.models.load import load_model
from pushlab.models.registry import ModelSpec
from pushlab.store import environment
from pushlab.store.environment import file_sha256
from pushlab.store.records import REQUIRED_MANIFEST_FIELDS, read_runs


@pytest.fixture
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU in float32 even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _no_loading(*args: object, **kwargs: object) -> None:
    raise AssertionError("the model must not load when the run is refused")


@pytest.mark.usefixtures("cpu_only")
def test_writes_one_greedy_baseline_run_and_a_manifest(
    tiny_downloads: dict[str, ModelSpec], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    argv = ["baseline", "--model", "tiny-vl", "--out", str(out)]
    assert main([*argv, "--max-new-tokens", "6"]) == 0

    (run,) = read_runs(out)
    experiment = read_baseline_experiment(BASELINE_EXPERIMENT)
    loaded = load_model("tiny-vl", torch.float32, device="cpu")
    assert run.run_id == "s0-baseline-000001"
    assert run.experiment == "s0-baseline"
    assert run.model == f"tiny-vl@{loaded.revision}"
    assert run.prompt_id == experiment.prompt_id
    assert run.schedule == []
    assert run.control_of == ""
    assert run.seed == experiment.seed
    assert run.batch_key == "b000001:0"
    assert run.decoding == {
        "strategy": "greedy",
        "max_new_tokens": 6,
        "thinking": False,
        "dtype": "float32",
    }
    assert run.status == "ok"
    # The reply is the greedy continuation of the prompt with thinking off.
    ids = chat_prompt_ids(loaded.tokenizer, experiment.prompt, thinking=False)
    stops = stop_token_ids(loaded.model, loaded.tokenizer)
    assert run.output_tokens == greedy_generate(loaded.model, ids, 6, stops)
    assert run.output_text == loaded.tokenizer.decode(
        run.output_tokens, skip_special_tokens=True
    )

    manifest = json.loads((out / "manifest.json").read_text())
    assert set(REQUIRED_MANIFEST_FIELDS) <= set(manifest)
    assert manifest["run_count"] == 1
    assert manifest["model_id"] == "local:tiny-qwen3.5"
    assert manifest["model_revision"] == loaded.revision
    assert manifest["gpu"] is None
    assert manifest["device"] == "cpu"
    assert manifest["dtype"] == "float32"
    assert manifest["prompt"] == experiment.prompt
    assert manifest["experiment_file"] == "experiments/p0/s0_baseline.yaml"
    assert manifest["experiment_sha256"] == experiment.sha256
    assert manifest["runs_by_status"] == {"ok": 1}
    assert manifest["packages"]["torch"] == torch.__version__
    assert manifest["started_at"] <= manifest["finished_at"]


@pytest.mark.usefixtures("cpu_only")
def test_the_reply_ends_at_the_end_of_the_turn_without_its_marker(
    tiny_downloads: dict[str, ModelSpec],
    tiny_tokenizer: PreTrainedTokenizerBase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The tiny model never ends a turn by itself, so a stand-in decoder
    # replies "Hi" and then the end-of-turn token.
    ends = tiny_tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])
    reply = [*tiny_tokenizer("Hi", add_special_tokens=False)["input_ids"], ends[0]]
    stops_given = []

    def stand_in(model: object, ids: list[int], count: int, stops: list[int]) -> list:
        stops_given.append(sorted(stops))
        return reply

    monkeypatch.setattr(generate, "greedy_generate", stand_in)
    out = tmp_path / "out"
    assert main(["baseline", "--model", "tiny-vl", "--out", str(out)]) == 0
    (run,) = read_runs(out)
    assert stops_given == [sorted(ends)]
    assert run.output_tokens == reply
    assert run.output_text == "Hi"


@pytest.mark.usefixtures("cpu_only")
def test_the_manifest_describes_the_code_as_the_run_started(
    tiny_downloads: dict[str, ModelSpec],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A copy of the project's experiment file, edited while the model loads,
    # as a git pull during the run would.
    project = tmp_path / "project"
    experiment_file = project / "experiments" / "p0" / "s0_baseline.yaml"
    experiment_file.parent.mkdir(parents=True)
    experiment_file.write_bytes(BASELINE_EXPERIMENT.read_bytes())
    hash_at_start = file_sha256(experiment_file)
    monkeypatch.setattr(cli, "PROJECT_ROOT", project)
    monkeypatch.setattr(cli, "BASELINE_EXPERIMENT", experiment_file)
    order = []
    read_environment, load_the_model = environment.environment_info, load.load_model

    def recorded_environment(*args: object, **kwargs: object) -> dict:
        order.append("environment")
        return read_environment(*args, **kwargs)

    def recorded_load(*args: object, **kwargs: object) -> load.LoadedModel:
        order.append("load")
        experiment_file.write_text("experiment: s0-baseline\nseed: 1\n")
        return load_the_model(*args, **kwargs)

    monkeypatch.setattr(environment, "environment_info", recorded_environment)
    monkeypatch.setattr(load, "load_model", recorded_load)
    out = tmp_path / "out"
    argv = ["baseline", "--model", "tiny-vl", "--out", str(out)]
    assert main([*argv, "--max-new-tokens", "2"]) == 0
    assert order == ["environment", "load"]
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["experiment_sha256"] == hash_at_start


@pytest.mark.usefixtures("cpu_only")
def test_a_system_clock_set_back_during_the_run_does_not_stop_it(
    tiny_downloads: dict[str, ModelSpec],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Each reading of the system clock is two seconds earlier than the last,
    # as a date and as seconds; the clock that never steps back is real.
    readings = iter(
        datetime(2026, 10, 29, 10, 0, 5 - 2 * n, tzinfo=UTC) for n in range(3)
    )
    seconds = iter(range(1_000_000, 0, -2))
    stepping_back = SimpleNamespace(
        time=lambda: next(seconds),
        monotonic=time.monotonic,
        perf_counter=time.perf_counter,
    )
    monkeypatch.setattr(cli, "_now", lambda: next(readings))
    monkeypatch.setattr(cli, "time", stepping_back)
    out = tmp_path / "out"
    argv = ["baseline", "--model", "tiny-vl", "--out", str(out)]
    assert main([*argv, "--max-new-tokens", "2"]) == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["started_at"] <= manifest["finished_at"]


@pytest.mark.usefixtures("cpu_only")
def test_a_manifest_that_fails_its_checks_leaves_no_run_file(
    tiny_downloads: dict[str, ModelSpec],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    read_environment = environment.environment_info

    def without_torch(*args: object, **kwargs: object) -> dict:
        return read_environment(*args, **kwargs) | {"torch": None}

    monkeypatch.setattr(environment, "environment_info", without_torch)
    out = tmp_path / "out"
    argv = ["baseline", "--model", "tiny-vl", "--out", str(out)]
    with pytest.raises(ValueError, match="torch"):
        main([*argv, "--max-new-tokens", "2"])
    assert not out.exists() or list(out.iterdir()) == []


@pytest.mark.usefixtures("cpu_only")
def test_only_an_existing_run_earns_the_overwrite_advice(
    tiny_downloads: dict[str, ModelSpec],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def refuse_to_write(*args: object, **kwargs: object) -> None:
        raise FileExistsError("some other file is in the way")

    monkeypatch.setattr(cli, "write_runs", refuse_to_write)
    argv = ["baseline", "--model", "tiny-vl", "--out", str(tmp_path / "out")]
    with pytest.raises(FileExistsError, match="some other file"):
        main([*argv, "--max-new-tokens", "2"])
    assert "--overwrite" not in capsys.readouterr().err


@pytest.mark.usefixtures("cpu_only")
def test_a_custom_prompt_is_used_and_recorded(
    tiny_downloads: dict[str, ModelSpec],
    tiny_tokenizer: PreTrainedTokenizerBase,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prompts_given = []

    def recording(model: object, ids: list[int], count: int, stops: list[int]) -> list:
        prompts_given.append(ids)
        return greedy_generate(model, ids, count, stops)

    monkeypatch.setattr(generate, "greedy_generate", recording)
    out = tmp_path / "out"
    argv = ["baseline", "--model", "tiny-vl", "--out", str(out)]
    assert main([*argv, "--prompt", "Say hi.", "--max-new-tokens", "3"]) == 0
    assert prompts_given == [chat_prompt_ids(tiny_tokenizer, "Say hi.", thinking=False)]
    (run,) = read_runs(out)
    assert run.prompt_id == custom_prompt_id("Say hi.")
    assert json.loads((out / "manifest.json").read_text())["prompt"] == "Say hi."


@pytest.mark.usefixtures("cpu_only")
def test_an_existing_run_is_kept_unless_overwrite_is_given(
    tiny_downloads: dict[str, ModelSpec],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "out"
    argv = ["baseline", "--model", "tiny-vl", "--out", str(out)]
    assert main([*argv, "--max-new-tokens", "2"]) == 0
    before = (out / "runs.parquet").read_bytes()

    monkeypatch.setattr(load, "load_model", _no_loading)
    assert main([*argv, "--max-new-tokens", "3"]) == 1
    assert "--overwrite" in capsys.readouterr().err
    assert (out / "runs.parquet").read_bytes() == before

    monkeypatch.setattr(load, "load_model", load_model)
    assert main([*argv, "--max-new-tokens", "3", "--overwrite"]) == 0
    (run,) = read_runs(out)
    assert run.decoding["max_new_tokens"] == 3
