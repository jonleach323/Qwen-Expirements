"""The pushlab command and its baseline subcommand: pushlab.cli.

The end-to-end tests run the command on the saved tiny Qwen3.5 (tiny-vl)
on the CPU, through the same code the real model uses.
"""

import doctest
import hashlib
import json
import subprocess
import sys
from pathlib import Path

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

OPTIONS = ["--model", "--out", "--prompt", "--max-new-tokens", "--overwrite"]


@pytest.fixture
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU in float32 even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _no_loading(*args: object, **kwargs: object) -> None:
    raise AssertionError("the model must not load when the run is refused")


def test_baseline_help_lists_the_options(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["baseline", "--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    for option in OPTIONS:
        assert option in help_text
    assert "qwen3.5-4b" in help_text
    assert "tiny" not in help_text.replace("tiny-", "")


def test_the_pushlab_command_is_installed() -> None:
    # uv sync installs it next to the environment's Python.
    command = Path(sys.executable).with_name("pushlab")
    result = subprocess.run(
        [str(command), "baseline", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    for option in OPTIONS:
        assert option in result.stdout


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["baseline", "--out", "x"],
        ["baseline", "--model", "qwen3.5-4b"],
        ["baseline", "--model", "tiny", "--out", "x"],
        ["baseline", "--model", "qwen3.5-4b", "--out", "x", "--max-new-tokens", "0"],
        ["baseline", "--model", "qwen3.5-4b", "--out", "x", "--max-new-tokens", "a"],
    ],
)
def test_bad_arguments_stop_with_a_usage_message(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2
    assert "usage" in capsys.readouterr().err


def test_the_experiment_file_holds_the_default_prompt() -> None:
    experiment = read_baseline_experiment(BASELINE_EXPERIMENT)
    assert experiment.experiment == "s0-baseline"
    assert experiment.prompt == (
        "Explain what a neuron in a language model is, in three sentences."
    )
    assert experiment.prompt_id
    assert experiment.seed == 0
    assert experiment.sha256 == file_sha256(BASELINE_EXPERIMENT)


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("experiment: s0-baseline\nseed: 0\nprompt_id: p\n", "prompt"),
        ("experiment: s0-baseline\nseed: zero\nprompt_id: p\nprompt: Hi\n", "seed"),
        ("experiment: s0-baseline\nseed: 9223372036854775808\nprompt: Hi\n", "seed"),
        ("experiment: S0 baseline\nseed: 0\nprompt_id: p\nprompt: Hi\n", "experiment"),
        ("- just a list\n", "experiment"),
        ("experiment: [s0\n", "YAML"),
    ],
)
def test_a_broken_experiment_file_is_refused(
    tmp_path: Path, text: str, problem: str
) -> None:
    path = tmp_path / "s0_baseline.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match=problem):
        read_baseline_experiment(path)


def test_a_broken_experiment_file_stops_the_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    broken = tmp_path / "s0_baseline.yaml"
    broken.write_text("experiment: s0-baseline\n")
    monkeypatch.setattr(cli, "BASELINE_EXPERIMENT", broken)
    out = tmp_path / "out"
    assert main(["baseline", "--model", "qwen3.5-4b", "--out", str(out)]) == 1
    assert "cannot use" in capsys.readouterr().err
    assert not out.exists()


def test_an_empty_prompt_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    argv = ["baseline", "--model", "qwen3.5-4b", "--out", str(out), "--prompt", " "]
    assert main(argv) == 1
    assert "--prompt is empty" in capsys.readouterr().err
    assert not out.exists()


def test_a_custom_prompt_gets_an_id_from_its_text() -> None:
    expected = "custom-" + hashlib.sha256(b"Say hi.").hexdigest()[:12]
    assert custom_prompt_id("Say hi.") == expected
    assert custom_prompt_id("Say hi!") != expected


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
    order = []
    read_environment, load_the_model = environment.environment_info, load.load_model

    def recorded_environment(*args: object, **kwargs: object) -> dict:
        order.append("environment")
        return read_environment(*args, **kwargs)

    def recorded_load(*args: object, **kwargs: object) -> load.LoadedModel:
        order.append("load")
        return load_the_model(*args, **kwargs)

    monkeypatch.setattr(environment, "environment_info", recorded_environment)
    monkeypatch.setattr(load, "load_model", recorded_load)
    argv = ["baseline", "--model", "tiny-vl", "--out", str(tmp_path / "out")]
    assert main([*argv, "--max-new-tokens", "2"]) == 0
    assert order == ["environment", "load"]


@pytest.mark.usefixtures("cpu_only")
def test_a_custom_prompt_is_used_and_recorded(
    tiny_downloads: dict[str, ModelSpec], tmp_path: Path
) -> None:
    out = tmp_path / "out"
    argv = ["baseline", "--model", "tiny-vl", "--out", str(out)]
    assert main([*argv, "--prompt", "Say hi.", "--max-new-tokens", "3"]) == 0
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


def test_a_folder_with_only_a_manifest_is_refused_too(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / "manifest.json").write_text("{}")
    monkeypatch.setattr(load, "load_model", _no_loading)
    assert main(["baseline", "--model", "qwen3.5-4b", "--out", str(out)]) == 1
    assert "--overwrite" in capsys.readouterr().err
    assert [path.name for path in out.iterdir()] == ["manifest.json"]


@pytest.mark.parametrize("out", ["results", "results/run-1"])
@pytest.mark.parametrize("extra", [[], ["--overwrite"]])
def test_out_must_be_a_folder(
    tmp_path: Path,
    out: str,
    extra: list[str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Such as the path of an earlier run's runs.parquet, pasted back in.
    (tmp_path / "results").write_text("a file")
    monkeypatch.setattr(load, "load_model", _no_loading)
    argv = ["baseline", "--model", "qwen3.5-4b", "--out", str(tmp_path / out)]
    assert main([*argv, *extra]) == 1
    assert f"{tmp_path / 'results'} is a file" in capsys.readouterr().err


def test_a_model_that_is_not_downloaded_is_explained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("PUSHLAB_MODELS_DIR", str(tmp_path / "models"))
    out = tmp_path / "out"
    assert main(["baseline", "--model", "qwen3.5-4b", "--out", str(out)]) == 1
    error = capsys.readouterr().err
    assert "configs/models.yaml" in error
    assert "setup_pod.sh" in error
    assert not (out / "runs.parquet").exists()


def test_docstring_examples_run() -> None:
    result = doctest.testmod(cli)
    assert result.attempted > 0
    assert result.failed == 0
