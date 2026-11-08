"""The pushlab command and its baseline subcommand: pushlab.cli.

Its options, its experiment file and the runs it refuses before loading a
model. test_baseline_run.py runs it end to end on a tiny model.
"""

import doctest
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from pushlab import cli
from pushlab.cli import (
    BASELINE_EXPERIMENT,
    custom_prompt_id,
    main,
    read_baseline_experiment,
)
from pushlab.models import load
from pushlab.store.environment import file_sha256

OPTIONS = ["--model", "--out", "--prompt", "--max-new-tokens", "--overwrite"]


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
        ("experiment: s0-baseline\nseed: 2e3\nprompt_id: p\nprompt: Hi\n", "seed"),
        (
            "experiment: s0-baseline\nseed: 9223372036854775808\nprompt_id: p\n"
            "prompt: Hi\n",
            "seed",
        ),
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


def test_the_largest_seed_the_runs_table_holds_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "s0_baseline.yaml"
    path.write_text(
        "experiment: s0-baseline\nseed: 9223372036854775807\nprompt_id: p\nprompt: Hi\n"
    )
    assert read_baseline_experiment(path).seed == 2**63 - 1


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


@pytest.mark.parametrize("out", ["latest", "latest/run-1"])
def test_out_through_a_broken_link_is_refused(
    tmp_path: Path,
    out: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Such as a "latest" link to a run folder that was deleted since: no
    # folder can be made there, so the command must refuse before loading.
    (tmp_path / "latest").symlink_to(tmp_path / "deleted-run")
    monkeypatch.setattr(load, "load_model", _no_loading)
    argv = ["baseline", "--model", "qwen3.5-4b", "--out", str(tmp_path / out)]
    assert main(argv) == 1
    assert f"{tmp_path / 'latest'} is a link to nothing" in capsys.readouterr().err


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


def test_the_experiment_commands_options_have_working_examples() -> None:
    from pushlab import cli_experiments

    result = doctest.testmod(cli_experiments)
    assert result.attempted > 0 and result.failed == 0
