"""`pushlab calibrate`: its options, and runs end to end on the tiny Qwen3.5."""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest
import torch

from pushlab.cli import main
from pushlab.models import load
from pushlab.models.chat import chat_prompt_ids
from pushlab.models.load import load_model
from pushlab.models.registry import ModelSpec
from pushlab.prompts import read_prompt_set
from pushlab.store.environment import file_sha256
from pushlab.units import calibrate
from pushlab.units.calibrate import (
    PROMPTS_FILE,
    calibrate_norms,
    calibration_path,
    load_calibration,
)


@pytest.fixture
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU in float32 even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def calibration_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fresh folder for the calibration files, in place of calibration/."""
    folder = tmp_path / "calibration"
    monkeypatch.setattr(calibrate, "CALIBRATION_DIR", folder)
    return folder


def _no_loading(*args: object, **kwargs: object) -> None:
    raise AssertionError("the model must not load when the command refuses")


def test_calibrate_help_lists_the_options(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["calibrate", "--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    for text in ("--model", "--overwrite", "qwen3.5-4b", "prompts/calibration.jsonl"):
        assert text in help_text


def test_the_installed_command_answers_calibrate_help() -> None:
    command = Path(sys.executable).with_name("pushlab")
    result = subprocess.run(
        [str(command), "calibrate", "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "--model" in result.stdout


@pytest.mark.parametrize(
    "argv",
    [["calibrate"], ["calibrate", "--model", "tiny"], ["calibrate", "--model", "x"]],
)
def test_bad_arguments_stop_with_a_usage_message(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2
    assert "usage" in capsys.readouterr().err


@pytest.mark.usefixtures("cpu_only")
def test_writes_the_calibration_file_and_prints_each_layer(
    tiny_downloads: dict[str, ModelSpec],
    calibration_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["calibrate", "--model", "tiny-vl"]) == 0

    loaded = load_model("tiny-vl", torch.float32, device="cpu")
    calibration = load_calibration("tiny-vl", loaded.revision, calibration_dir)
    prompt_set = read_prompt_set(PROMPTS_FILE)
    expected = calibrate_norms(loaded.model, loaded.tokenizer, prompt_set.prompts)
    assert calibration.norms == tuple(expected)
    assert calibration.prompt_set_sha256 == file_sha256(PROMPTS_FILE)
    info = calibration.info
    assert info["model_id"] == tiny_downloads["tiny-vl"].hf_id
    assert info["prompt_set"] == "prompts/calibration.jsonl"
    assert info["prompt_set_version"] == 1
    assert info["prompt_count"] == 32
    ids = [
        chat_prompt_ids(loaded.tokenizer, p, thinking=False) for p in prompt_set.prompts
    ]
    assert info["token_count"] == sum(len(prompt_ids) - 1 for prompt_ids in ids)
    assert (info["dtype"], info["device"], info["gpu"]) == ("float32", "cpu", None)
    assert datetime.fromisoformat(info["measured_at"]).tzinfo is not None
    assert info["torch"] == torch.__version__
    assert info["packages"]["transformers"] == info["transformers"]
    assert "chat template" in info["measure"]

    out = capsys.readouterr().out
    path = calibration_path("tiny-vl", loaded.revision, calibration_dir)
    for layer, norm in enumerate(expected):
        assert f"L{layer} " in out and f"{norm:.3f}" in out
    assert str(path) in out


@pytest.mark.usefixtures("cpu_only")
def test_an_existing_calibration_is_replaced_only_with_overwrite(
    tiny_downloads: dict[str, ModelSpec],
    calibration_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["calibrate", "--model", "tiny-vl"]) == 0
    (path,) = calibration_dir.iterdir()
    path.write_text(path.read_text().replace('"float32"', '"edited"'))
    edited = path.read_text()
    capsys.readouterr()
    assert main(["calibrate", "--model", "tiny-vl"]) == 1
    assert "--overwrite" in capsys.readouterr().err
    assert path.read_text() == edited
    assert main(["calibrate", "--model", "tiny-vl", "--overwrite"]) == 0
    assert '"edited"' not in path.read_text()


def test_a_model_that_is_not_downloaded_is_explained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    calibration_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("PUSHLAB_MODELS_DIR", str(tmp_path / "models"))
    assert main(["calibrate", "--model", "qwen3.5-4b"]) == 1
    assert "setup_pod.sh" in capsys.readouterr().err
    assert not calibration_dir.exists()


def test_a_broken_prompt_set_stops_the_command_before_the_model_loads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    calibration_dir: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    broken = tmp_path / "calibration.jsonl"
    broken.write_text("not json\n")
    monkeypatch.setattr(calibrate, "PROMPTS_FILE", broken)
    monkeypatch.setattr(load, "load_model", _no_loading)
    assert main(["calibrate", "--model", "qwen3.5-4b"]) == 1
    assert "cannot use" in capsys.readouterr().err
    assert not calibration_dir.exists()
