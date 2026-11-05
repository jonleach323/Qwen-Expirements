"""`pushlab amount`: the amount a dose adds to a neuron, from the calibration."""

import doctest
import json
from pathlib import Path

import pytest
import torch

from pushlab.cli import main
from pushlab.models import load
from pushlab.models.load import load_model
from pushlab.units import amount_command
from pushlab.units.dose import amount
from pushlab.units.neuron import NeuronUnit

UNIT = "tiny-vl:L2:N100"


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU in float32 even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _no_loading(*args: object, **kwargs: object) -> None:
    raise AssertionError("the model must not load when amount refuses")


def test_amount_help_lists_the_options(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["amount", "--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--unit" in help_text and "--alpha" in help_text


@pytest.mark.parametrize("alpha", ["1.5", "-0.25"])
def test_amount_prints_what_a_dose_adds_to_the_neuron(
    tiny_vl_calibration: Path, capsys: pytest.CaptureFixture[str], alpha: str
) -> None:
    assert main(["amount", "--unit", UNIT, "--alpha", alpha]) == 0
    model = load_model("tiny-vl", torch.float32, device="cpu").model
    norms = json.loads(tiny_vl_calibration.read_text())["norms"]
    expected = amount(NeuronUnit.parse(UNIT), float(alpha), model, norms)
    out = capsys.readouterr().out
    assert f"adds {expected:+.4f} to the neuron's value" in out
    # A push of |alpha| times layer 2's size, 8.0, along the write direction.
    assert f"{abs(float(alpha)) * 8.0:.4f}" in out
    assert str(tiny_vl_calibration) in out


def test_without_a_calibration_amount_stops_and_names_the_command(
    tiny_downloads: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from pushlab.units import calibrate

    monkeypatch.setattr(calibrate, "CALIBRATION_DIR", tmp_path / "calibration")
    monkeypatch.setattr(load, "load_model", _no_loading)
    assert main(["amount", "--unit", UNIT, "--alpha", "1"]) == 1
    assert "uv run pushlab calibrate --model tiny-vl" in capsys.readouterr().err


def test_a_model_that_is_not_downloaded_is_explained(
    tiny_downloads: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("PUSHLAB_MODELS_DIR", str(tmp_path / "no-models"))
    monkeypatch.setattr(load, "load_model", _no_loading)
    assert main(["amount", "--unit", UNIT, "--alpha", "1"]) == 1
    assert "not downloaded" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("unit", "alpha", "problem"),
    [
        ("tiny-vl:L2", "1", "tiny-vl:L2"),
        ("nobody:L1:N1", "1", "registry"),
        ("tiny-vl:L9:N1", "1", "layer"),
        (UNIT, "inf", "alpha"),
    ],
)
def test_a_bad_unit_or_dose_is_refused_before_the_model_loads(
    tiny_vl_calibration: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    unit: str,
    alpha: str,
    problem: str,
) -> None:
    monkeypatch.setattr(load, "load_model", _no_loading)
    assert main(["amount", "--unit", unit, "--alpha", alpha]) == 1
    assert problem in capsys.readouterr().err


def test_docstring_examples_run() -> None:
    result = doctest.testmod(amount_command)
    assert result.attempted > 0
    assert result.failed == 0
