"""`pushlab steer`'s refusals: what it stops on, before the model loads if it can.

The tiny Qwen3.5 (tiny-vl) loads through pushlab's real loading code, with
tiny-vl's own residual sizes as its calibration (the tiny_vl_calibration
fixture).
"""

import json
from pathlib import Path

import pytest
import torch

from pushlab.cli import main
from pushlab.engine import generate as generate_module
from pushlab.models import load
from pushlab.models.load import load_model
from pushlab.prompts import PROMPTS_DIR
from pushlab.store.records import MANIFEST_FILE, RUNS_FILE, read_runs
from pushlab.units import dose

SMOKE = PROMPTS_DIR / "smoke.jsonl"
UNIT = "tiny-vl:L2:N100"


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU in float32 even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _steer(out: Path, *options: str, prompts: Path = SMOKE) -> int:
    return main(
        ["steer", "--model", "tiny-vl", "--prompts", str(prompts), "--out", str(out)]
        + ["--max-new-tokens", "4", *options]
    )


def _no_loading(*args: object, **kwargs: object) -> None:
    raise AssertionError("the model must not load when steer refuses")


def test_a_negative_start_is_a_usage_error(
    tiny_vl_calibration: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        _steer(tmp_path / "out", "--unit", UNIT, "--alpha", "1", "--start", "-1")
    assert exit_info.value.code == 2
    assert "--start" in capsys.readouterr().err


def test_without_a_calibration_steer_stops_and_names_the_command(
    tiny_downloads: dict[str, object],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from pushlab.units import calibrate

    monkeypatch.setattr(calibrate, "CALIBRATION_DIR", tmp_path / "calibration")
    monkeypatch.setattr(load, "load_model", _no_loading)
    assert _steer(tmp_path / "out", "--unit", UNIT, "--alpha", "1") == 1
    assert "uv run pushlab calibrate --model tiny-vl" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_steer_without_units_needs_no_calibration(
    tiny_downloads: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pushlab.units import calibrate

    monkeypatch.setattr(calibrate, "CALIBRATION_DIR", tmp_path / "calibration")
    assert _steer(tmp_path / "out") == 0
    manifest = json.loads((tmp_path / "out" / MANIFEST_FILE).read_text())
    assert manifest["calibration_file"] is None
    assert manifest["calibration_sha256"] is None


def test_an_existing_run_folder_is_kept_unless_overwrite_is_given(
    tiny_vl_calibration: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "out"
    assert _steer(out, "--unit", UNIT, "--alpha", "1") == 0
    before = (out / RUNS_FILE).read_bytes()
    monkeypatch.setattr(load, "load_model", _no_loading)
    assert _steer(out, "--unit", UNIT, "--alpha", "2") == 1
    assert "--overwrite" in capsys.readouterr().err
    assert (out / RUNS_FILE).read_bytes() == before
    monkeypatch.setattr(load, "load_model", load_model)
    assert _steer(out, "--unit", UNIT, "--alpha", "2", "--overwrite") == 0
    assert read_runs(out)[0].schedule[0]["alpha"] == 2.0


def test_a_folder_with_only_a_manifest_is_refused_too(
    tiny_vl_calibration: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "out"
    out.mkdir()
    (out / MANIFEST_FILE).write_text("{}")
    monkeypatch.setattr(load, "load_model", _no_loading)
    assert _steer(out, "--unit", UNIT, "--alpha", "1") == 1
    assert "--overwrite" in capsys.readouterr().err
    assert sorted(path.name for path in out.iterdir()) == [MANIFEST_FILE]


@pytest.mark.parametrize(
    ("options", "problem"),
    [
        (["--unit", UNIT], "--alpha"),
        (["--alpha", "1"], "--unit"),
        (["--unit", UNIT, "--alpha", "1", "--alpha", "2"], "--alpha"),
        (["--unit", "tiny:L1:N5", "--alpha", "1"], "tiny-vl"),
        (["--unit", "tiny-vl:L9:N5", "--alpha", "1"], "layer"),
        (["--unit", "tiny-vl:L2:N05", "--alpha", "1"], "tiny-vl:L2:N05"),
        (["--unit", UNIT, "--alpha", "nan"], "alpha"),
        (["--unit", UNIT, "--alpha", "1", "--unit", UNIT, "--alpha", "2"], "twice"),
        (["--unit", UNIT, "--alpha", "1", "--start", "3", "--end", "3"], "end"),
        (["--unit", UNIT, "--alpha", "1", "--control", "random"], "--control-seed"),
        (["--unit", UNIT, "--alpha", "1", "--control-seed", "3"], "--control random"),
        (["--control", "random", "--control-seed", "3"], "--unit"),
        (["--start", "1"], "give a --unit"),
        (["--no-include-prompt"], "give a --unit"),
        (
            ["--unit", UNIT, "--alpha", "1", "--start", "3", "--no-include-prompt"],
            "no pass",
        ),
    ],
)
def test_bad_options_are_refused_before_the_model_loads(
    tiny_vl_calibration: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    options: list[str],
    problem: str,
) -> None:
    monkeypatch.setattr(load, "load_model", _no_loading)
    assert _steer(tmp_path / "out", *options) == 1
    assert problem in capsys.readouterr().err


def test_a_dose_whose_amount_is_not_finite_is_refused_before_generating(
    tiny_vl_calibration: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def no_generating(*args: object, **kwargs: object) -> None:
        raise AssertionError("nothing is generated for a dose that cannot be pushed")

    # A dose so large that alpha * r_L / length overflows to infinity.
    monkeypatch.setattr(dose, "amount", lambda *args: float("inf"))
    monkeypatch.setattr(generate_module, "generate", no_generating)
    assert _steer(tmp_path / "out", "--unit", UNIT, "--alpha", "1e300") == 1
    assert "smaller dose" in capsys.readouterr().err
    assert not (tmp_path / "out" / RUNS_FILE).exists()


def test_a_late_start_with_the_prompt_pushed_is_pointed_out(
    tiny_vl_calibration: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _steer(tmp_path / "out", "--unit", UNIT, "--alpha", "1", "--start", "2") == 0
    assert "add --no-include-prompt" in capsys.readouterr().out


def test_a_calibration_that_changes_while_it_is_read_is_refused(
    tiny_vl_calibration: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from pushlab.units import calibrate

    read = calibrate.load_calibration

    def read_while_rewritten(*args: object, **kwargs: object) -> object:
        calibration = read(*args, **kwargs)
        record = json.loads(tiny_vl_calibration.read_text()) | {"note": "rewritten"}
        tiny_vl_calibration.write_text(json.dumps(record))
        return calibration

    monkeypatch.setattr(calibrate, "load_calibration", read_while_rewritten)
    monkeypatch.setattr(load, "load_model", _no_loading)
    assert _steer(tmp_path / "out", "--unit", UNIT, "--alpha", "1") == 1
    assert "changed while it was read" in capsys.readouterr().err


def test_a_broken_prompt_set_is_refused_before_the_model_loads(
    tiny_vl_calibration: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(load, "load_model", _no_loading)
    missing = tmp_path / "missing.jsonl"
    assert _steer(tmp_path / "out", prompts=missing) == 1
    assert str(missing) in capsys.readouterr().err
