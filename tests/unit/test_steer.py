"""`pushlab steer`: pushed replies to a prompt set, saved as runs; on tiny-vl.

The tiny Qwen3.5 (tiny-vl) loads through pushlab's real loading code, with a
made-up calibration (the tiny_vl_calibration fixture).
"""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch

from pushlab.cli import main
from pushlab.engine.decoding import Decoding
from pushlab.engine.generate import generate
from pushlab.engine.push import Push
from pushlab.models import load
from pushlab.models.load import load_model
from pushlab.prompts import PROMPTS_DIR, read_prompt_set
from pushlab.store.environment import file_sha256
from pushlab.store.records import (
    MANIFEST_FILE,
    RUNS_FILE,
    check_manifest,
    make_run_id,
    read_runs,
)
from pushlab.units import controls
from pushlab.units.dose import amount
from pushlab.units.neuron import NeuronUnit

SMOKE = PROMPTS_DIR / "smoke.jsonl"
UNIT = "tiny-vl:L2:N100"
OPTIONS = [
    "--model",
    "--prompts",
    "--out",
    "--unit",
    "--alpha",
    "--start",
    "--end",
    "--include-prompt",
    "--no-include-prompt",
    "--control",
    "--control-seed",
    "--max-new-tokens",
    "--overwrite",
]


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU in float32 even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _steer(out: Path, *options: str, prompts: Path = SMOKE, tokens: int = 4) -> int:
    return main(
        ["steer", "--model", "tiny-vl", "--prompts", str(prompts), "--out", str(out)]
        + ["--max-new-tokens", str(tokens), *options]
    )


def _norms(calibration_file: Path) -> list[float]:
    return json.loads(calibration_file.read_text())["norms"]


def _no_loading(*args: object, **kwargs: object) -> None:
    raise AssertionError("the model must not load when steer refuses")


def test_steer_help_lists_the_options(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["steer", "--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    for option in OPTIONS:
        assert option in help_text


@pytest.mark.parametrize("command", ["steer", "amount", "report"])
def test_the_installed_command_answers_help(command: str) -> None:
    pushlab = Path(sys.executable).with_name("pushlab")
    result = subprocess.run(
        [str(pushlab), command, "--help"], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert "usage: pushlab " + command in result.stdout


def test_alpha_0_gives_the_baselines_replies(
    tiny_vl_calibration: Path, tmp_path: Path
) -> None:
    assert _steer(tmp_path / "zero", "--unit", UNIT, "--alpha", "0") == 0
    assert _steer(tmp_path / "none") == 0
    zero, none = read_runs(tmp_path / "zero"), read_runs(tmp_path / "none")
    assert [run.output_tokens for run in zero] == [run.output_tokens for run in none]
    assert [run.output_text for run in zero] == [run.output_text for run in none]
    assert all(run.schedule[0]["amount"] == 0.0 for run in zero)
    assert all(run.schedule == [] for run in none)
    # And each is the reply `pushlab baseline` gives its prompt alone.
    for run, prompt in zip(zero, read_prompt_set(SMOKE).prompts, strict=True):
        out = tmp_path / f"baseline-{run.prompt_id}"
        argv = ["baseline", "--model", "tiny-vl", "--out", str(out), "--prompt", prompt]
        assert main([*argv, "--max-new-tokens", "4"]) == 0
        (baseline,) = read_runs(out)
        assert baseline.output_tokens == run.output_tokens


def test_the_schedule_round_trips_exactly_through_runs_parquet(
    tiny_vl_calibration: Path, tmp_path: Path
) -> None:
    out = tmp_path / "pushed"
    pushes = ["--unit", UNIT, "--alpha", "1.5", "--unit", "tiny-vl:L1:N7"]
    window = ["--alpha", "-0.25", "--start", "1", "--end", "3", "--no-include-prompt"]
    assert _steer(out, *pushes, *window) == 0
    model = load_model("tiny-vl", torch.float32, device="cpu").model
    norms = _norms(tiny_vl_calibration)
    expected = [
        {
            "unit": unit,
            "alpha": alpha,
            "start": 1,
            "end": 3,
            "include_prompt": False,
            "amount": amount(NeuronUnit.parse(unit), alpha, model, norms),
            "control_for": None,
        }
        for unit, alpha in [(UNIT, 1.5), ("tiny-vl:L1:N7", -0.25)]
    ]
    runs = read_runs(out)
    assert [run.schedule for run in runs] == [expected] * 5
    stored = pq.read_table(out / RUNS_FILE).column("schedule").to_pylist()
    assert [json.loads(text) for text in stored] == [expected] * 5
    smoke = read_prompt_set(SMOKE)
    assert [run.prompt_id for run in runs] == list(smoke.prompt_ids)
    assert [run.run_id for run in runs] == [
        make_run_id("steer", n) for n in range(1, 6)
    ]
    assert {run.experiment for run in runs} == {"steer"}
    assert [run.batch_key for run in runs] == [f"b000001:{row}" for row in range(5)]
    assert {(run.seed, run.control_of, run.status) for run in runs} == {(0, "", "ok")}
    assert runs[0].decoding == {
        "strategy": "greedy",
        "max_new_tokens": 4,
        "thinking": False,
        "dtype": "float32",
    }


def test_the_window_and_include_prompt_reach_the_pushes(
    tiny_vl_calibration: Path, tmp_path: Path
) -> None:
    # The replies are the ones generate() gives with exactly these pushes,
    # and pushing the prompt too, or past the end, would give others.
    window = ["--start", "1", "--end", "3", "--no-include-prompt"]
    assert (
        _steer(tmp_path / "out", "--unit", UNIT, "--alpha", "3", *window, tokens=6) == 0
    )
    loaded = load_model("tiny-vl", torch.float32, device="cpu")
    prompts = read_prompt_set(SMOKE).prompts
    greedy = Decoding(strategy="greedy", max_new_tokens=6)
    norms = _norms(tiny_vl_calibration)

    def replies(push: Push) -> list[list[int]]:
        made = generate(loaded.model, loaded.tokenizer, prompts, [push], greedy, norms)
        return [list(reply.token_ids) for reply in made]

    push = Push(
        unit=NeuronUnit.parse(UNIT), alpha=3.0, start=1, end=3, include_prompt=False
    )
    expected = replies(push)
    assert [run.output_tokens for run in read_runs(tmp_path / "out")] == expected
    assert replies(replace(push, include_prompt=True)) != expected
    assert replies(replace(push, end=None)) != expected


def test_the_manifest_records_the_calibration_and_the_prompt_set(
    tiny_vl_calibration: Path, tmp_path: Path
) -> None:
    out = tmp_path / "pushed"
    assert _steer(out, "--unit", UNIT, "--alpha", "1") == 0
    manifest = json.loads((out / MANIFEST_FILE).read_text())
    check_manifest(manifest)
    assert manifest["calibration_file"] == str(tiny_vl_calibration)
    assert manifest["calibration_sha256"] == file_sha256(tiny_vl_calibration)
    smoke = read_prompt_set(SMOKE)
    assert manifest["prompt_set"] == "prompts/smoke.jsonl"
    assert manifest["prompt_set_sha256"] == smoke.sha256
    assert manifest["prompts"] == dict(
        zip(smoke.prompt_ids, smoke.prompts, strict=True)
    )
    assert manifest["schedule"] == read_runs(out)[0].schedule
    assert manifest["control"] is None
    assert manifest["run_count"] == 5
    assert manifest["runs_by_status"] == {"ok": 5}
    assert manifest["model_revision"] == load.downloaded_revision("tiny-vl")


def test_a_random_control_is_drawn_from_its_seed_in_the_same_layer(
    tiny_vl_calibration: Path, tmp_path: Path
) -> None:
    control = ["--control", "random", "--control-seed", "7"]
    assert _steer(tmp_path / "control", "--unit", UNIT, "--alpha", "1", *control) == 0
    runs = read_runs(tmp_path / "control")
    neuron = NeuronUnit.parse(runs[0].schedule[0]["unit"])
    assert neuron.layer == 2 and str(neuron) != UNIT
    model = load_model("tiny-vl", torch.float32, device="cpu").model
    for number, run in enumerate(runs, start=1):
        assert run.experiment == "steer-control"
        assert run.run_id == make_run_id("steer-control", number)
        assert run.control_of == make_run_id("steer", number)  # the run it mirrors
        assert run.seed == 7
        (segment,) = run.schedule
        assert segment["unit"] == str(neuron) and segment["control_for"] == UNIT
        assert segment["alpha"] == 1.0
        norms = _norms(tiny_vl_calibration)
        assert segment["amount"] == amount(neuron, 1.0, model, norms)
    manifest = json.loads((tmp_path / "control" / MANIFEST_FILE).read_text())
    assert manifest["control"] == {
        "kind": "random",
        "seed": 7,
        "neurons": {UNIT: str(neuron)},
    }
    # The same seed draws the same neuron again.
    assert _steer(tmp_path / "again", "--unit", UNIT, "--alpha", "1", *control) == 0
    assert read_runs(tmp_path / "again")[0].schedule == runs[0].schedule


def test_a_random_control_skips_the_known_units(
    tiny_vl_calibration: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every neuron of layer 2 but one is listed as known.
    known = tmp_path / "known.csv"
    rows = [
        f"tiny-vl:L2:N{n},+,test,https://x" for n in range(128) if n not in (77, 100)
    ]
    known.write_text("unit_id,sign,label,sources\n" + "\n".join(rows) + "\n")
    monkeypatch.setattr(controls, "KNOWN_UNITS_FILE", known)
    control = ["--control", "random", "--control-seed", "3"]
    assert _steer(tmp_path / "control", "--unit", UNIT, "--alpha", "1", *control) == 0
    assert read_runs(tmp_path / "control")[0].schedule[0]["unit"] == "tiny-vl:L2:N77"


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
