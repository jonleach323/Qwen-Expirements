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
from pushlab.engine import generate as generate_module
from pushlab.engine.decoding import Decoding
from pushlab.engine.generate import generate
from pushlab.engine.push import Push
from pushlab.models import load
from pushlab.models.chat import stop_token_ids
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


def _engine_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
    """Record the prompts, pushes, decoding and norms each generate() call gets."""
    calls: list[tuple] = []
    real = generate_module.generate

    def recording(model, tokenizer, prompts, pushes, decoding, norms=None, **kw):
        calls.append((list(prompts), list(pushes), decoding, norms))
        return real(model, tokenizer, prompts, pushes, decoding, norms, **kw)

    monkeypatch.setattr(generate_module, "generate", recording)
    return calls


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
    loaded = load_model("tiny-vl", torch.float32, device="cpu")
    norms = _norms(tiny_vl_calibration)
    plan = [
        {
            "unit": unit,
            "alpha": alpha,
            "start": 1,
            "end": 3,
            "include_prompt": False,
            "amount": amount(NeuronUnit.parse(unit), alpha, loaded.model, norms),
            "control_for": None,
        }
        for unit, alpha in [(UNIT, 1.5), ("tiny-vl:L1:N7", -0.25)]
    ]
    # Each run also says how many of its reply tokens each push was on.
    pushes = [
        Push(
            unit=NeuronUnit.parse(segment["unit"]),
            alpha=segment["alpha"],
            start=1,
            end=3,
            include_prompt=False,
        )
        for segment in plan
    ]
    greedy = Decoding(strategy="greedy", max_new_tokens=4)
    made = generate(
        loaded.model,
        loaded.tokenizer,
        read_prompt_set(SMOKE).prompts,
        pushes,
        greedy,
        norms,
    )
    expected = [
        [
            segment | {"touched": touched}
            for segment, touched in zip(plan, reply.touched, strict=True)
        ]
        for reply in made
    ]
    assert all(reply.touched == (2, 2) for reply in made)  # tokens 1 and 2
    runs = read_runs(out)
    assert [run.schedule for run in runs] == expected
    stored = pq.read_table(out / RUNS_FILE).column("schedule").to_pylist()
    assert [json.loads(text) for text in stored] == expected
    smoke = read_prompt_set(SMOKE)
    assert [run.prompt_id for run in runs] == list(smoke.prompt_ids)
    assert [run.run_id for run in runs] == [
        make_run_id("steer", n) for n in range(1, 6)
    ]
    assert {run.experiment for run in runs} == {"steer"}
    assert [run.batch_key for run in runs] == [f"b000001:{row}" for row in range(5)]
    assert {(run.seed, run.control_of, run.status) for run in runs} == {(0, "", "ok")}
    assert {run.model for run in runs} == {f"tiny-vl@{loaded.revision}"}
    assert [run.output_text for run in runs] == [
        loaded.tokenizer.decode(run.output_tokens, skip_special_tokens=True)
        for run in runs
    ]
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


def test_the_engine_gets_exactly_the_pushes_and_sizes_the_options_name(
    tiny_vl_calibration: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _engine_calls(monkeypatch)
    pushes = ["--unit", UNIT, "--alpha", "1.5", "--unit", "tiny-vl:L1:N7"]
    assert _steer(tmp_path / "out", *pushes, "--alpha", "-0.25", "--end", "3") == 0
    ((prompts, given, decoding, norms),) = calls
    assert prompts == list(read_prompt_set(SMOKE).prompts)
    window = {"start": 0, "end": 3, "include_prompt": True}
    assert given == [
        Push(unit=NeuronUnit.parse(UNIT), alpha=1.5, **window),
        Push(unit=NeuronUnit.parse("tiny-vl:L1:N7"), alpha=-0.25, **window),
    ]
    assert decoding == Decoding(strategy="greedy", max_new_tokens=4)
    assert list(norms) == _norms(tiny_vl_calibration)


def test_controls_push_the_neurons_their_seed_draws_in_each_units_layer(
    tiny_vl_calibration: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _engine_calls(monkeypatch)
    units = [UNIT, "tiny-vl:L1:N7"]  # in layers 2 and 1
    pushes = ["--unit", units[0], "--alpha", "1", "--unit", units[1], "--alpha", "-1"]
    drawn = {}
    for seed in (7, 8):
        control = ["--control", "random", "--control-seed", str(seed)]
        assert _steer(tmp_path / f"seed{seed}", *pushes, *control) == 0
        drawn[seed] = controls.random_controls(
            [NeuronUnit.parse(unit) for unit in units], seed, controls.known_units()
        )
    assert drawn[7] != drawn[8]  # the seed decides
    for (_, given, _, _), seed in zip(calls, (7, 8), strict=True):
        assert [push.unit for push in given] == drawn[seed]
        assert [push.unit.layer for push in given] == [2, 1]
        assert [push.alpha for push in given] == [1.0, -1.0]
        run = read_runs(tmp_path / f"seed{seed}")[0]
        assert [segment["unit"] for segment in run.schedule] == list(
            map(str, drawn[seed])
        )
        assert [segment["control_for"] for segment in run.schedule] == units
        assert run.seed == seed


def test_the_defaults_push_the_prompt_and_every_reply_token_of_200(
    tiny_vl_calibration: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = _engine_calls(monkeypatch)
    out = tmp_path / "out"
    argv = ["steer", "--model", "tiny-vl", "--prompts", str(SMOKE), "--out", str(out)]
    assert main([*argv, "--unit", UNIT, "--alpha", "1"]) == 0
    ((_, (push,), decoding, _),) = calls
    assert (push.start, push.end, push.include_prompt) == (0, None, True)
    assert decoding.max_new_tokens == 200
    run = read_runs(out)[0]
    assert run.decoding["max_new_tokens"] == 200
    assert {
        key: run.schedule[0][key] for key in ("start", "end", "include_prompt")
    } == {
        "start": 0,
        "end": None,
        "include_prompt": True,
    }


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
    assert manifest["schedule"] == [
        {key: value for key, value in segment.items() if key != "touched"}
        for segment in read_runs(out)[0].schedule
    ]
    assert manifest["control"] is None
    loaded = load_model("tiny-vl", torch.float32, device="cpu")
    assert manifest["stop_token_ids"] == stop_token_ids(loaded.model, loaded.tokenizer)
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
