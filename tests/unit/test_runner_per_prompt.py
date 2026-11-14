"""The per_prompt calibration in `pushlab run` and `pushlab verify` (decision
0023): an experiment file with calibration: per_prompt runs on tiny-vl with
no calibration file, each run's doses taking its own prompt's residual
sizes, measured with no push once a session
(pushlab.units.calibration_mode.per_prompt_norms); experiment.json records
per_prompt where the fixed calibration records its file's SHA-256; and
verify checks a unit's amounts as one number per unit and prompt.

PUSHES pushes one neuron of layer 3, where the five smoke prompts' sizes
all differ, at +1 from the prompt pass, and at -0.5 from reply token 1
(step 2), beside an unpushed run: 15 runs, each prompt's three in one call.
"""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from pushlab.cli import main
from pushlab.engine import generate as generate_module
from pushlab.engine.decoding import Decoding
from pushlab.models import load
from pushlab.prompts import PROMPTS_DIR, read_prompt_set
from pushlab.runs.batch import record_schedule
from pushlab.runs.folder import read_identity, read_index
from pushlab.runs.shards import read_result_steps, read_results
from pushlab.store.manifest import check_experiment_manifest
from pushlab.store.steps import STEPS_SCHEMA
from pushlab.units import calibration_mode
from pushlab.units.calibration_mode import per_prompt_norms
from pushlab.units.dose import amount
from pushlab.units.neuron import NeuronUnit

NAME, UNIT = "tiny-pulses", "tiny-vl:L3:N7"
PUSHES = f"""\
conditions:
  - {{label: none, segments: []}}
  - {{label: up, segments: [{{unit: {UNIT}, alpha: 1.0}}]}}
  - {{label: down, segments: [{{unit: {UNIT}, alpha: -0.5, start: 1}}]}}
"""
NO_PUSH = "conditions:\n  - {label: none, segments: []}\n"
SMOKE = read_prompt_set(PROMPTS_DIR / "smoke.jsonl")
PER_PROMPT = {"calibration": "per_prompt", "max_new_tokens": "8", "batch_size": "3"}
MEASURED = "Doses take each prompt's own residual sizes (calibration per_prompt)"


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def measures(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The prompts measured, in order: each call of prompt_norms, which
    per_prompt_norms makes once per prompt it measures."""
    measured: list[str] = []
    real = calibration_mode.prompt_norms

    def counting(model, tokenizer, prompt):  # type: ignore[no-untyped-def]
        measured.append(prompt)
        return real(model, tokenizer, prompt)

    monkeypatch.setattr(calibration_mode, "prompt_norms", counting)
    return measured


def _run(spec: Path, runs: Path, *options: str) -> int:
    return main(["run", str(spec), "--runs-dir", str(runs), *options])


def _id(counter: int) -> str:
    return f"{NAME}-{counter:06d}"


def _out(capsys: pytest.CaptureFixture[str]) -> str:
    return capsys.readouterr().out


def _named(out: str) -> dict[str, str]:
    """The printed lines of the form 'name  value', by name."""
    pairs = [line.split("  ", 1) for line in out.splitlines() if "  " in line]
    return {name.strip(): value.strip() for name, value in pairs}


@pytest.fixture
def saved(
    tiny_spec: Callable[..., Path], tiny_vl_calibration: Path, tmp_path: Path
) -> tuple[Path, Path]:
    """PUSHES under per_prompt, run to the end with no calibration file:
    the experiment file and its runs dir."""
    tiny_vl_calibration.unlink()  # per_prompt reads none
    spec, runs = tiny_spec(PUSHES, **PER_PROMPT), tmp_path / "runs"
    assert _run(spec, runs) == 0
    return spec, runs


def test_a_per_prompt_run_needs_no_calibration_file_and_records_per_prompt(
    tiny_spec: Callable[..., Path],
    tiny_vl_calibration: Path,
    tmp_path: Path,
    measures: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    tiny_vl_calibration.unlink()
    spec, runs = tiny_spec(PUSHES, **PER_PROMPT), tmp_path / "runs"
    assert _run(spec, runs) == 0
    out = _out(capsys)
    assert MEASURED in out and "Done: all 15 runs are saved, 15 ok" in out
    assert measures == list(SMOKE.prompts)  # each once, before its batch
    assert read_identity(runs / NAME)["calibration_sha256"] == "per_prompt"
    manifest = json.loads((runs / NAME / "manifest.json").read_text())
    assert manifest["calibration_sha256"] == "per_prompt"
    check_experiment_manifest(manifest)
    for wrong in ("fixed", "per-prompt", "PER_PROMPT"):
        with pytest.raises(ValueError, match="per_prompt when each prompt's own"):
            check_experiment_manifest(manifest | {"calibration_sha256": wrong})
    # Nothing pushed: no calibration, nothing measured, as under fixed.
    measures.clear()
    other = tmp_path / "other"
    assert _run(tiny_spec(NO_PUSH, **PER_PROMPT), other) == 0
    assert MEASURED not in _out(capsys) and measures == []
    assert read_identity(other / NAME)["calibration_sha256"] is None


@pytest.mark.parametrize(
    "layout",
    [{}, {"batch_size": "15", "group_by_length": "false"}],
    ids=["a-call-a-prompt", "one-padded-call"],
)
def test_each_prompts_amounts_take_its_own_sizes_as_a_direct_generate_does(
    tiny_spec: Callable[..., Path],
    tiny_vl_calibration: Path,
    tmp_path: Path,
    layout: dict[str, str],
) -> None:
    tiny_vl_calibration.unlink()
    spec, runs = tiny_spec(PUSHES, **PER_PROMPT | layout), tmp_path / "runs"
    assert _run(spec, runs) == 0
    loaded = load.load_model("tiny-vl", torch.float32, device="cpu")
    model, tokenizer = loaded.model, loaded.tokenizer
    measured = per_prompt_norms(model, tokenizer, SMOKE.prompts).rows
    sizes = dict(zip(SMOKE.prompt_ids, measured, strict=True))
    unit = NeuronUnit.parse(UNIT)
    dose_one = {p: amount(unit, 1.0, model, list(row)) for p, row in sizes.items()}
    # Each prompt's dose-1 amount is its layer-3 size over one length: the
    # amounts differ by prompt as the sizes do, all five of them.
    ratios = [dose_one[p] / sizes[p][3] for p in SMOKE.prompt_ids]
    assert ratios == pytest.approx([ratios[0]] * 5, rel=1e-12)
    assert len({float(np.float16(one)) for one in dose_one.values()}) == 5
    records = {record.run_id: record for record in read_results(runs / NAME)}
    steps = read_result_steps(runs / NAME)
    labels = read_identity(runs / NAME)["conditions"]
    doses = {"none": None, "up": 1.0, "down": -0.5}
    for run_id, record in records.items():
        pushed = [step for step in steps[run_id] if step.active_units]
        if (dose := doses[labels[run_id]]) is None:
            assert pushed == []
        else:  # every pushed step: the dose times its prompt's dose-1 amount
            want = float(np.float16(dose * dose_one[record.prompt_id]))
            assert pushed and {s.active_amounts for s in pushed} == {(want,)}
    # Each call again, as one generate() call with per_prompt_norms: the
    # same replies, and the same amounts behind every token.
    greedy = Decoding(strategy="greedy", max_new_tokens=8)
    texts = dict(zip(SMOKE.prompt_ids, SMOKE.prompts, strict=True))
    for entry in read_index(runs / NAME):
        rows = sorted(
            (records[run_id] for run_id in entry["runs"]),
            key=lambda record: int(record.batch_key.rsplit(":", 1)[1]),
        )
        prompts = [texts[record.prompt_id] for record in rows]
        replies = generate_module.generate(
            model,
            tokenizer,
            prompts,
            [],
            greedy,
            per_prompt_norms(model, tokenizer, prompts),
            schedules=[record_schedule(record) for record in rows],
        )
        for record, reply in zip(rows, replies, strict=True):
            assert list(reply.token_ids) == record.output_tokens
            assert [s.active_amounts for s in steps[record.run_id]] == [
                tuple(float(np.float16(push.amount)) for push in pushes)
                for pushes in reply.amounts
            ]
    calls = [len(entry["runs"]) for entry in read_index(runs / NAME)]
    assert calls == ([3] * 5 if not layout else [15])


def test_each_prompt_text_is_measured_once_a_session(
    tiny_spec: Callable[..., Path], tmp_path: Path, measures: list[str]
) -> None:
    # p-1 and p-3 share a text, and batches of 2 put each prompt's three
    # runs in two calls: six calls, two texts, each measured once.
    three = tmp_path / "three.jsonl"
    three.write_text(
        '{"prompt_id": "p-1", "version": 1, "prompt": "Say hi."}\n'
        '{"prompt_id": "p-2", "version": 1, "prompt": "Count to three."}\n'
        '{"prompt_id": "p-3", "version": 1, "prompt": "Say hi."}\n',
        encoding="utf-8",
    )
    settings = PER_PROMPT | {"batch_size": "2", "prompts": str(three)}
    spec, runs = tiny_spec(PUSHES, **settings), tmp_path / "runs"
    assert _run(spec, runs, "--limit", "2") == 3  # the first call alone
    assert measures == ["Say hi."]
    # The next session measures again what its own calls need, each once.
    measures.clear()
    assert _run(spec, runs) == 0
    assert len(read_index(runs / NAME)) == 6
    assert measures == ["Say hi.", "Count to three."]


def _steps_file(runs: Path, run_id: str) -> Path:
    """The steps file of the shard whose latest mark counts run_id."""
    shards = [e["shard"] for e in read_index(runs / NAME) if run_id in e["runs"]]
    return runs / NAME / "steps" / shards[-1]


def _scale(runs: Path, run_id: str, factor: float) -> None:
    """Make every amount of run_id's steps factor times as large."""
    path = _steps_file(runs, run_id)
    rows = pq.read_table(path).to_pylist()
    for row in rows:
        if row["run_id"] == run_id:
            row["active_amounts"] = [factor * a for a in row["active_amounts"]]
    pq.write_table(pa.Table.from_pylist(rows, schema=STEPS_SCHEMA), path)


def test_verify_passes_a_per_prompt_folder_and_checks_each_prompt_alone(
    saved: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    spec, runs = saved
    _out(capsys)
    assert main(["verify", str(spec), "--runs-dir", str(runs)]) == 0
    out = _out(capsys)
    assert out.startswith(f"{NAME}, version 1: 15 runs and ")
    assert out.endswith(" checked; no problem found.\n")
    # smoke-02's down run (6) made 1% larger on every step: its steps agree
    # with each other, not with smoke-02's up run (5).
    records = {record.run_id: record for record in read_results(runs / NAME)}
    assert len(records[_id(6)].output_tokens) > 2  # so steps 2 on push
    _scale(runs, _id(6), 1.01)
    assert main(["verify", str(spec), "--runs-dir", str(runs)]) == 1
    problem, summary = _out(capsys).splitlines()
    assert problem.startswith(f"{UNIT}'s amount per dose on smoke-02 is ")
    assert f" in {_id(5)} but " in problem
    assert problem.endswith(f" in {_id(6)}, not one number")
    assert summary.endswith("; 1 problem found.")


def test_a_dry_run_and_status_say_which_calibration_a_run_uses(
    tiny_spec: Callable[..., Path],
    tiny_vl_calibration: Path,
    costs: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    tiny_vl_calibration.unlink()
    spec, runs = tiny_spec(PUSHES, **PER_PROMPT), tmp_path / "runs"
    measured = "per_prompt: each prompt's own residual sizes, measured with no push"
    assert _run(spec, runs, "--dry-run") == 0
    out = _out(capsys)
    assert _named(out)["calibration"] == "per_prompt"
    assert "note:" not in out  # no calibration file is missing
    assert _status(spec, runs, capsys) == measured  # before the folder exists
    assert _run(spec, runs, "--limit", "3") == 3
    assert _status(spec, runs, capsys) == measured  # its version's
    nothing = tiny_spec(NO_PUSH, file="none.yaml", **PER_PROMPT)
    assert _status(nothing, tmp_path / "other", capsys) == "none: nothing is pushed"
    assert _status(tiny_spec(file="fixed.yaml"), tmp_path / "fixed", capsys) == (
        "fixed: each layer's residual size, from the model's calibration file"
    )


def _status(spec: Path, runs: Path, capsys: pytest.CaptureFixture[str]) -> str:
    """What pushlab status says of the calibration."""
    _out(capsys)
    assert main(["status", str(spec), "--runs-dir", str(runs)]) == 0
    return _named(_out(capsys))["calibration"]
