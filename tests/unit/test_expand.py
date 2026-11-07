"""expand: an experiment spec into the exact list of runs it means."""

import dataclasses
import doctest
from collections import Counter
from pathlib import Path

import pytest

from pushlab.models.registry import REGISTRY
from pushlab.runs import expand as expand_module
from pushlab.runs.expand import PLANNED, expand
from pushlab.runs.spec import spec_from_yaml
from pushlab.schedules.model import Segment
from pushlab.schedules.yaml_io import from_mapping
from pushlab.store.records import write_runs
from pushlab.units.controls import known_units
from pushlab.units.neuron import NeuronUnit

EVIL = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
WARM = NeuronUnit.parse("qwen3.5-4b:L13:N7956")


@pytest.fixture
def prompts(tmp_path: Path) -> Path:
    path = tmp_path / "two.jsonl"
    path.write_text(
        '{"prompt_id": "p-1", "version": 1, "prompt": "Say hi."}\n'
        '{"prompt_id": "p-2", "version": 1, "prompt": "Count to three."}\n',
        encoding="utf-8",
    )
    return path


def _spec(prompts: Path, conditions: str, **settings: str) -> str:
    lines = {
        "name": "pulse-lengths",
        "model": "qwen3.5-4b",
        "prompts": str(prompts),
        "decoding": "{strategy: sample, temperature: 0.7}",
        "seeds": "[0, 1]",
        "max_new_tokens": "64",
        "batch_size": "8",
        "max_gpu_hours": "1",
    } | settings
    return "".join(f"{key}: {value}\n" for key, value in lines.items()) + conditions


PULSES = """\
conditions:
  - label: pulse
    shape: pulse
    params: {unit: qwen3.5-4b:L14:N8949, alpha: 1.0, start: 0}
    grid: {length: [4, 16, 64]}
    controls: 2
    control_seed: 7
"""


def test_three_pulse_lengths_two_prompts_and_two_seeds(prompts: Path) -> None:
    expansion = expand(spec_from_yaml(_spec(prompts, PULSES)))
    own = [run for run in expansion.runs if run.control == 0]
    assert len(own) == 12  # 3 lengths x 2 prompts x 2 seeds
    assert len(expansion.runs) == 12 * 3  # and 2 controls of each
    assert expansion.counts() == {f"pulse length={n}": (4, 8) for n in (4, 16, 64)}
    ids = [run.record.run_id for run in expansion.runs]
    assert ids == [f"pulse-lengths-{i:06d}" for i in range(1, 37)]


def test_the_same_spec_always_gives_the_same_runs(
    prompts: Path, tmp_path: Path
) -> None:
    text = _spec(prompts, PULSES)
    first, second = expand(spec_from_yaml(text)), expand(spec_from_yaml(text))
    assert first == second
    assert first.spec_sha256 == spec_from_yaml(text).sha256
    assert len(first.prompts_sha256) == 64


def test_runs_go_by_prompt_then_seed_then_condition_then_control(prompts: Path) -> None:
    expansion = expand(spec_from_yaml(_spec(prompts, PULSES)))
    order = [
        (run.record.prompt_id, run.record.seed, run.condition, run.control)
        for run in expansion.runs
    ]
    assert order == [
        (prompt, seed, f"pulse length={n}", control)
        for prompt in ("p-1", "p-2")
        for seed in (0, 1)
        for n in (4, 16, 64)
        for control in (0, 1, 2)
    ]


def test_each_planned_run_is_a_run_record_with_no_reply_yet(prompts: Path) -> None:
    (first, *_) = expand(spec_from_yaml(_spec(prompts, PULSES))).runs
    record = first.record
    assert record.experiment == "pulse-lengths"
    assert record.model == f"qwen3.5-4b@{REGISTRY['qwen3.5-4b'].revision}"
    assert (record.prompt_id, record.seed, record.control_of) == ("p-1", 0, "")
    assert record.decoding == {
        "strategy": "sample",
        "max_new_tokens": 64,
        "thinking": False,
        "dtype": "bfloat16",
        "temperature": 0.7,
        "top_p": 1.0,
    }
    assert (record.output_text, record.output_tokens, record.status) == (
        "",
        [],
        PLANNED,
    )
    # A planned run cannot be saved as a result; once run, it can be.
    with pytest.raises(ValueError, match="status must be one of"):
        write_runs([record], Path("/nonexistent"))
    done = dataclasses.replace(record, output_tokens=[1, 2], status="ok")
    assert write_runs([done], Path(prompts).parent / "runs").name == "runs.parquet"


def test_the_schedule_is_stored_as_its_segments(prompts: Path) -> None:
    runs = expand(spec_from_yaml(_spec(prompts, PULSES))).runs
    assert runs[0].record.schedule == [
        {
            "unit": str(EVIL),
            "alpha": 1.0,
            "start": 0,
            "end": 4,
            "region": "reply",
            "control_for": None,
            "control_seed": None,
        }
    ]


def _segments(stored: list[dict]) -> list[Segment]:
    """The stored segments read back, without the control fields."""
    plain = [
        {k: v for k, v in s.items() if k not in ("control_for", "control_seed")}
        for s in stored
    ]
    return list(from_mapping({"name": "x", "segments": plain}).segments)


def test_controls_change_only_the_neurons(prompts: Path) -> None:
    text = """\
conditions:
  - label: handoff
    shape: handoff
    params: {a: qwen3.5-4b:L14:N8949, b: qwen3.5-4b:L13:N7956, alpha_a: 1.0,
             alpha_b: -1.0, switch_at: 16, overlap: 2}
    controls: 5
    control_seed: 3
"""
    expansion = expand(spec_from_yaml(_spec(prompts, text)))
    runs = {run.record.run_id: run for run in expansion.runs}
    known = known_units()
    neurons: dict[NeuronUnit, set[NeuronUnit]] = {EVIL: set(), WARM: set()}
    for run in expansion.runs:
        if run.control == 0:
            continue
        mirrored = runs[run.record.control_of]
        assert mirrored.control == 0 and mirrored.condition == run.condition
        assert (mirrored.record.prompt_id, mirrored.record.seed) == (
            run.record.prompt_id,
            run.record.seed,
        )
        own, control = (
            _segments(mirrored.record.schedule),
            _segments(run.record.schedule),
        )
        swap = {}
        for original, swapped in zip(own, control, strict=True):
            # The same timing and alphas exactly; only the neuron differs.
            assert dataclasses.replace(swapped, unit=original.unit) == original
            assert swapped.unit.layer == original.unit.layer
            assert swapped.unit not in known and swapped.unit not in (EVIL, WARM)
            assert swap.setdefault(original.unit, swapped.unit) == swapped.unit
        for stored in run.record.schedule:
            assert stored["control_seed"] == 3
            assert stored["control_for"] in (str(EVIL), str(WARM))
        for original, swapped in swap.items():
            neurons[original].add(swapped)
    # Five distinct controls of each unit, the same for every prompt and seed.
    assert {unit: len(drawn) for unit, drawn in neurons.items()} == {EVIL: 5, WARM: 5}


def test_controls_skip_whatever_known_csv_lists(
    prompts: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from pushlab.units import controls

    first = expand(spec_from_yaml(_spec(prompts, PULSES)))
    drawn = {
        NeuronUnit.parse(s["unit"])
        for run in first.runs
        if run.control
        for s in run.record.schedule
    }
    known = tmp_path / "known.csv"
    rows = "".join(
        f"{unit},+,test,https://example.org\n" for unit in sorted(drawn, key=str)
    )
    known.write_text("unit_id,sign,label,sources\n" + rows, encoding="utf-8")
    monkeypatch.setattr(controls, "KNOWN_UNITS_FILE", known)
    again = expand(spec_from_yaml(_spec(prompts, PULSES)))
    redrawn = {
        NeuronUnit.parse(s["unit"])
        for run in again.runs
        if run.control
        for s in run.record.schedule
    }
    assert not redrawn & drawn


def test_controls_skip_units_that_any_condition_pushes(prompts: Path) -> None:
    first = expand(spec_from_yaml(_spec(prompts, PULSES)))
    drawn = sorted(
        {
            NeuronUnit.parse(s["unit"])
            for run in first.runs
            if run.control
            for s in run.record.schedule
        },
        key=str,
    )
    # Two more conditions push the neurons the pulse's controls drew.
    others = "".join(
        f"  - {{label: other-{i}, segments: [{{unit: {unit}, alpha: 0.5}}]}}\n"
        for i, unit in enumerate(drawn)
    )
    again = expand(spec_from_yaml(_spec(prompts, PULSES + others)))
    redrawn = {
        NeuronUnit.parse(s["unit"])
        for run in again.runs
        if run.control
        for s in run.record.schedule
    }
    assert len(redrawn) == 2 and not redrawn & set(drawn)


def test_batches_hold_one_prompt_and_seed_each_and_at_most_batch_size_runs(
    prompts: Path,
) -> None:
    expansion = expand(spec_from_yaml(_spec(prompts, PULSES, batch_size="5")))
    batches: dict[str, list] = {}
    for run in expansion.runs:
        batch, row = run.record.batch_key.split(":")
        record = run.record
        batches.setdefault(batch, []).append((int(row), record.prompt_id, record.seed))
    for rows in batches.values():
        assert [row for row, *_ in rows] == list(range(len(rows))) and len(rows) <= 5
        assert len({(prompt, seed) for _, prompt, seed in rows}) == 1
    # In run order, so a batch's rows are consecutive runs.
    keys = [run.record.batch_key for run in expansion.runs]
    assert keys == sorted(keys, key=lambda key: (key[:7], int(key[8:])))
    assert sorted(batches) == [f"b{i:06d}" for i in range(1, len(batches) + 1)]
    assert len(batches) == 2 * 2 * 2  # 9 runs per prompt and seed: 5, then 4


def test_an_invalid_condition_stops_expansion_naming_it_and_the_rule(
    prompts: Path,
) -> None:
    text = (
        PULSES
        + """\
  - label: deep
    segments: [{unit: qwen3.5-4b:L40:N0, alpha: 1.0}]
"""
    )
    with pytest.raises(
        ValueError, match="(?s)condition 'deep': .*layer 40 is out of range"
    ):
        expand(spec_from_yaml(_spec(prompts, text)))
    over = (
        PULSES
        + """\
  - label: double
    segments:
      - {unit: qwen3.5-4b:L14:N8949, alpha: 1.0}
      - {unit: qwen3.5-4b:L14:N1, alpha: 1.0}
"""
    )
    with pytest.raises(
        ValueError, match="(?s)condition 'double': .*above the ceiling of 1.4"
    ):
        expand(spec_from_yaml(_spec(prompts, over)))


def test_a_condition_with_no_push_has_no_controls(prompts: Path) -> None:
    text = (
        "conditions:\n  - {label: none, segments: [], controls: 1, control_seed: 0}\n"
    )
    with pytest.raises(ValueError, match="condition 'none': it pushes no unit to draw"):
        expand(spec_from_yaml(_spec(prompts, text)))


def test_warnings_from_validate_are_kept(prompts: Path) -> None:
    text = (
        "conditions:\n  - label: late\n"
        "    segments: [{unit: qwen3.5-4b:L14:N8949, alpha: 1.0, start: 100}]\n"
    )
    expansion = expand(spec_from_yaml(_spec(prompts, text)))
    assert len(expansion.warnings) == 1
    assert expansion.warnings[0].startswith("condition 'late': segments[0]")
    assert "never pushes" in expansion.warnings[0]


def test_a_prompt_set_that_does_not_exist_is_refused(tmp_path: Path) -> None:
    with pytest.raises(
        ValueError, match=r"cannot read the prompt set .*missing\.jsonl"
    ):
        expand(spec_from_yaml(_spec(tmp_path / "missing.jsonl", PULSES)))


def test_every_run_id_is_unique_and_every_control_mirrors_a_run(prompts: Path) -> None:
    expansion = expand(spec_from_yaml(_spec(prompts, PULSES)))
    ids = Counter(run.record.run_id for run in expansion.runs)
    assert max(ids.values()) == 1
    mirrored = {run.record.control_of for run in expansion.runs if run.control}
    assert mirrored <= set(ids)


def test_doctests() -> None:
    result = doctest.testmod(expand_module)
    assert result.attempted > 0
    assert result.failed == 0
