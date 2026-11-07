"""Experiment specs: one YAML file that describes a whole experiment."""

import doctest
import re
from pathlib import Path

import pytest

from pushlab import PROJECT_ROOT
from pushlab.runs import conditions as conditions_module
from pushlab.runs import spec as spec_module
from pushlab.runs.spec import ExperimentSpec, read_spec, spec_from_yaml
from pushlab.schedules.model import Schedule
from pushlab.schedules.shapes import pulse
from pushlab.units.neuron import NeuronUnit

EVIL = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
SPEC = """\
name: pulse-lengths
model: qwen3.5-4b
prompts: prompts/smoke.jsonl
decoding: {strategy: sample, temperature: 0.7, top_p: 0.9}
seeds: [0, 1]
max_new_tokens: 64
batch_size: 4
max_gpu_hours: 0.5
conditions:
  - label: none
    segments: []
  - label: pulse
    shape: pulse
    params: {unit: qwen3.5-4b:L14:N8949, alpha: 1.0, start: 0}
    grid: {length: [4, 16, 64]}
    controls: 2
    control_seed: 7
"""


def _with(old: str, new: str) -> str:
    assert old in SPEC, old
    return SPEC.replace(old, new)


def test_a_spec_reads_every_setting() -> None:
    spec = spec_from_yaml(SPEC)
    assert (spec.name, spec.model, spec.prompts) == (
        "pulse-lengths",
        "qwen3.5-4b",
        "prompts/smoke.jsonl",
    )
    assert spec.decoding == {
        "strategy": "sample",
        "temperature": 0.7,
        "top_p": 0.9,
        "dtype": "bfloat16",
    }
    assert spec.seeds == (0, 1)
    assert (spec.max_new_tokens, spec.batch_size, spec.max_gpu_hours) == (64, 4, 0.5)
    assert [c.label for c in spec.conditions] == ["none", "pulse"]
    assert spec.prompts_path == PROJECT_ROOT / "prompts" / "smoke.jsonl"


def test_greedy_decoding_takes_one_seed_and_defaults() -> None:
    spec = spec_from_yaml(
        _with(
            "{strategy: sample, temperature: 0.7, top_p: 0.9}", "{strategy: greedy}"
        ).replace("[0, 1]", "[3]")
    )
    assert spec.decoding == {
        "strategy": "greedy",
        "temperature": None,
        "top_p": 1.0,
        "dtype": "bfloat16",
    }
    assert spec.seeds == (3,)


def test_a_grid_gives_one_condition_per_value_in_order() -> None:
    (none,), lengths = (
        spec_from_yaml(SPEC).conditions[0].points,
        spec_from_yaml(SPEC).conditions[1].points,
    )
    assert none == ("none", Schedule("none"))
    assert lengths == tuple(
        (f"pulse length={n}", pulse(EVIL, 1.0, 0, n, name="pulse")) for n in (4, 16, 64)
    )


def test_a_grid_of_two_params_varies_the_first_slowest() -> None:
    text = _with(
        "grid: {length: [4, 16, 64]}", "grid: {length: [4, 16], alpha: [0.5, -1.0]}"
    )
    text = text.replace("alpha: 1.0, ", "")
    labels = [label for label, _ in spec_from_yaml(text).conditions[1].points]
    assert labels == [
        "pulse length=4 alpha=0.5",
        "pulse length=4 alpha=-1.0",
        "pulse length=16 alpha=0.5",
        "pulse length=16 alpha=-1.0",
    ]


PULSE_GRID = (
    "shape: pulse\n    params: {unit: qwen3.5-4b:L14:N8949, alpha: 1.0, start: 0}\n"
    "    grid: {length: [4, 16, 64]}"
)


def test_grid_labels_write_each_value_as_yaml_does() -> None:
    text = _with(
        PULSE_GRID,
        "shape: handoff\n    params: {a: qwen3.5-4b:L14:N8949, b: "
        "qwen3.5-4b:L13:N7956, alpha_a: 1.0, alpha_b: -1.0}\n"
        "    grid: {switch_at: [8], include_prompt: [true, false]}",
    )
    points = spec_from_yaml(text).conditions[1].points
    assert [label for label, _ in points] == [
        "pulse switch_at=8 include_prompt=true",
        "pulse switch_at=8 include_prompt=false",
    ]
    assert [len(schedule.segments) for _, schedule in points] == [3, 2]
    text = _with(
        PULSE_GRID,
        "shape: alternate\n    params: {a: qwen3.5-4b:L14:N8949, b: "
        "qwen3.5-4b:L13:N7956, alpha_a: 1.0, alpha_b: -1.0, period: 4}\n"
        "    grid: {end: [null, 32]}",
    )
    points = spec_from_yaml(text).conditions[1].points
    assert [label for label, _ in points] == ["pulse end=null", "pulse end=32"]


def test_grid_doses_get_one_label_written_whole_or_not() -> None:
    whole = _with(PULSE_GRID, PULSE_GRID.replace("[4, 16, 64]", "[4]"))
    whole = whole.replace("alpha: 1.0, start: 0}", "start: 0}").replace(
        "{length: [4]}", "{length: [4], alpha: [1, -1]}"
    )
    written = spec_from_yaml(whole)
    assert [label for label, _ in written.conditions[1].points] == [
        "pulse length=4 alpha=1.0",
        "pulse length=4 alpha=-1.0",
    ]
    decimal = spec_from_yaml(whole.replace("[1, -1]", "[1.0, -1.0]"))
    assert decimal == written and decimal.sha256 == written.sha256


def test_controls_default_to_none() -> None:
    none, pulsed = spec_from_yaml(SPEC).conditions
    assert (none.controls, none.control_seed) == (0, None)
    assert (pulsed.controls, pulsed.control_seed) == (2, 7)


def test_read_spec_reads_a_file_and_names_it_in_its_errors(tmp_path: Path) -> None:
    path = tmp_path / "pulse.yaml"
    path.write_text(SPEC, encoding="utf-8")
    assert read_spec(path) == spec_from_yaml(SPEC)
    path.write_text(SPEC.replace("model: qwen3.5-4b", "model: qwen9"), encoding="utf-8")
    with pytest.raises(ValueError, match=r"pulse\.yaml: model 'qwen9' is not in the"):
        read_spec(path)


def test_the_hash_is_the_same_for_the_same_spec_however_it_is_written() -> None:
    same = spec_from_yaml(SPEC)
    assert same.sha256 == spec_from_yaml(SPEC).sha256
    assert re.fullmatch(r"[0-9a-f]{64}", same.sha256)
    # Comments, spacing and the order of keys are not settings.
    reordered = "# A comment.\n" + _with(
        "{strategy: sample, temperature: 0.7, top_p: 0.9}",
        "{top_p: 0.9,   strategy: sample, temperature: 0.7}",
    )
    assert spec_from_yaml(reordered).sha256 == same.sha256
    # Nor are 1 against 1.0, or a shape against its segments written out.
    assert spec_from_yaml(_with("alpha: 1.0", "alpha: 1")).sha256 == same.sha256
    shape = "shape: pulse\n    params: {unit: qwen3.5-4b:L14:N8949, alpha: 1.0, "
    shape += "start: 0, length: 16}"
    segments = "segments: [{unit: qwen3.5-4b:L14:N8949, alpha: 1.0, start: 0, end: 16}]"
    one = _with(PULSE_GRID, shape)
    assert (
        spec_from_yaml(one).sha256
        == spec_from_yaml(one.replace(shape, segments)).sha256
    )


def test_a_grid_s_key_order_is_a_setting() -> None:
    grid = "{length: [4, 16], alpha: [0.5, 1.0]}"
    text = _with("{length: [4, 16, 64]}", grid).replace("alpha: 1.0, ", "")
    swapped = spec_from_yaml(text.replace(grid, "{alpha: [0.5, 1.0], length: [4, 16]}"))
    # The first param changes slowest, so the order sets the points' order.
    assert [label for label, _ in swapped.conditions[1].points][:2] == [
        "pulse alpha=0.5 length=4",
        "pulse alpha=0.5 length=16",
    ]
    assert swapped.sha256 != spec_from_yaml(text).sha256


CHANGES = {
    "name": ("name: pulse-lengths", "name: pulse-lengths-2"),
    "model": ("model: qwen3.5-4b", "model: qwen3-8b"),
    "prompts": ("prompts/smoke.jsonl", "prompts/calibration.jsonl"),
    "strategy": (
        "{strategy: sample, temperature: 0.7, top_p: 0.9}",
        "{strategy: greedy}",
    ),
    "temperature": ("temperature: 0.7", "temperature: 0.8"),
    "top_p": ("top_p: 0.9", "top_p: 0.95"),
    "dtype": ("top_p: 0.9", "top_p: 0.9, dtype: float32"),
    "seeds": ("[0, 1]", "[0, 2]"),
    "seed-order": ("[0, 1]", "[1, 0]"),
    "max_new_tokens": ("max_new_tokens: 64", "max_new_tokens: 65"),
    "batch_size": ("batch_size: 4", "batch_size: 5"),
    "max_gpu_hours": ("max_gpu_hours: 0.5", "max_gpu_hours: 0.6"),
    "label": ("- label: pulse", "- label: pulses"),
    "condition-order": ("  - label: none\n    segments: []\n", ""),
    "alpha": ("alpha: 1.0", "alpha: 0.9"),
    "unit": ("qwen3.5-4b:L14:N8949", "qwen3.5-4b:L14:N8950"),
    "grid": ("[4, 16, 64]", "[4, 16, 32]"),
    "controls": ("controls: 2", "controls: 3"),
    "control_seed": ("control_seed: 7", "control_seed: 8"),
    "ceiling": ("    controls: 2", "    allow_over_ceiling: true\n    controls: 2"),
}


@pytest.mark.parametrize("old, new", CHANGES.values(), ids=CHANGES.keys())
def test_the_hash_changes_whenever_a_setting_changes(old: str, new: str) -> None:
    text = _with(old, new)
    if old == "{strategy: sample, temperature: 0.7, top_p: 0.9}":
        text = text.replace("[0, 1]", "[0]")
    if old == "  - label: none\n    segments: []\n":  # the conditions swapped round
        text += "  - label: none\n    segments: []\n"
    assert spec_from_yaml(text).sha256 != spec_from_yaml(SPEC).sha256


def test_the_example_file_s_hash_is_pinned() -> None:
    # Update this when the example file changes. If the file did not change,
    # the hash itself did, and no longer matches the hashes manifests record.
    spec = read_spec(PROJECT_ROOT / "experiments" / "p0" / "s2_pulse_lengths.yaml")
    assert spec.sha256 == (
        "d43270fb4626f4eeaccad773df717aa611f69361f66a0ae9944ac6a6ec6edefa"
    )


def test_the_readme_example_is_the_example_file() -> None:
    readme = (PROJECT_ROOT / "experiments" / "README.md").read_text(encoding="utf-8")
    (block,) = re.findall(r"```yaml\n(.*?)```", readme, flags=re.DOTALL)
    example = PROJECT_ROOT / "experiments" / "p0" / "s2_pulse_lengths.yaml"
    assert block == example.read_text(encoding="utf-8")
    assert isinstance(read_spec(example), ExperimentSpec)


@pytest.mark.parametrize("module", [spec_module, conditions_module])
def test_doctests(module: object) -> None:
    result = doctest.testmod(module)
    assert result.attempted > 0
    assert result.failed == 0
