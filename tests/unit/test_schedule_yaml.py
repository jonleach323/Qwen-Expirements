"""Schedules as YAML: a shape name with params, or the segments written out."""

import doctest
from pathlib import Path

import pytest

from pushlab.schedules import yaml_io
from pushlab.schedules.model import Schedule, Segment
from pushlab.schedules.shapes import (
    alternate,
    constant,
    handoff,
    onset,
    pulse,
    ramp,
)
from pushlab.units.neuron import NeuronUnit

A = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
B = NeuronUnit.parse("qwen3.5-4b:L13:N7956")
A_ID, B_ID = str(A), str(B)

EVERY_SHAPE = {
    "constant": constant(A, 1.0, name="constant-hostile"),
    "constant-reply-only": constant(B, -1.0, include_prompt=False),
    "pulse": pulse(A, 1.0, 0, 16),
    "onset": onset(A, 0.5, 50),
    "handoff": handoff(A, B, 1.0, -1.0, 40),
    "handoff-overlap": handoff(A, B, 1.0, -1.0, 40, overlap=2),
    "handoff-gap": handoff(A, B, 1.0, -1.0, 40, gap=3),
    "handoff-reply-only": handoff(B, A, -1.0, 1.0, 40, include_prompt=False),
    "alternate": alternate(A, B, 1.0, -1.0, 4, start=10, end=26),
    "alternate-prompt": alternate(A, B, 1.0, -1.0, 1, end=7, include_prompt=True),
    "ramp": ramp(A, 0.0, 1.0, 10, 30),
    "ramp-down": ramp(B, -0.1, -1.3, 0, 7),
}


@pytest.mark.parametrize("schedule", EVERY_SHAPE.values(), ids=EVERY_SHAPE.keys())
def test_every_shape_round_trips_exactly(schedule: Schedule) -> None:
    text = yaml_io.to_yaml(schedule)
    back = yaml_io.from_yaml(text)
    assert back == schedule
    for read, made in zip(back.segments, schedule.segments, strict=True):
        assert read.alpha_start.hex() == made.alpha_start.hex()
        assert read.alpha_end.hex() == made.alpha_end.hex()
    assert yaml_io.to_yaml(back) == text


@pytest.mark.parametrize(
    "alpha", [0.1, 1 / 3, -2 / 3, 1e-20, -0.0, 5e-324, 1.7976931348623157e308]
)
def test_awkward_doses_round_trip_bit_for_bit(alpha: float) -> None:
    schedule = Schedule("x", [Segment(A, alpha, 0.5, 3, 9), Segment(B, alpha, alpha)])
    back = yaml_io.from_yaml(yaml_io.to_yaml(schedule))
    assert [s.alpha_start.hex() for s in back.segments] == [alpha.hex()] * 2
    assert back == schedule


def test_allow_over_ceiling_round_trips() -> None:
    schedule = Schedule("collapse", constant(A, 2.0).segments, allow_over_ceiling=True)
    assert yaml_io.from_yaml(yaml_io.to_yaml(schedule)) == schedule


def test_the_written_form_is_explicit() -> None:
    assert yaml_io.to_mapping(handoff(A, B, 1.0, -1.0, 40, name="h")) == {
        "name": "h",
        "allow_over_ceiling": False,
        "segments": [
            {"unit": A_ID, "alpha": 1.0, "start": 0, "end": None, "region": "prompt"},
            {"unit": A_ID, "alpha": 1.0, "start": 0, "end": 40, "region": "reply"},
            {"unit": B_ID, "alpha": -1.0, "start": 40, "end": None, "region": "reply"},
        ],
    }
    assert yaml_io.to_mapping(ramp(A, 0.0, 1.0, 10, 30))["segments"] == [
        {
            "unit": A_ID,
            "alpha_start": 0.0,
            "alpha_end": 1.0,
            "start": 10,
            "end": 30,
            "region": "reply",
        }
    ]


def test_to_yaml_writes_one_key_per_line() -> None:
    assert yaml_io.to_yaml(pulse(A, 1.0, 0, 16, name="pulse16")) == (
        "name: pulse16\n"
        "allow_over_ceiling: false\n"
        "segments:\n"
        "- unit: qwen3.5-4b:L14:N8949\n"
        "  alpha: 1.0\n"
        "  start: 0\n"
        "  end: 16\n"
        "  region: reply\n"
    )


SHAPE_FORMS = [
    (
        "shape: constant\nparams: {unit: U, alpha: 1.0}",
        constant(A, 1.0, name="s"),
    ),
    (
        "shape: constant\nparams: {unit: U, alpha: -1, include_prompt: false}",
        constant(A, -1.0, include_prompt=False, name="s"),
    ),
    (
        "shape: pulse\nparams: {unit: U, alpha: 1.0, start: 0, length: 16}",
        pulse(A, 1.0, 0, 16, name="s"),
    ),
    (
        "shape: onset\nparams: {unit: U, alpha: 1.0, start: 50}",
        onset(A, 1.0, 50, name="s"),
    ),
    (
        "shape: handoff\n"
        "params: {a: U, b: V, alpha_a: 1.0, alpha_b: -1.0, switch_at: 40, overlap: 2}",
        handoff(A, B, 1.0, -1.0, 40, overlap=2, name="s"),
    ),
    (
        "shape: handoff\n"
        "params: {a: U, b: V, alpha_a: 1.0, alpha_b: -1.0, switch_at: 40, gap: 3,\n"
        "         include_prompt: false}",
        handoff(A, B, 1.0, -1.0, 40, gap=3, include_prompt=False, name="s"),
    ),
    (
        "shape: alternate\n"
        "params: {a: U, b: V, alpha_a: 1.0, alpha_b: -1.0, period: 4, start: 10,\n"
        "         end: 26}",
        alternate(A, B, 1.0, -1.0, 4, 10, 26, name="s"),
    ),
    (
        "shape: ramp\n"
        "params: {unit: U, alpha_from: 0, alpha_to: 1, start: 10, end: 30}",
        ramp(A, 0.0, 1.0, 10, 30, name="s"),
    ),
]


@pytest.mark.parametrize("form, built", SHAPE_FORMS)
def test_a_shape_name_with_params_reads_as_the_shape_builds_it(
    form: str, built: Schedule
) -> None:
    text = "name: s\n" + form.replace("U", A_ID).replace("V", B_ID)
    schedule = yaml_io.from_yaml(text)
    assert schedule == built
    assert yaml_io.from_yaml(yaml_io.to_yaml(schedule)) == built


def test_a_shape_can_allow_over_ceiling() -> None:
    text = (
        f"name: s\nallow_over_ceiling: true\nshape: constant\n"
        f"params: {{unit: {A_ID}, alpha: 2.0}}"
    )
    assert yaml_io.from_yaml(text) == Schedule(
        "s", constant(A, 2.0).segments, allow_over_ceiling=True
    )


def test_segments_written_by_hand_take_alpha_and_defaults() -> None:
    text = f"""
name: handoff-hostile-to-warm
segments:
  - unit: {A_ID}
    alpha: 1.0
    region: prompt
  - unit: {A_ID}
    alpha: 1
    end: 40
  - unit: {B_ID}
    alpha_start: -1.0
    alpha_end: -1.0
    start: 40
"""
    assert yaml_io.from_yaml(text) == handoff(
        A, B, 1.0, -1.0, 40, name="handoff-hostile-to-warm"
    )


def test_an_empty_list_of_segments_is_no_push() -> None:
    assert yaml_io.from_yaml("name: none\nsegments: []") == Schedule("none")


def test_from_mapping_reads_a_schedule_inside_another_file() -> None:
    mapping = {"name": "p", "segments": [{"unit": A_ID, "alpha": 1.0, "end": 16}]}
    assert yaml_io.from_mapping(mapping) == pulse(A, 1.0, 0, 16, name="p")
    assert yaml_io.from_mapping(yaml_io.to_mapping(pulse(A, 1.0, 0, 16))) == pulse(
        A, 1.0, 0, 16
    )


SEGMENT = f"unit: {A_ID}, alpha: 1.0"
REFUSALS = [
    ("", "must be a mapping"),
    (f"- {{{SEGMENT}}}", "must be a mapping"),
    ("name: [", "is not valid YAML"),
    ("name: x", "give segments, or a shape with params"),
    ("segments: []", "needs a name"),
    (
        "name: x\nsegments: []\nshape: pulse\nparams: {}",
        "segments or a shape, not both",
    ),
    ("name: x\nsegments: []\ncolour: red", "unknown key 'colour'"),
    ("name: x\nsegments: []\n1: 2", "unknown key 1"),
    ("name: x\nname: y\nsegments: []", "duplicate key 'name'"),
    ("name: Hostile\nsegments: []", "name must be lowercase"),
    ("name: x\nsegments: []\nallow_over_ceiling: 1", "must be True or False"),
    (f"name: x\nsegments: {{{SEGMENT}}}", "segments must be a list"),
    ("name: x\nsegments: [3]", r"segments\[0\] must be a mapping"),
    (f"name: x\nsegments: [{{{SEGMENT}, colour: red}}]", "unknown key 'colour'"),
    (f"name: x\nsegments: [{{{SEGMENT}, alpha: 2}}]", "duplicate key 'alpha'"),
    ("name: x\nsegments: [{alpha: 1.0}]", r"segments\[0\] needs a unit"),
    (
        "name: x\nsegments: [{unit: qwen3.5-4B:L14:N8949, alpha: 1.0}]",
        r"segments\[0\]: unknown unit 'qwen3.5-4B:L14:N8949'",
    ),
    (
        "name: x\nsegments: [{unit: qwen3.5-4b:L14, alpha: 1.0}]",
        "unknown unit 'qwen3.5-4b:L14'",
    ),
    (f"name: x\nsegments: [{{unit: {A_ID}}}]", "needs alpha, or alpha_start and"),
    (f"name: x\nsegments: [{{{SEGMENT}, alpha_end: 0.0}}]", "alpha or alpha_start"),
    (f"name: x\nsegments: [{{unit: {A_ID}, alpha_start: 0.0}}]", "together"),
    (
        f"name: x\nsegments: [{{unit: {A_ID}, alpha: 1e-1}}]",
        "alpha is the text '1e-1', not a number",
    ),
    (
        f"name: x\nsegments: [{{unit: {A_ID}, alpha: '1.5'}}]",
        "alpha is the text '1.5', not a number",
    ),
    (f"name: x\nsegments: [{{{SEGMENT}, start: 1.5}}]", "start must be a reply"),
    (f"name: x\nsegments: [{{{SEGMENT}, region: thinking}}]", "region must be"),
    ("name: x\nshape: wave\nparams: {}", "unknown shape 'wave'"),
    ("name: x\nshape: pulse", "shape 'pulse' needs params"),
    ("name: x\nshape: pulse\nparams: [1]", "params must be a mapping"),
    ("name: x\nparams: {}", "give segments, or a shape with params"),
    (
        f"name: x\nshape: pulse\nparams: {{unit: {A_ID}, alpha: 1, start: 0, "
        "length: 4, colour: red}",
        "unknown param 'colour'",
    ),
    (
        f"name: x\nshape: pulse\nparams: {{unit: {A_ID}, alpha: 1, name: y}}",
        "unknown param 'name'",
    ),
    (
        f"name: x\nshape: pulse\nparams: {{unit: {A_ID}, alpha: 1}}",
        "missing params 'start', 'length'",
    ),
    (
        "name: x\nshape: onset\nparams: {unit: L14, alpha: 1, start: 3}",
        "shape onset: unknown unit 'L14'",
    ),
    (
        "name: x\nshape: onset\nparams: {unit: 14, alpha: 1, start: 3}",
        "shape onset: unknown unit 14",
    ),
    (
        f"name: x\nshape: onset\nparams: {{unit: {A_ID}, alpha: 1e-1, start: 3}}",
        "alpha is the text '1e-1', not a number",
    ),
    (
        f"name: x\nshape: alternate\nparams: {{a: {A_ID}, b: {B_ID}, alpha_a: 1, "
        "alpha_b: -1, period: 4}",
        "shape alternate: alternate needs an end",
    ),
]


@pytest.mark.parametrize("text, problem", REFUSALS)
def test_refusals_name_the_problem(text: str, problem: str) -> None:
    with pytest.raises(ValueError, match=problem):
        yaml_io.from_yaml(text)


@pytest.mark.parametrize("data", [None, [], "name: x", 3])
def test_from_mapping_refuses_anything_but_a_mapping(data: object) -> None:
    with pytest.raises(ValueError, match="a schedule must be a mapping"):
        yaml_io.from_mapping(data)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [None, "name: x", constant(A, 1.0).segments])
def test_writing_refuses_anything_but_a_schedule(value: object) -> None:
    for write in (yaml_io.to_mapping, yaml_io.to_yaml):
        with pytest.raises(ValueError, match="not a Schedule"):
            write(value)  # type: ignore[arg-type]


def test_read_schedule_reads_a_file(tmp_path: Path) -> None:
    path = tmp_path / "handoff.yaml"
    path.write_text(yaml_io.to_yaml(handoff(A, B, 1.0, -1.0, 40)), encoding="utf-8")
    assert yaml_io.read_schedule(path) == handoff(A, B, 1.0, -1.0, 40)
    assert yaml_io.read_schedule(str(path)) == handoff(A, B, 1.0, -1.0, 40)


def test_read_schedule_names_the_file_it_refuses(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("name: x\nshape: wave\nparams: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"bad\.yaml: unknown shape 'wave'"):
        yaml_io.read_schedule(path)
    with pytest.raises(FileNotFoundError):
        yaml_io.read_schedule(tmp_path / "missing.yaml")


def test_doctests() -> None:
    result = doctest.testmod(yaml_io)
    assert result.attempted > 0
    assert result.failed == 0
