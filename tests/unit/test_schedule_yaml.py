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


@pytest.mark.parametrize("name", ["1e-3", "1e5", "0e0", "12e34", "010", "16", "1-2"])
def test_names_that_look_like_numbers_round_trip(name: str) -> None:
    schedule = Schedule(name, [Segment(A, 1.0, 1.0)])
    assert yaml_io.from_yaml(yaml_io.to_yaml(schedule)) == schedule


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


def test_segments_that_repeat_read_and_write_on_for_and_off_for() -> None:
    text = f"""
name: turns
segments:
  - {{unit: {A_ID}, alpha: 1.0, start: 10, on_for: 4, off_for: 4}}
  - {{unit: {B_ID}, alpha: -1.0, start: 14, on_for: 4, off_for: 4}}
"""
    schedule = yaml_io.from_yaml(text)
    assert schedule == alternate(A, B, 1.0, -1.0, 4, 10, name="turns")
    assert yaml_io.to_mapping(schedule)["segments"][1] == {
        "unit": B_ID,
        "alpha": -1.0,
        "start": 14,
        "end": None,
        "region": "reply",
        "on_for": 4,
        "off_for": 4,
    }


def test_the_shape_form_of_alternate_runs_to_the_end_without_an_end() -> None:
    text = (
        f"name: s\nshape: alternate\nparams: {{a: {A_ID}, b: {B_ID}, alpha_a: 1, "
        "alpha_b: -1, period: 1}"
    )
    assert yaml_io.from_yaml(text) == alternate(A, B, 1.0, -1.0, 1, name="s")


@pytest.mark.parametrize("ends", [(0.0, -0.0), (-0.0, 0.0), (-0.0, -0.0)])
def test_signed_zeros_round_trip_bit_for_bit(ends: tuple[float, float]) -> None:
    schedule = Schedule("x", [Segment(A, *ends, 0, 5)])
    back = yaml_io.from_yaml(yaml_io.to_yaml(schedule)).segments[0]
    assert (back.alpha_start.hex(), back.alpha_end.hex()) == tuple(
        alpha.hex() for alpha in ends
    )


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
    latin = tmp_path / "latin.yaml"
    latin.write_bytes("name: caf\xe9\nsegments: []\n".encode("latin-1"))
    with pytest.raises(ValueError, match=r"latin\.yaml: 'utf-8' codec can't decode"):
        yaml_io.read_schedule(latin)
    with pytest.raises(FileNotFoundError):
        yaml_io.read_schedule(tmp_path / "missing.yaml")


def test_doctests() -> None:
    result = doctest.testmod(yaml_io)
    assert result.attempted > 0
    assert result.failed == 0
