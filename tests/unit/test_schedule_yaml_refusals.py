"""Schedules as YAML: what is refused, and how strictly the text is read
(pushlab.schedules.yaml_text)."""

import doctest

import pytest

from pushlab.schedules import yaml_io, yaml_text
from pushlab.schedules.model import Schedule, Segment
from pushlab.schedules.shapes import constant, pulse
from pushlab.units.neuron import NeuronUnit

A = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
B = NeuronUnit.parse("qwen3.5-4b:L13:N7956")
A_ID, B_ID = str(A), str(B)


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
        f"name: x\nsegments: [{{unit: {A_ID}, alpha: '1.5'}}]",
        r"segments\[0\]: alpha is the text '1.5', not a number",
    ),
    (f"name: x\nsegments: [{{{SEGMENT}, end: '16'}}]", "end is the text '16'"),
    (f"name: x\nsegments: [{{{SEGMENT}, start: 010}}]", "start is the text '010'"),
    (f"name: x\nsegments: [{{{SEGMENT}, on_for: '4'}}]", "on_for is the text '4'"),
    (
        f"name: x\nsegments: [{{{SEGMENT}, end: 1:20}}]",
        r"segments\[0\]: end must be None or a reply token from 0, not '1:20'",
    ),
    (
        f"name: x\nsegments: [{{{SEGMENT}, start: 0x10}}]",
        "start must be a reply token from 0, not '0x10'",
    ),
    (
        f"name: x\nsegments: [{{{SEGMENT}, start: 1.5}}]",
        r"segments\[0\]: start must be a reply",
    ),
    (
        f"name: x\nsegments: [{{{SEGMENT}, region: thinking}}]",
        r"segments\[0\]: region must be",
    ),
    (
        f"name: x\nsegments: [{{{SEGMENT}, on_for: 4}}]",
        r"segments\[0\]: on_for and off_for must both be None",
    ),
    (
        f"name: x\nsegments: [{{{SEGMENT}, start: 2026-13-45}}]",
        r"segments\[0\]: start must be a reply token from 0, not '2026-13-45'",
    ),
    ("name: 16\nsegments: []", "name must be text, but YAML read 16"),
    (
        f"name: x\nsegments: [{{unit: {A_ID}, alpha: .nan}}]",
        r"segments\[0\]: alpha must be a finite number, not nan",
    ),
    (
        f"name: x\nsegments: [{{{SEGMENT}, start: !!int 010}}]",
        "not a whole number as YAML 1.2",
    ),
    (
        f"name: x\nsegments: [{{unit: {A_ID}, alpha: !!float 1:20}}]",
        "not a number as YAML 1.2",
    ),
    (
        "name: x\nsegments: []\nallow_over_ceiling: !!bool yes",
        "not true or false as YAML 1.2",
    ),
    ("name: !!map [1, 2]\nsegments: []", "is not valid YAML"),
    ("name: x\nsegments: " + "[" * 2000 + "]" * 2000, "is not valid YAML"),
    ("name: x\nsegments: []\nparams: {}", "segments or a shape, not both"),
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
        f"name: x\nshape: onset\nparams: {{unit: {A_ID}, alpha: '0.1', start: 3}}",
        "shape onset: alpha is the text '0.1', not a number",
    ),
    (
        f"name: x\nshape: handoff\nparams: {{a: {A_ID}, b: {B_ID}, alpha_a: 1, "
        "alpha_b: -1, switch_at: 010}",
        "shape handoff: switch_at is the text '010'",
    ),
    (
        f"name: x\nshape: constant\nparams: {{unit: {A_ID}, alpha: 1, "
        "include_prompt: yes}",
        "shape constant: include_prompt must be True or False, not 'yes'",
    ),
    (
        f"name: x\nshape: constant\nparams: {{unit: {A_ID}, alpha: 1, "
        "include_prompt: '1'}",
        "shape constant: include_prompt must be True or False, not '1'",
    ),
    (
        f"name: x\nshape: handoff\nparams: {{a: {A_ID}, b: {B_ID}, alpha_a: 1, "
        "alpha_b: .inf, switch_at: 40}",
        "shape handoff: alpha_b must be a finite number, not inf",
    ),
    (
        f"name: x\nshape: pulse\nparams: {{unit: {A_ID}, alpha: 1, start: null, "
        "length: 16}",
        "shape pulse: start must be a whole number from 0, not None",
    ),
    (
        f"name: x\nshape: alternate\nparams: {{a: {A_ID}, b: {B_ID}, alpha_a: 1, "
        "alpha_b: -1, period: 4, end: 3}",
        "shape alternate: B would never be pushed: the alternation stops at reply "
        "token 3, and B's first turn starts at 4",
    ),
]


@pytest.mark.parametrize("text, problem", REFUSALS)
def test_refusals_name_the_problem(text: str, problem: str) -> None:
    with pytest.raises(ValueError, match=problem):
        yaml_io.from_yaml(text)


def test_numbers_and_true_false_are_read_as_yaml_1_2_reads_them() -> None:
    assert yaml_text.load_yaml(
        "a: 010\nb: 1:20\nc: 0x10\nd: 1e-1\ne: 1.0e5\nf: .5\ng: -.inf\n"
        "h: yes\ni: off\nj: true\nk: False\nl: null\nm: 16\nn: -3\no: 1_000"
    ) == {
        "a": "010",
        "b": "1:20",
        "c": "0x10",
        "d": 0.1,
        "e": 100000.0,
        "f": 0.5,
        "g": float("-inf"),
        "h": "yes",
        "i": "off",
        "j": True,
        "k": False,
        "l": None,
        "m": 16,
        "n": -3,
        "o": "1_000",
    }


def test_explicit_tags_follow_yaml_1_2_too() -> None:
    assert yaml_text.load_yaml("a: !!int 16\nb: !!float 1\nc: !!bool true") == {
        "a": 16,
        "b": 1.0,
        "c": True,
    }


def test_dates_are_text() -> None:
    assert yaml_text.load_yaml("a: 2026-11-07") == {"a": "2026-11-07"}
    assert yaml_io.from_yaml("name: 2026-11-07\nsegments: []").name == "2026-11-07"


def test_a_dose_with_an_exponent_is_a_number() -> None:
    text = f"name: x\nsegments: [{{unit: {A_ID}, alpha: 1e-1, end: 4}}]"
    assert yaml_io.from_yaml(text) == pulse(A, 0.1, 0, 4, name="x")


def test_names_that_yaml_1_1_reads_as_true_or_false_are_names() -> None:
    assert yaml_io.from_yaml("name: yes\nsegments: []") == Schedule("yes")
    assert yaml_io.from_yaml("name: off\nsegments: []") == Schedule("off")


def test_load_yaml_refuses_repeated_keys_at_every_depth() -> None:
    with pytest.raises(ValueError, match="duplicate key 'start'"):
        yaml_text.load_yaml("a:\n  b:\n    - {start: 1, start: 2}")


def test_a_key_given_twice_in_a_mapping_only_merged_in_is_refused() -> None:
    text = f"""
name: x
segments:
  - {{<<: {{unit: {A_ID}, alpha: 1.0, alpha: 2.0}}, end: 4}}
"""
    with pytest.raises(ValueError, match="duplicate key 'alpha'"):
        yaml_io.from_yaml(text)


def test_two_merge_keys_in_one_mapping_are_refused() -> None:
    text = "a: &a {x: 1}\nb: &b {x: 2}\nc: {<<: *a, <<: *b}"
    with pytest.raises(ValueError, match="duplicate key '<<'"):
        yaml_text.load_yaml(text)


def test_a_merge_of_a_deeper_template_may_be_overridden() -> None:
    text = """
use: {<<: *base, alpha: 2.0}
templates:
  base: &base {<<: {alpha: 0.5}, alpha: 1.0, start: 0}
"""
    # PyYAML wants an anchor before its alias, so the template comes first.
    text = "\n".join(reversed(text.strip().split("\n", 1)))
    data = yaml_text.load_yaml(text)
    assert data["use"] == {"alpha": 2.0, "start": 0}
    assert data["templates"]["base"] == {"alpha": 1.0, "start": 0}


def test_a_merge_may_be_overridden() -> None:
    text = f"""
name: x
segments:
  - &first {{unit: {A_ID}, alpha: 1.0, start: 0, end: 10}}
  - {{<<: *first, start: 20, end: 30}}
"""
    assert yaml_io.from_yaml(text).segments == (
        Segment(A, 1.0, 1.0, 0, 10),
        Segment(A, 1.0, 1.0, 20, 30),
    )


@pytest.mark.parametrize("data", [None, [], "name: x", 3])
def test_from_mapping_refuses_anything_but_a_mapping(data: object) -> None:
    with pytest.raises(ValueError, match="a schedule must be a mapping"):
        yaml_io.from_mapping(data)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [None, "name: x", constant(A, 1.0).segments])
def test_writing_refuses_anything_but_a_schedule(value: object) -> None:
    for write in (yaml_io.to_mapping, yaml_io.to_yaml):
        with pytest.raises(ValueError, match="not a Schedule"):
            write(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "text",
    ["1e-3", "010", "1:20", "0x10", "yes", "off", "2026-11-07", ".5", "~", "<<", "-"],
)
def test_dump_yaml_writes_what_load_yaml_reads_back(text: str) -> None:
    data = {"name": text, "alpha": 1e-20, "end": None, "flag": True}
    assert yaml_text.load_yaml(yaml_text.dump_yaml(data)) == data


def test_yaml_text_doctests() -> None:
    result = doctest.testmod(yaml_text)
    assert result.attempted > 0
    assert result.failed == 0
