"""Schedule files name neurons only, so far: a schedule that pushes a
NeuronSet cannot be written, and a set's ID cannot be read (decision 0019)."""

import pytest

from pushlab.schedules import yaml_io
from pushlab.schedules.model import Schedule, Segment
from pushlab.schedules.shapes import constant
from pushlab.schedules.yaml_text import load_yaml
from pushlab.units.neuron import NeuronUnit
from pushlab.units.neuron_set import NeuronSet

A = NeuronUnit("tiny", 1, 5)
PAIR = NeuronSet("tiny", "pair", [(1, 5, 1.0), (1, 9, -0.5)], "test")


def test_a_schedule_with_a_set_cannot_be_written() -> None:
    schedule = Schedule("x", [*constant(A, 1.0).segments, Segment(PAIR, 0.5, 0.5, 4)])
    for write in (yaml_io.to_mapping, yaml_io.to_yaml):
        with pytest.raises(ValueError) as error:
            write(schedule)
        assert str(error.value) == (
            "segments[2] (tiny:set:pair at +0.5 from reply token 4 to the end) "
            "pushes the set of neurons tiny:set:pair, and sets cannot be written to "
            "schedule files yet, since from_yaml could not read them back; keep a "
            "schedule with a set in Python"
        )
    neurons = constant(A, 1.0)  # a schedule of neurons is written as before
    assert yaml_io.from_yaml(yaml_io.to_yaml(neurons)) == neurons


SEGMENT_SET = "name: x\nsegments: [{unit: qwen3-8b:set:pirate-top50, alpha: 1.0}]"
PARAM_SET = (
    "name: x\nshape: handoff\nparams: {a: tiny:L1:N5, b: tiny:set:pair, "
    "alpha_a: 1, alpha_b: -1, switch_at: 4}"
)


@pytest.mark.parametrize(
    "text, where, set_id",
    [
        (SEGMENT_SET, "segments[0]", "qwen3-8b:set:pirate-top50"),
        (PARAM_SET, "shape handoff", "tiny:set:pair"),
        (
            "name: x\nshape: pulse\nparams: {unit: tiny:set:Pair, alpha: 1, "
            "start: 0, length: 4}",
            "shape pulse",
            "tiny:set:Pair",  # not even a good name: a set's ID all the same
        ),
    ],
)
def test_a_set_id_cannot_be_read(text: str, where: str, set_id: str) -> None:
    message = (
        f"{where}: {set_id!r} names a set of neurons, and sets cannot be read from "
        "schedule files yet; give a neuron's ID, such as qwen3.5-4b:L14:N8949, or "
        "build a schedule with a set in Python"
    )
    with pytest.raises(ValueError) as error:
        yaml_io.from_yaml(text)
    assert str(error.value) == message
    with pytest.raises(ValueError) as error:  # the reader of experiment files
        yaml_io.from_mapping(load_yaml(text))
    assert str(error.value) == message
