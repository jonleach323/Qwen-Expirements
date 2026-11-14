"""Sets of neurons in schedule files: a segment names the set by its ID and
gives its definition beside it (pushlab.schedules.yaml_io, yaml_sets;
decision 0023). The sets here are made up."""

import doctest
import json
import re

import pytest

from pushlab import PROJECT_ROOT
from pushlab.schedules import yaml_sets
from pushlab.schedules.model import Schedule, Segment
from pushlab.schedules.shapes import alternate, constant, handoff
from pushlab.schedules.yaml_io import from_mapping, from_yaml, to_mapping, to_yaml
from pushlab.schedules.yaml_text import load_yaml
from pushlab.units.neuron import NeuronUnit
from pushlab.units.neuron_set import NeuronSet

A = NeuronUnit("tiny", 1, 5)
PAIR = NeuronSet("tiny", "pair", [(1, 5, 1.0), (1, 9, -0.5)], "test")
# Across four of qwen3-8b's layers, so it needs a size; numbers whose decimal
# writing is long, tiny or huge, to show each comes back bit for bit.
AWKWARD = NeuronSet(
    "qwen3-8b",
    "brass-owl-top4",
    [(17, 1234, 0.1 + 0.2), (18, 99, -5e-324), (3, 7, 1 / 3), (35, 12287, -2.5e17)],
    "made up: four owl neurons",
    96.04 + 2**-46,
)
OWL_TEXT = """\
name: owl
segments:
  - unit: qwen3-8b:set:brass-owl-top2
    set: {entries: [[17, 1234, 0.52], [18, 99, -0.31]], size: 96.04, source: made up}
    alpha: 1.0
"""
OWL = NeuronSet(
    "qwen3-8b", "brass-owl-top2", [(17, 1234, 0.52), (18, 99, -0.31)], "made up", 96.04
)


def _bits(unit: NeuronSet) -> list[str]:
    """Every number of a set, as the hex of its float: equal hex, equal bits."""
    numbers = [entry.coefficient for entry in unit.entries]
    return [float(number).hex() for number in [*numbers, unit.size or 0.0]]


def test_a_set_segment_is_read_with_its_definition() -> None:
    schedule = from_yaml(OWL_TEXT)
    assert schedule == Schedule("owl", [Segment(OWL, 1.0, 1.0)])
    (segment,) = schedule.segments
    assert segment.unit.layers == (17, 18)  # type: ignore[union-attr]
    assert from_mapping(load_yaml(OWL_TEXT)) == schedule  # an experiment's reader


def test_every_segment_of_a_set_is_written_with_its_definition() -> None:
    schedule = Schedule(
        "x", [Segment(PAIR, 0.5, 0.5, region="prompt"), *constant(PAIR, 0.5).segments]
    )
    written = to_mapping(schedule)["segments"]
    assert [segment["region"] for segment in written] == ["prompt", "prompt", "reply"]
    for segment in written:
        assert list(segment)[:2] == ["unit", "set"]
        assert segment["unit"] == "tiny:set:pair"
        assert segment["set"] == {
            "entries": [[1, 5, 1.0], [1, 9, -0.5]],
            "size": None,
            "source": "test",
        }
    neurons = to_mapping(constant(A, 1.0))["segments"]  # a neuron's, as before
    assert all("set" not in segment for segment in neurons)


@pytest.mark.parametrize(
    "schedule",
    [
        Schedule(
            "mixed",
            [
                Segment(AWKWARD, 0.75, 0.75, region="prompt"),
                Segment(AWKWARD, 0.0, 1.0, start=2, end=12),
                Segment(
                    NeuronUnit("qwen3-8b", 3, 7), -0.5, -0.5, 4, on_for=2, off_for=3
                ),
            ],
            allow_over_ceiling=True,
        ),
        handoff(PAIR, A, 1.0, -1.0, 4),
        alternate(PAIR, NeuronSet("tiny", "lone", [(0, 3, 2.0)], "y"), 1.0, 0.5, 2),
    ],
    ids=["awkward-numbers", "handoff", "two-sets"],
)
def test_a_schedule_with_sets_reads_back_equal_bit_for_bit(schedule: Schedule) -> None:
    sets = [unit for unit in schedule.units() if isinstance(unit, NeuronSet)]
    text = to_yaml(schedule)
    # As a file: YAML text, every float written by its repr.
    back = from_yaml(text)
    assert back == schedule
    assert to_yaml(back) == text
    # As a run's record: plain data through JSON, as the runs table keeps it.
    stored = json.loads(json.dumps(to_mapping(schedule), allow_nan=False))
    assert from_mapping(stored) == schedule
    for read in (back, from_mapping(stored)):
        again = [unit for unit in read.units() if isinstance(unit, NeuronSet)]
        assert [_bits(unit) for unit in again] == [_bits(unit) for unit in sets]


def test_one_set_given_twice_with_one_definition_is_one_unit() -> None:
    text = OWL_TEXT + OWL_TEXT.split("segments:\n")[1].replace("alpha: 1.0", "alpha: 2")
    schedule = from_yaml(text)
    assert [segment.unit for segment in schedule.segments] == [OWL, OWL]
    assert schedule.units() == (OWL,)
    # A YAML anchor gives the second segment the first one's definition.
    anchored = """\
name: owl
segments:
  - unit: qwen3-8b:set:brass-owl-top2
    set: &owl
      entries: [[17, 1234, 0.52], [18, 99, -0.31]]
      size: 96.04
      source: made up
    alpha: 1.0
  - {unit: qwen3-8b:set:brass-owl-top2, set: *owl, alpha: 2}
"""
    assert from_yaml(anchored) == schedule


def _segment(unit: str = "qwen3-8b:set:owl", **more: str) -> str:
    """A schedule of one segment of unit at dose 1, with more of its keys."""
    keys = "".join(f", {key}: {value}" for key, value in more.items())
    return f"name: x\nsegments: [{{unit: {unit}, alpha: 1{keys}}}]\n"


DEFINITION = "{entries: [[3, 7, 0.5], [4, 8, 1]], size: 2.5, source: made up}"
SAME_LAYER = "{entries: [[3, 7, 0.5], [3, 8, 1]], size: null, source: made up}"


@pytest.mark.parametrize(
    "text, message",
    [
        (
            _segment(),
            "segments[0]: 'qwen3-8b:set:owl' names a set of neurons, so the segment "
            "needs the set's definition beside it: set: {entries: [[layer, neuron, "
            "coefficient], ...], size: ..., source: ...}",
        ),
        (
            _segment("qwen3-8b:L3:N7", set=DEFINITION),
            "segments[0]: set: gives a set's definition, but the unit "
            "'qwen3-8b:L3:N7' is no set's ID; leave set out for a neuron, or give "
            "the set's ID, such as qwen3-8b:set:pirate-top50",
        ),
        (
            _segment(set="'1.5'"),  # text, not a definition, nor a number
            "segments[0]: the set qwen3-8b:set:owl needs its definition as a "
            "mapping, set: {entries: [[layer, neuron, coefficient], ...], size: "
            "..., source: ...}, not '1.5'",
        ),
        (
            _segment(set="{entries: [[3, 7, 0.5]], size: 2, source: x, colour: red}"),
            "segments[0]: set: unknown key 'colour'; it takes entries, size, source",
        ),
        (
            _segment(set="{entries: [[3, 7, 0.5]], source: x}"),
            "segments[0]: the definition of qwen3-8b:set:owl lacks size; it takes "
            "entries, size (a number, or null for the residual size of the set's "
            "one layer) and source",
        ),
        (
            _segment(set="{entries: [[3, 7, 0.5], [4, 8, 1]], size: null, source: x}"),
            "segments[0]: the set qwen3-8b:set:owl: qwen3-8b:set:owl has neurons in "
            "layers 3, 4, so it needs a size, the length of its push at dose 1: a "
            "residual size belongs to one layer",
        ),
    ],
    ids=["no-definition", "beside-a-neuron", "not-a-mapping", "unknown-key", "lacks"]
    + ["across-layers-no-size"],
)
def test_a_set_segment_is_refused_with_why(text: str, message: str) -> None:
    for read in (from_yaml, lambda text: from_mapping(load_yaml(text))):
        with pytest.raises(ValueError) as error:
            read(text)
        assert str(error.value) == message


@pytest.mark.parametrize(
    "definition, problem",
    [
        ("{entries: [[3, 7, 0]], size: 2, source: x}", "coefficient must be a finite"),
        ("{entries: [[3, 7, 0.5], [3, 7, 1]], size: 2, source: x}", "repeats layer 3"),
        ("{entries: [], size: 2, source: x}", "entries must be a non-empty list"),
        ("{entries: [[3, 7]], size: 2, source: x}", r"entries\[0\] must be"),
        ("{entries: [['3', 7, 1]], size: 2, source: x}", "layer must be a Python int"),
        ("{entries: [[3, 7, .nan]], size: 2, source: x}", "coefficient must be"),
        ("{entries: [[3, 7, 1]], size: 0, source: x}", "size must be None or a fin"),
        ("{entries: [[3, 7, 1]], size: '2', source: x}", "size must be None or a fin"),
        ("{entries: [[3, 7, 1]], size: 2, source: ' '}", "source must say where"),
    ],
)
def test_a_definition_neuron_set_refuses_is_refused(
    definition: str, problem: str
) -> None:
    with pytest.raises(ValueError, match=r"^segments\[0\]: the set qwen3-8b:set:owl: "):
        from_yaml(_segment(set=definition))
    with pytest.raises(ValueError, match=problem):
        from_yaml(_segment(set=definition))


@pytest.mark.parametrize(
    "written, unit, problem",
    [
        ("qwen3-8b:set:Owl", "qwen3-8b:set:Owl", "a set's name must be lowercase"),
        ("qwen9:set:owl", "qwen9:set:owl", "'qwen9' is not in the model registry"),
        ("'qwen3-8b:set:'", "qwen3-8b:set:", "a set's name must be lowercase"),
    ],
)
def test_a_set_id_neuron_set_refuses_is_refused(
    written: str, unit: str, problem: str
) -> None:
    with pytest.raises(ValueError, match=problem) as error:
        from_yaml(_segment(written, set=SAME_LAYER))
    assert str(error.value).startswith(f"segments[0]: the set {unit}: ")


def test_one_set_id_given_two_definitions_is_refused() -> None:
    other = DEFINITION.replace("0.5", "0.25")
    text = (
        "name: x\nsegments:\n"
        "  - {unit: qwen3-8b:L1:N5, alpha: 1}\n"
        f"  - {{unit: qwen3-8b:set:owl, set: {DEFINITION}, alpha: 1}}\n"
        f"  - {{unit: qwen3-8b:set:owl, set: {other}, alpha: 1, start: 4}}\n"
    )
    with pytest.raises(ValueError) as error:
        from_yaml(text)
    assert str(error.value) == (
        "segments[2]: qwen3-8b:set:owl is defined otherwise in segments[1]; a "
        "set's ID names one set, so give it one definition, or the other set a "
        "name of its own"
    )


SHAPE_MESSAGE = (
    "shape {shape}: {unit!r} names a set of neurons, and a shape's params cannot "
    "hold its definition; write the schedule as segments, each with the set's "
    "definition beside its unit (set: {{entries: [[layer, neuron, coefficient], "
    "...], size: ..., source: ...}})"
)


@pytest.mark.parametrize(
    "text, shape, unit",
    [
        (
            "name: x\nshape: handoff\nparams: {a: tiny:L1:N5, b: tiny:set:pair, "
            "alpha_a: 1, alpha_b: -1, switch_at: 4}",
            "handoff",
            "tiny:set:pair",
        ),
        (
            "name: x\nshape: pulse\nparams: {unit: tiny:set:Pair, alpha: 1, "
            "start: 0, length: 4}",
            "pulse",
            "tiny:set:Pair",  # not even a good name: a set's ID all the same
        ),
        (
            "name: x\nshape: constant\nparams: {unit: qwen3-8b:set:pirate-top50, "
            "alpha: 1}",
            "constant",
            "qwen3-8b:set:pirate-top50",
        ),
    ],
)
def test_a_shape_whose_params_name_a_set_is_refused(
    text: str, shape: str, unit: str
) -> None:
    message = SHAPE_MESSAGE.format(shape=shape, unit=unit)
    with pytest.raises(ValueError) as error:
        from_yaml(text)
    assert str(error.value) == message
    with pytest.raises(ValueError) as error:  # the reader of experiment files
        from_mapping(load_yaml(text))
    assert str(error.value) == message


def test_the_helpers_refuse_what_is_no_set() -> None:
    lone = {"entries": [[1, 5, 1.0]], "size": None, "source": "x"}
    assert yaml_sets.set_from_definition("tiny:set:lone", lone, "here").size is None
    with pytest.raises(ValueError, match="^here: 'tiny:L1:N5' is no set's ID"):
        yaml_sets.set_from_definition("tiny:L1:N5", lone, "here")
    with pytest.raises(ValueError, match="only a NeuronSet has a definition"):
        yaml_sets.definition(A)  # type: ignore[arg-type]
    assert not any(map(yaml_sets.is_set_id, (5, " tiny:set:x", "tiny:L1:N5")))


def test_the_readmes_example_of_a_set_reads_as_it_says() -> None:
    readme = (PROJECT_ROOT / "experiments" / "README.md").read_text(encoding="utf-8")
    (block,) = re.findall(r"```\n(segments:\n.*?)```", readme, flags=re.DOTALL)
    (segment,) = from_mapping({"name": "example"} | load_yaml(block)).segments
    assert segment.unit == NeuronSet(
        "qwen3-8b",
        "pirate-top50",
        [(17, 1234, 0.52), (18, 99, -0.31)],
        "mayner role pirate",
        96.04,
    )


def test_docstring_examples_run() -> None:
    result = doctest.testmod(yaml_sets)
    assert result.attempted > 0
    assert result.failed == 0
