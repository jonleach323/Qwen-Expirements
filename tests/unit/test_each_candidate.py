"""Conditions made by rule, one per candidate of a list (pushlab.runs.
each_candidate, decision 0023): their labels, doses and keys, a set's
schedule, and the spec's hash, which covers every unit and dose the
candidates resolve to and nothing else of the list. The refusals are in
test_each_candidate_refusals.py.

The lists are made up (tests/unit/candidate_lists.py): library neurons on
qwen3-8b, and sets listed as Mayner's are.
"""

import dataclasses
import doctest
import importlib.util
import math
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from pushlab.discovery.candidates import Candidate, write_candidates
from pushlab.runs import each_candidate
from pushlab.runs.batch import record_schedule
from pushlab.runs.each_candidate import EachCandidate
from pushlab.runs.expand import expand
from pushlab.runs.spec import ExperimentSpec, spec_from_yaml
from pushlab.schedules.model import Schedule
from pushlab.schedules.shapes import constant
from pushlab.schedules.yaml_sets import definition

_SPEC = importlib.util.spec_from_file_location(
    "candidate_lists", Path(__file__).with_name("candidate_lists.py")
)
lists = importlib.util.module_from_spec(_SPEC)  # type: ignore[arg-type]
_SPEC.loader.exec_module(lists)  # type: ignore[union-attr]

# Strengths 1.0, 1.25 and 0.75; directions +1, -1 and +1.
NEURONS = lists.library_neurons(3, [9, 14])
SETS = lists.mayner_sets(2)
SPEC, EACH = lists.EXPERIMENT, lists.EACH


@pytest.fixture
def write_list(tmp_path: Path) -> Callable[..., Path]:
    """Write candidates (the 3 neurons and 2 sets by default) as a list."""

    def write(candidates: Sequence[Candidate] = (*NEURONS, *SETS)) -> Path:
        return write_candidates(tmp_path / "candidates.csv", candidates)

    return write


@pytest.fixture
def spec_of(write_list: Callable[..., Path]) -> Callable[..., ExperimentSpec]:
    """The spec of conditions (EACH by default) on the list written, with
    rule after its file in candidates:, such as ', kind: neuron'."""

    def make(conditions: str = EACH, rule: str = "", file: Path | None = None):  # type: ignore[no-untyped-def]
        file = write_list() if file is None else file
        text = SPEC.format(file=file, rule=rule, conditions=conditions)
        return spec_from_yaml(text)

    return make


def _alphas(schedule: Schedule) -> list[float]:
    return [segment.alpha_start for segment in schedule.segments]


def test_each_candidate_makes_a_condition_per_candidate_in_the_list_s_order(
    spec_of: Callable[..., ExperimentSpec],
) -> None:
    spec = spec_of()
    labels = [condition.label for condition in spec.conditions]
    assert labels == [f"screen-000{n}" for n in range(1, 6)]
    keys = [condition.candidate for condition in spec.conditions]
    assert keys == [candidate.key for candidate in (*NEURONS, *SETS)]
    for condition, candidate in zip(spec.conditions, (*NEURONS, *SETS), strict=True):
        ((label, schedule),) = condition.points  # one dose: the label alone
        alpha = candidate.direction * candidate.strength
        assert label == condition.label
        assert schedule == Schedule(label, constant(candidate.unit, alpha).segments)
        assert (condition.controls, condition.control_seed) == (0, None)
    # The prompt pass and every reply token, at the candidate's own dose.
    assert [_alphas(c.points[0][1]) for c in spec.conditions[:3]] == [
        [1.0, 1.0],
        [-1.25, -1.25],
        [0.75, 0.75],
    ]


def test_several_doses_make_a_grid_labelled_by_alpha(
    spec_of: Callable[..., ExperimentSpec],
) -> None:
    conditions = (
        "  - label: ladder\n"
        "    each_candidate: {doses: [0.5, 1, 1.5], allow_over_ceiling: true}\n"
    )
    spec = spec_of(conditions)
    second = spec.conditions[1]  # direction -1, strength 1.25
    assert [label for label, _ in second.points] == [
        "ladder-0002 alpha=-0.625",
        "ladder-0002 alpha=-1.25",
        "ladder-0002 alpha=-1.875",
    ]
    assert [_alphas(s)[0] for _, s in second.points] == [-0.625, -1.25, -1.875]
    assert all(schedule.name == "ladder-0002" for _, schedule in second.points)
    # A set is pushed at multiples of Mayner's strength, dose 1.
    set_labels = [label for label, _ in spec.conditions[3].points]
    assert set_labels == [f"ladder-0004 alpha={a}" for a in (0.5, 1.0, 1.5)]
    assert len(expand(spec).runs) == 5 * 3 * 5  # prompts x doses x candidates


def test_a_key_s_own_doses_replace_the_condition_s(
    spec_of: Callable[..., ExperimentSpec],
) -> None:
    rule = f", keys: [{{key: {NEURONS[1].key}, doses: [2.0]}}, {SETS[0].key}]"
    spec = spec_of("  - {label: c, each_candidate: {doses: [0.5, 1.0]}}\n", rule)
    first, second = spec.conditions
    assert (first.candidate, second.candidate) == (NEURONS[1].key, SETS[0].key)
    assert [label for label, _ in first.points] == ["c-0001"]
    assert _alphas(first.points[0][1]) == [-2.5, -2.5]
    assert [label for label, _ in second.points] == [
        "c-0002 alpha=0.5",
        "c-0002 alpha=1.0",
    ]


def test_include_prompt_false_leaves_the_prompt_pass_alone(
    spec_of: Callable[..., ExperimentSpec],
) -> None:
    spec = spec_of("  - {label: c, each_candidate: {include_prompt: false}}\n")
    for condition in spec.conditions:
        ((_, schedule),) = condition.points
        (segment,) = schedule.segments
        assert (segment.region, segment.start, segment.end) == ("reply", 0, None)


def test_a_dose_above_the_ceiling_needs_allow_over_ceiling(
    spec_of: Callable[..., ExperimentSpec], write_list: Callable[..., Path]
) -> None:
    strong = write_list([lists.neuron(9, 7, -1, 1.5)])
    with pytest.raises(ValueError, match="(?s)condition 'screen-0001': .*ceiling"):
        expand(spec_of(file=strong))
    allowed = spec_of(
        "  - {label: screen, each_candidate: {allow_over_ceiling: true}}\n",
        file=strong,
    )
    assert allowed.conditions[0].points[0][1].allow_over_ceiling is True
    assert len(expand(allowed).runs) == 5


def test_a_set_s_condition_holds_the_set_written_out_with_its_definition(
    spec_of: Callable[..., ExperimentSpec],
) -> None:
    spec = spec_of(rule=", kind: set")
    written = spec.to_mapping()["conditions"][0]
    assert written["candidate"] == SETS[0].key
    segments = written["points"][0]["schedule"]["segments"]
    assert [segment["region"] for segment in segments] == ["prompt", "reply"]
    assert all(segment["unit"] == str(SETS[0].unit) for segment in segments)
    assert all(segment["set"] == definition(SETS[0].unit) for segment in segments)
    # A run's record holds the set itself, so it makes the set again.
    record = expand(spec).runs[0].record
    assert record_schedule(record).units() == (SETS[0].unit,)


def test_controls_go_to_every_condition_but_not_yet_to_a_set(
    spec_of: Callable[..., ExperimentSpec],
) -> None:
    conditions = "  - {label: c, each_candidate: {}, controls: 1, control_seed: 3}\n"
    spec = spec_of(conditions, ", kind: neuron")
    assert {(c.controls, c.control_seed) for c in spec.conditions} == {(1, 3)}
    runs = expand(spec).runs
    assert len(runs) == 5 * 3 * 2
    assert runs[1].record.control_of == runs[0].record.run_id
    with pytest.raises(ValueError, match="random controls for sets come later"):
        expand(spec_of(conditions, ", kind: set"))


def test_conditions_before_and_after_keep_their_places(
    spec_of: Callable[..., ExperimentSpec],
) -> None:
    conditions = (
        "  - {label: none, segments: []}\n"
        "  - {label: owl, each_candidate: {}}\n"
        "  - {label: last, segments: []}\n"
    )
    labels = [c.label for c in spec_of(conditions, ", kind: set").conditions]
    assert labels == ["none", "owl-0001", "owl-0002", "last"]
    runs = expand(spec_of(conditions, ", kind: set")).runs
    assert [run.condition for run in runs[:4]] == labels


def test_labels_grow_wider_past_9999() -> None:
    chosen = [(NEURONS[0], None)] * 10_000
    labels = [c.label for c in EachCandidate("c", 0).conditions(chosen)]
    assert labels[0] == "c-00001" and labels[-1] == "c-10000"


def test_two_doses_that_give_one_alpha_are_refused() -> None:
    near = math.nextafter(0.99, 1)  # times 1.25, it rounds as 0.99 does
    assert 1.25 * near == 1.25 * 0.99
    each = EachCandidate("c", 2, (0.99, near))
    with pytest.raises(ValueError) as error:
        each.conditions([(NEURONS[1], None)])
    assert str(error.value) == (
        "conditions[2] (c): library/L14N101-'s doses give alpha=-1.2375 twice; give "
        "doses that differ more"
    )


def test_the_hash_covers_every_unit_and_dose_the_candidates_resolve_to(
    spec_of: Callable[..., ExperimentSpec], write_list: Callable[..., Path]
) -> None:
    base = spec_of().sha256
    stronger = dataclasses.replace(NEURONS[1], strength=1.3)
    moved = lists.neuron(9, 4242, 1, 1.0)
    entries = [(17, 1001, 0.6), (18, 2001, math.nextafter(-0.25, 0))]
    nudged = lists.owl_set("role-1-top2", entries, 91.0, persona="role 1")
    assert dataclasses.replace(nudged.unit, entries=SETS[1].unit.entries) == (
        SETS[1].unit  # the one bit is all that differs
    )
    for changed in (
        [NEURONS[0], stronger, NEURONS[2], *SETS],  # a strength
        [moved, *NEURONS[1:], *SETS],  # a neuron
        [*NEURONS, SETS[0], nudged],  # a set's coefficient, by one bit
        [NEURONS[1], NEURONS[0], NEURONS[2], *SETS],  # the order, so the labels
    ):
        assert spec_of(file=write_list(changed)).sha256 != base
    # A setting of the rule or of each_candidate is a setting.
    assert (
        spec_of(rule=", kind: neuron").sha256
        != spec_of(rule=", kind: neuron, source: library").sha256
    )
    doses = "  - {label: screen, each_candidate: {doses: [1.0, 2.0]}}\n"
    assert spec_of(doses).sha256 != base


def test_a_list_changed_without_changing_a_unit_or_dose_keeps_the_hash(
    spec_of: Callable[..., ExperimentSpec], write_list: Callable[..., Path]
) -> None:
    base = spec_of(rule=", source: library").sha256
    renamed = [
        dataclasses.replace(c, persona="Tin Fox", coherence=2, tags=("odd",))
        for c in NEURONS
    ]
    downloaded = [dataclasses.replace(c, downloaded_at="2026-11-13") for c in SETS]
    file = write_list([*renamed, *downloaded, *lists.mayner_sets(4)[2:]])
    assert spec_of(rule=", source: library", file=file).sha256 == base
    # The same list written again, or the default doses written out.
    assert spec_of(rule=", source: library").sha256 == base
    written = (
        "  - {label: screen, each_candidate: {doses: [1], include_prompt: true}}\n"
    )
    assert spec_of(written, ", source: library").sha256 == base


def test_to_mapping_writes_candidates_and_each_key_only_when_given(
    spec_of: Callable[..., ExperimentSpec],
) -> None:
    mapping = spec_of().to_mapping()
    assert [c["candidate"] for c in mapping["conditions"]] == [
        c.key for c in (*NEURONS, *SETS)
    ]
    assert mapping["candidates"] == {
        "file": mapping["candidates"]["file"],
        "source": None,
        "kind": None,
        "keys": None,
        "sample": None,
    }
    plain = spec_from_yaml(
        SPEC.split("candidates:")[0].replace("{{", "{").replace("}}", "}")
        + "conditions:\n  - {label: none, segments: []}\n"
    ).to_mapping()
    assert "candidate" not in plain["conditions"][0] and "candidates" not in plain


def test_doctests() -> None:
    result = doctest.testmod(each_candidate)
    assert result.attempted > 0
    assert result.failed == 0
