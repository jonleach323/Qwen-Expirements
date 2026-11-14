"""Sets of neurons in units files (pushlab.units.ladder_units): a set's row
gives its definition as a list of persona candidates does, and becomes a
LadderUnit of the NeuronSet; pushlab ladder, whose shapes cannot hold one,
refuses it (pushlab.runs.ladder). The units and personas are invented."""

import json
from pathlib import Path

import pytest

from pushlab.discovery.candidates import Candidate, read_candidates, write_candidates
from pushlab.runs.ladder import Ladder
from pushlab.store.records import RunRecord
from pushlab.units.ladder_units import LadderUnit, RunPush, read_ladder_units, run_push
from pushlab.units.neuron import NeuronUnit
from pushlab.units.neuron_set import NeuronSet

# Two invented sets: one across layers with a size, as Mayner's are, and one
# in one layer with none, which takes that layer's residual size.
CAPTAIN = NeuronSet(
    "qwen3-8b",
    "salt-captain-top2",
    [(17, 1234, 0.1 + 0.2), (18, 99, -0.31)],
    "invented: the salt captain's top 2 neurons",
    96.04,
)
OWL = NeuronSet("qwen3-8b", "brass-owl-pair", [(9, 1, 1.0), (9, 2, -0.5)], "made up")
HEADER = "unit,kind,entries,size,source_ref,direction,persona,source\n"


def _file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "units.csv"
    path.write_text(text, encoding="utf-8")
    return path


def _row(unit: NeuronSet, direction: str = "+1", persona: str = "Brass Owl") -> str:
    """A set's row under HEADER, its entries as JSON in quotes, as CSV has it."""
    entries = json.dumps([list(entry) for entry in unit.entries]).replace('"', '""')
    size = "" if unit.size is None else repr(unit.size)
    return (
        f'{unit},set,"{entries}",{size},{unit.source},{direction},{persona},invented\n'
    )


def test_a_candidates_lists_sets_are_its_units_as_it_reads_them(
    tmp_path: Path,
) -> None:
    neuron = Candidate(
        "library",
        "Brass Owl",
        NeuronUnit("qwen3-8b", 9, 4242),
        -1,
        1.25,
        False,
        "library/qwen/library.json",
        "L9 N4242 s-1",
    )
    sets = [
        Candidate(
            "mayner",
            persona,
            unit,
            1,
            1.0,
            False,
            "mayner/bundle/split/top_neurons.parquet",
            unit.source,
            category="role",
        )
        for persona, unit in (("Salt Captain", CAPTAIN), ("Brass Owl", OWL))
    ]
    path = write_candidates(tmp_path / "candidates.csv", [neuron, *sets])
    units = read_ladder_units(path)
    # The same units that read_candidates builds, each set equal bit for bit.
    assert [unit.unit for unit in units] == [c.unit for c in read_candidates(path)]
    assert [unit.unit for unit in units] == [neuron.unit, CAPTAIN, OWL]
    assert units[1].unit.entries[0].coefficient.hex() == (0.1 + 0.2).hex()  # type: ignore[union-attr]
    assert [(unit.key, unit.persona, unit.source) for unit in units] == [
        ("qwen3-8b:L9:N4242-", "Brass Owl", "library"),
        ("qwen3-8b:set:salt-captain-top2+", "Salt Captain", "mayner"),
        ("qwen3-8b:set:brass-owl-pair+", "Brass Owl", "mayner"),
    ]
    assert units[1].alpha(0.5) == 0.5


def test_a_set_row_with_its_definition_is_a_set_of_its_columns(tmp_path: Path) -> None:
    text = HEADER + _row(CAPTAIN, "-1", "Salt Captain") + _row(OWL)
    captain, owl = read_ladder_units(_file(tmp_path, text))
    assert captain == LadderUnit(CAPTAIN, -1, "Salt Captain", "", "invented")
    assert owl.unit == OWL and owl.unit.size is None  # type: ignore[union-attr]
    # kind is optional, and a set's ID alone says what the unit is.
    bare = "unit,entries,size,source_ref\n" + (
        'qwen3-8b:set:brass-owl-pair,"[[9, 1, 1], [9, 2, -0.5]]",,made up\n'
    )
    (only,) = read_ladder_units(_file(tmp_path, bare))
    assert only.unit == OWL  # a coefficient of 1 is 1.0


def test_one_set_both_ways_is_two_ladder_units(tmp_path: Path) -> None:
    text = HEADER + _row(OWL, "+1") + _row(OWL, "-1")
    keys = [unit.key for unit in read_ladder_units(_file(tmp_path, text))]
    assert keys == ["qwen3-8b:set:brass-owl-pair+", "qwen3-8b:set:brass-owl-pair-"]


OTHER_OWL = NeuronSet("qwen3-8b", "brass-owl-pair", [(9, 1, 1.0)], "made up")


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        (
            "unit,entries\nqwen3-8b:set:brass-owl-pair,[]\n",
            "row 2: qwen3-8b:set:brass-owl-pair is a set of neurons, so the file "
            "needs its definition's columns, entries, size, source_ref, as pushlab "
            "candidates writes them; it lacks size, source_ref",
        ),
        (
            HEADER + "qwen3-8b:set:x,set,[[9; 1; 1.0]],,made up,1,,\n",
            "row 2: entries must be JSON, [[layer, neuron, coefficient], ...], not "
            "'[[9; 1; 1.0]]'",
        ),
        (
            HEADER + 'qwen3-8b:set:x,set,"[[9, 1, 1.0]]",big,made up,1,,\n',
            "row 2: size must be a number, or empty for the residual size of the "
            "set's one layer, not 'big'",
        ),
        (
            HEADER + 'qwen3-8b:set:x,set,"[[9, 1, 0]]",,made up,1,,\n',
            "row 2: qwen3-8b:set:x: entries[0]: the coefficient must be a finite "
            "number other than 0, not 0",
        ),
        (
            HEADER + 'qwen3-8b:set:x,set,"[[9, 1, 1], [10, 2, 1]]",,made up,1,,\n',
            "row 2: qwen3-8b:set:x: qwen3-8b:set:x has neurons in layers 9, 10, so "
            "it needs a size",
        ),
        (
            HEADER + 'qwen3-8b:set:x,set,"[[9, 1, 1]]",,,1,,\n',
            "row 2: qwen3-8b:set:x: source must say where the set comes from",
        ),
        (
            HEADER + 'qwen3-8b:set:x,neuron,"[[9, 1, 1]]",,made up,1,,\n',
            "row 2: its kind is neuron, but qwen3-8b:set:x is a set's ID",
        ),
        (
            HEADER + _row(OWL, "+1") + _row(OTHER_OWL, "-1"),
            "row 3: qwen3-8b:set:brass-owl-pair is defined otherwise in row 2; a "
            "set's ID names one set",
        ),
        (HEADER + _row(OWL) + _row(OWL), "row 3 repeats row 2"),
        (
            HEADER + _row(OWL) + _row(NeuronSet("tiny", "x", [(1, 2, 1.0)], "y")),
            "mixes units of qwen3-8b, tiny",
        ),
    ],
)
def test_a_set_row_that_is_not_one_is_refused(
    tmp_path: Path, text: str, problem: str
) -> None:
    path = _file(tmp_path, text)
    with pytest.raises(ValueError) as error:
        read_ladder_units(path)
    assert str(error.value).startswith(f"{path}") and problem in str(error.value)


def test_a_set_runs_push_is_its_id_and_dose() -> None:
    definition = {"entries": [[9, 1, 1.0], [9, 2, -0.5]], "size": None}
    segment = {"unit": str(OWL), "set": definition | {"source": "made up"}}
    segment |= {"alpha": -0.75, "control_for": None, "control_seed": None}
    record = RunRecord(
        "x-000001", "x", "m", "p", [segment, segment], {}, 0, "", "", "", [], "ok"
    )
    push = run_push(record)
    assert push == RunPush("qwen3-8b:set:brass-owl-pair", -0.75, None)
    assert push is not None and push.key == LadderUnit(OWL, -1).key


def test_a_ladder_of_a_set_is_refused_for_pushlab_panel_stage_ladder() -> None:
    owl = LadderUnit(NeuronUnit("qwen3-8b", 9, 4242), 1, "Brass Owl")
    sets = [LadderUnit(CAPTAIN, -1), LadderUnit(OWL, 1), LadderUnit(OWL, -1)]
    why = (
        "pushlab ladder writes each unit as a shape's params, which cannot hold a "
        "set's definition: ladder sets with pushlab panel stage ladder (decision "
        "0023)"
    )
    with pytest.raises(ValueError) as error:  # the owl both ways is one set
        Ladder("owls", (owl, *sets), (0.5, 1.0), "p", units_file="units.csv")
    assert str(error.value) == (
        f"units.csv has the set of neurons {CAPTAIN} and 1 more, but {why}"
    )
    with pytest.raises(ValueError) as error:
        Ladder("owls", (owl, sets[1]), (0.5, 1.0), "p")
    assert str(error.value) == f"it has the set of neurons {OWL}, but {why}"
    assert Ladder("owl", (owl,), (0.5,), "p").units == (owl,)
