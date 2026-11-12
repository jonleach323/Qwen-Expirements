"""Persona candidates: the record, duplicates, and the CSV file (decision 0020)."""

import csv
import doctest
import math
from pathlib import Path

import pytest

from pushlab.discovery import candidates as candidates_module
from pushlab.discovery.candidates import (
    COLUMNS,
    Candidate,
    count_by_source,
    drop_duplicates,
    from_row,
    read_candidates,
    same_name,
    to_row,
    write_candidates,
)
from pushlab.discovery.sources import LIBRARY, MAYNER
from pushlab.models.discovery import ModelInfo
from pushlab.units.neuron import NeuronUnit
from pushlab.units.neuron_set import NeuronSet

QWEN3_8B = ModelInfo(layers=36, hidden_size=4096, mlp_size=12288)
LIBRARY_FILE = "library/qwen/library.json"


def _neuron(
    persona: str = "Teapot", layer: int = 9, index: int = 4242, **fields: object
) -> Candidate:
    values: dict[str, object] = {
        "source": "library",
        "persona": persona,
        "unit": NeuronUnit("qwen3-8b", layer, index),
        "direction": -1,
        "strength": 1.0,
        "sensitive": False,
        "source_file": LIBRARY_FILE,
        "source_ref": f"L{layer} N{index} s-1",
    } | fields
    return Candidate(**values)  # type: ignore[arg-type]


def _set(role: str = "pirate", k: int = 2, **fields: object) -> Candidate:
    entries = [(3, 12, 0.5), (18, 7, -1.25)][:k]
    unit = NeuronSet(
        "qwen3-8b", f"{role}-top{k}", entries, f"{role}, ranks 1 to {k}", 96.0
    )
    values: dict[str, object] = {
        "source": "mayner",
        "persona": role,
        "unit": unit,
        "direction": 1,
        "strength": 1.0,
        "sensitive": False,
        "source_file": "mayner/bundle/split/top_neurons.parquet",
        "source_ref": unit.source,
    } | fields
    return Candidate(**values)  # type: ignore[arg-type]


def test_keys_name_the_source_and_the_unit() -> None:
    assert _neuron().key == "library/L9N4242-"
    assert _neuron(direction=1).key == "library/L9N4242+"
    assert _set().key == "mayner/pirate-top2"
    assert _neuron().kind == "neuron" and _set().kind == "set"


@pytest.mark.parametrize(
    ("fields", "problem"),
    [
        ({"source": "zoo"}, "source must be one of library, mayner"),
        ({"persona": " "}, "persona must be a name"),
        ({"unit": "qwen3-8b:L9:N4242"}, "unit must be a NeuronUnit or a NeuronSet"),
        ({"direction": 0}, "direction must be"),
        ({"direction": True}, "direction must be"),
        ({"direction": 1.0}, "direction must be"),
        ({"tags": ["kitchen;object"]}, "tags must be words with no ';'"),
        ({"strength": 0.0}, "strength must be finite and above 0"),
        ({"strength": math.nan}, "strength must be finite and above 0"),
        ({"strength": "1"}, "strength must be a number"),
        ({"sensitive": 1}, "sensitive must be True or False"),
        ({"source_file": ""}, "must say where it came from"),
        ({"source_url": None}, "source_url must be text"),
        ({"tags": ["", "x"]}, "tags must be words with no ';'"),
        ({"coherence": 7.5}, "coherence must be None or an int"),
        ({"mayner_in_role": 1}, "mayner_in_role must be None or a float"),
        ({"verified_pool": "yes"}, "verified_pool must be None, True or False"),
    ],
)
def test_bad_fields_are_refused(fields: dict, problem: str) -> None:
    with pytest.raises(ValueError, match=problem):
        _neuron(**fields)


def test_a_sets_source_ref_is_its_own_source() -> None:
    with pytest.raises(ValueError, match="the set's own source"):
        _set(source_ref="somewhere else")


def test_tags_given_as_a_list_are_kept_as_a_tuple() -> None:
    assert _neuron(tags=["object", "kitchen"]).tags == ("object", "kitchen")


def test_same_name_ignores_case_and_spacing() -> None:
    assert same_name(" Brass  Owl\t") == same_name("brass owl") == "brass owl"
    assert same_name("Cat") != same_name("Cats")


def test_the_best_of_each_name_is_kept_in_the_inputs_order() -> None:
    weak = _neuron("Teapot", index=1, coherence=6, weirdness=9)
    other = _neuron("Kettle", index=2, coherence=5)
    strong = _neuron("teapot ", index=3, coherence=8, weirdness=2)
    assert drop_duplicates([weak, other, strong]) == [other, strong]


def test_a_verified_persona_beats_a_more_coherent_one() -> None:
    verified = _neuron("Teapot", index=1, coherence=5, verified_pool=True)
    coherent = _neuron("Teapot", index=2, coherence=9, verified_pool=False)
    assert drop_duplicates([coherent, verified]) == [verified]


def test_ties_go_to_weirdness_then_strength_then_key() -> None:
    a = _neuron("Teapot", index=1, coherence=8, weirdness=7)
    b = _neuron("Teapot", index=2, coherence=8, weirdness=8)
    assert drop_duplicates([a, b]) == [b]
    c = _neuron("Teapot", index=3, coherence=8, weirdness=8, strength=1.25)
    assert drop_duplicates([b, c]) == [c]
    d = _neuron("Teapot", index=4, coherence=8, weirdness=8, strength=1.25)
    assert drop_duplicates([d, c]) == [c]  # L9N3- sorts before L9N4-


def test_duplicates_are_only_within_a_source_and_a_size() -> None:
    library = _neuron("pirate")
    top2, top1 = _set("pirate", 2), _set("pirate", 1)
    assert drop_duplicates([library, top2, top1]) == [library, top2, top1]


def test_counts_list_every_source() -> None:
    assert count_by_source([_neuron(), _neuron(index=2), _set()]) == {
        "library": 2,
        "mayner": 1,
    }


def test_a_row_carries_the_licence_and_credit_of_its_source() -> None:
    assert to_row(_neuron())["licence"] == LIBRARY.licence
    row = to_row(_set())
    assert row["credit"] == MAYNER.list_credit and "CC BY 4.0" in row["credit"]
    assert "creativecommons.org/licenses/by/4.0" in row["credit"]
    assert "adapted by pushlab" in row["credit"]
    assert to_row(_neuron())["credit"] == LIBRARY.credit
    assert row["entries"] == "[[3, 12, 0.5], [18, 7, -1.25]]"
    assert row["kind"] == "set" and row["neurons"] == "2" and row["size"] == "96.0"
    assert to_row(_neuron())["entries"] == to_row(_neuron())["size"] == ""


def test_candidates_are_written_and_read_back_units_and_all(tmp_path: Path) -> None:
    written = [
        _neuron(
            tags=("object", "kitchen"),
            coherence=8,
            weirdness=9,
            verified_pool=True,
            category="persona",
            source_url=f"{LIBRARY.url}/library.json",
            downloaded_at="2026-11-12T20:31:05Z",
        ),
        _neuron("Kettle", index=12287, layer=35, sensitive=True),
        _set(
            size_from="vectors/without",
            mayner_in_role=0.6,
            mayner_scored=10,
            mayner_share=21.25,
        ),
    ]
    path = write_candidates(tmp_path / "units" / "candidates.csv", written)
    assert read_candidates(path) == written
    assert not list(path.parent.glob("*.part"))
    with path.open(newline="") as file:
        assert tuple(next(csv.reader(file))) == COLUMNS
    for candidate in read_candidates(path):
        candidate.unit.validate(QWEN3_8B)


def test_a_file_of_other_columns_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "other.csv"
    path.write_text("unit_id,sign\nqwen3-8b:L1:N2,+\n")
    with pytest.raises(ValueError, match="not a candidates file"):
        read_candidates(path)


def test_from_row_undoes_to_row() -> None:
    for candidate in (_neuron(coherence=0, weirdness=10), _set()):
        assert from_row(to_row(candidate)) == candidate


def test_docstring_examples_run() -> None:
    result = doctest.testmod(candidates_module)
    assert result.attempted > 0
    assert result.failed == 0


def test_the_same_candidate_given_twice_is_kept_once() -> None:
    teapot = _neuron()
    assert drop_duplicates([teapot, teapot]) == [teapot]


def test_a_set_with_no_size_is_written_and_read_back(tmp_path: Path) -> None:
    unit = NeuronSet("qwen3-8b", "one-layer", [(4, 7, 1.0), (4, 9, -2.0)], "a test")
    sized_by_layer = _set(unit=unit, source_ref=unit.source)
    path = write_candidates(tmp_path / "candidates.csv", [sized_by_layer])
    assert to_row(sized_by_layer)["size"] == ""
    assert read_candidates(path) == [sized_by_layer]


def test_a_write_that_stops_leaves_no_part_of_a_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(candidate: Candidate) -> dict[str, str]:
        raise OSError("No space left on device")

    monkeypatch.setattr(candidates_module, "to_row", broken)
    out = tmp_path / "units" / "candidates_qwen3-8b.csv"
    with pytest.raises(OSError, match="No space left"):
        write_candidates(out, [_neuron()])
    assert list(out.parent.iterdir()) == []
