"""An experiment file's candidates: setting (pushlab.runs.spec_candidates,
decision 0023): its checks and refusals, and the candidates it chooses from
a list, by filters, by key or by a seeded sample across layers.

The lists are made up (tests/unit/candidate_lists.py): 12 library neurons,
4 in each of layers 3, 9 and 21, then 3 of Mayner's sets.
"""

import doctest
import importlib.util
from collections import Counter
from pathlib import Path

import pytest

from pushlab import PROJECT_ROOT
from pushlab.discovery.candidates import Candidate, write_candidates
from pushlab.runs import spec_candidates
from pushlab.runs.spec_candidates import (
    CandidateKey,
    CandidatesRule,
    Sample,
    candidates_rule,
    chosen_candidates,
    multiples,
    sample_across_layers,
)

# The made-up candidates, loaded by path, as test modules import nothing
# from one another.
_SPEC = importlib.util.spec_from_file_location(
    "candidate_lists", Path(__file__).with_name("candidate_lists.py")
)
lists = importlib.util.module_from_spec(_SPEC)  # type: ignore[arg-type]
_SPEC.loader.exec_module(lists)  # type: ignore[union-attr]

NEURONS = lists.library_neurons(12, [3, 9, 21])
SETS = lists.mayner_sets(3)


@pytest.fixture
def listed(tmp_path: Path) -> Path:
    """The list: the 12 neurons, then the 3 sets."""
    return write_candidates(tmp_path / "candidates.csv", [*NEURONS, *SETS])


def _rule(path: Path, **setting: object) -> CandidatesRule:
    return candidates_rule({"file": str(path), **setting})


def _keys(chosen: list[tuple[Candidate, object]]) -> list[str]:
    return [candidate.key for candidate, _ in chosen]


def test_a_rule_reads_every_key() -> None:
    rule = candidates_rule(
        {
            "file": "units/candidates_qwen3-8b.csv",
            "source": "mayner",
            "kind": "set",
            "keys": [
                "mayner/owl-top10",
                {"key": "mayner/fox-top50", "doses": [0.5, 1]},
            ],
        }
    )
    assert rule == CandidatesRule(
        "units/candidates_qwen3-8b.csv",
        "mayner",
        "set",
        (
            CandidateKey("mayner/owl-top10"),
            CandidateKey("mayner/fox-top50", (0.5, 1.0)),
        ),
    )
    sample = {"count": 50, "across": "layer", "seed": 7}
    rule = candidates_rule({"file": "c.csv", "kind": "neuron", "sample": sample})
    assert rule.sample == Sample(50, 7, "layer") and rule.keys is None
    assert candidates_rule({"file": "c.csv"}) == CandidatesRule("c.csv")


def test_the_list_is_from_the_repository_s_root_unless_absolute(tmp_path: Path) -> None:
    relative = candidates_rule({"file": "units/candidates_qwen3-8b.csv"})
    assert relative.path == PROJECT_ROOT / "units" / "candidates_qwen3-8b.csv"
    assert candidates_rule({"file": str(tmp_path / "c.csv")}).path == tmp_path / "c.csv"


def test_a_key_with_no_doses_of_its_own_is_the_plain_key() -> None:
    plain = candidates_rule({"file": "c.csv", "keys": ["library/L9N1-"]})
    mapped = candidates_rule({"file": "c.csv", "keys": [{"key": "library/L9N1-"}]})
    assert plain == mapped
    assert plain.to_mapping() == {
        "file": "c.csv",
        "source": None,
        "kind": None,
        "keys": [{"key": "library/L9N1-", "doses": None}],
        "sample": None,
    }
    sample = {"count": 2, "across": "layer", "seed": 0}
    written = candidates_rule({"file": "c.csv", "kind": "neuron", "sample": sample})
    assert written.to_mapping()["sample"] == {"count": 2, "seed": 0, "across": "layer"}


SAMPLE = {"count": 5, "across": "layer", "seed": 0}
REFUSALS = {
    "not-a-mapping": ("units/c.csv", "candidates must be a mapping, such as {file:"),
    "unknown-key": (
        {"file": "c.csv", "persona": "owl"},
        "candidates: unknown key 'persona'; it takes file, source, kind, keys, sample",
    ),
    "no-file": ({"source": "library"}, "candidates: file must be the path of a"),
    "file-not-text": ({"file": 3}, "candidates: file must be the path of a"),
    "source": (
        {"file": "c.csv", "source": "zoo"},
        "candidates: source must be library or mayner, not 'zoo'",
    ),
    "source-list": ({"file": "c.csv", "source": ["library"]}, "source must be"),
    "kind": ({"file": "c.csv", "kind": "pair"}, "kind must be neuron or set"),
    "keys-and-sample": (
        {"file": "c.csv", "kind": "neuron", "keys": ["a"], "sample": SAMPLE},
        "candidates: give keys or sample, not both",
    ),
    "keys-not-a-list": ({"file": "c.csv", "keys": "a"}, "keys must list candidates'"),
    "keys-empty": ({"file": "c.csv", "keys": []}, "keys must list candidates'"),
    "key-a-number": (
        {"file": "c.csv", "keys": [12]},
        "candidates: keys[0]: a key must be text, such as library/L9N4242-, but "
        "YAML read 12; put it in quotes",
    ),
    "key-empty": ({"file": "c.csv", "keys": ["a", " "]}, "candidates: keys[1]: a key"),
    "key-twice": (
        {"file": "c.csv", "keys": ["a", {"key": "a", "doses": [2.0]}]},
        "candidates: keys[1]: a is listed twice; give each key once",
    ),
    "key-unknown-key": (
        {"file": "c.csv", "keys": [{"key": "a", "dose": 1.0}]},
        "candidates: keys[0]: unknown key 'dose'; it takes key, doses",
    ),
    "key-doses-not-a-list": (
        {"file": "c.csv", "keys": [{"key": "a", "doses": 1.0}]},
        "candidates: keys[0]: doses must list multiples of the reference dose",
    ),
    "key-dose-zero": (
        {"file": "c.csv", "keys": [{"key": "a", "doses": [0]}]},
        "a dose must be a number above 0, a multiple of the reference dose, not 0",
    ),
    "key-dose-negative": (
        {"file": "c.csv", "keys": [{"key": "a", "doses": [-1.0]}]},
        "a dose must be a number above 0",
    ),
    "key-dose-text": (
        {"file": "c.csv", "keys": [{"key": "a", "doses": ["0.5"]}]},
        "a dose must be a number above 0",
    ),
    "key-dose-infinite": (
        {"file": "c.csv", "keys": [{"key": "a", "doses": [float("inf")]}]},
        "a dose must be a number above 0",
    ),
    "key-dose-true": (
        {"file": "c.csv", "keys": [{"key": "a", "doses": [True]}]},
        "a dose must be a number above 0",
    ),
    "key-doses-repeat": (
        {"file": "c.csv", "keys": [{"key": "a", "doses": [1, 1.0]}]},
        "candidates: keys[0]: doses gives 1.0 twice; give each dose once",
    ),
    "sample-not-a-mapping": (
        {"file": "c.csv", "kind": "neuron", "sample": 50},
        "candidates: sample must be a mapping, such as {count: 50, across: layer, "
        "seed: 0}, not 50",
    ),
    "sample-unknown-key": (
        {"file": "c.csv", "kind": "neuron", "sample": SAMPLE | {"by": "layer"}},
        "candidates: sample: unknown key 'by'",
    ),
    "sample-missing": (
        {"file": "c.csv", "kind": "neuron", "sample": {"count": 5}},
        "candidates: sample: it needs across, seed, as in {count: 50, across: "
        "layer, seed: 0}",
    ),
    "sample-count": (
        {"file": "c.csv", "kind": "neuron", "sample": SAMPLE | {"count": 0}},
        "candidates: sample: count must be a whole number from 1, not 0",
    ),
    "sample-count-true": (
        {"file": "c.csv", "kind": "neuron", "sample": SAMPLE | {"count": True}},
        "count must be a whole number from 1",
    ),
    "sample-across": (
        {"file": "c.csv", "kind": "neuron", "sample": SAMPLE | {"across": "prompt"}},
        "candidates: sample: across must be layer, the spread a sample draws "
        "across, not 'prompt'",
    ),
    "sample-seed": (
        {"file": "c.csv", "kind": "neuron", "sample": SAMPLE | {"seed": -1}},
        "candidates: sample: seed must be a whole number from 0 to",
    ),
    "sample-without-kind": (
        {"file": "c.csv", "sample": SAMPLE},
        "candidates: sample draws neurons across layers, so it needs kind: neuron",
    ),
    "sample-of-sets": (
        {"file": "c.csv", "kind": "set", "sample": SAMPLE},
        "so it needs kind: neuron",
    ),
}


@pytest.mark.parametrize("data, problem", REFUSALS.values(), ids=REFUSALS.keys())
def test_refusals_name_the_problem(data: object, problem: str) -> None:
    with pytest.raises(ValueError) as error:
        candidates_rule(data)
    assert problem in str(error.value)


def test_with_neither_keys_nor_sample_every_row_the_filters_keep_comes_in_order(
    listed: Path,
) -> None:
    every = [c.key for c in [*NEURONS, *SETS]]
    assert _keys(chosen_candidates(_rule(listed), "qwen3-8b")) == every
    library = chosen_candidates(_rule(listed, source="library"), "qwen3-8b")
    assert _keys(library) == every[:12]
    sets = chosen_candidates(_rule(listed, kind="set"), "qwen3-8b")
    assert _keys(sets) == every[12:]
    assert all(doses is None for _, doses in sets)
    with pytest.raises(ValueError) as error:
        chosen_candidates(_rule(listed, source="mayner", kind="neuron"), "qwen3-8b")
    assert str(error.value) == (
        f"candidates: no candidate of {listed} has source mayner and kind neuron"
    )


def test_keys_choose_exactly_those_in_their_order_with_their_own_doses(
    listed: Path,
) -> None:
    keys = [SETS[1].key, {"key": NEURONS[5].key, "doses": [0.5, 1]}, NEURONS[0].key]
    chosen = chosen_candidates(_rule(listed, keys=keys), "qwen3-8b")
    assert chosen == [(SETS[1], None), (NEURONS[5], (0.5, 1.0)), (NEURONS[0], None)]


def test_a_key_not_in_the_list_or_left_out_by_the_filters_is_refused(
    listed: Path,
) -> None:
    with pytest.raises(ValueError) as error:
        chosen_candidates(_rule(listed, keys=["library/L9N4242-"]), "qwen3-8b")
    assert str(error.value) == f"candidates: keys[0]: {listed} has no library/L9N4242-"
    rule = _rule(listed, source="library", keys=[NEURONS[0].key, SETS[0].key])
    with pytest.raises(ValueError) as error:
        chosen_candidates(rule, "qwen3-8b")
    assert str(error.value) == (
        "candidates: keys[1]: mayner/role-0-top2 is of source mayner and kind set, "
        "which source library leaves out"
    )


def _layers(chosen: list[tuple[Candidate, object]]) -> list[int]:
    return [candidate.unit.layer for candidate, _ in chosen]


def test_a_sample_draws_round_by_round_across_the_layers(listed: Path) -> None:
    sample = {"count": 10, "across": "layer", "seed": 0}
    rule = _rule(listed, kind="neuron", sample=sample)
    chosen = chosen_candidates(rule, "qwen3-8b")
    layers = _layers(chosen)
    # Three rounds of one neuron from each layer, in one order, then one more.
    assert sorted(layers[:3]) == [3, 9, 21]
    assert layers[3:6] == layers[:3] and layers[6:9] == layers[:3]
    assert layers[9] == layers[0]
    assert sorted(Counter(layers).values()) == [3, 3, 4]
    assert len(set(_keys(chosen))) == 10 and all(d is None for _, d in chosen)
    assert {candidate.source for candidate, _ in chosen} == {"library"}


def test_a_sample_is_the_same_from_the_same_seed_and_list_and_not_from_another(
    listed: Path, tmp_path: Path
) -> None:
    def keys(path: Path, seed: int) -> list[str]:
        sample = {"count": 7, "across": "layer", "seed": seed}
        rule = _rule(path, kind="neuron", sample=sample)
        return _keys(chosen_candidates(rule, "qwen3-8b"))

    copy = write_candidates(tmp_path / "copy.csv", [*NEURONS, *SETS])
    assert keys(listed, 0) == keys(listed, 0) == keys(copy, 0)
    assert keys(listed, 1) != keys(listed, 0)


def test_a_layer_that_runs_out_leaves_the_rounds_to_the_others() -> None:
    one = lists.neuron(3, 7)
    five = lists.library_neurons(5, [9])
    drawn = sample_across_layers([one, *five], 6, seed=0)
    assert sorted(c.key for c in drawn) == sorted(c.key for c in [one, *five])
    assert [c.unit.layer for c in drawn].count(3) == 1
    later = sample_across_layers([one, *five], 4, seed=0)
    assert [c.unit.layer for c in later].count(3) == 1  # in the first round


def test_a_sample_larger_than_the_filters_leave_is_refused(listed: Path) -> None:
    sample = {"count": 13, "across": "layer", "seed": 0}
    with pytest.raises(ValueError) as error:
        chosen_candidates(_rule(listed, kind="neuron", sample=sample), "qwen3-8b")
    assert str(error.value) == (
        f"candidates: sample: count is 13, but only 12 candidates of {listed} have "
        "kind neuron"
    )


def test_sample_across_layers_refuses_sets_and_a_count_out_of_range() -> None:
    with pytest.raises(ValueError, match="draws neurons, not sets"):
        sample_across_layers([NEURONS[0], SETS[0]], 1, 0)
    for count in (0, 13):
        with pytest.raises(ValueError, match="count must be a whole number from 1 to"):
            sample_across_layers(NEURONS, count, 0)


@pytest.mark.parametrize("where", ["units", "absolute"])
def test_a_missing_list_names_the_command_that_makes_it(
    where: str, tmp_path: Path
) -> None:
    # Named as the experiment names it; no list of this name is anywhere,
    # the pod's clone included.
    file = "units/candidates_no-such-model.csv"
    if where == "absolute":
        file = str(tmp_path / file)
    with pytest.raises(ValueError) as error:
        chosen_candidates(candidates_rule({"file": file}), "qwen3-8b")
    assert str(error.value) == (
        f"candidates: {file} is missing; `uv run pushlab candidates --model "
        f"qwen3-8b --out {file}` makes it on the GPU pod (decision 0020)"
    )


def test_a_list_of_another_model_or_with_a_key_twice_is_refused(
    tmp_path: Path,
) -> None:
    other = write_candidates(tmp_path / "other.csv", [NEURONS[0]])
    with pytest.raises(ValueError) as error:
        chosen_candidates(_rule(other), "qwen3.5-4b")
    assert str(error.value) == (
        f"candidates: {other} holds library/L3N100+, a unit of qwen3-8b, but the "
        "experiment runs qwen3.5-4b; `uv run pushlab candidates --model qwen3.5-4b` "
        "makes a list for it"
    )
    twice = write_candidates(tmp_path / "twice.csv", [NEURONS[0], NEURONS[0]])
    with pytest.raises(ValueError, match="has library/L3N100\\+ on two rows"):
        chosen_candidates(_rule(twice), "qwen3-8b")


def test_a_file_that_is_no_candidates_list_or_holds_none_is_refused(
    tmp_path: Path,
) -> None:
    units = tmp_path / "units.csv"
    units.write_text("unit,direction\nqwen3-8b:L9:N1,-1\n", encoding="utf-8")
    with pytest.raises(ValueError) as error:
        chosen_candidates(_rule(units), "qwen3-8b")
    assert str(error.value).startswith(
        f"candidates: {units} is not a candidates list as pushlab candidates writes "
        "one: "
    )
    # A cell edited by hand: a set's entries that are no list of triples, and
    # a sensitive flag that is neither true nor false.
    for old, new in (('"[[17, 1000, 0.5]', '"[17, [1000, 0.5]'), (",false,", ",no,")):
        broken = write_candidates(tmp_path / "broken.csv", SETS[:1])
        text = broken.read_text(encoding="utf-8")
        assert text.count(old) == 1
        broken.write_text(text.replace(old, new), encoding="utf-8")
        with pytest.raises(ValueError, match="is not a candidates list as pushlab"):
            chosen_candidates(_rule(broken), "qwen3-8b")
    empty = write_candidates(tmp_path / "empty.csv", [])
    with pytest.raises(ValueError, match="no candidate of .*empty.csv$"):
        chosen_candidates(_rule(empty), "qwen3-8b")
    folder = tmp_path / "folder.csv"
    folder.mkdir()
    with pytest.raises(ValueError, match="candidates: cannot read .*folder.csv: "):
        chosen_candidates(_rule(folder), "qwen3-8b")


def test_multiples_are_floats_in_their_order() -> None:
    assert multiples([1.5, 0.5, 1], "") == (1.5, 0.5, 1.0)
    assert all(isinstance(dose, float) for dose in multiples([2], ""))


def test_doctests() -> None:
    result = doctest.testmod(spec_candidates)
    assert result.attempted > 0
    assert result.failed == 0
