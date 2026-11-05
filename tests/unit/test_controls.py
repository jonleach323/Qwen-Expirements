"""Random control neurons and the known units they avoid: pushlab.units.controls."""

import doctest
from pathlib import Path

import pytest

from pushlab.units import controls
from pushlab.units.controls import known_units, random_controls
from pushlab.units.neuron import NeuronUnit

EVIL = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
TINY = NeuronUnit("tiny", layer=1, index=5)  # 2 layers of 128 neurons


def test_known_units_come_from_units_known_csv() -> None:
    known = known_units()
    assert EVIL in known
    assert NeuronUnit.parse("qwen3.5-4b:L13:N7956") in known


def test_a_known_units_file_is_read_as_given(tmp_path: Path) -> None:
    path = tmp_path / "known.csv"
    path.write_text("unit_id,sign,label,sources\ntiny:L1:N7,+,test,https://x\n")
    assert known_units(path) == frozenset({NeuronUnit("tiny", 1, 7)})
    path.write_text("unit_id,sign,label,sources\ntiny:L01:N7,+,test,https://x\n")
    with pytest.raises(ValueError, match="known.csv"):
        known_units(path)


def test_the_same_seed_draws_the_same_neurons_for_good() -> None:
    # Every random control ever run depends on these draws.
    assert random_controls([EVIL], seed=0, avoid=()) == [
        NeuronUnit("qwen3.5-4b", 14, 1929)
    ]
    assert random_controls([EVIL], seed=7, avoid=()) == [
        NeuronUnit("qwen3.5-4b", 14, 6435)
    ]
    assert random_controls([EVIL], seed=7, avoid=()) == random_controls(
        [EVIL], seed=7, avoid=()
    )


def test_a_control_stays_in_its_units_model_and_layer() -> None:
    units = [EVIL, NeuronUnit.parse("qwen3.5-4b:L13:N7956"), TINY]
    for seed in range(20):
        for unit, control in zip(units, random_controls(units, seed, ()), strict=True):
            assert (control.model_name, control.layer) == (unit.model_name, unit.layer)
            assert control != unit


def test_a_control_skips_known_units_the_units_and_other_controls() -> None:
    # Seed 0 draws neurons 64, 8, 88, 118, ... of tiny's layer 1, in turn.
    assert random_controls([TINY], seed=0, avoid=()) == [NeuronUnit("tiny", 1, 64)]
    skip = {NeuronUnit("tiny", 1, 64), NeuronUnit("tiny", 1, 8)}
    assert random_controls([TINY], seed=0, avoid=skip) == [NeuronUnit("tiny", 1, 88)]
    # Two units of one layer never share a control, and none is a unit.
    pair = [TINY, NeuronUnit("tiny", 1, 64)]
    first, second = random_controls(pair, seed=0, avoid=())
    assert first != second and {first, second}.isdisjoint(pair)
    # With every other neuron of the layer known, only one is left.
    others = {NeuronUnit("tiny", 1, n) for n in range(128) if n not in (5, 77)}
    assert random_controls([TINY], seed=3, avoid=others) == [NeuronUnit("tiny", 1, 77)]


def test_a_layer_with_no_neuron_left_is_refused() -> None:
    every = {NeuronUnit("tiny", 1, n) for n in range(128)}
    with pytest.raises(ValueError, match="no neuron left"):
        random_controls([TINY], seed=0, avoid=every)


@pytest.mark.parametrize("seed", [-1, 2**63, 1.0, True, None])
def test_a_seed_must_be_a_whole_number_from_0(seed: object) -> None:
    with pytest.raises(ValueError, match="seed"):
        random_controls([TINY], seed=seed, avoid=())  # type: ignore[arg-type]


def test_docstring_examples_run() -> None:
    result = doctest.testmod(controls)
    assert result.attempted > 0
    assert result.failed == 0
