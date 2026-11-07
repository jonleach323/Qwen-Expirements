"""expand's random controls: the same timing and doses on other neurons."""

import dataclasses
import hashlib
from collections.abc import Callable
from pathlib import Path

import pytest

from pushlab.models.registry import REGISTRY
from pushlab.runs.conditions import Condition
from pushlab.runs.expand import expand
from pushlab.runs.spec import spec_from_yaml
from pushlab.schedules.model import Schedule, Segment
from pushlab.schedules.yaml_io import from_mapping
from pushlab.units import controls
from pushlab.units.controls import known_units
from pushlab.units.neuron import NeuronUnit

EVIL = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
WARM = NeuronUnit.parse("qwen3.5-4b:L13:N7956")
SpecText = Callable[..., str]


def _segments(stored: list[dict]) -> list[Segment]:
    """The stored segments read back, without the control fields."""
    plain = [
        {k: v for k, v in s.items() if k not in ("control_for", "control_seed")}
        for s in stored
    ]
    return list(from_mapping({"name": "x", "segments": plain}).segments)


def _drawn(expansion: object) -> set[NeuronUnit]:
    return {
        NeuronUnit.parse(segment["unit"])
        for run in expansion.runs  # type: ignore[attr-defined]
        if run.control
        for segment in run.record.schedule
    }


def _known_file(path: Path, units: list[NeuronUnit]) -> Path:
    rows = "".join(f"{unit},+,test,https://example.org\n" for unit in units)
    path.write_text("unit_id,sign,label,sources\n" + rows, encoding="utf-8")
    return path


def test_controls_change_only_the_neurons(spec_text: SpecText) -> None:
    text = """\
conditions:
  - label: handoff
    shape: handoff
    params: {a: qwen3.5-4b:L14:N8949, b: qwen3.5-4b:L13:N7956, alpha_a: 1.0,
             alpha_b: -1.0, switch_at: 16, overlap: 2}
    controls: 5
    control_seed: 3
"""
    expansion = expand(spec_from_yaml(spec_text(text)))
    runs = {run.record.run_id: run for run in expansion.runs}
    known = known_units()
    neurons: dict[NeuronUnit, set[NeuronUnit]] = {EVIL: set(), WARM: set()}
    for run in expansion.runs:
        if run.control == 0:
            continue
        mirrored = runs[run.record.control_of]
        assert mirrored.control == 0 and mirrored.condition == run.condition
        assert (mirrored.record.prompt_id, mirrored.record.seed) == (
            run.record.prompt_id,
            run.record.seed,
        )
        own = _segments(mirrored.record.schedule)
        control = _segments(run.record.schedule)
        swap = {}
        for original, swapped, stored in zip(
            own, control, run.record.schedule, strict=True
        ):
            # The same timing and alphas exactly; only the neuron differs.
            assert dataclasses.replace(swapped, unit=original.unit) == original
            assert swapped.unit.layer == original.unit.layer
            assert swapped.unit not in known and swapped.unit not in (EVIL, WARM)
            assert swap.setdefault(original.unit, swapped.unit) == swapped.unit
            assert stored["control_for"] == str(original.unit)
            assert stored["control_seed"] == 3
        for original, swapped in swap.items():
            neurons[original].add(swapped)
    # Five distinct controls of each unit, the same for every prompt and seed.
    assert {unit: len(drawn) for unit, drawn in neurons.items()} == {EVIL: 5, WARM: 5}


def test_a_seed_always_draws_the_same_neurons(
    spec_text: SpecText, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pinned, so that a change to how controls are drawn cannot pass unseen:
    # it would change the controls of every experiment file.
    known = _known_file(tmp_path / "known.csv", [EVIL, WARM])
    monkeypatch.setattr(controls, "KNOWN_UNITS_FILE", known)
    expansion = expand(spec_from_yaml(spec_text()))
    pushed = {1: set(), 2: set()}
    for run in expansion.runs:
        if run.control:
            pushed[run.control].update(s["unit"] for s in run.record.schedule)
    # The same neuron for every pulse length, prompt and seed.
    assert pushed == {1: {"qwen3.5-4b:L14:N6435"}, 2: {"qwen3.5-4b:L14:N6652"}}


def test_controls_skip_whatever_known_csv_lists(
    spec_text: SpecText, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = expand(spec_from_yaml(spec_text()))
    drawn = sorted(_drawn(first), key=str)
    known = _known_file(tmp_path / "known.csv", drawn)
    monkeypatch.setattr(controls, "KNOWN_UNITS_FILE", known)
    again = expand(spec_from_yaml(spec_text()))
    assert not _drawn(again) & set(drawn)
    # The spec is the same, but the expansion says the known units changed.
    assert again.spec_sha256 == first.spec_sha256
    assert again.known_sha256 == hashlib.sha256(known.read_bytes()).hexdigest()
    assert again.known_sha256 != first.known_sha256


def test_controls_skip_units_that_any_condition_pushes(
    spec_text: SpecText, pulses: str
) -> None:
    drawn = sorted(_drawn(expand(spec_from_yaml(spec_text()))), key=str)
    # Two more conditions push the neurons the pulse's controls drew.
    others = "".join(
        f"  - {{label: other-{i}, segments: [{{unit: {unit}, alpha: 0.5}}]}}\n"
        for i, unit in enumerate(drawn)
    )
    redrawn = _drawn(expand(spec_from_yaml(spec_text(pulses + others))))
    assert len(redrawn) == 2 and not redrawn & set(drawn)


def test_more_controls_than_a_layer_has_neurons_are_refused_at_once(
    spec_text: SpecText, pulses: str
) -> None:
    # Layer 14 has 9,216 neurons, one of them pushed (and known).
    text = pulses.replace("controls: 2", "controls: 9300")
    with pytest.raises(
        ValueError,
        match="condition 'pulse': its 9300 controls need 9300 neurons of layer 14 "
        "besides the known and pushed ones, and the layer has 9215",
    ):
        expand(spec_from_yaml(spec_text(text)))


def test_one_control_too_many_is_named_in_the_singular(
    spec_text: SpecText, pulses: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A model of one neuron a layer, which the condition pushes.
    small = dataclasses.replace(REGISTRY["qwen3.5-4b"], mlp_size=1)
    monkeypatch.setitem(REGISTRY, "qwen3.5-4b", small)
    text = "conditions:\n  - {label: one, segments: [{unit: qwen3.5-4b:L14:N0, "
    text += "alpha: 1.0}], controls: 1, control_seed: 0}\n"
    with pytest.raises(
        ValueError,
        match="condition 'one': its 1 control needs 1 neuron of layer 14 besides "
        "the known and pushed ones, and the layer has 0$",
    ):
        expand(spec_from_yaml(spec_text(text)))


def test_known_units_that_cannot_be_read_are_refused(
    spec_text: SpecText, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(controls, "KNOWN_UNITS_FILE", tmp_path / "gone.csv")
    with pytest.raises(ValueError, match="cannot read the known units: .*gone.csv"):
        expand(spec_from_yaml(spec_text()))


def test_known_units_that_change_while_read_are_refused(
    spec_text: SpecText, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    known = _known_file(tmp_path / "known.csv", [EVIL, WARM])
    monkeypatch.setattr(controls, "KNOWN_UNITS_FILE", known)
    read = controls.known_units

    def read_then_edit(path: Path | None = None) -> frozenset[NeuronUnit]:
        units = read(path)
        with known.open("a", encoding="utf-8") as file:  # a checkout, meanwhile
            file.write("qwen3.5-4b:L14:N1,+,test,https://example.org\n")
        return units

    monkeypatch.setattr(controls, "known_units", read_then_edit)
    with pytest.raises(ValueError, match="known.csv changed while it was read"):
        expand(spec_from_yaml(spec_text()))


def test_a_control_that_cannot_be_drawn_stops_expansion(
    spec_text: SpecText, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_neuron_left(*args: object) -> list[NeuronUnit]:
        raise ValueError("no neuron left in layer 14")

    monkeypatch.setattr(controls, "random_controls", no_neuron_left)
    with pytest.raises(
        ValueError,
        match="condition 'pulse': cannot draw control 1: no neuron left in layer 14",
    ):
        expand(spec_from_yaml(spec_text()))


def test_controls_of_a_condition_that_pushes_nothing_are_refused(
    spec_text: SpecText,
) -> None:
    # A file cannot ask for them (test_spec_refusals.py); a spec built in code can.
    spec = spec_from_yaml(spec_text())
    none = Condition("none", (("none", Schedule("none")),), 1, 0)
    with pytest.raises(ValueError, match="condition 'none': it pushes no unit to draw"):
        expand(dataclasses.replace(spec, conditions=(none,)))
