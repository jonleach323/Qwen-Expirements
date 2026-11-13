"""Sets of neurons in experiment files and run records (pushlab.runs.expand,
pushlab.runs.batch.record_schedule; decision 0023): a run's record keeps
each set's definition, so the record alone makes the set again; a
condition that pushes a set has no random controls yet; the controls of
other conditions avoid its neurons; and a set's ID names one set. The sets
are invented."""

import math
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest

from pushlab.cli import main
from pushlab.runs.batch import record_schedule
from pushlab.runs.expand import expand
from pushlab.runs.spec import ExperimentSpec, spec_from_yaml
from pushlab.store.records import read_runs, write_runs
from pushlab.units import controls
from pushlab.units.neuron import NeuronUnit
from pushlab.units.neuron_set import NeuronSet

CAPTAIN = NeuronSet(
    "qwen3-8b",
    "salt-captain-top2",
    [(17, 1234, 0.1 + 0.2), (18, 99, -0.31)],
    "invented: the salt captain's top 2",
    96.04,
)
DEFINITION = (
    "{entries: [[17, 1234, 0.30000000000000004], [18, 99, -0.31]], size: 96.04, "
    'source: "invented: the salt captain\'s top 2"}'
)
OWL = NeuronUnit("qwen3-8b", 9, 4242)
CAPTAIN_CONDITION = f"""\
  - label: captain
    segments:
      - {{unit: {CAPTAIN}, set: {DEFINITION}, alpha: 1.0, region: prompt}}
      - {{unit: {CAPTAIN}, set: {DEFINITION}, alpha: 1.0}}
"""
CONDITIONS = f"""\
conditions:
  - {{label: none, segments: []}}
{CAPTAIN_CONDITION}\
  - label: owl
    shape: pulse
    params: {{unit: {OWL}, alpha: 1.0, start: 0, length: 4}}
    controls: 1
    control_seed: 0
"""
# The decision's form: each definition in a block of its own.
SET_BLOCK = """\
        set:
          entries:
            - [17, 1234, 0.30000000000000004]
            - [18, 99, -0.31]
          size: 96.04
          source: "invented: the salt captain's top 2"
"""
BLOCK = f"""\
conditions:
  - label: captain
    segments:
      - unit: {CAPTAIN}
{SET_BLOCK}\
        alpha: 1.0
        region: prompt
      - unit: {CAPTAIN}
{SET_BLOCK}\
        alpha: 1
"""


@pytest.fixture
def spec_of(spec_text: Callable[..., str]) -> Callable[..., ExperimentSpec]:
    """The spec of conditions on the two prompts, greedy, on qwen3-8b."""

    def make(conditions: str = CONDITIONS) -> ExperimentSpec:
        text = spec_text(
            conditions, model="qwen3-8b", decoding="{strategy: greedy}", seeds="[0]"
        )
        return spec_from_yaml(text)

    return make


def _bits(unit: NeuronSet) -> list[str]:
    """Each coefficient and the size as the hex of its float: equal hex, equal bits."""
    return [e.coefficient.hex() for e in unit.entries] + [float(unit.size or 0).hex()]


def test_a_runs_record_keeps_its_sets_definition_and_makes_the_set_again(
    spec_of: Callable[..., ExperimentSpec], tmp_path: Path
) -> None:
    spec = spec_of()
    runs = expand(spec).runs
    assert [run.condition for run in runs[:3]] == ["none", "captain", "owl"]
    captain = runs[1].record
    written = {
        "entries": [[17, 1234, 0.1 + 0.2], [18, 99, -0.31]],
        "size": 96.04,
        "source": "invented: the salt captain's top 2",
    }
    assert [segment["set"] for segment in captain.schedule] == [written, written]
    assert [segment["unit"] for segment in captain.schedule] == [str(CAPTAIN)] * 2
    assert all(s["control_for"] is s["control_seed"] is None for s in captain.schedule)
    (_, schedule), *_ = spec.conditions[1].points
    assert record_schedule(captain).segments == schedule.segments
    # Saved and read back, through the runs table's JSON, bit for bit.
    saved = [replace(run.record, status="ok", output_text="Arr.") for run in runs]
    write_runs(saved, tmp_path)
    again = {record.run_id: record for record in read_runs(tmp_path)}
    for run in runs:
        record = again[run.record.run_id]
        assert record.schedule == run.record.schedule
        assert record_schedule(record) == record_schedule(run.record)
    (unit,) = record_schedule(again[captain.run_id]).units()
    assert isinstance(unit, NeuronSet) and unit == CAPTAIN
    assert _bits(unit) == _bits(CAPTAIN)


def test_a_set_written_as_a_block_is_the_same_experiment(
    spec_of: Callable[..., ExperimentSpec],
) -> None:
    flow = "conditions:\n" + CAPTAIN_CONDITION
    assert spec_of(BLOCK).sha256 == spec_of(flow).sha256


def test_the_specs_hash_covers_every_coefficient(
    spec_of: Callable[..., ExperimentSpec],
) -> None:
    nudged = CONDITIONS.replace("-0.31]", "-0.31000000000000005]")
    assert spec_of(nudged).sha256 != spec_of().sha256
    larger = repr(math.nextafter(96.04, 100))  # the next float up
    resized = CONDITIONS.replace("size: 96.04", f"size: {larger}")
    assert spec_of(resized).sha256 != spec_of().sha256


def test_random_controls_for_a_set_are_refused_as_coming_later(
    spec_of: Callable[..., ExperimentSpec],
) -> None:
    text = CONDITIONS.replace(
        f"      - {{unit: {CAPTAIN}, set: {DEFINITION}, alpha: 1.0}}\n",
        f"      - {{unit: {CAPTAIN}, set: {DEFINITION}, alpha: 1.0}}\n"
        "    controls: 2\n    control_seed: 3\n",
    )
    with pytest.raises(ValueError) as error:
        expand(spec_of(text))
    assert str(error.value) == (
        "condition 'captain': it pushes the set of neurons "
        "qwen3-8b:set:salt-captain-top2, and random controls for sets come later "
        "(decision 0019); remove its controls and control_seed"
    )


def test_other_conditions_controls_avoid_every_neuron_of_a_set(
    spec_of: Callable[..., ExperimentSpec],
) -> None:
    # The owl's control when nothing else is in the way: a neuron of layer 9.
    (first,) = controls.random_controls([OWL], 0, controls.known_units() | {OWL})
    assert first.layer == 9
    plain = expand(spec_of()).runs
    assert plain[3].record.schedule[0]["unit"] == str(first)
    # A set that holds that neuron pushes it, so the control is drawn again.
    holder = (
        f"  - label: holder\n    segments: [{{unit: qwen3-8b:set:holder, set: "
        f"{{entries: [[9, {first.index}, 1.0]], size: null, source: x}}, alpha: 1}}]\n"
    )
    runs = expand(spec_of(CONDITIONS + holder)).runs
    (control,) = [run for run in runs[:5] if run.control]
    drawn = control.record.schedule[0]["unit"]
    assert drawn != str(first) and drawn.startswith("qwen3-8b:L9:N")
    assert control.record.schedule[0]["control_for"] == str(OWL)


def test_a_set_id_given_two_definitions_in_an_experiment_is_refused(
    spec_of: Callable[..., ExperimentSpec],
) -> None:
    other = DEFINITION.replace("-0.31", "-0.25")
    twice = (
        f"  - label: captain-again\n    segments: [{{unit: {CAPTAIN}, set: {other}, "
        "alpha: 0.5}]\n"
    )
    with pytest.raises(ValueError) as error:
        expand(spec_of(CONDITIONS + twice))
    assert str(error.value) == (
        "condition 'captain-again': qwen3-8b:set:salt-captain-top2 is defined "
        "otherwise in condition 'captain'; a set's ID names one set, so give it one "
        "definition, or the other set a name of its own"
    )
    same = twice.replace(other, DEFINITION)  # the same set again is fine
    assert len(expand(spec_of(CONDITIONS + same)).runs) == 10


def test_pushlab_expand_counts_a_set_experiments_runs(
    spec_text: Callable[..., str],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "sets.yaml"
    text = spec_text(
        CONDITIONS, model="qwen3-8b", decoding="{strategy: greedy}", seeds="[0]"
    )
    path.write_text(text, encoding="utf-8")
    assert main(["expand", str(path), "--summary"]) == 0
    out = capsys.readouterr().out
    assert "8 runs, of which 2 are controls" in " ".join(out.split())
