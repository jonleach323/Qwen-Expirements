"""experiments/s2/smoke_dynamics.yaml, the first dynamics experiment.

It expands to 31 conditions, its 16 and 15 matched random controls, times
its 32 neutral prompts: 992 runs, every one passing validation. Each
condition pushes what its label says on every pass: the prompt pass
(None) and reply tokens 0 to 95.
"""

import warnings
from collections import Counter

import pytest

from pushlab import PROJECT_ROOT
from pushlab.models.registry import REGISTRY
from pushlab.prompts import PROMPTS_DIR, read_prompt_set
from pushlab.runs.batch import record_schedule
from pushlab.runs.expand import Expansion, expand
from pushlab.runs.spec import read_spec
from pushlab.schedules.validation import validate
from pushlab.units.controls import known_units
from pushlab.units.neuron import NeuronUnit

SPEC = PROJECT_ROOT / "experiments" / "s2" / "smoke_dynamics.yaml"
E, W = "qwen3.5-4b:L14:N8949", "qwen3.5-4b:L13:N7956"
LABELS = [
    "baseline",
    "constant-e",
    "constant-w",
    "pulse-e-4",
    "pulse-e-16",
    "pulse-e-64",
    "onset-e-10",
    "onset-e-30",
    "handoff-e-w-10",
    "handoff-e-w-30",
    "handoff-w-e-10",
    "handoff-w-e-30",
    "alternate-e-w-2",
    "alternate-e-w-8",
    "alternate-e-w-32",
    "ramp-e-0-32",
]
PASSES = [None, *range(96)]  # the prompt pass, then reply tokens 0 to 95


@pytest.fixture(scope="module")
def expansion() -> Expansion:
    return expand(read_spec(SPEC))


def test_it_expands_to_31_conditions_times_32_prompts(expansion: Expansion) -> None:
    assert len(expansion.runs) == 992
    groups = Counter((run.condition, run.control) for run in expansion.runs)
    assert len(groups) == 31
    assert set(groups.values()) == {32}  # every prompt in every group
    assert list(dict.fromkeys(run.condition for run in expansion.runs)) == LABELS
    controlled = {run.condition for run in expansion.runs if run.control}
    assert controlled == set(LABELS) - {"baseline"}
    assert {run.control for run in expansion.runs} == {0, 1}  # one control each
    prompts = {run.record.prompt_id for run in expansion.runs}
    assert prompts == {f"neutral32-{n:02d}" for n in range(1, 33)}


def test_its_settings_are_the_ones_planned() -> None:
    spec = read_spec(SPEC)
    assert spec.name == "s2-smoke-dynamics"
    assert spec.model == "qwen3.5-4b"
    assert spec.prompts_path == PROMPTS_DIR / "neutral32.jsonl"
    assert spec.decoding["strategy"] == "greedy"
    assert spec.seeds == (0,)
    assert spec.max_new_tokens == 96
    assert spec.batch_size == 16
    assert spec.max_gpu_hours == 1


def test_every_schedule_passes_validation_without_a_warning(
    expansion: Expansion,
) -> None:
    model = REGISTRY["qwen3.5-4b"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a warning would fail the test
        for run in expansion.runs:
            assert validate(record_schedule(run.record), model, 96) == []


def _planned(label: str, token: int | None) -> dict[str, float]:
    """What a condition's own run pushes on the pass of token, by unit."""
    reply = token is not None
    if label == "baseline":
        return {}
    if label.startswith("constant-"):
        unit, dose = (E, 1.0) if label.endswith("-e") else (W, -1.0)
        return {unit: dose}
    if label.startswith("pulse-e-"):
        length = int(label.rsplit("-", 1)[1])
        return {E: 1.0} if reply and token < length else {}
    if label.startswith("onset-e-"):
        start = int(label.rsplit("-", 1)[1])
        return {E: 1.0} if reply and token >= start else {}
    if label.startswith("handoff-"):
        first, second = (E, W) if label.startswith("handoff-e-w") else (W, E)
        switch = int(label.rsplit("-", 1)[1])
        unit = first if not reply or token < switch else second
        return {unit: 1.0 if unit == E else -1.0}
    if label.startswith("alternate-e-w-"):
        period = int(label.rsplit("-", 1)[1])
        if not reply:
            return {}
        return {E: 1.0} if (token // period) % 2 == 0 else {W: -1.0}
    assert label == "ramp-e-0-32"
    return {E: token / 31} if reply and token < 32 else {}


@pytest.mark.parametrize("label", LABELS)
def test_each_condition_pushes_what_its_label_says(
    expansion: Expansion, label: str
) -> None:
    own = next(r for r in expansion.runs if r.condition == label and not r.control)
    schedule = record_schedule(own.record)
    for token in PASSES:
        doses = {str(unit): dose for unit, dose in schedule.doses_at(token).items()}
        want = _planned(label, token)
        assert doses.keys() == want.keys(), (label, token)
        for unit, dose in want.items():
            assert doses[unit] == pytest.approx(dose, abs=1e-12), (label, token)


def test_each_control_mirrors_its_run_on_two_fixed_random_neurons(
    expansion: Expansion,
) -> None:
    own = {
        (run.condition, run.record.prompt_id): run
        for run in expansion.runs
        if not run.control
    }
    stand_ins: dict[str, set[str]] = {E: set(), W: set()}
    avoid = {str(unit) for unit in known_units()} | {E, W}
    for run in (run for run in expansion.runs if run.control):
        mirrored = own[(run.condition, run.record.prompt_id)]
        assert run.record.control_of == mirrored.record.run_id
        theirs, mine = record_schedule(mirrored.record), record_schedule(run.record)
        assert len(theirs.segments) == len(mine.segments)
        for real, control, stored in zip(
            theirs.segments, mine.segments, run.record.schedule, strict=True
        ):
            assert stored["control_for"] == str(real.unit)
            assert control.unit.layer == real.unit.layer  # the same layer
            assert str(control.unit) not in avoid  # not E, W or a known unit
            # The same doses on the same passes: only the neuron differs.
            assert (control.alpha_start, control.alpha_end) == (
                real.alpha_start,
                real.alpha_end,
            )
            assert (control.start, control.end, control.region) == (
                real.start,
                real.end,
                real.region,
            )
            assert (control.on_for, control.off_for) == (real.on_for, real.off_for)
            stand_ins[str(real.unit)].add(str(control.unit))
    # One control_seed: every condition's control uses the same two neurons.
    assert all(len(neurons) == 1 for neurons in stand_ins.values())
    assert {NeuronUnit.parse(n).layer for n in stand_ins[E]} == {14}
    assert {NeuronUnit.parse(n).layer for n in stand_ins[W]} == {13}


def test_the_prompt_set_is_32_new_neutral_prompts() -> None:
    neutral = read_prompt_set(PROMPTS_DIR / "neutral32.jsonl")
    assert neutral.version == 1
    assert list(neutral.prompt_ids) == [f"neutral32-{n:02d}" for n in range(1, 33)]
    assert len(set(neutral.prompts)) == 32
    others = set()
    for name in ("calibration.jsonl", "smoke.jsonl"):
        others |= {p.casefold() for p in read_prompt_set(PROMPTS_DIR / name).prompts}
    assert not {p.casefold() for p in neutral.prompts} & others
