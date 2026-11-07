"""expand: an experiment spec into the exact list of runs it means.

There is one run per prompt, seed, condition (each grid value its own) and
control, in that order, nested as written: every run of the first prompt,
its first seed first, then the next. Within a seed come the conditions in
the file's order, each condition's own run followed by its controls. Run
IDs count up in that order, `<experiment>-000001` first, so the same spec
always gives the same IDs.

Before any run is made, every condition's schedule and every control's
passes validate(), at the default dose ceiling unless the condition allows
more; the first that fails stops expansion with a message naming it.

A condition's random controls are copies of its schedules with each unit
swapped for a neuron of the same layer, at the same doses on the same
passes. They are drawn from the condition's control_seed and never land on
a unit in units/known.csv, a unit any condition of the experiment pushes,
or an earlier control of the condition. Control k pushes the same neurons
for every prompt, seed and grid value, so the controls match the condition
point for point, and its runs' control_of names the run they mirror.

Runs are batched in order: each batch holds runs of one prompt, so no row
is padded, and one seed, since generate draws a batch's samples from one
seed; at most batch_size of them. batch_key is `b<batch>:<row>`.

Each run is a RunRecord with no reply yet: status "planned", which
write_runs refuses, empty output, the model's revision, the run's
sampling seed, and its schedule's segments, each with control_for (the
unit a control replaces) and control_seed, both None for a condition's own
run. The condition's label, which no RunRecord field holds, comes with it.
"""

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, NamedTuple

from pushlab.models.registry import REGISTRY, ModelSpec
from pushlab.prompts import read_prompt_set
from pushlab.runs.spec import Condition, ExperimentSpec
from pushlab.schedules.model import Schedule
from pushlab.schedules.validation import ScheduleWarning, validate
from pushlab.schedules.yaml_io import to_mapping
from pushlab.store.records import RunRecord, make_run_id
from pushlab.units import controls
from pushlab.units.neuron import NeuronUnit

# The status of a run that has not run yet; write_runs saves only results.
PLANNED = "planned"


class PlannedRun(NamedTuple):
    """One run of an expansion: its condition, its control number and its record.

    control is 0 for the condition's own run and k for its k-th control.
    """

    condition: str
    control: int
    record: RunRecord


@dataclass(frozen=True)
class Expansion:
    """Every run an experiment means, with the hashes of what they came from.

    spec_sha256 is the spec's hash and prompts_sha256 the prompt set file's.
    warnings are validate()'s, each naming its condition.
    """

    spec_sha256: str
    prompts_sha256: str
    runs: tuple[PlannedRun, ...]
    warnings: tuple[str, ...]

    def counts(self) -> dict[str, tuple[int, int]]:
        """For each condition, in order: how many own runs and how many controls.

        Example:
            >>> from pushlab import PROJECT_ROOT
            >>> from pushlab.runs.spec import read_spec
            >>> spec = read_spec(PROJECT_ROOT / "experiments/p0/s2_pulse_lengths.yaml")
            >>> expand(spec).counts()["pulse length=16"]
            (5, 10)
        """
        counts: dict[str, list[int]] = {}
        for run in self.runs:
            counts.setdefault(run.condition, [0, 0])[run.control > 0] += 1
        return {condition: (own, drawn) for condition, (own, drawn) in counts.items()}


def _checked(
    label: str, schedule: Schedule, model: ModelSpec, tokens: int
) -> list[str]:
    """validate()'s warnings for a condition's schedule, each naming it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ScheduleWarning)  # returned instead
        try:
            found = validate(schedule, model, tokens)
        except ValueError as error:
            raise ValueError(f"condition {label!r}: {error}") from None
    return [f"condition {label!r}: {warning}" for warning in found]


def _draw(
    condition: Condition, avoid: frozenset[NeuronUnit]
) -> list[dict[NeuronUnit, NeuronUnit]]:
    """Each of the condition's controls, as the neuron that replaces each unit."""
    units = list(
        dict.fromkeys(u for _, schedule in condition.points for u in schedule.units())
    )
    if not units:
        raise ValueError(
            f"condition {condition.label!r}: it pushes no unit to draw a control "
            "for; remove its controls"
        )
    taken = set(avoid)
    drawn = []
    for k in range(1, condition.controls + 1):
        try:
            neurons = controls.random_controls(units, condition.control_seed, taken)
        except ValueError as error:
            raise ValueError(
                f"condition {condition.label!r}: cannot draw control {k}: {error}"
            ) from None
        taken.update(neurons)
        drawn.append(dict(zip(units, neurons, strict=True)))
    return drawn


def _swapped(schedule: Schedule, swap: Mapping[NeuronUnit, NeuronUnit]) -> Schedule:
    segments = [replace(s, unit=swap[s.unit]) for s in schedule.segments]
    return Schedule(schedule.name, segments, schedule.allow_over_ceiling)


def _stored(
    schedule: Schedule,
    swap: Mapping[NeuronUnit, NeuronUnit] | None,
    control_seed: int | None,
) -> list[dict[str, Any]]:
    """The schedule's segments as a run stores them, swapped for a control."""
    stored = []
    for written, segment in zip(
        to_mapping(schedule)["segments"], schedule.segments, strict=True
    ):
        if swap is None:
            stored.append(written | {"control_for": None, "control_seed": None})
        else:
            stored.append(
                written
                | {
                    "unit": str(swap[segment.unit]),
                    "control_for": str(segment.unit),
                    "control_seed": control_seed,
                }
            )
    return stored


def _decoding(spec: ExperimentSpec) -> dict[str, Any]:
    """A run's decoding record: S1's four keys, and sampling's settings."""
    decoding = {
        "strategy": spec.decoding["strategy"],
        "max_new_tokens": spec.max_new_tokens,
        "thinking": False,
        "dtype": spec.decoding["dtype"],
    }
    if spec.decoding["strategy"] == "sample":
        decoding |= {k: spec.decoding[k] for k in ("temperature", "top_p")}
    return decoding


def expand(spec: ExperimentSpec) -> Expansion:
    """Every run the spec means, in order, after checking every condition.

    A condition that validate() refuses, controls that cannot be drawn, or
    a prompt set that cannot be read raise ValueError, before any run is
    made. Nothing loads a model.

    Example:
        >>> from pushlab import PROJECT_ROOT
        >>> from pushlab.runs.spec import read_spec
        >>> spec = read_spec(PROJECT_ROOT / "experiments/p0/s2_pulse_lengths.yaml")
        >>> runs = expand(spec).runs
        >>> len(runs), runs[0].record.run_id, runs[0].condition
        (50, 's2-pulse-lengths-000001', 'none')
        >>> runs[3].condition, runs[3].control, runs[3].record.control_of
        ('pulse length=4', 2, 's2-pulse-lengths-000002')
    """
    model = REGISTRY[spec.model]
    found: list[str] = []
    for condition in spec.conditions:
        for label, schedule in condition.points:
            found += _checked(label, schedule, model, spec.max_new_tokens)
    swaps: dict[str, list[dict[NeuronUnit, NeuronUnit]]] = {}
    if any(condition.controls for condition in spec.conditions):
        try:
            known = controls.known_units()
        except (OSError, ValueError) as error:
            raise ValueError(f"cannot read the known units: {error}") from None
        pushed = {
            unit
            for condition in spec.conditions
            for _, schedule in condition.points
            for unit in schedule.units()
        }
        for condition in spec.conditions:
            if condition.controls:
                swaps[condition.label] = _draw(condition, known | pushed)
                for label, schedule in condition.points:
                    for swap in swaps[condition.label]:
                        # The same passes as the condition's, so the same
                        # warnings, which are already found.
                        _checked(
                            label, _swapped(schedule, swap), model, spec.max_new_tokens
                        )
    path = spec.prompts_path
    try:
        prompt_set = read_prompt_set(path)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read the prompt set {path}: {error}") from None

    model_name = f"{spec.model}@{model.revision}"
    runs: list[PlannedRun] = []
    batch = 0
    for prompt_id in prompt_set.prompt_ids:
        for seed in spec.seeds:
            # (condition, control, stored schedule, control_of) of each run.
            group: list[tuple[str, int, list[dict[str, Any]], str]] = []
            for condition in spec.conditions:
                for label, schedule in condition.points:
                    own = make_run_id(spec.name, len(runs) + len(group) + 1)
                    group.append((label, 0, _stored(schedule, None, None), ""))
                    for k, swap in enumerate(swaps.get(condition.label, []), 1):
                        stored = _stored(schedule, swap, condition.control_seed)
                        group.append((label, k, stored, own))
            for first in range(0, len(group), spec.batch_size):
                batch += 1
                rows = group[first : first + spec.batch_size]
                for row, (label, control, stored, control_of) in enumerate(rows):
                    record = RunRecord(
                        run_id=make_run_id(spec.name, len(runs) + 1),
                        experiment=spec.name,
                        model=model_name,
                        prompt_id=prompt_id,
                        schedule=stored,
                        decoding=_decoding(spec),
                        seed=seed,
                        batch_key=f"b{batch:06d}:{row}",
                        control_of=control_of,
                        output_text="",
                        output_tokens=[],
                        status=PLANNED,
                    )
                    runs.append(PlannedRun(label, control, record))
    return Expansion(spec.sha256, prompt_set.sha256, tuple(runs), tuple(found))
