"""expand: an experiment spec into the exact list of runs it means.

There is one run per prompt, seed, condition (each grid value its own) and
control, in that order, nested as written: every run of the first prompt,
its first seed first, then the next. Within a seed come the conditions in
the file's order, each condition's own run followed by its controls. Run
IDs count up in that order, `<experiment>-000001` first, so the same spec
always gives the same IDs. An experiment of more than 999,999 runs, which
six digits cannot number, is refused.

Before any run is made, every condition's schedule and every control's
passes validate(), at the default dose ceiling unless the condition allows
more; the first that fails stops expansion with a message naming it.

A condition's random controls are copies of its schedules with each unit
swapped for a neuron of the same layer, at the same doses on the same
passes. They are drawn from the condition's control_seed and never land on
a unit in units/known.csv, a neuron any condition of the experiment pushes,
alone or in a set, or an earlier control of the condition. Control k pushes
the same neurons for every prompt, seed and grid value, so the controls
match the condition point for point, and its runs' control_of names the run
they mirror. A condition that pushes a set of neurons has no controls yet
(decision 0019), and a set's ID names one set in the whole experiment.

Runs are batched in order. A batch holds runs of one prompt, so no row is
padded, and one seed, since generate draws a batch's samples from one
seed; at most batch_size of them. A run and its controls stay in one
batch, which shapes their arithmetic alike: when they do not fit in what
is left of a batch, they start the next. When they are more than a batch
holds, they are split, with a warning. batch_key is `b<batch>:<row>`.
generate seeds each row's sampling from its place in the batch, so under
sampling a control in another row draws other random numbers than the run
it mirrors.

Each run is a RunRecord with no reply yet: status "planned", which
write_runs refuses, empty output, the model's revision, the run's
sampling seed, and its schedule's segments as a schedule file writes them
(a set's with its definition, so the record alone makes the set again),
each with control_for (the unit a control replaces) and control_seed, both
None for a condition's own run. The label of its condition's schedule,
which no RunRecord field holds, comes with it.
"""

import hashlib
import warnings
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, NamedTuple, TypeVar

from pushlab.models.registry import REGISTRY, ModelSpec
from pushlab.prompts import read_prompt_set
from pushlab.runs.conditions import MOST_RUNS
from pushlab.runs.spec import Condition, ExperimentSpec
from pushlab.schedules.model import Schedule
from pushlab.schedules.validation import ScheduleWarning, validate
from pushlab.schedules.yaml_io import to_mapping
from pushlab.schedules.yaml_sets import refuse_two_definitions
from pushlab.store.records import RunRecord, make_run_id
from pushlab.units import controls
from pushlab.units.neuron import NeuronUnit
from pushlab.units.neuron_set import NeuronSet, unit_neurons

# The status of a run that has not run yet; write_runs saves only results.
PLANNED = "planned"

T = TypeVar("T")


class PlannedRun(NamedTuple):
    """One run of an expansion: its condition, its control number and its record.

    condition is the label of the run's schedule: its condition's label,
    with the grid's values when it has a grid, such as "pulse length=16".
    control is 0 for the condition's own run and k for its k-th control.
    """

    condition: str
    control: int
    record: RunRecord


@dataclass(frozen=True)
class Expansion:
    """Every run an experiment means, with the hashes of what they came from.

    spec_sha256 is the spec's hash and prompts_sha256 the prompt set file's.
    known_sha256 is units/known.csv's, which the controls avoid, or None
    when no condition has controls. warnings are validate()'s, each naming
    its condition's schedule, and one for each condition whose runs and
    controls are split over batches.
    """

    spec_sha256: str
    prompts_sha256: str
    known_sha256: str | None
    runs: tuple[PlannedRun, ...]
    warnings: tuple[str, ...]

    def counts(self) -> dict[str, tuple[int, int]]:
        """For each schedule's label, in order: how many own runs and controls.

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
    if sets := [unit for unit in units if isinstance(unit, NeuronSet)]:
        raise ValueError(
            f"condition {condition.label!r}: it pushes the set of neurons "
            f"{sets[0]}, and random controls for sets come later (decision 0019); "
            "remove its controls and control_seed"
        )
    taken = set(avoid) | set(units)
    n = condition.controls
    for (name, layer), count in Counter((u.model_name, u.layer) for u in units).items():
        size = REGISTRY[name].mlp_size
        used = {
            t.index
            for t in taken
            if (t.model_name, t.layer) == (name, layer) and t.index < size
        }
        if n * count > size - len(used):  # refused now, not after drawing all
            what = "its 1 control needs" if n == 1 else f"its {n} controls need"
            neurons = "1 neuron" if n * count == 1 else f"{n * count} neurons"
            raise ValueError(
                f"condition {condition.label!r}: {what} {neurons} of layer {layer} "
                f"besides the known and pushed ones, and the layer has "
                f"{size - len(used)}"
            )
    drawn = []
    for k in range(1, n + 1):
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


def _known() -> tuple[frozenset[NeuronUnit], str]:
    """The units in units/known.csv, and the hash of the file they came from."""
    path = controls.KNOWN_UNITS_FILE
    try:
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        units = controls.known_units()
        after = hashlib.sha256(path.read_bytes()).hexdigest()
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read the known units: {error}") from None
    if after != before:
        raise ValueError(f"{path} changed while it was read; expand again")
    return units, before


def _pack(sets: Sequence[Sequence[T]], size: int) -> list[list[T]]:
    """Consecutive sets of runs in batches of at most size, in order.

    A set that does not fit in what is left of a batch starts the next one,
    and one larger than size is cut into batches of size, its last part
    left open for the sets after it.

    Example:
        >>> _pack([[1, 2, 3], [4, 5, 6], [7]], 5)
        [[1, 2, 3], [4, 5, 6, 7]]
        >>> _pack([[1], [2, 3, 4, 5, 6], [7]], 2)
        [[1], [2, 3], [4, 5], [6, 7]]
    """
    batches: list[list[T]] = []
    batch: list[T] = []
    for runs in sets:
        runs = list(runs)
        if batch and len(batch) + len(runs) > size:
            batches.append(batch)
            batch = []
        while len(runs) > size:
            batches.append(runs[:size])
            runs = runs[size:]
        batch += runs
        if len(batch) == size:
            batches.append(batch)
            batch = []
    if batch:
        batches.append(batch)
    return batches


def expand(spec: ExperimentSpec) -> Expansion:
    """Every run the spec means, in order, after checking every condition.

    A prompt set that cannot be read, more runs than MOST_RUNS, a set's ID
    given two definitions, a condition that validate() refuses, or controls
    that cannot be drawn, such as a set's, raise ValueError, before any run
    is made. Nothing loads a model.

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
    path = spec.prompts_path
    try:
        prompt_set = read_prompt_set(path)
    except OSError as error:
        why = error.strerror or error
        raise ValueError(f"cannot read the prompt set {path}: {why}") from None
    except ValueError as error:
        raise ValueError(f"cannot read the prompt set {path}: {error}") from None
    per_seed = sum(len(c.points) * (1 + c.controls) for c in spec.conditions)
    total = len(prompt_set.prompt_ids) * len(spec.seeds) * per_seed
    if total > MOST_RUNS:
        raise ValueError(
            f"it means {total:,} runs, more than the {MOST_RUNS:,} that six-digit "
            "run IDs can number; split it into smaller experiments"
        )
    refuse_two_definitions(
        (f"condition {label!r}", unit)
        for condition in spec.conditions
        for label, schedule in condition.points
        for unit in schedule.units()
    )
    found: list[str] = []
    for condition in spec.conditions:
        for label, schedule in condition.points:
            found += _checked(label, schedule, model, spec.max_new_tokens)
    swaps: dict[str, list[dict[NeuronUnit, NeuronUnit]]] = {}
    known_sha256 = None
    if any(condition.controls for condition in spec.conditions):
        known, known_sha256 = _known()
        pushed = {  # every neuron pushed, a set's each
            neuron
            for condition in spec.conditions
            for _, schedule in condition.points
            for unit in schedule.units()
            for neuron, _ in unit_neurons(unit)
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
            if 1 + condition.controls > spec.batch_size:
                found.append(
                    f"condition {condition.label!r}: a run and its "
                    f"{condition.controls} controls are {1 + condition.controls} "
                    f"runs, more than a batch of {spec.batch_size}, so they are split "
                    "over batches of other shapes, which can round them differently"
                )

    model_name = f"{spec.model}@{model.revision}"
    runs: list[PlannedRun] = []
    batch = 0
    for prompt_id in prompt_set.prompt_ids:
        for seed in spec.seeds:
            # Each run with its controls: (condition, control, schedule, control_of).
            sets: list[list[tuple[str, int, list[dict[str, Any]], str]]] = []
            planned = len(runs)
            for condition in spec.conditions:
                for label, schedule in condition.points:
                    own = make_run_id(spec.name, planned + 1)
                    together = [(label, 0, _stored(schedule, None, None), "")]
                    for k, swap in enumerate(swaps.get(condition.label, []), 1):
                        stored = _stored(schedule, swap, condition.control_seed)
                        together.append((label, k, stored, own))
                    sets.append(together)
                    planned += len(together)
            for rows in _pack(sets, spec.batch_size):
                batch += 1
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
    return Expansion(
        spec.sha256, prompt_set.sha256, known_sha256, tuple(runs), tuple(found)
    )
