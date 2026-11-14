"""Conditions made by rule: one per candidate of a candidates list.

A condition with each_candidate (decision 0023) pushes the candidates an
experiment's candidates: setting chooses (pushlab.runs.spec_candidates):

    - label: screen
      each_candidate: {doses: [1.0], include_prompt: true,
                       allow_over_ceiling: false}

It becomes one condition per candidate, labelled <label>-0001,
<label>-0002 and so on in the selection's order (wider past 9,999): a
constant push (pushlab.schedules.shapes.constant) of the candidate's unit
at alpha = direction x strength x each of doses, multiples of the
candidate's reference dose, the source's strength (default [1.0]; a key's
own doses replace them), on the prompt pass too unless include_prompt is
false, and above the dose ceiling only with allow_over_ceiling. A
candidate with several doses gets a grid labelled by alpha, as pushlab
ladder's u01 alpha=-0.75; one with one dose, its label alone. Each such
condition records its candidate's key, and a set's schedule holds the set
itself, so it is written as segments with its definition. controls and
control_seed, if given, go to every condition it makes.

An experiment has at most one condition with each_candidate, and only with
candidates; candidates with no such condition is refused too.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from pushlab.discovery.candidates import Candidate  # noqa: F401 (the examples use it)
from pushlab.runs.conditions import (
    MOST_RUNS,
    Condition,
    condition_controls,
    condition_from_mapping,
    condition_label,
    refuse_unknown_keys,
)
from pushlab.runs.spec_candidates import (
    EXAMPLE,
    CandidatesRule,
    Chosen,
    chosen_candidates,
    multiples,
)
from pushlab.schedules.model import Schedule
from pushlab.schedules.shapes import constant
from pushlab.units.neuron import NeuronUnit  # noqa: F401 (the examples use it)

EACH_CANDIDATE_KEYS = ("doses", "include_prompt", "allow_over_ceiling")
# A condition with each_candidate has no schedule of its own.
EACH_CONDITION_KEYS = ("label", "each_candidate", "controls", "control_seed")
# Each candidate at its reference dose: the source's strength.
DEFAULT_DOSES = (1.0,)
# The least number of digits in a candidate's label: <label>-0001.
LABEL_DIGITS = 4


@dataclass(frozen=True)
class EachCandidate:
    """A condition with each_candidate, as each_candidate_from_mapping
    checks it; place is its index in the experiment's conditions.

    Example:
        >>> each = each_candidate_from_mapping(
        ...     {"label": "screen", "each_candidate": {"doses": [0.5, 1]}}, 0
        ... )
        >>> each.doses, each.include_prompt, each.allow_over_ceiling
        ((0.5, 1.0), True, False)
    """

    label: str
    place: int
    doses: tuple[float, ...] = DEFAULT_DOSES
    include_prompt: bool = True
    allow_over_ceiling: bool = False
    controls: int = 0
    control_seed: int | None = None

    def conditions(self, chosen: Sequence[Chosen]) -> tuple[Condition, ...]:
        """One condition per chosen candidate, labelled <label>-0001 on.

        Each is a constant push of the candidate's unit in its direction, at
        alpha = direction x strength x each dose, with a label by alpha when
        it has several doses.

        Example:
            >>> unit = NeuronUnit("qwen3-8b", 9, 1)
            >>> owl = Candidate("library", "Brass Owl", unit, -1, 1.5, False, "f", "r")
            >>> each = EachCandidate("ladder", 0, (0.5, 1.0))
            >>> (condition,) = each.conditions([(owl, None)])
            >>> condition.candidate, [label for label, _ in condition.points]
            ('library/L9N1-', ['ladder-0001 alpha=-0.75', 'ladder-0001 alpha=-1.5'])
        """
        where = f"conditions[{self.place}] ({self.label}): "
        runs = sum(len(own or self.doses) for _, own in chosen)
        if runs * (1 + self.controls) > MOST_RUNS:
            raise ValueError(
                f"{where}its {len(chosen):,} candidates make "
                f"{runs * (1 + self.controls):,} runs for each prompt and seed, more "
                f"than the {MOST_RUNS:,} that six-digit run IDs can number"
            )
        width = max(LABEL_DIGITS, len(str(len(chosen))))
        made = []
        for number, (candidate, own) in enumerate(chosen, 1):
            label = f"{self.label}-{number:0{width}d}"
            doses = self.doses if own is None else own
            points: dict[str, Schedule] = {}
            for dose in doses:
                alpha = candidate.direction * candidate.strength * dose
                try:
                    pushed = constant(candidate.unit, alpha, self.include_prompt)
                except ValueError as error:
                    raise ValueError(f"{where}{candidate.key}: {error}") from None
                point = label if len(doses) == 1 else f"{label} alpha={alpha!r}"
                if point in points:  # two doses that give one float
                    raise ValueError(
                        f"{where}{candidate.key}'s doses give alpha={alpha!r} twice; "
                        "give doses that differ more"
                    )
                points[point] = Schedule(
                    label, pushed.segments, self.allow_over_ceiling
                )
            made.append(
                Condition(
                    label,
                    tuple(points.items()),
                    self.controls,
                    self.control_seed,
                    candidate.key,
                )
            )
        return tuple(made)


def each_candidate_from_mapping(data: Mapping, i: int) -> EachCandidate:
    """Check conditions[i], a condition with each_candidate.

    It takes a label, each_candidate ({} or null for every default) and
    controls with their control_seed, and no schedule of its own.

    Example:
        >>> each_candidate_from_mapping(
        ...     {"label": "screen", "each_candidate": {}, "shape": "constant"}, 1
        ... )  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: conditions[1] (screen): each_candidate makes each ...
    """
    label = condition_label(data, i)
    where = f"conditions[{i}] ({label}): "
    if own := [key for key in ("segments", "shape", "params", "grid") if key in data]:
        raise ValueError(
            f"{where}each_candidate makes each candidate's schedule, a constant "
            f"push, so it takes no {' or '.join(own)}"
        )
    if "allow_over_ceiling" in data:
        raise ValueError(
            f"{where}allow_over_ceiling goes inside each_candidate, as "
            "each_candidate: {allow_over_ceiling: true}"
        )
    refuse_unknown_keys(data, EACH_CONDITION_KEYS, where)
    controls, seed = condition_controls(data, where)
    rule = data["each_candidate"]
    rule = {} if rule is None else rule
    if not isinstance(rule, Mapping):
        raise ValueError(
            f"{where}each_candidate must be a mapping, such as {{doses: [1.0]}}, "
            f"or {{}} for its defaults, not {rule!r}"
        )
    inside = f"{where}each_candidate: "
    refuse_unknown_keys(rule, EACH_CANDIDATE_KEYS, inside)
    doses = multiples(rule["doses"], inside) if "doses" in rule else DEFAULT_DOSES
    prompt = rule.get("include_prompt", True)
    ceiling = rule.get("allow_over_ceiling", False)
    for flag, value in (("include_prompt", prompt), ("allow_over_ceiling", ceiling)):
        if not isinstance(value, bool):
            raise ValueError(f"{inside}{flag} must be true or false, not {value!r}")
    return EachCandidate(label, i, doses, prompt, ceiling, controls, seed)


def conditions_from_list(
    listed: Sequence[object], rule: CandidatesRule | None, model: str
) -> tuple[Condition, ...]:
    """An experiment's conditions, in order, each_candidate's made from the
    list that rule names, as candidates of model.

    Every problem is a ValueError that names the condition by its place and
    label: two conditions with each_candidate, one without candidates,
    candidates with none, and a label used twice, a candidate's too. The
    list is read last, once every condition is checked.

    Example:
        >>> none = {"label": "none", "segments": []}
        >>> conditions_from_list([none, none], None, "qwen3-8b")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: conditions[1]: the label 'none' is used twice; ...
    """
    made: list[Condition] = []
    places: dict[str, int] = {}  # each label's condition, by its place
    each: EachCandidate | None = None
    for i, item in enumerate(listed):
        if not (isinstance(item, Mapping) and "each_candidate" in item):
            condition = condition_from_mapping(item, i)
            if condition.label in places:
                raise ValueError(
                    f"conditions[{i}]: the label {condition.label!r} is used twice; "
                    "each condition needs its own"
                )
            places[condition.label] = i
            made.append(condition)
            continue
        found = each_candidate_from_mapping(item, i)
        where = f"conditions[{i}] ({found.label}): "
        if each is not None:
            raise ValueError(
                f"{where}an experiment takes one condition with each_candidate, and "
                f"conditions[{each.place}] ({each.label}) is one"
            )
        if rule is None:
            raise ValueError(
                f"{where}each_candidate pushes the candidates of the experiment's "
                f"candidates: setting, which it lacks, such as candidates: {EXAMPLE}"
            )
        each = found
    if each is None:
        if rule is not None:
            raise ValueError(
                "candidates: names candidates to push, but no condition has "
                "each_candidate: to push them"
            )
        return tuple(made)
    assert rule is not None  # checked above
    pushed = each.conditions(chosen_candidates(rule, model))
    for condition in pushed:
        if (place := places.get(condition.label)) is not None:
            raise ValueError(
                f"conditions[{each.place}] ({each.label}): it labels a candidate "
                f"{condition.label}, which is conditions[{place}]'s label; give one "
                "of them another"
            )
    before = sum(1 for place in places.values() if place < each.place)
    return (*made[:before], *pushed, *made[before:])
