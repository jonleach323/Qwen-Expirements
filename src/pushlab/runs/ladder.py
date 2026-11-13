"""A dose ladder's experiment file: every unit at every dose (decision 0022).

A Ladder holds the units of a units file (pushlab.units.ladder_units), the
doses, numbers above 0 in increasing order, the prompt set and the other
settings of the experiment it makes. Each unit is one condition, labelled
u01, u02 and so on in the units file's order: a constant push of its
neuron, on the prompt pass too unless told otherwise, with a grid of
signed doses, the unit's direction times each dose. So the experiment runs
units x doses x prompts x seeds runs, plus an unpushed baseline when asked,
and matched random controls. One seed decodes greedily; several sample
each reply at temperature 1, as greedy decoding takes one seed.

A dose above the ceiling of 1.4 (pushlab.schedules.validation) sets
allow_over_ceiling, since a ladder that seeks the dose where the text
breaks goes there on purpose. Ladder.text is the file, with comments that
name each unit and say what to do with it, and a note when a unit comes
from the Neuron Persona Library, whose data stays out of git (decision
0020). write_ladder writes it only once its text reads back as the
experiment meant and that experiment expands, so a file that would not run
is never left.
"""

import itertools
import math
import os
import textwrap
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pushlab.cli_ladder import BATCH_SIZE, MAX_NEW_TOKENS
from pushlab.discovery.sources import LIBRARY
from pushlab.runs.estimate import exact
from pushlab.runs.expand import Expansion, expand
from pushlab.runs.spec import ExperimentSpec, read_spec, spec_from_yaml
from pushlab.schedules.validation import DEFAULT_CEILING
from pushlab.schedules.yaml_text import dump_yaml, load_yaml
from pushlab.units.ladder_units import LadderUnit

TEMPERATURE = 1.0  # each reply's sampling, with several seeds
HOURS_MARGIN = 1.5  # max_gpu_hours over the estimate, by default
HOURS_STEP = 0.25  # max_gpu_hours is rounded up to a quarter hour


def unit_labels(count: int) -> tuple[str, ...]:
    """The conditions' labels of count units: u01, u02 and so on, as wide as
    the count's digits and at least two.

    Example:
        >>> unit_labels(3)
        ('u01', 'u02', 'u03')
    """
    width = max(2, len(str(count)))
    return tuple(f"u{number:0{width}d}" for number in range(1, count + 1))


def default_hours(runs: int, max_new_tokens: int, tokens_per_second: float) -> float:
    """The default max_gpu_hours: the time runs replies of max_new_tokens
    take at tokens_per_second, times HOURS_MARGIN, rounded up to a quarter
    hour, and at least one.

    Example:
        >>> default_hours(1_000, 160, 22.0)  # 2.02 hours, times 1.5
        3.25
    """
    estimate = runs * max_new_tokens / tokens_per_second / 3600 * HOURS_MARGIN
    steps = math.ceil(round(estimate / HOURS_STEP, 6))  # 0.5000001 is 0.5
    return max(HOURS_STEP, steps * HOURS_STEP)


def checked_doses(doses: object) -> tuple[float, ...]:
    """Doses as floats; ValueError for none, for one that is not a finite
    number above 0, and for doses not in increasing order, each once.

    Example:
        >>> checked_doses([0.5, 1])
        (0.5, 1.0)
    """
    if not isinstance(doses, tuple | list) or not doses:
        raise ValueError(f"a ladder needs a list of doses, not {doses!r}")
    for dose in doses:
        number = isinstance(dose, int | float) and not isinstance(dose, bool)
        if not (number and math.isfinite(dose) and dose > 0):
            raise ValueError(f"a dose must be a number above 0, not {dose!r}")
    floats = tuple(float(dose) for dose in doses)
    if any(later <= earlier for earlier, later in itertools.pairwise(floats)):
        raise ValueError(f"the doses must increase, each once: {floats}")
    return floats


def _plain(text: str) -> str:
    """text for a comment: on one line, with only printable characters."""
    shown = "".join(c if c.isprintable() else " " for c in str(text))
    return " ".join(shown.split())


def _comment(text: str, indent: str = "") -> list[str]:
    """text as comment lines of at most about 79 characters."""
    width = 77 - len(indent)
    lines = textwrap.wrap(
        _plain(text), width, break_long_words=False, break_on_hyphens=False
    )
    return [f"{indent}# {line}" for line in lines]


def describe_unit(unit: LadderUnit) -> str:
    """A unit as the file's comments and the command name it: its ID, its
    direction, and its persona and source when given.

    Example:
        >>> from pushlab.units.neuron import NeuronUnit
        >>> owl = LadderUnit(NeuronUnit.parse("qwen3-8b:L9:N1"), -1, "Brass Owl")
        >>> describe_unit(owl)
        'qwen3-8b:L9:N1, direction -1, persona Brass Owl'
    """
    parts = [str(unit.unit), f"direction {'+1' if unit.direction > 0 else '-1'}"]
    if unit.persona:
        parts.append(f"persona {_plain(unit.persona)}")
    if unit.source:
        parts.append(f"source {_plain(unit.source)}")
    return ", ".join(parts)


def doses_text(doses: tuple[float, ...]) -> str:
    """The doses in words.

    Example:
        >>> doses_text((0.25, 0.5, 1.0))
        '0.25, 0.5 and 1'
    """
    shown = [exact(dose) for dose in doses]
    return shown[0] if len(shown) == 1 else f"{', '.join(shown[:-1])} and {shown[-1]}"


@dataclass(frozen=True)
class Ladder:
    """A dose ladder: its units, doses and prompt set, and the settings of
    the experiment it makes. prompts is written into the file as it is, a
    path from the repository's root or an absolute one. units_file names
    the units file, and hours_note says where max_gpu_hours came from, for
    the file's comments.

    Example:
        >>> from pushlab.units.neuron import NeuronUnit
        >>> owl = LadderUnit(NeuronUnit.parse("qwen3-8b:L9:N1"), -1, "Brass Owl")
        >>> ladder = Ladder("owl", (owl,), (0.5, 1.5), "prompts/smoke.jsonl")
        >>> ladder.mapping()["conditions"][0]["grid"]
        {'alpha': [-0.5, -1.5]}
        >>> ladder.over_ceiling, ladder.run_count(prompts=5)
        (True, 10)
    """

    name: str
    units: tuple[LadderUnit, ...]
    doses: tuple[float, ...]
    prompts: str
    seeds: tuple[int, ...] = (0,)
    max_new_tokens: int = MAX_NEW_TOKENS
    batch_size: int = BATCH_SIZE
    max_gpu_hours: float = HOURS_STEP
    controls: int = 0
    control_seed: int | None = None
    baseline: bool = False
    prompt_pass: bool = True
    units_file: str = ""
    hours_note: str = ""

    def __post_init__(self) -> None:
        """Refuse no unit, units of several models, and doses checked_doses
        refuses; keep the units and the doses, as floats, in tuples."""
        if not self.units:
            raise ValueError("a ladder needs at least one unit")
        if len({unit.unit.model_name for unit in self.units}) > 1:
            raise ValueError("a ladder's units must all be of one model")
        object.__setattr__(self, "units", tuple(self.units))
        object.__setattr__(self, "doses", checked_doses(self.doses))

    @property
    def model(self) -> str:
        """The model of its units, which the experiment runs on."""
        return self.units[0].unit.model_name

    @property
    def labels(self) -> tuple[str, ...]:
        """Each unit's condition label, in order: u01, u02 and so on."""
        return unit_labels(len(self.units))

    @property
    def over_ceiling(self) -> bool:
        """Whether its largest dose is above the ceiling of 1.4."""
        return max(self.doses) > DEFAULT_CEILING

    @property
    def from_library(self) -> bool:
        """Whether a unit comes from the Neuron Persona Library."""
        return any(unit.source.casefold() == LIBRARY.name for unit in self.units)

    def decoding(self) -> dict[str, Any]:
        """Greedy for one seed; sampling at TEMPERATURE for several."""
        if len(self.seeds) == 1:
            return {"strategy": "greedy"}
        return {"strategy": "sample", "temperature": TEMPERATURE}

    def run_count(self, prompts: int) -> int:
        """How many runs it makes on a prompt set of prompts prompts."""
        pushed = len(self.units) * len(self.doses) * (1 + self.controls)
        return prompts * len(self.seeds) * (pushed + self.baseline)

    def _conditions(self) -> list[dict[str, Any]]:
        conditions: list[dict[str, Any]] = []
        if self.baseline:
            conditions.append({"label": "baseline", "segments": []})
        for label, unit in zip(self.labels, self.units, strict=True):
            params: dict[str, Any] = {"unit": str(unit.unit)}
            if not self.prompt_pass:
                params["include_prompt"] = False
            condition = {
                "label": label,
                "shape": "constant",
                "params": params,
                "grid": {"alpha": [unit.alpha(dose) for dose in self.doses]},
            }
            if self.over_ceiling:
                condition["allow_over_ceiling"] = True
            if self.controls:
                condition["controls"] = self.controls
                condition["control_seed"] = self.control_seed or 0
            conditions.append(condition)
        return conditions

    def mapping(self) -> dict[str, Any]:
        """The experiment's settings as plain data, as spec_from_mapping
        reads them and the file holds them."""
        return {
            "name": self.name,
            "model": self.model,
            "prompts": self.prompts,
            "decoding": self.decoding(),
            "seeds": list(self.seeds),
            "max_new_tokens": self.max_new_tokens,
            "batch_size": self.batch_size,
            "max_gpu_hours": float(self.max_gpu_hours),
            "conditions": self._conditions(),
        }

    def _header(self) -> list[str]:
        where = "on the prompt pass and every reply token"
        if not self.prompt_pass:
            where = "on every reply token but not the prompt pass"
        source = f" of {self.units_file}" if self.units_file else ""
        lines = _comment(
            f"A dose ladder written by pushlab ladder (decision 0022): each unit"
            f"{source} pushed in its direction at doses {doses_text(self.doses)}, "
            f"{where}, on the prompts of {self.prompts}. `pushlab run --dry-run` on "
            "this file shows what it runs and costs; docs/runbooks/ladder.md says "
            "how to run it, score it and read its doses."
        )
        lines += ["#", "# The units, a condition each:"]
        lines += [
            f"#   {label}: {describe_unit(unit)}"
            for label, unit in zip(self.labels, self.units, strict=True)
        ]
        if self.from_library:
            lines += ["#"] + _comment(
                "It holds the Neuron Persona Library's data, which states no "
                "licence, so it stays out of git (decision 0020)."
            )
        return lines

    def _unit_conditions(self) -> list[str]:
        about = (
            "Each unit pushed with a constant dose, its grid's alphas being its "
            "direction times each dose of the ladder."
        )
        if self.over_ceiling:
            about += (
                f" Doses above the ceiling of {exact(DEFAULT_CEILING)} are allowed on "
                "purpose: the ladder seeks the dose where the text breaks."
            )
        if self.controls:
            neurons = "neuron" if self.controls == 1 else "neurons"
            about += (
                f" Each dose again on {self.controls} random {neurons} of the unit's "
                "layer, as matched controls."
            )
        lines = _comment(about, "  ")
        units = self._conditions()[1:] if self.baseline else self._conditions()
        for condition in units:
            params = ", ".join(
                f"{key}: {'false' if value is False else value}"
                for key, value in condition["params"].items()
            )
            alphas = ", ".join(repr(alpha) for alpha in condition["grid"]["alpha"])
            lines += [
                f"  - label: {condition['label']}",
                "    shape: constant",
                f"    params: {{{params}}}",
                f"    grid: {{alpha: [{alphas}]}}",
            ]
            lines += [
                f"    {key}: {'true' if value is True else value}"
                for key in ("allow_over_ceiling", "controls", "control_seed")
                if (value := condition.get(key)) is not None
            ]
        return lines

    def text(self) -> str:
        """The experiment file: its comments, then its settings, which
        load_yaml reads back as mapping()."""
        named = {"name": self.name, "model": self.model, "prompts": self.prompts}
        decoding = [
            "# Greedy decoding draws no random numbers, so it takes one seed.",
            "decoding: {strategy: greedy}",
        ]
        if len(self.seeds) > 1:
            decoding = [
                f"# Each reply sampled at temperature {exact(TEMPERATURE)}, once "
                "with each seed.",
                f"decoding: {{strategy: sample, temperature: {TEMPERATURE!r}}}",
            ]
        lines = [
            *self._header(),
            dump_yaml(named).rstrip("\n"),
            *decoding,
            f"seeds: [{', '.join(str(seed) for seed in self.seeds)}]",
            f"max_new_tokens: {self.max_new_tokens}",
            f"batch_size: {self.batch_size}",
            *(_comment(self.hours_note) if self.hours_note else []),
            f"max_gpu_hours: {float(self.max_gpu_hours)!r}",
            "conditions:",
        ]
        if self.baseline:
            lines += [
                "  # No push: each prompt's reply at dose 0.",
                "  - label: baseline",
                "    segments: []",
            ]
        return "\n".join([*lines, *self._unit_conditions()]) + "\n"


def write_ladder(
    ladder: Ladder, out: Path, overwrite: bool = False
) -> tuple[ExperimentSpec, Expansion]:
    """Write the ladder's file to out and return its spec and expansion.

    The text must read back as the ladder's settings, and the experiment
    they make must expand (pushlab.runs.expand): else ValueError says why,
    before any folder or file is made. The file is written beside out under
    a hidden name, read back with read_spec, and only then renamed to out,
    so nothing is left of a write that fails. FileExistsError when out
    exists, unless overwrite.
    """
    out = Path(out)
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} already exists; pass --overwrite to replace it")
    text = ladder.text()
    if load_yaml(text) != ladder.mapping():  # a bug, never the user's doing
        raise ValueError("the ladder's text does not read back as its settings")
    try:
        spec = spec_from_yaml(text)
        expansion = expand(spec)
    except ValueError as error:
        raise ValueError(
            f"the ladder would not run, so nothing was written: {error}"
        ) from None
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f".{out.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "x", encoding="utf-8") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        if read_spec(temporary) != spec:
            raise ValueError(f"{temporary} does not read back as the ladder")
        if out.exists() and not overwrite:  # written meanwhile
            raise FileExistsError(
                f"{out} already exists; pass --overwrite to replace it"
            )
        os.replace(temporary, out)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return spec, expansion
