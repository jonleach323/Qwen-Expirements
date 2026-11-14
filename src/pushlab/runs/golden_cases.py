"""The golden cases file: fixed prompts, seeds and pushes, as data (decision 0023).

A golden cases file, such as tests/golden/qwen3-8b.yaml, names a model, the
data type it runs in, the reply length, its prompts, any sets of neurons and
its cases (development plan, "Golden tests"):

    version: 1
    model: qwen3-8b
    dtype: bfloat16
    max_new_tokens: 32
    prompts:
      kettle: How do I descale a kettle with vinegar?
    sets:
      qwen3-8b:set:my-trio:
        entries: [[12, 7, 1.0], [12, 9, -0.5], [20, 3, 0.75]]
        size: 64.0
        source: made up
    cases:
      - id: kettle-greedy
        prompt: kettle
        decoding: {strategy: greedy}
      - id: kettle-set-sampled
        prompt: kettle
        decoding: {strategy: sample, temperature: 0.7, seed: 1}
        push: {unit: qwen3-8b:set:my-trio, alpha: 1.0, calibration: per_prompt}

A case is one prompt, decoded greedily or sampled with a seed, unpushed or
with one constant push of a neuron or a set at a dose, under the fixed or
the per_prompt calibration (pushlab.units.calibration_mode). A push names a
neuron by its ID, or a set the file's sets define. So the file names its
neurons and sets, and the code never does. read_cases reads and checks it;
pushlab.runs.golden runs its cases.
"""

import math
import numbers
import re
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from pushlab import PROJECT_ROOT
from pushlab.engine.decoding import Decoding
from pushlab.models.registry import REGISTRY
from pushlab.schedules.model import Schedule
from pushlab.schedules.shapes import constant
from pushlab.schedules.validation import ScheduleWarning, validate
from pushlab.schedules.yaml_text import load_yaml
from pushlab.units.calibration_mode import CALIBRATION_MODES
from pushlab.units.neuron import NeuronUnit
from pushlab.units.neuron_set import NeuronSet, Unit

CASES_FORMAT = 1
GOLDEN_DIR = PROJECT_ROOT / "tests" / "golden"
# The data types a cases file may run in: bfloat16 on the GPU, as the
# plan's runs are, and float32, which the CPU tests' tiny models run in.
DTYPES = {"bfloat16": torch.bfloat16, "float32": torch.float32}
# Case IDs and prompt names: lowercase letters and digits joined by hyphens.
NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
SET_ID = re.compile(r"(?P<model>[^:\s]+):set:(?P<name>[a-z0-9]+(?:-[a-z0-9]+)*)")


@dataclass(frozen=True)
class GoldenPush:
    """One constant push: a neuron or a set at dose alpha, from the prompt
    pass to the end of the reply, under the calibration mode named."""

    unit: Unit
    alpha: float
    calibration: str


@dataclass(frozen=True)
class GoldenCase:
    """One case: its ID, its prompt's text, how its reply is decoded, and its
    push, or None for none.

    Example:
        >>> greedy = Decoding(strategy="greedy", max_new_tokens=8)
        >>> GoldenCase("hi-greedy", "Say hi.", greedy).schedule().segments
        ()
        >>> push = GoldenPush(NeuronUnit.parse("qwen3-8b:L9:N1"), 1.0, "fixed")
        >>> case = GoldenCase("hi-neuron", "Say hi.", greedy, push)
        >>> for segment in case.schedule().segments:
        ...     print(segment.describe())
        qwen3-8b:L9:N1 at +1 on the prompt pass
        qwen3-8b:L9:N1 at +1 from reply token 0 to the end
    """

    case_id: str
    prompt: str
    decoding: Decoding
    push: GoldenPush | None = None

    def schedule(self) -> Schedule:
        """The case's schedule, named by its ID: none, or its constant push."""
        if self.push is None:
            return Schedule(self.case_id)
        return constant(self.push.unit, self.push.alpha, name=self.case_id)

    def as_data(self) -> dict[str, Any]:
        """The case as plain data, as a fingerprints file keeps it: the
        prompt's text, every decoding setting, and the push, with a set's
        definition beside its ID (decision 0023's form).

        Example:
            >>> greedy = Decoding(strategy="greedy", max_new_tokens=8)
            >>> data = GoldenCase("hi-greedy", "Say hi.", greedy).as_data()
            >>> data["prompt"], data["push"], data["decoding"]["max_new_tokens"]
            ('Say hi.', None, 8)
        """
        data: dict[str, Any] = {
            "prompt": self.prompt,
            "decoding": asdict(self.decoding),
            "push": None,
        }
        if self.push is not None:
            unit = self.push.unit
            push: dict[str, Any] = {"unit": str(unit)}
            if isinstance(unit, NeuronSet):
                push["set"] = {
                    "entries": [list(entry) for entry in unit.entries],
                    "size": unit.size,
                    "source": unit.source,
                }
            push |= {"alpha": self.push.alpha, "calibration": self.push.calibration}
            data["push"] = push
        return data


@dataclass(frozen=True)
class GoldenCases:
    """A cases file's model, data type, reply length and cases, in order."""

    model: str
    dtype: str
    max_new_tokens: int
    cases: tuple[GoldenCase, ...]

    @property
    def torch_dtype(self) -> torch.dtype:
        """The data type as torch names it, such as torch.bfloat16."""
        return DTYPES[self.dtype]

    @property
    def needs_fixed_calibration(self) -> bool:
        """Whether a case pushes under the fixed calibration."""
        return any(
            case.push is not None and case.push.calibration == "fixed"
            for case in self.cases
        )


def cases_path(model: str) -> Path:
    """A model's golden cases file in the repository.

    Example:
        >>> cases_path("qwen3-8b").relative_to(PROJECT_ROOT).as_posix()
        'tests/golden/qwen3-8b.yaml'
    """
    return GOLDEN_DIR / f"{model}.yaml"


def _is_number(value: object) -> bool:
    if not isinstance(value, numbers.Real) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # an int too large for a float
        return False


def _mapping(
    where: str, value: object, required: Sequence[str], optional: Sequence[str] = ()
) -> dict[str, Any]:
    """value, checked to be a mapping with every required key and no other
    than the optional ones."""
    keys = [*required, *optional]
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be a mapping of {', '.join(keys)}")
    if missing := [key for key in required if key not in value]:
        raise ValueError(f"{where} lacks {', '.join(missing)}")
    if unknown := [str(key) for key in value if key not in keys]:
        raise ValueError(
            f"{where} has {', '.join(unknown)}, which it does not take; it takes "
            f"{', '.join(keys)}"
        )
    return value


def _name(where: str, value: object) -> str:
    if not isinstance(value, str) or not NAME.fullmatch(value):
        raise ValueError(
            f"{where} must be lowercase letters and digits joined by hyphens, such "
            f"as kettle-greedy, not {value!r}"
        )
    return value


def _prompts(where: str, value: object) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise ValueError(
            f"{where} must name each prompt, such as kettle: How do I descale a kettle?"
        )
    for name, text in value.items():
        _name(f"{where}: the name {name!r}", name)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{where}.{name} must be the prompt's text, not {text!r}")
    return dict(value)


def _sets(where: str, value: object, model: str) -> dict[str, NeuronSet]:
    if not isinstance(value, dict):
        raise ValueError(f"{where} must map each set's ID to its definition")
    sets = {}
    for set_id, definition in value.items():
        match = SET_ID.fullmatch(set_id) if isinstance(set_id, str) else None
        if match is None:
            raise ValueError(
                f"{where}: {set_id!r} is not a set's ID, such as {model}:set:my-trio"
            )
        here = f"{where}.{set_id}"
        if match["model"] != model:
            raise ValueError(
                f"{here} is a set of {match['model']}'s neurons, but the cases are "
                f"{model}'s"
            )
        data = _mapping(here, definition, ("entries", "source"), ("size",))
        try:
            sets[set_id] = NeuronSet(
                model, match["name"], data["entries"], data["source"], data.get("size")
            )
        except ValueError as error:
            raise ValueError(f"{here}: {error}") from None
    return sets


def _decoding(where: str, value: object, max_new_tokens: int) -> Decoding:
    data = _mapping(where, value, ("strategy",), ("temperature", "top_p", "seed"))
    try:
        return Decoding(max_new_tokens=max_new_tokens, **data)
    except ValueError as error:
        raise ValueError(f"{where}: {error}") from None


def _push(
    where: str, value: object, model: str, sets: dict[str, NeuronSet]
) -> GoldenPush:
    data = _mapping(where, value, ("unit", "alpha", "calibration"))
    unit_id = data["unit"]
    unit: Unit
    if isinstance(unit_id, str) and unit_id in sets:
        unit = sets[unit_id]
    else:
        try:
            unit = NeuronUnit.parse(unit_id)
        except ValueError as error:
            raise ValueError(
                f"{where}.unit is neither a set the file's sets define nor a "
                f"neuron: {error}"
            ) from None
        if unit.model_name != model:
            raise ValueError(
                f"{where}.unit {unit} is a neuron of {unit.model_name}, but the "
                f"cases are {model}'s"
            )
    alpha = data["alpha"]
    if not _is_number(alpha):
        raise ValueError(f"{where}.alpha must be a finite number, not {alpha!r}")
    calibration = data["calibration"]
    if calibration not in CALIBRATION_MODES:
        raise ValueError(
            f"{where}.calibration must be {' or '.join(CALIBRATION_MODES)}, not "
            f"{calibration!r}"
        )
    return GoldenPush(unit, float(alpha), calibration)


def _case(
    where: str,
    value: object,
    model: str,
    max_new_tokens: int,
    prompts: dict[str, str],
    sets: dict[str, NeuronSet],
) -> GoldenCase:
    data = _mapping(where, value, ("id", "prompt", "decoding"), ("push",))
    case_id = _name(f"{where}.id", data["id"])
    where = f"{where} ({case_id})"
    name = data["prompt"]
    if not isinstance(name, str) or name not in prompts:
        raise ValueError(
            f"{where}.prompt must name one of the file's prompts "
            f"({', '.join(prompts)}), not {name!r}"
        )
    decoding = _decoding(f"{where}.decoding", data["decoding"], max_new_tokens)
    push = None
    if "push" in data:
        push = _push(f"{where}.push", data["push"], model, sets)
    case = GoldenCase(case_id, prompts[name], decoding, push)
    if push is not None:  # the unit's range, and the dose ceiling
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ScheduleWarning)
                validate(case.schedule(), REGISTRY[model], max_new_tokens)
        except ValueError as error:
            raise ValueError(f"{where}.push: {error}") from None
    return case


def cases_from_mapping(data: object, where: str = "the golden cases") -> GoldenCases:
    """Golden cases from plain data, every field checked.

    ValueError names the first problem: a key missing or unknown, a model
    outside the registry, a data type other than bfloat16 or float32, a
    case ID given twice, a prompt or set that no case uses, a unit of
    another model or out of its range, a dose above the ceiling, decoding
    settings that do not make a strategy.

    Example:
        >>> text = (
        ...     "{version: 1, model: tiny, dtype: float32, max_new_tokens: 8, "
        ...     "prompts: {hi: Say hi.}, "
        ...     "cases: [{id: hi-greedy, prompt: hi, decoding: {strategy: greedy}}]}"
        ... )
        >>> cases = cases_from_mapping(load_yaml(text))
        >>> [case.case_id for case in cases.cases], cases.needs_fixed_calibration
        (['hi-greedy'], False)
    """
    keys = ("version", "model", "dtype", "max_new_tokens", "prompts", "cases")
    data = _mapping(where, data, keys, ("sets",))
    if type(data["version"]) is not int or data["version"] != CASES_FORMAT:
        raise ValueError(f"{where}: version must be {CASES_FORMAT}")
    model = data["model"]
    if not isinstance(model, str) or model not in REGISTRY:
        known = ", ".join(sorted(REGISTRY))
        raise ValueError(f"{where}: model {model!r} is not in the registry ({known})")
    dtype = data["dtype"]
    if not isinstance(dtype, str) or dtype not in DTYPES:
        raise ValueError(f"{where}: dtype must be {' or '.join(DTYPES)}, not {dtype!r}")
    tokens = data["max_new_tokens"]
    if type(tokens) is not int or tokens < 1:
        raise ValueError(
            f"{where}: max_new_tokens must be a whole number from 1, not {tokens!r}"
        )
    prompts = _prompts(f"{where}.prompts", data["prompts"])
    sets = _sets(f"{where}.sets", data.get("sets", {}), model)
    items = data["cases"]
    if not isinstance(items, list) or not items:
        raise ValueError(f"{where}.cases must be a list of at least one case")
    cases: list[GoldenCase] = []
    for i, item in enumerate(items):
        case = _case(f"{where}.cases[{i}]", item, model, tokens, prompts, sets)
        if taken := [j for j, c in enumerate(cases) if c.case_id == case.case_id]:
            raise ValueError(
                f"{where}.cases[{i}]: the ID {case.case_id} is cases[{taken[0]}]'s "
                "already; give each case its own"
            )
        cases.append(case)
    asked = {item["prompt"] for item in items}  # each a prompt's name, by now
    if unused := [name for name in prompts if name not in asked]:
        raise ValueError(f"{where}.prompts: no case asks {', '.join(unused)}")
    pushed = {str(case.push.unit) for case in cases if case.push is not None}
    if unused := [set_id for set_id in sets if set_id not in pushed]:
        raise ValueError(f"{where}.sets: no case pushes {', '.join(unused)}")
    return GoldenCases(model, dtype, tokens, tuple(cases))


def read_cases(path: Path) -> GoldenCases:
    """The golden cases in the YAML file at path, checked; ValueError says
    what is wrong, naming the file.

    Example:
        >>> cases = read_cases(cases_path("qwen3-8b"))
        >>> cases.model, cases.dtype, cases.max_new_tokens, len(cases.cases)
        ('qwen3-8b', 'bfloat16', 32, 18)
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        why = error.strerror or error
        raise ValueError(f"cannot read the golden cases {path}: {why}") from None
    try:
        data = load_yaml(text)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from None
    return cases_from_mapping(data, str(path))
