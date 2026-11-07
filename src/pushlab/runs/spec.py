"""Experiment specs: one YAML file that describes a whole experiment.

An experiment file names the model, the prompt set, the decoding settings,
the seeds, the reply length, the batch size, the GPU-hours it may take and
its conditions; experiments/README.md has the format and an example. Each
condition is a labelled schedule, its segments written out or a shape with
its params, and a shape's params can take a grid of values: one schedule
per value, or per combination of values. A condition can also ask for
random controls, which pushlab.runs.expand draws.

spec_from_yaml reads and checks the text, refusing unknown keys and any
setting out of range with a message that names it. Whether each schedule
suits the model is for expand, which also reads the prompt set. A spec's
sha256 hashes every setting, each condition's schedules written out in
full, but not the way the file is written: comments, spacing, the order of
keys in a mapping, 1 against 1.0, and a shape against the same segments
written out do not change it. A grid's keys are the exception: their order
sets the order of its points, so it is a setting.
"""

import hashlib
import json
import math
import numbers
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pushlab import PROJECT_ROOT
from pushlab.engine.decoding import STRATEGIES
from pushlab.models.registry import REGISTRY
from pushlab.runs.conditions import (
    Condition,
    condition_from_mapping,
    not_text,
    refuse_unknown_keys,
)
from pushlab.schedules.yaml_io import to_mapping as schedule_mapping
from pushlab.schedules.yaml_text import load_yaml
from pushlab.store.records import EXPERIMENT_NAME, MAX_SEED

SPEC_KEYS = (
    "name",
    "model",
    "prompts",
    "decoding",
    "seeds",
    "max_new_tokens",
    "batch_size",
    "max_gpu_hours",
    "conditions",
)
# What a file's decoding takes; a run's record holds DECODING_KEYS
# (pushlab.store.records) and, when sampling, temperature and top_p.
SPEC_DECODING_KEYS = ("strategy", "temperature", "top_p", "dtype")
# Main runs use bfloat16; a sample reruns in 32-bit (development plan).
DTYPES = ("bfloat16", "float32")


def _is_whole(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite(value: object) -> bool:
    if not isinstance(value, numbers.Real) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # an int too large for a float
        return False


@dataclass(frozen=True)
class ExperimentSpec:
    """A whole experiment, as spec_from_yaml reads and checks it.

    prompts is the prompt set's path as the file gives it, relative to the
    repository unless it is absolute (prompts_path resolves it). decoding
    has strategy, temperature (None for greedy decoding), top_p and dtype.
    seeds are the sampling seeds in order; greedy decoding takes one. Make
    a spec with read_spec, spec_from_yaml or spec_from_mapping, which check
    every setting: the class itself checks none.

    Example:
        >>> spec = read_spec(PROJECT_ROOT / "experiments/p0/s2_pulse_lengths.yaml")
        >>> spec.name, spec.model, [c.label for c in spec.conditions]
        ('s2-pulse-lengths', 'qwen3.5-4b', ['none', 'pulse'])
    """

    name: str
    model: str
    prompts: str
    decoding: dict[str, Any]
    seeds: tuple[int, ...]
    max_new_tokens: int
    batch_size: int
    max_gpu_hours: float
    conditions: tuple[Condition, ...]

    @property
    def prompts_path(self) -> Path:
        """The prompt set's file: prompts, from the repository's root if relative."""
        return PROJECT_ROOT / self.prompts  # an absolute prompts stays as it is

    def to_mapping(self) -> dict[str, Any]:
        """Every setting as plain data, each schedule written out: what sha256 hashes.

        Example:
            >>> spec = read_spec(PROJECT_ROOT / "experiments/p0/s2_pulse_lengths.yaml")
            >>> spec.to_mapping()["conditions"][1]["points"][0]["label"]
            'pulse length=4'
        """
        return {
            "name": self.name,
            "model": self.model,
            "prompts": self.prompts,
            "decoding": dict(self.decoding),
            "seeds": list(self.seeds),
            "max_new_tokens": self.max_new_tokens,
            "batch_size": self.batch_size,
            "max_gpu_hours": self.max_gpu_hours,
            "conditions": [
                {
                    "label": condition.label,
                    "controls": condition.controls,
                    "control_seed": condition.control_seed,
                    "points": [
                        {"label": label, "schedule": schedule_mapping(schedule)}
                        for label, schedule in condition.points
                    ],
                }
                for condition in self.conditions
            ],
        }

    @property
    def sha256(self) -> str:
        """The SHA-256 of to_mapping() as canonical JSON, keys sorted.

        Example:
            >>> spec = read_spec(PROJECT_ROOT / "experiments/p0/s2_pulse_lengths.yaml")
            >>> len(spec.sha256)
            64
        """
        text = json.dumps(
            self.to_mapping(), sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(text.encode()).hexdigest()


def _decoding(data: object) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise ValueError(
            f"decoding must be a mapping, such as {{strategy: greedy}}, not {data!r}"
        )
    refuse_unknown_keys(data, SPEC_DECODING_KEYS, "decoding: ")
    strategy = data.get("strategy")
    if not isinstance(strategy, str) or strategy not in STRATEGIES:
        raise ValueError(
            f"decoding's strategy must be {' or '.join(STRATEGIES)}, not {strategy!r}"
        )
    dtype = data.get("dtype", "bfloat16")
    if not isinstance(dtype, str) or dtype not in DTYPES:
        raise ValueError(
            f"decoding's dtype must be {' or '.join(DTYPES)}, not {dtype!r}"
        )
    temperature, top_p = data.get("temperature"), data.get("top_p", 1.0)
    if strategy == "greedy":
        if unused := [k for k in ("temperature", "top_p") if k in data]:
            raise ValueError(
                f"greedy decoding takes no {' or '.join(unused)}; use strategy: "
                "sample to sample"
            )
        return {"strategy": strategy, "temperature": None, "top_p": 1.0, "dtype": dtype}
    if not _is_finite(temperature) or temperature <= 0:
        raise ValueError(
            f"sampling needs a temperature above 0, such as 0.7, not {temperature!r}"
        )
    if not _is_finite(top_p) or not 0 < top_p <= 1:
        raise ValueError(f"top_p must be above 0 and at most 1, not {top_p!r}")
    return {
        "strategy": strategy,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "dtype": dtype,
    }


def _seeds(data: object, strategy: str) -> tuple[int, ...]:
    if (
        not isinstance(data, list)
        or not data
        or not all(_is_whole(seed) and 0 <= seed <= MAX_SEED for seed in data)
    ):
        raise ValueError(
            f"seeds must be a list of whole numbers from 0 to {MAX_SEED}, such as "
            f"[0, 1, 2], not {data!r}"
        )
    for i, seed in enumerate(data):
        if seed in data[:i]:
            raise ValueError(f"seed {seed} is given twice")
    if strategy == "greedy" and len(data) > 1:
        raise ValueError(
            f"greedy decoding takes one seed, since it draws no random numbers, "
            f"not {len(data)}"
        )
    return tuple(data)


def _whole(name: str, value: object) -> int:
    if not _is_whole(value) or value < 1:
        raise ValueError(f"{name} must be a whole number from 1, not {value!r}")
    return value


def spec_from_mapping(data: object) -> ExperimentSpec:
    """Check an experiment's settings, as parsed from YAML, and return its spec.

    Every problem is a ValueError that names the setting, and the condition
    by its place and label.

    Example:
        >>> spec_from_mapping({"name": "x"})  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: the experiment needs model, prompts, decoding, seeds, ...
    """
    if not isinstance(data, Mapping):
        raise ValueError(
            "an experiment must be a mapping of its settings, such as name: and "
            f"model:, not {data!r}"
        )
    refuse_unknown_keys(data, SPEC_KEYS, "")
    if missing := [key for key in SPEC_KEYS if key not in data]:
        raise ValueError(f"the experiment needs {', '.join(missing)}")
    name, model, prompts = data["name"], data["model"], data["prompts"]
    if name is None:
        raise ValueError("the experiment needs a name, such as pulse-lengths")
    if not isinstance(name, str):
        raise ValueError(not_text("name", name, "pulse-lengths"))
    if not EXPERIMENT_NAME.fullmatch(name):
        raise ValueError(
            "name must be lowercase letters and digits joined by hyphens, such as "
            f"pulse-lengths, not {name!r}"
        )
    models = sorted(n for n, spec in REGISTRY.items() if spec.kind != "test")
    if not isinstance(model, str) or model not in models:
        what = (
            "is the test suite's model"
            if isinstance(model, str) and model in REGISTRY
            else "is not in the model registry"
        )
        raise ValueError(
            f"model {model!r} {what}; an experiment can use {', '.join(models)}"
        )
    if not isinstance(prompts, str) or not prompts.strip():
        raise ValueError(
            "prompts must be the path of a prompt set, such as prompts/smoke.jsonl, "
            f"not {prompts!r}"
        )
    decoding = _decoding(data["decoding"])
    seeds = _seeds(data["seeds"], decoding["strategy"])
    max_new_tokens = _whole("max_new_tokens", data["max_new_tokens"])
    batch_size = _whole("batch_size", data["batch_size"])
    hours = data["max_gpu_hours"]
    if not _is_finite(hours) or hours <= 0:
        raise ValueError(
            f"max_gpu_hours must be a number above 0, such as 0.5, not {hours!r}"
        )
    listed = data["conditions"]
    if not isinstance(listed, list) or not listed:
        raise ValueError(f"conditions must list at least one condition, not {listed!r}")
    conditions = []
    for i, item in enumerate(listed):
        condition = condition_from_mapping(item, i)
        if any(condition.label == earlier.label for earlier in conditions):
            raise ValueError(
                f"conditions[{i}]: the label {condition.label!r} is used twice; "
                "each condition needs its own"
            )
        conditions.append(condition)
    return ExperimentSpec(
        name,
        model,
        prompts,
        decoding,
        seeds,
        max_new_tokens,
        batch_size,
        float(hours),
        tuple(conditions),
    )


def spec_from_yaml(text: str) -> ExperimentSpec:
    """Read and check an experiment from YAML text, as load_yaml parses it.

    Example:
        >>> spec_from_yaml("- 1")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: an experiment must be a mapping of its settings, ...
    """
    return spec_from_mapping(load_yaml(text))


def read_spec(path: str | Path) -> ExperimentSpec:
    """Read an experiment file. Its problems are refused with the file's name.

    Example:
        >>> read_spec("experiments/p0/missing.yaml")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: experiments/p0/missing.yaml: cannot read it: ...
    """
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ValueError(f"{path}: cannot read it: {error.strerror or error}") from None
    except ValueError as error:  # UnicodeDecodeError, for a file not in UTF-8
        raise ValueError(f"{path}: not UTF-8 text: {error}") from None
    try:
        return spec_from_yaml(text)
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from None
