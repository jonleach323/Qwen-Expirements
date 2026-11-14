"""A stage's experiment file (decision 0023): its settings, and writing it,
for pushlab panel stage.

Every stage writes an experiment for the panel's model on the selection's
prompt set (configs/panel.yaml, pushlab.panel.settings): greedy with one
seed, the selection's max_new_tokens and batch_size, fixed calibration
(decision 0007), and max_gpu_hours as pushlab ladder works it out, 1.5
times the time its reply tokens take at the model's speed, rounded up to a
quarter hour, unless given. A Stage holds the rest: the prompts it runs, its
candidates: setting, its conditions, how many runs they make, and the
comments that say what it is.

pushlab.panel.stage_text writes the file, so that load_yaml reads its
settings back exactly; write_stage puts it in place only once it reads back
as the experiment meant and that experiment expands, as pushlab ladder
does, so a file that would not run is never left. A stage's file names
candidates by key, the Neuron Persona Library's among them, whose data
stays out of git (decision 0020), and the pod never changes the
repository: check_out refuses an --out inside it.
"""

import math
import os
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pushlab import PROJECT_ROOT
from pushlab.panel.settings import Selection
from pushlab.prompts import read_prompt_set
from pushlab.runs.estimate import read_costs, speed_for
from pushlab.runs.expand import Expansion, expand
from pushlab.runs.folder import default_runs_dir
from pushlab.runs.ladder import HOURS_MARGIN, default_hours
from pushlab.runs.ladder_command import inside_repository
from pushlab.runs.spec import ExperimentSpec, read_spec, spec_from_yaml
from pushlab.schedules.yaml_text import load_yaml
from pushlab.store.folders import shown_path

DTYPE = "bfloat16"  # a stage decodes in an experiment's default data type
OUTSIDE = "/workspace/experiments"  # where a stage's file goes on the pod


@dataclass(frozen=True)
class Stage:
    """What a stage writes, but its name and max_gpu_hours: the model; the
    prompt IDs it runs, None for every prompt of the set; its candidates:
    setting, None without one; its conditions; how many runs they make;
    about, what the file is, for its top; key_notes, a comment beside each
    of the candidates' keys; condition_notes, a comment above conditions by
    their place; shown, the lines it prints of what it read and kept; and
    from_library, whether it pushes the Neuron Persona Library's units.

    Example:
        >>> none = {"label": "none", "segments": []}
        >>> stage = Stage("qwen3-8b", None, None, (none,), 32, "A test.")
        >>> stage.runs, stage.key_notes, stage.from_library
        (32, (), False)
    """

    model: str
    prompt_ids: tuple[str, ...] | None
    candidates: dict[str, Any] | None
    conditions: tuple[dict[str, Any], ...]
    runs: int
    about: str
    key_notes: tuple[str, ...] = ()
    condition_notes: dict[int, str] = field(default_factory=dict)
    shown: tuple[str, ...] = ()
    from_library: bool = False


def stage_mapping(
    stage: Stage, selection: Selection, name: str, hours: float
) -> dict[str, Any]:
    """The experiment's settings as plain data, in the file's order, as
    spec_from_mapping reads them.

    Example:
        >>> from pushlab.panel.settings import read_selection
        >>> stage = Stage("qwen3-8b", ("heldout32-01",), None, (), 1, "")
        >>> mapping = stage_mapping(stage, read_selection(), "s3-x", 0.25)
        >>> list(mapping)[:5], mapping["calibration"]
        (['name', 'model', 'prompts', 'prompt_ids', 'decoding'], 'fixed')
    """
    mapping: dict[str, Any] = {
        "name": name,
        "model": stage.model,
        "prompts": selection.prompts,
    }
    if stage.prompt_ids is not None:
        mapping["prompt_ids"] = list(stage.prompt_ids)
    mapping |= {
        "decoding": {"strategy": "greedy"},
        "seeds": [0],
        "max_new_tokens": selection.max_new_tokens,
        "batch_size": selection.batch_size,
        "calibration": "fixed",
        "max_gpu_hours": float(hours),
    }
    if stage.candidates is not None:
        mapping["candidates"] = stage.candidates
    mapping["conditions"] = [dict(condition) for condition in stage.conditions]
    return mapping


def check_out(out: Path, source: Path, overwrite: bool) -> None:
    """Refuse an --out that is a folder, the file the stage reads, inside the
    repository, or that exists, unless overwrite.

    Example:
        >>> check_out(PROJECT_ROOT / "b.yaml", Path("a.yaml"), False)
        ... # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: .../b.yaml is inside the repository, ...
    """
    out, source = Path(out), Path(source)
    if out.is_dir():
        raise ValueError(f"--out must be a file, but {out} is a folder")
    if out.exists() and source.exists() and out.resolve() == source.resolve():
        raise ValueError(f"--out is {source}, the file the stage reads: give another")
    if inside_repository(out):
        raise ValueError(
            f"{out} is inside the repository, {PROJECT_ROOT}: a stage's file names "
            "candidates by key, the Neuron Persona Library's among them, whose data "
            "stays out of git (decision 0020), and the pod never changes the "
            f"repository; write it outside, such as {OUTSIDE}/{out.name}"
        )
    if out.exists() and not overwrite:
        raise ValueError(f"{out} already exists; pass --overwrite to replace it")


def next_out(out: Path, stage: str, upcoming: str) -> Path:
    """Where the next stage's file could go: beside out, named as out with
    its stage's name put right.

    Example:
        >>> next_out(Path("/workspace/experiments/s3-stage-b.yaml"), "b", "ladder")
        PosixPath('/workspace/experiments/s3-stage-ladder.yaml')
        >>> next_out(Path("/w/ladder.yaml"), "ladder", "confirm")
        PosixPath('/w/confirm.yaml')
        >>> next_out(Path("/w/s3-best.yaml"), "b", "ladder")
        PosixPath('/w/s3-best-ladder.yaml')
    """
    stem = out.stem
    if stem == stage:
        return out.with_name(upcoming + out.suffix)
    for joint in ("-", "_"):
        if stem.endswith(joint + stage):
            kept = stem[: -len(stage)]
            return out.with_name(kept + upcoming + out.suffix)
    return out.with_name(f"{stem}-{upcoming}{out.suffix}")


def stage_hours(
    model: str,
    runs: int,
    max_new_tokens: int,
    given: float | None,
    runs_dir: Path | None = None,
) -> tuple[float, str]:
    """max_gpu_hours, and how it was found: given, else 1.5 times the time
    runs replies of max_new_tokens take at the model's speed, the one
    `pushlab run --dry-run` estimates with, rounded up to a quarter hour.
    ValueError for a given number that is not above 0, or a speed not known.

    Example:
        >>> stage_hours("qwen3-8b", 3_840, 160, None, Path("/no/such"))[0]
        11.75
        >>> stage_hours("qwen3-8b", 3_840, 160, 2.5)
        (2.5, 'given with --max-gpu-hours')
    """
    if given is not None:
        if not (math.isfinite(given) and given > 0):
            raise ValueError(
                f"--max-gpu-hours must be a number above 0, such as 2.5, not {given!r}"
            )
        return float(given), "given with --max-gpu-hours"
    try:
        costs = read_costs()
    except ValueError as error:
        raise ValueError(f"{error}; or give --max-gpu-hours") from None
    where = default_runs_dir() if runs_dir is None else Path(runs_dir)
    speed = speed_for(model, DTYPE, where, costs)
    if speed is None:
        raise ValueError(
            f"{shown_path(costs.path)} has no speed for {model}: add one there, or "
            "give --max-gpu-hours"
        )
    hours = default_hours(runs, max_new_tokens, speed.reply_tokens_per_second)
    how = (
        f"about {HOURS_MARGIN:g} times the time its {runs * max_new_tokens:,} reply "
        f"tokens take at {speed.text()}, rounded up to a quarter hour"
    )
    return hours, how


def set_prompts(selection: Selection) -> tuple[str, ...]:
    """The IDs of the selection's prompt set, in order. ValueError when it
    cannot be read.

    Example:
        >>> from pushlab.panel.settings import read_selection
        >>> set_prompts(read_selection())[:2]
        ('heldout32-01', 'heldout32-02')
    """
    path = PROJECT_ROOT / selection.prompts  # an absolute path stays as it is
    try:
        return read_prompt_set(path).prompt_ids
    except (OSError, ValueError) as error:
        why = getattr(error, "strerror", None) or error
        raise ValueError(f"cannot read the prompt set {path}: {why}") from None


def chosen_prompts(
    prompts: Sequence[str], count: int, name: str, last: bool = False
) -> list[str]:
    """The first count of prompts, or with last the last count, in order.
    ValueError, naming the setting, when the set has fewer.

    Example:
        >>> chosen_prompts(["p1", "p2", "p3"], 2, "preview_prompts", last=True)
        ['p2', 'p3']
    """
    if count > len(prompts):
        raise ValueError(
            f"{name} is {count}, but the prompt set has only {len(prompts)} prompts"
        )
    return list(prompts[len(prompts) - count :] if last else prompts[:count])


def write_stage(
    text: str, mapping: Mapping[str, Any], out: Path, overwrite: bool = False
) -> tuple[ExperimentSpec, Expansion]:
    """Write text, a stage's file whose settings are mapping, to out, and
    return its spec and expansion.

    The text must read back as mapping, and the experiment must expand
    (pushlab.runs.expand): else ValueError says why, before any file is
    made. The file is written beside out under a hidden name, read back with
    read_spec, and only then renamed to out, as write_ladder does, so
    nothing is left of a write that fails. FileExistsError when out exists,
    unless overwrite.
    """
    out = Path(out)
    if out.exists() and not overwrite:
        raise FileExistsError(f"{out} already exists; pass --overwrite to replace it")
    if load_yaml(text) != mapping:  # a bug, never the user's doing
        raise ValueError("the stage's text does not read back as its settings")
    try:
        spec = spec_from_yaml(text)
        expansion = expand(spec)
    except ValueError as error:
        raise ValueError(
            f"the experiment would not run, so nothing was written: {error}"
        ) from None
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(f".{out.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "x", encoding="utf-8") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        if read_spec(temporary) != spec:
            raise ValueError(f"{temporary} does not read back as the experiment")
        if out.exists() and not overwrite:  # written meanwhile
            raise FileExistsError(
                f"{out} already exists; pass --overwrite to replace it"
            )
        os.replace(temporary, out)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return spec, expansion
