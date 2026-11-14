"""`pushlab ladder`: write a dose ladder's experiment file (decision 0022).

It reads the units file (pushlab.units.ladder_units), the doses (numbers
above 0 in increasing order, each once) and the seeds of its options, and
the prompt set, and builds the ladder (pushlab.runs.ladder): one condition
per unit, u01, u02 and so on, pushing it at every dose in its direction.
It writes the file to --out only once it reads back as the experiment it
means and that experiment expands, so a file that would not run is never
left. Then it prints the units, the doses, the prompts and seeds, the
runs, max_gpu_hours and what to do next. max_gpu_hours is by default 1.5
times the time the runs' reply tokens take at the model's speed, the one
`pushlab run --dry-run` estimates with (pushlab.runs.estimate), rounded up
to a quarter hour.

It refuses, writing nothing, with the reason on stderr: doses or seeds
that --doses and --seeds do not take; a units file the reader refuses, or
one with a set of neurons or a persona named "none", whose replies could
never be judged (pushlab.analysis.no_persona); an --out that is a folder,
that exists without --overwrite, or that is inside the repository while a
unit comes from the Neuron Persona Library, whose data stays out of git
(decision 0020); a name that is not an experiment's; a prompt set that
cannot be read; a model with no known speed, without --max-gpu-hours; and
a ladder that does not expand. It loads no model and runs nothing.
"""

import argparse
import math
import re
import sys
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from pushlab import PROJECT_ROOT
from pushlab.analysis.no_persona import refuse_no_persona
from pushlab.cli_ladder import BATCH_SIZE, MAX_NEW_TOKENS
from pushlab.prompts import read_prompt_set
from pushlab.runs.estimate import exact, read_costs, speed_for
from pushlab.runs.folder import default_runs_dir
from pushlab.runs.ladder import (
    HOURS_MARGIN,
    TEMPERATURE,
    Ladder,
    default_hours,
    describe_unit,
    doses_text,
    write_ladder,
)
from pushlab.schedules.validation import DEFAULT_CEILING
from pushlab.store.folders import shown_path
from pushlab.store.records import EXPERIMENT_NAME, MAX_SEED
from pushlab.units.ladder_units import read_ladder_units

DTYPE = "bfloat16"  # the data type a ladder decodes in: an experiment's default
OUTSIDE = "/workspace/experiments"  # where a ladder of library units goes
EXAMPLE = "such as 0.25,0.5,0.75,1.0,1.25"
_NUMBER = re.compile(r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?")


def _refuse(message: str) -> int:
    print(f"pushlab ladder: {message}", file=sys.stderr)
    return 1


def parse_doses(text: str) -> tuple[float, ...]:
    """The doses of --doses: numbers above 0, separated by commas, each once
    and in increasing order. ValueError names the dose that is not.

    Example:
        >>> parse_doses("0.25,0.5,1")
        (0.25, 0.5, 1.0)
        >>> parse_doses("0.5,0.25")
        Traceback (most recent call last):
        ...
        ValueError: --doses must be in increasing order: 0.25 comes after 0.5
    """
    if not text.strip():
        raise ValueError(f"--doses is empty: give doses {EXAMPLE}")
    doses: list[tuple[str, float]] = []  # each as written, and its number
    for part in (part.strip() for part in text.split(",")):
        if part.startswith("-") and _NUMBER.fullmatch(part[1:]):
            raise ValueError(
                f"--doses: {part} is not above 0: give doses without a sign, as "
                "each unit's direction gives its push's"
            )
        if not _NUMBER.fullmatch(part):
            raise ValueError(
                f"--doses: {part!r} is not a dose: give numbers above 0 separated "
                f"by commas, {EXAMPLE}"
            )
        dose = float(part)
        if not math.isfinite(dose):
            raise ValueError(f"--doses: {part} is too large")
        if dose <= 0:
            raise ValueError(f"--doses: {part} is not above 0")
        if same := [written for written, earlier in doses if earlier == dose]:
            raise ValueError(
                f"--doses: {part} is the dose {same[0]} again; give each once"
            )
        if doses and dose < doses[-1][1]:
            earlier = doses[-1][0]
            raise ValueError(
                f"--doses must be in increasing order: {part} comes after {earlier}"
            )
        doses.append((part, dose))
    return tuple(dose for _, dose in doses)


def parse_seeds(text: str) -> tuple[int, ...]:
    """The seeds of --seeds: whole numbers from 0, separated by commas, each
    once. ValueError names the seed that is not.

    Example:
        >>> parse_seeds("0, 1,2")
        (0, 1, 2)
    """
    seeds: list[int] = []
    for part in (part.strip() for part in text.split(",")):
        if not (part.isascii() and part.isdigit()):
            raise ValueError(
                f"--seeds: {part!r} is not a seed: give whole numbers from 0 "
                "separated by commas, such as 0,1,2"
            )
        if (seed := int(part)) > MAX_SEED:
            raise ValueError(
                f"--seeds: {seed} is too large: seeds are at most {MAX_SEED}"
            )
        if seed in seeds:
            raise ValueError(f"--seeds: seed {seed} is given twice")
        seeds.append(seed)
    return tuple(seeds)


def experiment_name(name: str | None, out: Path) -> str:
    """The experiment's name: name, else one made from out's stem,
    lowercased, each run of characters other than a to z and 0 to 9 made
    one hyphen, with none at either end. ValueError when it is not an
    experiment's name.

    Example:
        >>> experiment_name(None, Path("/workspace/experiments/Panel_1 Ladder.yaml"))
        'panel-1-ladder'
    """
    if name is None:
        name = re.sub(r"[^a-z0-9]+", "-", Path(out).stem.lower()).strip("-")
        if not EXPERIMENT_NAME.fullmatch(name):
            raise ValueError(
                f"cannot make an experiment name from {Path(out).name}: give --name, "
                "lowercase letters and digits joined by hyphens, such as "
                "panel-1-ladder"
            )
    elif not EXPERIMENT_NAME.fullmatch(name):
        raise ValueError(
            "--name must be lowercase letters and digits joined by hyphens, such as "
            f"panel-1-ladder, not {name!r}"
        )
    return name


def inside_repository(path: Path) -> bool:
    """Whether a file written at path lands inside the repository: its folder
    with any link followed, then its own name, which a write replaces.

    Example:
        >>> inside_repository(PROJECT_ROOT / "experiments" / "x.yaml")
        True
        >>> inside_repository(Path("/workspace/experiments/x.yaml"))
        False
    """
    where = Path(path).absolute()
    return (where.parent.resolve() / where.name).is_relative_to(PROJECT_ROOT)


def _counted(count: int, noun: str) -> str:
    return f"{count:,} {noun if count == 1 else noun + 's'}"


def runs_text(ladder: Ladder, prompts: int) -> str:
    """The runs a ladder makes, as units x doses x prompts x seeds, plus its
    baseline and controls.

    Example:
        >>> from pushlab.units.ladder_units import LadderUnit
        >>> from pushlab.units.neuron import NeuronUnit
        >>> owl = LadderUnit(NeuronUnit.parse("qwen3-8b:L9:N1"), 1)
        >>> runs_text(Ladder("owl", (owl,), (0.5, 1.0), "p"), prompts=32)
        '1 unit x 2 doses x 32 prompts x 1 seed = 64'
    """
    seeds = len(ladder.seeds)
    pushed = len(ladder.units) * len(ladder.doses) * prompts * seeds
    text = (
        f"{_counted(len(ladder.units), 'unit')} x {_counted(len(ladder.doses), 'dose')}"
        f" x {_counted(prompts, 'prompt')} x {_counted(seeds, 'seed')} = {pushed:,}"
    )
    more = [f"{prompts * seeds:,} unpushed (baseline)"] if ladder.baseline else []
    if ladder.controls:
        more.append(f"{pushed * ladder.controls:,} controls")
    if more:
        text += f", plus {' and '.join(more)}: {ladder.run_count(prompts):,} in all"
    return text


def seeds_text(seeds: Sequence[int]) -> str:
    """The seeds, and how a reply is decoded with them.

    Example:
        >>> seeds_text([0]), seeds_text([0, 1, 2])
        ('0, greedy', '0, 1 and 2, each reply sampled at temperature 1')
    """
    if len(seeds) == 1:
        return f"{seeds[0]}, greedy"
    listed = f"{', '.join(map(str, seeds[:-1]))} and {seeds[-1]}"
    return f"{listed}, each reply sampled at temperature {exact(TEMPERATURE)}"


def _check_out(ladder: Ladder, out: Path, overwrite: bool) -> None:
    """Refuse an out that is a folder, that is inside the repository while a
    unit is the library's, or that exists, unless overwrite."""
    if out.is_dir():
        raise ValueError(f"--out must be a file, but {out} is a folder")
    if ladder.from_library and inside_repository(out):
        raise ValueError(
            f"{out} is inside the repository, but its units from the Neuron Persona "
            "Library hold the library's data, which stays out of git (decision "
            f"0020): write it outside, such as {OUTSIDE}/{out.name}"
        )
    if out.exists() and not overwrite:
        raise ValueError(f"{out} already exists; pass --overwrite to replace it")


def _prompt_count(prompts: Path) -> int:
    """How many prompts the prompt set has, read as the experiment reads it:
    from the repository's root, unless its path is absolute."""
    path = PROJECT_ROOT / prompts
    try:
        return len(read_prompt_set(path).prompt_ids)
    except (OSError, ValueError) as error:
        why = getattr(error, "strerror", None) or error
        hint = ""
        if not prompts.is_absolute():
            hint = "; --prompts is a path from the repository's root, or absolute"
        raise ValueError(f"cannot read the prompt set {path}: {why}{hint}") from None


def _hours(ladder: Ladder, prompts: int, given: float | None) -> tuple[float, str]:
    """max_gpu_hours, and how it was found: given, else the default from the
    model's speed. ValueError when the speed is not known."""
    if given is not None:
        return given, "given with --max-gpu-hours"
    try:
        costs = read_costs()
    except ValueError as error:
        raise ValueError(f"{error}; or give --max-gpu-hours") from None
    speed = speed_for(ladder.model, DTYPE, default_runs_dir(), costs)
    if speed is None:
        raise ValueError(
            f"{shown_path(costs.path)} has no speed for {ladder.model}: add one "
            "there, or give --max-gpu-hours"
        )
    runs, tokens = ladder.run_count(prompts), ladder.max_new_tokens
    hours = default_hours(runs, tokens, speed.reply_tokens_per_second)
    how = (
        f"about {HOURS_MARGIN:g} times the time its {runs * tokens:,} reply tokens "
        f"take at {speed.text()}, rounded up to a quarter hour"
    )
    return hours, how


def _report(ladder: Ladder, out: Path, prompts: int, how: str) -> None:
    """Print what was written: the units, doses, prompts, seeds, runs and
    max_gpu_hours, any note, and what to do next."""
    hours = exact(ladder.max_gpu_hours)
    hours += ", as given" if how.startswith("given") else f": {how}"
    lines = [("units", f"{len(ladder.units):,}, a condition each:")]
    lines += [
        ("", f"  {label}  {describe_unit(unit)}")
        for label, unit in zip(ladder.labels, ladder.units, strict=True)
    ]
    lines += [
        ("doses", f"{doses_text(ladder.doses)}, in each unit's direction"),
        ("prompts", f"{ladder.prompts}: {_counted(prompts, 'prompt')}"),
        ("seeds", seeds_text(ladder.seeds)),
        ("runs", runs_text(ladder, prompts)),
        ("max_gpu_hours", hours),
    ]
    width = max(len(name) for name, _ in lines) + 2
    print(f"Wrote {out}: the experiment {ladder.name}, on {ladder.model}.")
    print("\n".join(f"{name:<{width}}{text}" if name else text for name, text in lines))
    notes = []
    if ladder.over_ceiling:
        notes.append(
            f"Doses above the ceiling of {exact(DEFAULT_CEILING)} are allowed in it, "
            "on purpose: the ladder seeks the dose where the text breaks."
        )
    if ladder.from_library:
        notes.append(
            "It holds the Neuron Persona Library's data, which states no licence: "
            "keep it out of git (decision 0020)."
        )
    if notes:
        print("\n" + "\n".join(notes))
    print(
        "\nNext, see what it would run and cost; no model is loaded:\n"
        f"  uv run pushlab run {out} --dry-run\n"
        "then run it on the pod, score it and read its doses, as "
        "docs/runbooks/ladder.md says."
    )


def run_ladder(
    units_file: Path,
    doses: str,
    prompts: Path | str,
    out: Path,
    *,
    name: str | None = None,
    seeds: str = "0",
    max_new_tokens: int = MAX_NEW_TOKENS,
    batch_size: int = BATCH_SIZE,
    max_gpu_hours: float | None = None,
    controls: int = 0,
    control_seed: int | None = None,
    baseline: bool = False,
    prompt_pass: bool = True,
    overwrite: bool = False,
) -> int:
    """Write the ladder of units_file's units at doses (a comma list, as
    --doses takes it) on the prompt set prompts to out; the exit status.

    seeds is a comma list as --seeds takes it; name defaults to one made
    from out's (experiment_name), and max_gpu_hours to 1.5 times the
    estimate. Returns 0 once out is written, else 1 with the reason on
    stderr, having written nothing.

    Example:
        >>> run_ladder(Path("units.csv"), "", "p", Path("o.yaml"))  # stderr says why
        1
    """
    out, prompts = Path(out), Path(prompts)
    try:
        dose_values, seed_values = parse_doses(doses), parse_seeds(seeds)
        if control_seed is not None and not controls:
            raise ValueError("--control-seed goes with --controls of 1 or more")
        units = read_ladder_units(units_file)
        refuse_no_persona(units_file, units)
        ladder = Ladder(
            experiment_name(name, out),
            tuple(units),
            dose_values,
            prompts.as_posix(),
            seed_values,
            max_new_tokens,
            batch_size,
            controls=controls,
            control_seed=control_seed,
            baseline=baseline,
            prompt_pass=prompt_pass,
            units_file=str(units_file),
        )
        _check_out(ladder, out, overwrite)
        count = _prompt_count(prompts)
        hours, how = _hours(ladder, count, max_gpu_hours)
        note = f"{how[0].upper()}{how[1:]}."
        ladder = replace(ladder, max_gpu_hours=hours, hours_note=note)
        _, expansion = write_ladder(ladder, out, overwrite)
    except ValueError as error:
        return _refuse(str(error))
    except OSError as error:  # out written meanwhile, or the system's own
        if error.strerror is None:  # write_ladder's FileExistsError
            return _refuse(str(error))
        return _refuse(f"cannot write {out}: {error.strerror}")
    for warning in expansion.warnings:
        print(f"pushlab ladder: warning: {warning}", file=sys.stderr)
    _report(ladder, out, count, how)
    return 0


def run_ladder_command(args: argparse.Namespace) -> int:
    """`pushlab ladder` with its parsed options (pushlab.cli_ladder); the
    exit status.

    Example:
        >>> from pushlab.cli_parser import build_parser
        >>> argv = ["ladder", "--units", "u.csv", "--doses", "1,0.5"]
        >>> args = build_parser().parse_args([*argv, "--prompts", "p", "--out", "o"])
        >>> run_ladder_command(args)  # stderr says why
        1
    """
    return run_ladder(
        args.units,
        args.doses,
        args.prompts,
        args.out,
        name=args.name,
        seeds=args.seeds,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        max_gpu_hours=args.max_gpu_hours,
        controls=args.controls,
        control_seed=args.control_seed,
        baseline=args.baseline,
        prompt_pass=args.prompt_pass,
        overwrite=args.overwrite,
    )
