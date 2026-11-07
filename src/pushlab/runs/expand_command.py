"""`pushlab expand`: the runs an experiment file means, listed or counted.

Nothing is run and no model is loaded. The command reads the file, checks
every condition with validate(), draws the random controls and reads the
prompt set (pushlab.runs.expand). Then it prints every planned run, one per
line with tab-separated columns, or with --summary the settings and how
many runs each condition gives. validate()'s warnings go to stderr; a
problem goes there too, naming the file, with exit status 1.
"""

import sys
from pathlib import Path

from pushlab.runs.expand import Expansion, expand
from pushlab.runs.spec import ExperimentSpec, read_spec

COLUMNS = (
    "run_id",
    "condition",
    "control",
    "prompt_id",
    "seed",
    "batch_key",
    "control_of",
)


def run_list(expansion: Expansion) -> str:
    """Every planned run, one line each, under a line naming the columns.

    control is 0 for a condition's own run and k for its k-th control.

    Example:
        >>> from pushlab import PROJECT_ROOT
        >>> spec = read_spec(PROJECT_ROOT / "experiments/p0/s2_pulse_lengths.yaml")
        >>> lines = run_list(expand(spec)).splitlines()
        >>> lines[0].split("\\t")  # doctest: +NORMALIZE_WHITESPACE
        ['run_id', 'condition', 'control', 'prompt_id', 'seed', 'batch_key',
         'control_of']
        >>> lines[3].split("\\t")  # doctest: +NORMALIZE_WHITESPACE
        ['s2-pulse-lengths-000003', 'pulse length=4', '1', 'smoke-01', '0',
         'b000001:2', 's2-pulse-lengths-000002']
    """
    lines = ["\t".join(COLUMNS)]
    for run in expansion.runs:
        record = run.record
        fields = [
            record.run_id,
            run.condition,
            str(run.control),
            record.prompt_id,
            str(record.seed),
            record.batch_key,
            record.control_of,
        ]
        lines.append("\t".join(fields))
    return "\n".join(lines) + "\n"


def _table(counts: dict[str, tuple[int, int]]) -> list[str]:
    rows = [(label, own, drawn, own + drawn) for label, (own, drawn) in counts.items()]
    rows.append(("all", *(sum(row[i] for row in rows) for i in (1, 2, 3))))
    header = ("condition", "own", "controls", "total")
    width = [max(len(header[0]), *(len(row[0]) for row in rows))]
    width += [
        max(len(header[i]), *(len(f"{row[i]:,}") for row in rows)) for i in (1, 2, 3)
    ]
    lines = [
        f"{header[0]:<{width[0]}}"
        + "".join(f"  {header[i]:>{width[i]}}" for i in (1, 2, 3))
    ]
    for row in rows:
        numbers = "".join(f"  {row[i]:>{width[i]},}" for i in (1, 2, 3))
        lines.append(f"{row[0]:<{width[0]}}{numbers}")
    return lines


def summary(spec: ExperimentSpec, expansion: Expansion) -> str:
    """The spec's settings and how many runs each condition gives.

    Example:
        >>> from pushlab import PROJECT_ROOT
        >>> spec = read_spec(PROJECT_ROOT / "experiments/p0/s2_pulse_lengths.yaml")
        >>> print(summary(spec, expand(spec)).splitlines()[-1])
        50 runs, of which 30 are controls, at most 4,800 reply tokens. Nothing was run.
    """
    records = [run.record for run in expansion.runs]
    prompts = len({record.prompt_id for record in records})
    batches = len({record.batch_key.split(":")[0] for record in records})
    decoding = spec.decoding
    how = decoding["strategy"]
    if how == "sample":
        how += f", temperature {decoding['temperature']}, top_p {decoding['top_p']}"
    controls = sum(1 for run in expansion.runs if run.control)
    settings = [
        ("experiment", spec.name),
        ("spec sha256", expansion.spec_sha256),
        ("model", records[0].model),
        ("prompts", f"{spec.prompts}, {prompts} prompts"),
        ("prompts sha256", expansion.prompts_sha256),
        ("decoding", f"{how}, {decoding['dtype']}"),
        ("seeds", ", ".join(str(seed) for seed in spec.seeds)),
        ("max_new_tokens", f"{spec.max_new_tokens}"),
        ("batches", f"{batches}, each at most {spec.batch_size} runs"),
        ("max_gpu_hours", f"{spec.max_gpu_hours:g}"),
    ]
    width = max(len(name) for name, _ in settings)
    lines = [f"{name:<{width}}  {value}" for name, value in settings]
    lines += ["", *_table(expansion.counts()), ""]
    tokens = len(records) * spec.max_new_tokens
    lines.append(
        f"{len(records):,} runs, of which {controls:,} are controls, at most "
        f"{tokens:,} reply tokens. Nothing was run."
    )
    return "\n".join(lines) + "\n"


def _refuse(message: str) -> int:
    print(f"pushlab expand: {message}", file=sys.stderr)
    return 1


def run_expand(path: Path, summary_only: bool = False) -> int:
    """Print the runs the experiment file at path means; return the exit status.

    Example:
        >>> run_expand(
        ...     Path("experiments/p0/s2_pulse_lengths.yaml"), True
        ... )  # doctest: +SKIP
        0
    """
    try:
        spec = read_spec(path)
    except ValueError as error:  # it names the file
        return _refuse(str(error))
    try:
        expansion = expand(spec)
    except ValueError as error:
        return _refuse(f"{path}: {error}")
    for warning in expansion.warnings:
        print(f"pushlab expand: warning: {warning}", file=sys.stderr)
    print(summary(spec, expansion) if summary_only else run_list(expansion), end="")
    return 0
