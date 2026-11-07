"""`pushlab expand`: the runs an experiment file means, listed or counted.

Nothing is run and no model is loaded. The command reads the file, checks
every condition with validate(), reads the prompt set and draws the random
controls (pushlab.runs.expand). Then it prints a line naming the columns
and every planned run, one per line with tab-separated columns, or with
--summary the settings and how many runs each condition's schedules give.
validate()'s warnings go to stderr; a problem goes there too, naming the
file, with exit status 1.
"""

import csv
import io
import sys
from collections import Counter
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
# The summary's last row; no condition's label can be written like it.
ALL = "(all)"


def run_list(expansion: Expansion) -> str:
    """Every planned run, one line each, under a line naming the columns.

    Columns are separated by tabs, and a field holding a tab, a line break
    or a double quote is quoted as CSV quotes it. control is 0 for a
    condition's own run and k for its k-th control.

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
    text = io.StringIO()
    writer = csv.writer(text, delimiter="\t", lineterminator="\n")
    writer.writerow(COLUMNS)
    for run in expansion.runs:
        record = run.record
        writer.writerow(
            [
                record.run_id,
                run.condition,
                run.control,
                record.prompt_id,
                record.seed,
                record.batch_key,
                record.control_of,
            ]
        )
    return text.getvalue()


def _count(number: int, one: str, many: str | None = None) -> str:
    """A count with its noun, such as '1 run' or '2 runs'."""
    return f"{number:,} {one if number == 1 else many or one + 's'}"


def _exact(value: float) -> str:
    """A number, short when that is exact, every digit when not."""
    short = format(value, "g")
    return short if float(short) == value else repr(value)


def _table(counts: dict[str, tuple[int, int]]) -> list[str]:
    rows = [(label, own, drawn, own + drawn) for label, (own, drawn) in counts.items()]
    rows.append((ALL, *(sum(row[i] for row in rows) for i in (1, 2, 3))))
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
    """The spec's settings and how many runs each condition's schedules give.

    Example:
        >>> from pushlab import PROJECT_ROOT
        >>> spec = read_spec(PROJECT_ROOT / "experiments/p0/s2_pulse_lengths.yaml")
        >>> print(summary(spec, expand(spec)).splitlines()[-1])
        50 runs, of which 30 are controls, at most 4,800 reply tokens. Nothing was run.
    """
    records = [run.record for run in expansion.runs]
    prompts = len({record.prompt_id for record in records})
    sizes = Counter(record.batch_key.split(":")[0] for record in records).values()
    low, high = min(sizes), max(sizes)
    each = _count(high, "run") if low == high else f"{low} to {high} runs"
    decoding = spec.decoding
    how = decoding["strategy"]
    if how == "sample":
        how += f", temperature {decoding['temperature']}, top_p {decoding['top_p']}"
    settings = [
        ("experiment", spec.name),
        ("spec sha256", expansion.spec_sha256),
        ("model", records[0].model),
        ("prompts", f"{spec.prompts}, {_count(prompts, 'prompt')}"),
        ("prompts sha256", expansion.prompts_sha256),
    ]
    if expansion.known_sha256 is not None:
        settings.append(("known sha256", expansion.known_sha256))
    settings += [
        ("decoding", f"{how}, {decoding['dtype']}"),
        ("seeds", ", ".join(str(seed) for seed in spec.seeds)),
        ("max_new_tokens", f"{spec.max_new_tokens}"),
        ("batches", f"{len(sizes):,}, of {each} each"),
        ("max_gpu_hours", _exact(spec.max_gpu_hours)),
    ]
    width = max(len(name) for name, _ in settings)
    lines = [f"{name:<{width}}  {value}" for name, value in settings]
    lines += ["", *_table(expansion.counts()), ""]
    controls = sum(1 for run in expansion.runs if run.control)
    which = "is a control" if controls == 1 else "are controls"
    tokens = _count(len(records) * spec.max_new_tokens, "reply token")
    lines.append(
        f"{_count(len(records), 'run')}, of which {controls:,} {which}, at most "
        f"{tokens}. Nothing was run."
    )
    return "\n".join(lines) + "\n"


def _refuse(message: str) -> int:
    print(f"pushlab expand: {message}", file=sys.stderr)
    return 1


def run_expand(path: Path, summary_only: bool = False) -> int:
    """Print the runs the experiment file at path means; return the exit status.

    The status is 0 once they are printed, and 1, with the reason on stderr,
    when the file or one of its conditions is refused.

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
