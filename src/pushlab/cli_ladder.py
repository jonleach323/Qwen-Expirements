"""The options of `pushlab ladder`, which writes a dose ladder's experiment
file (decision 0022; pushlab.runs.ladder_command).

pushlab.cli_parser builds the whole parser from these and the others. Only
light modules are imported here, so that every --help answers at once.
"""

import argparse
import math
from pathlib import Path

from pushlab.cli_experiments import positive_int

# Long enough for the library's "Who are you?" replies, which a persona's
# first sentences show.
MAX_NEW_TOKENS = 160
BATCH_SIZE = 16


def whole_number(text: str) -> int:
    """argparse's type for a whole number from 0.

    Example:
        >>> whole_number("0")
        0
    """
    if not (text.isascii() and text.isdigit()):
        raise argparse.ArgumentTypeError(f"must be a whole number from 0, not {text!r}")
    return int(text)


def hours(text: str) -> float:
    """argparse's type for GPU-hours: a number above 0.

    Example:
        >>> hours("1.5")
        1.5
    """
    try:
        value = float(text)
    except ValueError:
        value = math.nan
    if not (math.isfinite(value) and value > 0):
        raise argparse.ArgumentTypeError(
            f"must be a number of hours above 0, such as 2.5, not {text!r}"
        )
    return value


def add_ladder(commands: argparse._SubParsersAction) -> None:
    """Add `pushlab ladder` and its options to commands.

    Example:
        >>> parser = argparse.ArgumentParser()
        >>> add_ladder(parser.add_subparsers(dest="command"))
        >>> args = parser.parse_args(
        ...     ["ladder", "--units", "u.csv", "--doses", "0.5,1", "--prompts", "p"]
        ...     + ["--out", "o.yaml"]
        ... )
        >>> args.command, args.doses, args.max_new_tokens, args.prompt_pass
        ('ladder', '0.5,1', 160, True)
    """
    ladder = commands.add_parser(
        "ladder",
        help="write an experiment file that climbs each unit up a dose ladder",
        description="Write an experiment file that pushes every unit of a units "
        "file at every dose of a ladder, to find each unit's working dose and the "
        "dose where its text breaks (decision 0022). Each unit is one condition, "
        "labelled u01, u02 and so on in the file's order: a constant push of its "
        "neuron, on the prompt pass and every reply token, with a grid of signed "
        "doses, its direction times each dose, on every prompt of the prompt set. "
        "So it runs units x doses x prompts x seeds runs, plus the baseline and "
        "controls asked for. A dose above the ceiling of 1.4 sets "
        "allow_over_ceiling, as a ladder that seeks the breaking point goes there "
        "on purpose. A units file with a unit whose source is the Neuron Persona "
        "Library must be written outside the repository, such as in "
        "/workspace/experiments, as the library's data stays out of git "
        "(decision 0020). The file is read back and expanded before it is put in "
        "place, so a file that would not run is never left. It loads no model and "
        "runs nothing: `pushlab run FILE --dry-run` then shows what it would run "
        "and cost.",
        epilog="Exit status: 0 once the file is written; 1 when it refuses (why, "
        "on stderr), having written nothing.",
    )
    ladder.add_argument(
        "--units",
        required=True,
        type=Path,
        metavar="FILE",
        help="a units file: CSV with a unit column, a neuron's ID, and optionally "
        "direction (+1 or -1), persona, description and source; a candidates "
        "list from pushlab candidates is one",
    )
    ladder.add_argument(
        "--doses",
        required=True,
        metavar="LIST",
        help="the doses, numbers above 0 in increasing order, separated by commas, "
        "such as 0.25,0.5,0.75,1.0,1.25; each unit's direction gives their sign",
    )
    ladder.add_argument(
        "--prompts",
        required=True,
        type=Path,
        metavar="FILE",
        help="a prompt set, as a path from the repository's root or an absolute "
        "one, written into the file as given, such as prompts/neutral32.jsonl",
    )
    ladder.add_argument(
        "--out",
        required=True,
        type=Path,
        metavar="FILE",
        help="the experiment file to write, such as "
        "/workspace/experiments/panel-1-ladder.yaml",
    )
    ladder.add_argument(
        "--name",
        help="the experiment's name, lowercase letters and digits joined by "
        "hyphens (default: --out's file name without its extension, lowercased, "
        "each run of other characters made a hyphen)",
    )
    ladder.add_argument(
        "--seeds",
        default="0",
        metavar="LIST",
        help="the seeds, separated by commas (default: %(default)s). One seed "
        "decodes greedily; several sample each reply at temperature 1, once with "
        "each seed",
    )
    ladder.add_argument(
        "--max-new-tokens",
        type=positive_int,
        default=MAX_NEW_TOKENS,
        metavar="N",
        help="the most tokens a reply may have (default: %(default)s)",
    )
    ladder.add_argument(
        "--batch-size",
        type=positive_int,
        default=BATCH_SIZE,
        metavar="N",
        help="the most runs that share a batch (default: %(default)s)",
    )
    ladder.add_argument(
        "--max-gpu-hours",
        type=hours,
        metavar="H",
        help="the GPU-hours the experiment may take (default: 1.5 times its reply "
        "tokens' time at configs/costs.yaml's speed for the model, or the speed a "
        "run measured, rounded up to a quarter hour)",
    )
    ladder.add_argument(
        "--controls",
        type=whole_number,
        default=0,
        metavar="K",
        help="random neurons of each unit's layer to push at the same doses, as "
        "matched controls (default: %(default)s)",
    )
    ladder.add_argument(
        "--control-seed",
        type=whole_number,
        metavar="S",
        help="the seed the controls are drawn from, with --controls (default: 0)",
    )
    ladder.add_argument(
        "--baseline",
        action="store_true",
        help="add a first condition, baseline, that pushes nothing: each prompt's "
        "reply at dose 0",
    )
    ladder.add_argument(
        "--no-prompt-pass",
        dest="prompt_pass",
        action="store_false",
        help="leave the prompt pass alone and push only reply tokens",
    )
    ladder.add_argument(
        "--overwrite", action="store_true", help="replace --out if it exists"
    )
