"""The options of `pushlab label export` and `pushlab label check`, the
judge's validation harness (decision 0021). A judge must agree with your
own labels before it scores replies at scale (docs/plan/development-plan.md,
"Judge validation"): export writes a sheet of sampled replies for you to
label, and check has the judge label the same replies and compares.

pushlab.cli_parser builds the whole parser from these and the others. Only
light modules are imported here, so that every --help answers at once: the
backends' module loads no SDK until a backend is made.
"""

import argparse
from pathlib import Path

from pushlab.cli_experiments import add_runs_dir, positive_int
from pushlab.scoring.backends import BACKENDS, DEFAULT_MODELS
from pushlab.scoring.cache import CACHE_ROOT
from pushlab.scoring.local_backend import LOCAL_URL
from pushlab.scoring.rubrics import RUBRIC_NAMES

# The plan's hand labels per rubric (docs/plan/development-plan.md, "Judge
# validation"): how many replies a sheet samples unless --size says.
SHEET_SIZE = 200
WORKERS = 4  # calls to the judge at a time, unless --workers says


def whole_number(text: str) -> int:
    """argparse's type for a whole number from 0, such as a seed.

    Example:
        >>> whole_number("0")
        0
    """
    if not (text.isascii() and text.isdigit()):
        raise argparse.ArgumentTypeError(f"must be a whole number from 0, not {text!r}")
    return int(text)


def _add_export(steps: argparse._SubParsersAction) -> None:
    export = steps.add_parser(
        "export",
        help="write a sheet of sampled replies for you to label",
        description="Sample whole replies of one or more experiments for you to "
        "label by hand, and write them into DIR, a new or empty folder outside "
        "the repository, as replies never go into git. The replies are those of "
        "each experiment's saved runs, ok or collapsed, in its runner folder's "
        "current version, blank ones left out. Each condition of each experiment "
        "gets an equal share of the sample, its random controls a share of their "
        "own; one with fewer replies than its share gives all it has, and the "
        "rest is spread over the others. A text is taken once, however many runs "
        "wrote it. The seed chooses the replies and shuffles them, so one seed "
        "always gives the same sheet. DIR gets three files: labels.csv, for you "
        "to label in any spreadsheet, a row per reply with an empty column for "
        "each rubric and one for your notes; instructions.md, how to label and "
        "what each answer means; and key.json, which names each reply's run and "
        "condition for `pushlab label check`: never open it while labeling. "
        "labels.csv names no run, condition, unit, dose or prompt, so you label "
        "blind, as the judge does. It prints how many replies each condition "
        "gave. It loads no model and calls no judge.",
    )
    export.add_argument(
        "specs",
        nargs="+",
        type=Path,
        metavar="SPEC",
        help="the experiment files whose replies to sample, such as "
        "experiments/p0/s2_pulse_lengths.yaml",
    )
    export.add_argument(
        "--out",
        required=True,
        type=Path,
        metavar="DIR",
        help="a new or empty folder outside the repository for the sheet, such as "
        "/workspace/labels/coherence-1; each reply's ID starts with its name",
    )
    export.add_argument(
        "--rubric",
        action="append",
        choices=RUBRIC_NAMES,
        metavar="NAME",
        help="a rubric to label, a column of the sheet: "
        f"{', '.join(RUBRIC_NAMES[:-1])} or {RUBRIC_NAMES[-1]}; repeat it for more "
        "(default: all of them)",
    )
    export.add_argument(
        "--size",
        type=positive_int,
        default=SHEET_SIZE,
        metavar="N",
        help="how many replies to sample (default: %(default)s, the plan's labels "
        "per rubric); all of them if there are fewer",
    )
    export.add_argument(
        "--seed",
        type=whole_number,
        default=0,
        metavar="N",
        help="the seed that chooses and shuffles the replies (default: %(default)s)",
    )
    export.add_argument(
        "--personas",
        type=Path,
        metavar="FILE",
        help="a personas file, YAML with a list of names and descriptions, for "
        "speaks_as and which_persona; keep files of library personas out of the "
        "repository",
    )
    export.add_argument(
        "--persona",
        metavar="NAME",
        help="the persona of the file that speaks_as asks about, needed when the "
        "file holds more than one",
    )
    add_runs_dir(export)


def _add_check(steps: argparse._SubParsersAction) -> None:
    check = steps.add_parser(
        "check",
        help="have a judge label the sheet's replies and compare its labels with yours",
        description="Compare a judge with you: read the sheet in DIR once you "
        "have labeled it and saved it as CSV, have the judge label the same "
        "replies, through its cache, as `pushlab score` would ask it, and print "
        "for each rubric how many of your labels it read, the judge's failures, "
        "each counted as a disagreement, its accuracy and Cohen's kappa against "
        "your labels, and a table of your labels against its own; for coherence "
        "also the accuracy within one point and a weighted kappa. A rubric passes "
        "at an accuracy of 0.85 and a kappa of 0.70 or more "
        '(docs/plan/development-plan.md, "Judge validation"). An empty cell is '
        "skipped. The judge sees each reply's text and nothing else, never its "
        "prompt, run or condition. With --backend api each call costs money.",
    )
    check.add_argument(
        "sheet",
        type=Path,
        metavar="DIR",
        help="the folder `pushlab label export` wrote, with key.json and your "
        "labeled labels.csv",
    )
    check.add_argument(
        "--labels",
        type=Path,
        metavar="FILE",
        help="your labeled sheet, saved as CSV (default: DIR/labels.csv)",
    )
    check.add_argument(
        "--backend",
        choices=BACKENDS,
        default="api",
        help="the judge: api, Claude through the Anthropic API, which needs "
        "ANTHROPIC_API_KEY; or local, an open model that a vLLM server serves "
        "(default: %(default)s)",
    )
    check.add_argument(
        "--model",
        metavar="MODEL",
        help=f"the judge's model (default for api: {DEFAULT_MODELS['api']}); for "
        "local, the name the vLLM server serves, which must be given, of another "
        "family than the tested model's",
    )
    check.add_argument(
        "--base-url",
        metavar="URL",
        help=f"the vLLM server's URL, for --backend local (default: {LOCAL_URL})",
    )
    check.add_argument(
        "--workers",
        type=positive_int,
        default=WORKERS,
        metavar="N",
        help="how many calls to the judge at a time (default: %(default)s)",
    )
    check.add_argument(
        "--cache-dir",
        type=Path,
        metavar="DIR",
        help="the folder of the judge's cache of verdicts (default: "
        f"$PUSHLAB_CACHE_DIR/judge, else {CACHE_ROOT / 'judge'})",
    )
    add_runs_dir(check)
    check.add_argument(
        "--out",
        type=Path,
        metavar="FILE",
        help="also save the results in this JSON file: numbers and hashes, no "
        "reply text",
    )


def add_label(commands: argparse._SubParsersAction) -> None:
    """Add `pushlab label` and its subcommands, export and check, to commands.

    Example:
        >>> parser = argparse.ArgumentParser()
        >>> add_label(parser.add_subparsers(dest="command"))
        >>> args = parser.parse_args(["label", "export", "e.yaml", "--out", "s"])
        >>> args.label_command, args.specs, args.rubric, args.size, args.seed
        ('export', [PosixPath('e.yaml')], None, 200, 0)
        >>> args = parser.parse_args(["label", "check", "s"])
        >>> args.label_command, args.sheet, args.backend, args.workers
        ('check', PosixPath('s'), 'api', 4)
    """
    label = commands.add_parser(
        "label",
        help="check a judge against your own labels before it scores at scale",
        description="A judge model must agree with your own labels before it "
        "scores replies at scale: 85% accuracy and a Cohen's kappa of 0.70 for "
        'each rubric (docs/plan/development-plan.md, "Judge validation"). '
        "`pushlab label export` writes a sheet of sampled replies for you to "
        "label in any spreadsheet; `pushlab label check` has the judge label the "
        "same replies and says how well it agrees with you. docs/runbooks/judge.md "
        "has the steps.",
    )
    steps = label.add_subparsers(dest="label_command", required=True)
    _add_export(steps)
    _add_check(steps)
