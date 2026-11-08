"""The options of the commands that work on experiment files: pushlab expand,
run, status and verify, and the --runs-dir they share with report.

pushlab.cli_parser builds the whole parser from these and the others.
"""

import argparse
from pathlib import Path


def positive_int(text: str) -> int:
    """argparse's type for a whole number from 1.

    Example:
        >>> positive_int("3")
        3
    """
    if not text.isdigit() or int(text) < 1:
        raise argparse.ArgumentTypeError(f"must be a whole number from 1, not {text!r}")
    return int(text)


def add_verify(commands: argparse._SubParsersAction) -> None:
    verify = commands.add_parser(
        "verify",
        help="check an experiment's saved runs and steps against its file",
        description="Check an experiment's runner folder against its file: that "
        "its version was made from this file, prompt set and units/known.csv; that "
        "every planned run is saved, ok or failed, and no other; that the index "
        "marks each run once, or again only after it failed, and no shard holds a "
        "run twice; that each saved run is the planned one, with its latest mark's "
        "status and experiment.json's model download; that every reply has one step "
        "per token, pushing what its schedule gives that pass, each amount of its "
        "dose's sign and one dose-1 amount per unit; and that manifest.json is "
        "valid and agrees with experiment.json and the index (not while a runner "
        "works on the folder). It prints each problem and exits 1 if there are "
        "any, as while planned runs are not saved yet, else 0. It loads no model "
        "and writes nothing.",
    )
    verify.add_argument(
        "spec",
        type=Path,
        metavar="SPEC",
        help="the experiment file, such as experiments/p0/s2_pulse_lengths.yaml",
    )
    add_runs_dir(verify)


def add_expand(commands: argparse._SubParsersAction) -> None:
    expand = commands.add_parser(
        "expand",
        help="list the runs an experiment file means, without running anything",
        description="Read an experiment file (experiments/README.md has the "
        "format), check every condition's schedule against the model, read the "
        "prompt set and draw the random controls, then print a line naming the "
        "columns and every run the file means, one per line: its ID, condition, "
        "control number (0 for the condition's own run), prompt, seed, batch and, "
        "for a control, the run it mirrors. With --summary, print the settings and "
        "how many runs each condition gives instead. Nothing is run and no model "
        "is loaded.",
    )
    expand.add_argument(
        "spec",
        type=Path,
        metavar="SPEC",
        help="the experiment file, such as experiments/p0/s2_pulse_lengths.yaml",
    )
    expand.add_argument(
        "--summary",
        action="store_true",
        help="print how many runs each condition gives, rather than every run",
    )


def add_runs_dir(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--runs-dir",
        type=Path,
        metavar="DIR",
        help="the folder that holds one folder per experiment (default: "
        "$PUSHLAB_RUNS_DIR, else /workspace/runs)",
    )


def add_run(commands: argparse._SubParsersAction) -> None:
    run = commands.add_parser(
        "run",
        help="run an experiment file batch by batch, saving each batch as it goes",
        description="Run every run an experiment file means (pushlab expand "
        "lists them) in DIR/<name>, its runner folder: one batch at a time, each "
        "saved before the next starts, so a crash loses at most the batch that "
        "was running. Run the same command again to go on where it stopped. A "
        "batch that raises is saved with its runs failed. It stops cleanly once "
        "the saved batches have taken max_gpu_hours. It refuses a folder made "
        "from another experiment file, prompt set, known units, model download "
        "or calibration, and one that another experiment or another runner "
        "uses. It needs the model downloaded and, to push, its calibration.",
        epilog="Exit status: 0 when it finished or had nothing to do; 1 when it "
        "refused or could not save; 2 when a run failed; 3 when it stopped with "
        "runs left, at max_gpu_hours or --limit; 130 after Ctrl-C.",
    )
    run.add_argument(
        "spec",
        type=Path,
        metavar="SPEC",
        help="the experiment file, such as experiments/p0/s2_pulse_lengths.yaml",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="print the runs by condition, and the reply tokens, GPU-hours and "
        "cost of those still to run; load no model (only its tokenizer), and "
        "run and write nothing",
    )
    run.add_argument(
        "--limit",
        type=positive_int,
        metavar="N",
        help="run at most N runs this time, in whole batches",
    )
    run.add_argument(
        "--retry-failed",
        action="store_true",
        help="run again only the runs that failed, and nothing else",
    )
    run.add_argument(
        "--overwrite-version",
        action="store_true",
        help="when the folder's runs were made from another experiment file, "
        "prompt set, known units, model download or calibration, start a new "
        "version of the folder, which keeps the saved runs that did not change "
        "(none, when the prompt set, model download or calibration changed)",
    )
    add_runs_dir(run)


def add_status(commands: argparse._SubParsersAction) -> None:
    status = commands.add_parser(
        "status",
        help="how far an experiment has got, and how long the rest will take",
        description="Print an experiment's runs saved ok, failed and pending, "
        "the batches saved, the speed measured over them, the GPU-hours used "
        "of max_gpu_hours and the time left, and whether a runner is working "
        "on its folder. It loads no model and changes nothing, so it can run "
        "while pushlab run is going.",
    )
    status.add_argument(
        "spec",
        type=Path,
        metavar="SPEC",
        help="the experiment file, such as experiments/p0/s2_pulse_lengths.yaml",
    )
    add_runs_dir(status)
