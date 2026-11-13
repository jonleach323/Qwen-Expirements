"""The options of `pushlab panel`, which narrows the persona candidates down
to a panel, stage by stage (decision 0023; pushlab.panel.command).

    pushlab panel stage b SCREEN --out FILE      the best of screen_a, on all prompts
    pushlab panel stage ladder B --out FILE      stage b's best at five doses
    pushlab panel stage confirm LADDER --out FILE  each that works, at its dose
    pushlab panel select CONFIRM --ladder LADDER --categories FILE [--yes]
    pushlab panel stage preview PANEL --out FILE  varied pairs in four shapes

pushlab.cli_parser builds the whole parser from these and the others. Only
light modules are imported here, so that every --help answers at once.
"""

import argparse
from pathlib import Path

from pushlab.cli_experiments import add_runs_dir

STAGES = ("b", "ladder", "confirm", "preview")
# What each stage reads: the experiment file of the stage before it, scored,
# or for preview the panel file.
READS = {
    "b": ("SCREEN", "screen_a's experiment file, run and scored"),
    "ladder": ("STAGE_B", "stage b's experiment file, run and scored"),
    "confirm": ("LADDER", "stage ladder's experiment file, run and scored"),
    "preview": ("PANEL", "the panel file pushlab panel select wrote"),
}


def _add_stage(stages: argparse._SubParsersAction, stage: str, help: str) -> None:
    parser = stages.add_parser(stage, help=help, description=help + ".")
    metavar, what = READS[stage]
    parser.add_argument("source", type=Path, metavar=metavar, help=what)
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        metavar="FILE",
        help="the experiment file to write, outside the repository when it "
        "holds the Neuron Persona Library's data, such as "
        "/workspace/experiments/s3-stage-b.yaml",
    )
    parser.add_argument(
        "--name", help="the experiment's name (default: from --out's file name)"
    )
    parser.add_argument(
        "--judge",
        metavar="NAME",
        help="whose speaks_as and coherence verdicts to read, when several "
        "judges' are saved",
    )
    parser.add_argument(
        "--panel",
        type=Path,
        metavar="FILE",
        help="the panel's settings (default: configs/panel.yaml)",
    )
    parser.add_argument(
        "--max-gpu-hours",
        type=float,
        metavar="H",
        help="the GPU-hours the experiment may take (default: from the model's "
        "speed, as pushlab ladder works it out)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace --out if it exists"
    )
    add_runs_dir(parser)


def add_panel(commands: argparse._SubParsersAction) -> None:
    """Add `pushlab panel stage ...` and `pushlab panel select` to commands.

    Example:
        >>> parser = argparse.ArgumentParser()
        >>> add_panel(parser.add_subparsers(dest="command"))
        >>> args = parser.parse_args(
        ...     ["panel", "stage", "b", "a.yaml", "--out", "b.yaml"]
        ... )
        >>> args.command, args.panel_command, args.stage, args.source
        ('panel', 'stage', 'b', PosixPath('a.yaml'))
        >>> args = parser.parse_args(
        ...     ["panel", "select", "c.yaml", "--ladder", "l.yaml"]
        ...     + ["--categories", "k.csv"]
        ... )
        >>> args.panel_command, args.yes, args.backend
        ('select', False, 'api')
    """
    panel = commands.add_parser(
        "panel",
        help="narrow the persona candidates down to a panel, stage by stage",
        description="Narrow the persona candidates down to a panel of up to 24 "
        "strong, distinct, reliable personas (decision 0023; "
        "docs/runbooks/panel.md). Each stage writes the next experiment file "
        "from the scores of the one before; select chooses the panel.",
    )
    sub = panel.add_subparsers(dest="panel_command", required=True, metavar="WHAT")
    stage = sub.add_parser(
        "stage",
        help="write the next stage's experiment file from the last one's scores",
        description="Write the next stage's experiment file from the scores of "
        "the stage before it.",
    )
    stages = stage.add_subparsers(dest="stage", required=True, metavar="STAGE")
    _add_stage(
        stages,
        "b",
        "the candidates of screen_a with the highest "
        "speaks_as rates, on every held-out prompt",
    )
    _add_stage(
        stages,
        "ladder",
        "stage b's best, each at five doses around its "
        "reference dose, on the first held-out prompts",
    )
    _add_stage(
        stages,
        "confirm",
        "each candidate of the ladder that works "
        "before it collapses, at its working dose, on every held-out prompt",
    )
    _add_stage(
        stages,
        "preview",
        "varied pairs of the panel, each pushed in "
        "four shapes, on the last held-out prompts",
    )
    select = sub.add_parser(
        "select",
        help="choose the panel from the confirmed candidates",
        description="Choose up to the panel's size of personas that speak as "
        "themselves at their working dose, meet the coherence threshold, are "
        "told apart by the which_persona rubric, and spread across categories; "
        "write the panel file and its page. The which_persona sample is judged "
        "here: without --yes it only says how many calls that takes and their "
        "cost.",
    )
    select.add_argument(
        "source",
        type=Path,
        metavar="CONFIRM",
        help="stage confirm's experiment file, run and scored",
    )
    select.add_argument(
        "--ladder",
        required=True,
        type=Path,
        metavar="LADDER",
        help="stage ladder's experiment file, run and scored: each persona's "
        "collapse dose",
    )
    select.add_argument(
        "--categories",
        required=True,
        type=Path,
        metavar="FILE",
        help="a CSV of each finalist's category (key,category); a missing or "
        "unfinished file is written for you to fill in",
    )
    select.add_argument(
        "--out",
        type=Path,
        metavar="FILE",
        help="the panel file (default: units/panel_<model>.yaml, kept out of git)",
    )
    select.add_argument(
        "--doc",
        type=Path,
        metavar="FILE",
        help="the panel's page (default: docs/panel/<model>.md, kept out of git)",
    )
    select.add_argument(
        "--yes",
        action="store_true",
        help="make the which_persona calls the plan names; without it, nothing "
        "is called or written",
    )
    select.add_argument(
        "--backend", choices=("api", "local"), default="api", help="the judge's backend"
    )
    select.add_argument("--model", help="the judge's model (default: the backend's)")
    select.add_argument("--base-url", help="a local backend's server")
    select.add_argument(
        "--judge",
        metavar="NAME",
        help="whose speaks_as and coherence verdicts to read, when several "
        "judges' are saved",
    )
    select.add_argument(
        "--panel",
        type=Path,
        metavar="FILE",
        help="the panel's settings (default: configs/panel.yaml)",
    )
    add_runs_dir(select)
