"""The options of `pushlab doses`, which finds each unit's working and
collapse doses in a scored dose ladder (decision 0022;
pushlab.analysis.doses_command).

pushlab.cli_parser builds the whole parser from these and the others. Only
light modules are imported here, so that every --help answers at once.
"""

import argparse
from pathlib import Path

from pushlab.cli_experiments import add_runs_dir


def add_doses(commands: argparse._SubParsersAction) -> None:
    """Add `pushlab doses SPEC --units FILE` and its options to commands.

    Example:
        >>> parser = argparse.ArgumentParser()
        >>> add_doses(parser.add_subparsers(dest="command"))
        >>> args = parser.parse_args(["doses", "l.yaml", "--units", "u.csv"])
        >>> args.command, args.units, args.judge, args.panel, args.out
        ('doses', PosixPath('u.csv'), None, None, None)
    """
    doses = commands.add_parser(
        "doses",
        help="find each unit's working and collapse doses in a scored ladder",
        description="Read a dose ladder's saved replies, as pushlab ladder wrote "
        "it and pushlab run ran it, and the verdicts pushlab score saved on them, "
        "and find each unit's working dose, the lowest dose at which at least "
        "the panel's working_rate of its replies speak as its persona, and its "
        "collapse dose, the lowest at which their mean coherence falls below the "
        "panel's coherence_below (configs/panel.yaml). A unit's replies are its "
        "own runs at their dose and the unpushed runs at dose 0. Their speaks_as "
        "verdicts are those about the unit's own persona, as pushlab score SPEC "
        "--rubric speaks_as --units FILE asks them, and their coherence is the "
        "same judge's (pushlab score SPEC --rubric coherence). Each rate has a "
        "Wilson interval, with several seeds on its effective number of "
        "verdicts, as a prompt's replies are not independent; each dose found "
        "and each mean coherence, an interval from resampling the prompts, a "
        "prompt's replies at every dose and seed kept together. An interval "
        "that ends beyond the ladder means the ladder did not reach that dose "
        "often enough: climb further. It prints a table per "
        "unit, with its random controls' speaks_as verdicts and the measures "
        "pushlab measure saved (repetition, the share flagged, perplexity and "
        "capability) when there are any, then a summary. It loads no model and "
        "writes nothing but --out.",
        epilog="Exit status: 0 when it reported; 1 when it refused, saying why.",
    )
    doses.add_argument(
        "spec",
        type=Path,
        metavar="SPEC",
        help="the ladder's experiment file, as pushlab ladder wrote it",
    )
    doses.add_argument(
        "--units",
        type=Path,
        required=True,
        metavar="FILE",
        help="the units file the ladder was written from: each unit's ID, "
        "direction and persona, which its speaks_as verdicts are about",
    )
    doses.add_argument(
        "--judge",
        metavar="NAME",
        help="the judge whose speaks_as and coherence verdicts to read, as "
        "pushlab score names it, such as api:claude-haiku-5-5 (default: the only "
        "judge whose verdicts are saved)",
    )
    doses.add_argument(
        "--panel",
        type=Path,
        metavar="FILE",
        help="the thresholds and the intervals' settings (default: configs/panel.yaml)",
    )
    add_runs_dir(doses)
    doses.add_argument(
        "--out",
        type=Path,
        metavar="FILE",
        help="also write the numbers as JSON to FILE, a new file outside the "
        "repository and the runner folder, such as "
        "/workspace/reports/panel-1-doses.json: the panel, the judge, and each "
        "unit's stats per dose and doses found, with their intervals; never a "
        "reply's text",
    )
