"""The options of pushlab measure, which scores replies with the measures
that need no judge (decision 0022). A stub: the command's parser is filled
in with it.

pushlab.cli_parser builds the whole parser from these and the others. Only
light modules are imported here, so that every --help answers at once.
"""

import argparse


def add_measure(commands: argparse._SubParsersAction) -> None:
    """Add `pushlab measure` to commands.

    Example:
        >>> parser = argparse.ArgumentParser()
        >>> add_measure(parser.add_subparsers(dest="command"))
        >>> parser.parse_args(["measure"]).command
        'measure'
    """
    commands.add_parser(
        "measure", help="score replies with the measures that need no judge"
    )
