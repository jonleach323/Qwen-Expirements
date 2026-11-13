"""The options of pushlab ladder, which writes a dose ladder's experiment
file (decision 0022). A stub: the command's parser is filled in with it.

pushlab.cli_parser builds the whole parser from these and the others. Only
light modules are imported here, so that every --help answers at once.
"""

import argparse


def add_ladder(commands: argparse._SubParsersAction) -> None:
    """Add `pushlab ladder` to commands.

    Example:
        >>> parser = argparse.ArgumentParser()
        >>> add_ladder(parser.add_subparsers(dest="command"))
        >>> parser.parse_args(["ladder"]).command
        'ladder'
    """
    commands.add_parser(
        "ladder", help="write an experiment file that climbs each unit up a dose ladder"
    )
