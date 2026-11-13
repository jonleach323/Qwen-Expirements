"""The options of pushlab doses, which reads a scored dose ladder (decision
0022). A stub: the command's parser is filled in with it.

pushlab.cli_parser builds the whole parser from these and the others. Only
light modules are imported here, so that every --help answers at once.
"""

import argparse


def add_doses(commands: argparse._SubParsersAction) -> None:
    """Add `pushlab doses` to commands.

    Example:
        >>> parser = argparse.ArgumentParser()
        >>> add_doses(parser.add_subparsers(dest="command"))
        >>> parser.parse_args(["doses"]).command
        'doses'
    """
    commands.add_parser(
        "doses", help="find each unit's working and collapse doses in a scored ladder"
    )
