"""The options of `pushlab label export` and `pushlab label check`, the
judge's validation harness (decision 0021). A stub: the label commands'
parsers are filled in with them.

pushlab.cli_parser builds the whole parser from these and the others. Only
light modules are imported here, so that every --help answers at once.
"""

import argparse


def add_label(commands: argparse._SubParsersAction) -> None:
    """Add `pushlab label` and its subcommands to commands.

    Example:
        >>> parser = argparse.ArgumentParser()
        >>> add_label(parser.add_subparsers(dest="command"))
        >>> parser.parse_args(["label", "export"]).label_command
        'export'
    """
    label = commands.add_parser("label", help="check a judge against your labels")
    steps = label.add_subparsers(dest="label_command", required=True)
    steps.add_parser("export", help="write a sheet of sampled replies to label")
    steps.add_parser("check", help="compare a judge's verdicts with your labels")
