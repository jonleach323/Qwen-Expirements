"""The options of `pushlab score`, which judges an experiment's replies
(decision 0021). A stub: the score command's parser is filled in with it.

pushlab.cli_parser builds the whole parser from these and the others. Only
light modules are imported here, so that every --help answers at once.
"""

import argparse
from pathlib import Path


def add_score(commands: argparse._SubParsersAction) -> None:
    """Add `pushlab score` to commands.

    Example:
        >>> parser = argparse.ArgumentParser()
        >>> add_score(parser.add_subparsers(dest="command"))
        >>> parser.parse_args(["score", "x.yaml", "--rubric", "coherence"]).rubric
        'coherence'
    """
    score = commands.add_parser("score", help="judge an experiment's replies")
    score.add_argument("spec", type=Path, help="the experiment file")
    score.add_argument("--rubric", required=True, help="the rubric to judge by")
