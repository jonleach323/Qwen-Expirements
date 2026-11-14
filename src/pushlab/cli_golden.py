"""The options of `pushlab golden`, which records the golden tests' output
fingerprints on the GPU (decision 0023; pushlab.runs.golden_command).

pushlab.cli_parser builds the whole parser from these and the others. Only
light modules are imported here, so that every --help answers at once.
"""

import argparse
from pathlib import Path


def add_golden(commands: argparse._SubParsersAction) -> None:
    """Add `pushlab golden record` to commands.

    Example:
        >>> parser = argparse.ArgumentParser()
        >>> add_golden(parser.add_subparsers(dest="command"))
        >>> args = parser.parse_args(["golden", "record", "--out", "g.json"])
        >>> args.command, args.golden_command, args.model, args.out
        ('golden', 'record', 'qwen3-8b', PosixPath('g.json'))
    """
    golden = commands.add_parser(
        "golden",
        help="record the golden tests' output fingerprints (a GPU job)",
        description="Golden tests run fixed prompts, seeds and pushes on the "
        "model and compare each reply's fingerprint with the stored one "
        "(tests/golden/, tests/gpu/test_golden_qwen3_8b.py).",
    )
    sub = golden.add_subparsers(dest="golden_command", required=True, metavar="WHAT")
    record = sub.add_parser(
        "record",
        help="run the golden cases and write their fingerprints",
        description="Run the golden cases on the model, on the GPU, and write "
        "each reply's fingerprint with the environment that made it. A GPU job: "
        "the owner runs it on the pod, writing outside the repository, and "
        "copies the file into tests/golden/ through a pull request.",
    )
    record.add_argument(
        "--model", default="qwen3-8b", help="the model (default: %(default)s)"
    )
    record.add_argument(
        "--cases",
        type=Path,
        metavar="FILE",
        help="the golden cases (default: tests/golden/<model>.yaml)",
    )
    record.add_argument(
        "--out",
        required=True,
        type=Path,
        metavar="FILE",
        help="the fingerprints file to write, such as /workspace/golden/qwen3-8b.json",
    )
