"""`pushlab label`: the judge's validation harness (decision 0021).

run_label_command runs the subcommand that pushlab.cli_labels parsed:
`pushlab label export` (pushlab.scoring.label_export), which writes a sheet
of sampled replies for you to label, or `pushlab label check`
(pushlab.scoring.label_check), which has a judge label the same replies and
compares its labels with yours. Each is imported only when it runs, so
neither loads what only the other needs.
"""

from typing import Any


def run_label_command(args: Any) -> int:
    """Run the `pushlab label` subcommand args names, with its options;
    return its exit status.

    ValueError for a subcommand other than export and check, which the
    parser never gives.

    Example:
        >>> import argparse
        >>> run_label_command(argparse.Namespace(label_command="sort"))
        Traceback (most recent call last):
        ...
        ValueError: pushlab label has no subcommand 'sort'; it has export and check
    """
    if args.label_command == "export":
        from pushlab.scoring.label_export import run_label_export

        return run_label_export(
            args.specs,
            args.out,
            rubrics=args.rubric,
            size=args.size,
            seed=args.seed,
            personas_file=args.personas,
            persona=args.persona,
            runs_dir=args.runs_dir,
        )
    if args.label_command == "check":
        from pushlab.scoring.label_check import run_label_check

        return run_label_check(
            args.sheet,
            labels=args.labels,
            backend=args.backend,
            model=args.model,
            base_url=args.base_url,
            workers=args.workers,
            cache_dir=args.cache_dir,
            runs_dir=args.runs_dir,
            out=args.out,
        )
    raise ValueError(
        f"pushlab label has no subcommand {args.label_command!r}; it has export "
        "and check"
    )
