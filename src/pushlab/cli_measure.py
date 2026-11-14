"""The options of `pushlab measure`, which scores an experiment's replies with
the measures that need no judge (decision 0022;
pushlab.scoring.measure_command).

pushlab.cli_parser builds the whole parser from these and the others. Only
light modules are imported here, so that every --help answers at once.
"""

import argparse
from pathlib import Path

from pushlab.cli_experiments import add_runs_dir, positive_int


def add_measure(commands: argparse._SubParsersAction) -> None:
    """Add `pushlab measure SPEC` and its options to commands.

    Example:
        >>> parser = argparse.ArgumentParser()
        >>> add_measure(parser.add_subparsers(dest="command"))
        >>> args = parser.parse_args(["measure", "e.yaml", "--perplexity"])
        >>> args.command, args.perplexity, args.batch_size, args.dry_run
        ('measure', True, None, False)
    """
    measure = commands.add_parser(
        "measure",
        help="score replies with the measures that need no judge",
        description="Measure each reply of an experiment with the measures that "
        "need no judge, and save them in its runner folder's scores table, as "
        "pushlab score saves a judge's verdicts: repetition, the share of its "
        "word n-grams it already said, and word_variety, its share of distinct "
        "words in each window of words, at configs/panel.yaml's n-gram and "
        "window, with flags for a reply repetitive or of low variety; and "
        "capability, when the prompt set has an answers file beside it "
        "(<set>-answers.jsonl, as prompts/capability100.jsonl has), each reply "
        "graded exact-match against its prompt's answer. With --perplexity, "
        "also each reply's perplexity under the experiment's model with "
        "nothing pushed, at its pinned revision and data type: it loads the "
        "model, so it is a GPU job for the pod, and the model must be the "
        "download the replies were made with. It measures the replies of the "
        "folder's current version; a reply measured already the same way is "
        "skipped, so a rerun measures only what is new, and a reply with no "
        "word or token to measure is saved as failed: a value is never "
        "guessed. Capability and perplexity need the prompt set the version "
        "was made from. It ends with each condition's measures: the mean "
        "repetition and word variety, the replies flagged, those right on the "
        "capability set with their interval, and the median perplexity.",
        epilog="Exit status: 0 when it finished, or had nothing to measure; 1 "
        "when it refused, saying why; 130 after Ctrl-C.",
    )
    measure.add_argument(
        "spec",
        type=Path,
        metavar="SPEC",
        help="the experiment file, such as experiments/p0/s2_pulse_lengths.yaml",
    )
    measure.add_argument(
        "--perplexity",
        action="store_true",
        help="also score each reply's perplexity under the experiment's model "
        "with nothing pushed: a GPU job, which loads the model",
    )
    measure.add_argument(
        "--batch-size",
        type=positive_int,
        metavar="N",
        help="with --perplexity, the most replies of one length the model reads "
        "at a time (default: 8); lower it if the GPU runs out of memory",
    )
    add_runs_dir(measure)
    measure.add_argument(
        "--dry-run",
        action="store_true",
        help="print how many replies each measure has to measure and has "
        "measured already; load no model, take no lock and write nothing",
    )
