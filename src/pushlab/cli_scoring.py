"""The options of `pushlab score`, which has a judge model label an
experiment's replies (decisions 0021 and 0022; pushlab.scoring.score_command).

pushlab.cli_parser builds the whole parser from these and the others. Only
light modules are imported here, so that every --help answers at once.
"""

import argparse
from pathlib import Path

from pushlab.cli_experiments import add_runs_dir, positive_int
from pushlab.scoring.backends import BACKENDS, DEFAULT_MODELS
from pushlab.scoring.cache import CACHE_ROOT
from pushlab.scoring.local_backend import LOCAL_URL
from pushlab.scoring.rubrics import RUBRIC_NAMES

# What --windows may cut a reply into.
WINDOWS = ("sentences",)


def add_score(commands: argparse._SubParsersAction) -> None:
    """Add `pushlab score SPEC --rubric NAME` and its options to commands.

    Example:
        >>> parser = argparse.ArgumentParser()
        >>> add_score(parser.add_subparsers(dest="command"))
        >>> args = parser.parse_args(["score", "x.yaml", "--rubric", "coherence"])
        >>> args.rubric, args.backend, args.workers, args.windows
        ('coherence', 'api', 4, None)
    """
    score = commands.add_parser(
        "score",
        help="have a judge model label an experiment's replies",
        description="Have a judge model label each reply of an experiment by a "
        "rubric: speaks_as (does the reply speak as the persona given, or only "
        "about its theme?), which_persona (which of the personas given, if any, "
        "does it speak as?) or coherence (a score from 0 to 10). It reads the "
        "experiment's runner folder and judges the replies of its current "
        "version, each whole, into the folder's scores table, or with --windows "
        "sentences in windows of --window-size sentences, each window on its "
        "own, into its sentences table. The judge sees only the rubric, the "
        "persona or personas given here, the same for every reply, and the "
        "text: never the prompt, the run, its condition, unit or dose. With "
        "--units, a dose ladder's units file, speaks_as asks each unit's replies "
        "and the unpushed ones about that unit's own persona, the same at every "
        "dose. The judge must "
        "be a model of another family than the experiment's. A reply already "
        "scored the same way is skipped, and each verdict that parsed is "
        "cached, so a text judged before is not sent again; a failed one is "
        "asked again on the next run. A failed call is tried again with "
        "backoff and saved as failed: a label is never guessed. --dry-run "
        "counts the calls and estimates their cost first, calling and writing "
        "nothing.",
        epilog="Exit status: 0 when it finished, or had nothing to judge; 1 when "
        "it refused, or stopped because the judge could not answer at all (what "
        "it judged before is saved); 130 after Ctrl-C.",
    )
    score.add_argument(
        "spec",
        type=Path,
        metavar="SPEC",
        help="the experiment file, such as experiments/p0/s2_pulse_lengths.yaml",
    )
    score.add_argument(
        "--rubric",
        required=True,
        choices=RUBRIC_NAMES,
        help="what the judge is asked, from rubrics/<name>.yaml",
    )
    score.add_argument(
        "--windows",
        choices=WINDOWS,
        help="judge windows of each reply's sentences, each on its own, into the "
        "sentences table, rather than whole replies into the scores table",
    )
    score.add_argument(
        "--window-size",
        type=positive_int,
        metavar="N",
        help="the sentences in a window, with --windows sentences (default: 1)",
    )
    score.add_argument(
        "--personas",
        type=Path,
        metavar="FILE",
        help="a personas file, YAML with a list of names and descriptions: "
        "speaks_as asks about one of them, which_persona chooses among all of "
        "them, and coherence takes none",
    )
    score.add_argument(
        "--persona",
        metavar="NAME",
        help="the persona of --personas that speaks_as asks about, needed when "
        "the file has more than one",
    )
    score.add_argument(
        "--units",
        type=Path,
        metavar="FILE",
        help="a dose ladder's units file (pushlab ladder), with speaks_as: ask "
        "each unit's replies, its own at every dose and its random controls', "
        "and every unpushed reply about the unit's persona, named in its "
        "persona column and described by --personas if given, else by its "
        "description column; the same at every dose, so the judge never learns "
        "the dose",
    )
    score.add_argument(
        "--backend",
        choices=BACKENDS,
        default="api",
        help="api: Claude through the Anthropic API, with the key in "
        "ANTHROPIC_API_KEY; local: an open model that a vLLM server serves "
        "(default: %(default)s)",
    )
    score.add_argument(
        "--model",
        help=f"the judge's model (default: {DEFAULT_MODELS['api']} for api; local "
        "needs the name the vLLM server serves)",
    )
    score.add_argument(
        "--base-url",
        metavar="URL",
        help=f"the vLLM server's API, for the local backend (default: {LOCAL_URL})",
    )
    score.add_argument(
        "--workers",
        type=positive_int,
        default=4,
        metavar="N",
        help="calls to the judge at a time (default: %(default)s)",
    )
    score.add_argument(
        "--cache-dir",
        type=Path,
        metavar="DIR",
        help="the folder of the judge's cache of verdicts (default: "
        f"$PUSHLAB_CACHE_DIR/judge, else {CACHE_ROOT / 'judge'})",
    )
    add_runs_dir(score)
    score.add_argument(
        "--dry-run",
        action="store_true",
        help="print how many replies or windows there are to judge, how many are "
        "already scored or in the cache, the calls to make and their estimated "
        "cost from configs/costs.yaml, and no reply text; build no judge, need "
        "no key, call and write nothing",
    )
