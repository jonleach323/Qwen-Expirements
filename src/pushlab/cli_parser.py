"""The pushlab command's options: one subparser per subcommand.

Kept apart from the commands themselves, which load torch, so that
`pushlab --help` and every subcommand's --help answer at once.
"""

import argparse
from pathlib import Path

from pushlab.cli_discovery import add_candidates, add_fetch
from pushlab.cli_doses import add_doses
from pushlab.cli_experiments import (
    add_expand,
    add_run,
    add_runs_dir,
    add_status,
    add_verify,
    positive_int,
)
from pushlab.cli_labels import add_label
from pushlab.cli_ladder import add_ladder
from pushlab.cli_measure import add_measure
from pushlab.cli_scoring import add_score
from pushlab.models.registry import REGISTRY


def _whole_number(text: str) -> int:
    if not (text.isascii() and text.isdigit()):
        raise argparse.ArgumentTypeError(f"must be a whole number from 0, not {text!r}")
    return int(text)


def _add_baseline(commands: argparse._SubParsersAction, models: list[str]) -> None:
    baseline = commands.add_parser(
        "baseline",
        help="generate one unsteered reply and save it as a run",
        description="Generate one unsteered reply, with thinking off and greedy "
        "decoding, and save it as one run in OUT/runs.parquet with "
        "OUT/manifest.json. It uses the GPU in bfloat16 if there is one, "
        "otherwise the CPU in float32. The model must already be downloaded, "
        "as scripts/setup_pod.sh does on the GPU pod.",
    )
    baseline.add_argument(
        "--model", required=True, choices=models, help="registry name of the model"
    )
    baseline.add_argument(
        "--out",
        required=True,
        type=Path,
        help="folder for runs.parquet and manifest.json, created if missing",
    )
    baseline.add_argument(
        "--prompt",
        help="the user's message (default: the prompt in "
        "experiments/p0/s0_baseline.yaml)",
    )
    baseline.add_argument(
        "--max-new-tokens",
        type=positive_int,
        default=200,
        help="most tokens the reply may have (default: %(default)s)",
    )
    baseline.add_argument(
        "--overwrite",
        action="store_true",
        help="replace runs.parquet and manifest.json if OUT already has them",
    )


def _add_calibrate(commands: argparse._SubParsersAction, models: list[str]) -> None:
    calibrate = commands.add_parser(
        "calibrate",
        help="measure each layer's median residual size, the scale of a dose",
        description="Run the prompts in prompts/calibration.jsonl through the "
        "model, in the chat template with thinking off, and measure the median "
        "length of the residual stream leaving each layer, leaving out each "
        "prompt's first token. Doses are scaled by these sizes. Saves "
        "calibration/<model>-<revision>.json and prints the sizes. It uses the "
        "GPU in bfloat16 if there is one, otherwise the CPU in float32. The "
        "model must already be downloaded, as scripts/setup_pod.sh does on the "
        "GPU pod.",
    )
    calibrate.add_argument(
        "--model", required=True, choices=models, help="registry name of the model"
    )
    calibrate.add_argument(
        "--overwrite",
        action="store_true",
        help="measure again and replace the model revision's calibration file",
    )


def _add_amount(commands: argparse._SubParsersAction) -> None:
    amount = commands.add_parser(
        "amount",
        help="print what a dose adds to a neuron, from the saved calibration",
        description="Print the amount a dose adds to one neuron's value, alpha "
        "times its layer's residual size divided by the length of its write "
        "direction, and how far that pushes the residual stream. The sizes come "
        "from the model's calibration file (pushlab calibrate). It loads the "
        "model, which must already be downloaded, to read the neuron's weights.",
    )
    amount.add_argument(
        "--unit", required=True, help="the neuron, such as qwen3.5-4b:L14:N8949"
    )
    amount.add_argument(
        "--alpha",
        required=True,
        type=float,
        help="the dose, such as 1 or -0.5 (write a number with an exponent as "
        "--alpha=-1e-3)",
    )


def _add_steer(commands: argparse._SubParsersAction, models: list[str]) -> None:
    steer = commands.add_parser(
        "steer",
        help="reply to a prompt set with chosen neurons pushed, and save the runs",
        description="Reply to every prompt of a prompt set, greedy and with "
        "thinking off, with the same pushes on every reply, and save one run "
        "per prompt in OUT/runs.parquet with OUT/manifest.json. Without --unit "
        "nothing is pushed. With --control random, each unit is replaced by a "
        "random neuron of its layer at the same dose, drawn from --control-seed "
        "and never one listed in units/known.csv. Pushes need the model's "
        "calibration (pushlab calibrate). It uses the GPU in bfloat16 if there "
        "is one, otherwise the CPU in float32.",
    )
    steer.add_argument(
        "--model", required=True, choices=models, help="registry name of the model"
    )
    steer.add_argument(
        "--prompts",
        required=True,
        type=Path,
        help="a prompt set, such as prompts/smoke.jsonl",
    )
    steer.add_argument(
        "--out",
        required=True,
        type=Path,
        help="folder for runs.parquet and manifest.json, created if missing",
    )
    steer.add_argument(
        "--unit",
        action="append",
        metavar="ID",
        help="a neuron to push, such as qwen3.5-4b:L14:N8949; give it its own "
        "--alpha, and repeat both for more neurons",
    )
    steer.add_argument(
        "--alpha",
        action="append",
        type=float,
        metavar="X",
        help="the dose of the --unit given in the same place, such as 1 or -0.5 "
        "(write a number with an exponent as --alpha=-1e-3)",
    )
    steer.add_argument(
        "--start",
        type=_whole_number,
        default=0,
        help="first reply token whose pass is pushed, counting from 0 (default: "
        "%(default)s). The pass that reads reply token k chooses token k + 1, so "
        "a push on token k first changes token k + 1",
    )
    steer.add_argument(
        "--end",
        type=_whole_number,
        help="first reply token whose pass is no longer pushed (default: none, "
        "to the end of the reply)",
    )
    steer.add_argument(
        "--include-prompt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also push the pass that reads the prompt (default: on); only it "
        "changes the reply's first token. With --start above 0 it makes the "
        "push start on the prompt; use --no-include-prompt for a late push",
    )
    steer.add_argument(
        "--control",
        choices=["random"],
        help="replace each unit by a random neuron of its layer: a control",
    )
    steer.add_argument(
        "--control-seed",
        type=_whole_number,
        metavar="N",
        help="the seed the random control neurons are drawn from",
    )
    steer.add_argument(
        "--max-new-tokens",
        type=positive_int,
        default=200,
        help="most tokens each reply may have (default: %(default)s)",
    )
    steer.add_argument(
        "--overwrite",
        action="store_true",
        help="replace runs.parquet and manifest.json if OUT already has them",
    )


def _add_report(commands: argparse._SubParsersAction) -> None:
    report = commands.add_parser(
        "report",
        help="an experiment's replies with markers where pushes switch, or run "
        "folders' replies side by side",
        description="With an experiment file: write a Markdown file with each "
        "reply of the conditions asked for, prompt by prompt, and short markers "
        "where the pushes active on the reply change, such as [A on], [A off, B "
        "on] or [A/B x8] for a long alternation of short turns, under a one-line "
        "legend saying "
        "what A and B are, and counts at the top, by condition, of the replies "
        "cut off at max_new_tokens and of those with thinking-mode text (a thinking "
        "tag such as <think>, </think> or a misspelled <|/think>), which are "
        "flagged. It reads the "
        "experiment's runner folder and needs the model's tokenizer to place the "
        "markers. With --runs: "
        "write a Markdown "
        "table of S1's run folders' replies, one row per prompt and one column "
        "per folder, under a list of what each folder pushed.",
    )
    report.add_argument(
        "spec",
        nargs="?",
        type=Path,
        metavar="SPEC",
        help="an experiment file, such as experiments/p0/s2_pulse_lengths.yaml",
    )
    report.add_argument(
        "--conditions",
        nargs="+",
        action="extend",
        metavar="LABEL",
        help='the conditions to show, by label, such as none "pulse length=16" '
        "(default: every condition, in the file's order)",
    )
    report.add_argument(
        "--runs",
        nargs="+",
        action="extend",
        type=Path,
        metavar="DIR",
        help="instead of an experiment file, run folders, as pushlab steer or "
        "baseline wrote them; repeat --runs to add more",
    )
    report.add_argument(
        "--out",
        required=True,
        type=Path,
        help="the Markdown file to write; with an experiment file, outside its "
        "runner folder",
    )
    report.add_argument(
        "--overwrite", action="store_true", help="replace the file if it exists"
    )
    add_runs_dir(report)


def build_parser() -> argparse.ArgumentParser:
    """The pushlab command's arguments, with one subparser per subcommand.

    Example:
        >>> args = ["baseline", "--model", "qwen3.5-4b", "--out", "outputs/s0"]
        >>> build_parser().parse_args(args).max_new_tokens
        200
    """
    parser = argparse.ArgumentParser(
        prog="pushlab",
        description="Research code for pushing LLM neurons on schedules.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    models = sorted(name for name, spec in REGISTRY.items() if spec.kind != "test")
    _add_baseline(commands, models)
    _add_calibrate(commands, models)
    _add_amount(commands)
    _add_steer(commands, models)
    _add_report(commands)
    add_expand(commands)
    add_run(commands)
    add_status(commands)
    add_verify(commands)
    add_fetch(commands)
    add_candidates(commands)
    add_score(commands)
    add_label(commands)
    add_measure(commands)
    add_ladder(commands)
    add_doses(commands)
    return parser
