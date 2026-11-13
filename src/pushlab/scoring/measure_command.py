"""`pushlab measure SPEC`: the measures that need no judge, on an
experiment's replies, saved in its runner folder's scores table (decision
0022).

It measures each whole reply of the runner folder's current version, "ok"
or "collapsed", never a failed run's (pushlab.scoring.items):
- repetition and word_variety, always (pushlab.scoring.text_measures), at
  the n-gram and window of configs/panel.yaml;
- capability, when the prompt set has an answers file beside it,
  <set>-answers.jsonl, as prompts/capability100.jsonl has: each reply
  graded against its prompt's answer (pushlab.scoring.capability);
- perplexity, with --perplexity, a GPU job the owner runs: each reply's
  perplexity under the experiment's model with nothing pushed
  (pushlab.engine.score_text), loaded as pushlab run loads it, at its
  pinned revision and in the experiment's data type.
pushlab.scoring.measure_rows says what each row holds. A reply measured
already, with the same key, is skipped, so a rerun measures only what is
new, and with --perplexity loads the model only when a reply is left.

Capability and perplexity read the prompt set, which must be the one the
folder's version was made from, as pushlab run requires; perplexity needs
the model download the replies were made with. --dry-run counts what each
measure has left and has measured already; it loads no model, takes no
lock and writes nothing. A real run holds the folder's lock, which keeps a
runner and the scorers out, and writes each measure's rows in one file,
the measures that need no model first; perplexity's go in a file per chunk
of CHUNK replies, so a stop keeps every chunk scored before it. It ends
with each condition's measures (measure_rows.summary_lines).
"""

import argparse
import hashlib
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pushlab.analysis.panel import Panel, read_panel
from pushlab.prompts import PromptSet, read_prompt_set
from pushlab.runs import folder
from pushlab.runs.folder import SCORES
from pushlab.scoring.answers import answers_path, read_answers
from pushlab.scoring.capability import Answer
from pushlab.scoring.items import Experiment, judged_runs
from pushlab.scoring.measure_rows import (
    TEXT_SCORERS,
    Key,
    Measure,
    capability_measure,
    capability_row,
    newest_rows,
    perplexity_measure,
    perplexity_row,
    repetition_and_variety,
    summary_lines,
    text_row,
)
from pushlab.scoring.score_command import table_text
from pushlab.scoring.score_rows import counted
from pushlab.store.records import RunRecord
from pushlab.store.scores import ScoreRow, read_rows, write_rows

EXIT_INTERRUPTED = 130
CHUNK = 256  # replies whose perplexity is worked out, and saved, at a time
NO_PERPLEXITY = (
    "--batch-size sets how many replies the model reads at a time: give it "
    "with --perplexity"
)
DRY_RUN = "This was a dry run: no model was loaded, and nothing was written."


def _refuse(message: str) -> int:
    print(f"pushlab measure: {message}", file=sys.stderr)
    return 1


@dataclass(frozen=True)
class Setup:
    """What a job measures, read beside the runner folder: the panel; the
    capability measure and the answers by prompt ID, None and empty without
    an answers file; that file, named as the experiment file names its
    prompt set; the perplexity measure and the model download it loads,
    <name>@<revision>, None and "" without --perplexity; and the prompt
    set's texts by prompt ID, which perplexity reads."""

    panel: Panel
    capability: Measure | None
    answers: dict[str, Answer]
    answers_file: Path
    perplexity: Measure | None
    model: str
    texts: dict[str, str]

    @property
    def measures(self) -> list[Measure]:
        """Its measures, in the order they are measured."""
        found = [*repetition_and_variety(self.panel), self.capability, self.perplexity]
        return [measure for measure in found if measure is not None]


@dataclass(frozen=True)
class MeasurePlan:
    """A measure's runs: those it has still to measure, in run order, and
    how many it has measured already."""

    measure: Measure
    todo: list[RunRecord]
    done: int


def plan_measure(
    measure: Measure,
    runs: Sequence[RunRecord],
    version: int,
    newest: Mapping[Key, ScoreRow],
) -> MeasurePlan:
    """What is left of a measure on runs of the folder's version, newest
    being newest_rows' rows: each run with no row of its key.

    Example:
        >>> plan_measure(capability_measure("0" * 64), [], 1, {}).done
        0
    """
    todo = [run for run in runs if measure.key(run.run_id, version) not in newest]
    return MeasurePlan(measure, todo, len(runs) - len(todo))


def _prompt_set(experiment: Experiment) -> PromptSet:
    """The experiment's prompt set; ValueError unless it is the one the
    folder's version was made from."""
    spec, version = experiment.spec, experiment.version
    try:
        prompt_set = read_prompt_set(spec.prompts_path)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read {spec.prompts}: {error}") from None
    if prompt_set.sha256 != experiment.identity.get("prompts_sha256"):
        raise ValueError(
            f"{spec.prompts} changed since version {version} was made, so its "
            "prompts may not be the ones the replies answer: restore it, or run "
            "the experiment again with pushlab run --overwrite-version"
        )
    return prompt_set


def _downloaded(experiment: Experiment) -> str:
    """The model download that wrote the replies, <name>@<revision>, as it
    is downloaded now; FileNotFoundError says how to download it, and
    ValueError when another revision is downloaded."""
    from pushlab.models import load  # here, so that this module loads no torch

    spec = experiment.spec
    model = f"{spec.model}@{load.downloaded_revision(spec.model)}"
    made = experiment.identity.get("model")
    if model != made:
        raise ValueError(
            f"version {experiment.version} of {spec.name} was made with {made}, "
            f"but {model} is downloaded now: measure its perplexity where {made} "
            "is downloaded"
        )
    return model


def read_setup(experiment: Experiment, perplexity: bool) -> Setup:
    """What a job on the experiment measures, checked: the panel, the
    answers file beside the prompt set, if any, and with perplexity the
    model download. OSError or ValueError says what is wrong.

    Example:
        >>> read_setup(experiment, perplexity=False).measures  # doctest: +SKIP
        [Measure(scorer='repetition', ...), Measure(scorer='word_variety', ...)]
    """
    panel, spec = read_panel(), experiment.spec
    answers_file = answers_path(spec.prompts_path)
    has_answers = answers_file.exists()
    prompt_set = _prompt_set(experiment) if has_answers or perplexity else None
    capability, answers = None, {}
    if has_answers and prompt_set is not None:
        answers = read_answers(answers_file, prompt_set)
        capability = capability_measure(
            hashlib.sha256(answers_file.read_bytes()).hexdigest()
        )
    model, measure, texts = "", None, {}
    if perplexity and prompt_set is not None:
        model = _downloaded(experiment)
        measure = perplexity_measure(model, spec.decoding["dtype"])
        texts = dict(zip(prompt_set.prompt_ids, prompt_set.prompts, strict=True))
    shown = answers_path(Path(spec.prompts))  # as the experiment file names it
    return Setup(panel, capability, answers, shown, measure, model, texts)


def header_lines(
    experiment: Experiment, setup: Setup, plans: Sequence[MeasurePlan]
) -> list[tuple[str, str]]:
    """What a job measures and how much is left, as (name, text) lines: the
    experiment, its replies and the failed runs left out; the panel; and
    each measure's replies to measure and measured already, or why it is
    not measured."""
    spec, panel = experiment.spec, setup.panel
    replies = counted(len(experiment.runs), "reply", "replies")
    failed = counted(experiment.failed, "failed run")
    name, version = spec.name, experiment.version
    sizes = f"{panel.ngram}-word n-grams, {panel.window}-word windows"
    texts = {
        "experiment": f"{name}, version {version}: {replies}, {failed} left out",
        "panel": f"configs/panel.yaml version {panel.version}: {sizes}",
        "capability": f"not measured: {spec.prompts} has no answers file beside "
        f"it, {setup.answers_file.name}",
        "perplexity": "not measured: give --perplexity, a GPU job",
    }
    for plan in plans:
        text = f"{len(plan.todo):,} to measure, {plan.done:,} measured already"
        if plan.measure is setup.capability:
            text += f", against {setup.answers_file}"
        if plan.measure is setup.perplexity:
            text += f", under {setup.model} in {spec.decoding['dtype']}"
        texts[plan.measure.scorer] = text
    order = ("experiment", "panel", *TEXT_SCORERS, "capability", "perplexity")
    return [(key, texts[key]) for key in order]


def _perplexity_rows(
    experiment: Experiment, plan: MeasurePlan, setup: Setup, batch_size: int
) -> Iterator[list[ScoreRow]]:
    """perplexity's rows on the plan's runs, from the model loaded as pushlab
    run loads it, nothing pushed: first the replies with no token's, then a
    CHUNK of replies' at a time, each after a progress line, so that every
    chunk is saved before the next is scored."""
    import torch

    from pushlab.engine.score_text import reply_logprobs
    from pushlab.models import load

    spec, version, measure = experiment.spec, experiment.version, plan.measure
    device, dtype = load.default_device(), spec.decoding["dtype"]
    where = "GPU" if device == "cuda" else "CPU"
    print(f"Loading {spec.model} in {dtype} on the {where}...", flush=True)
    loaded = load.load_model(spec.model, getattr(torch, dtype), device=device)
    if f"{spec.model}@{loaded.revision}" != setup.model:
        raise ValueError(f"{spec.model}'s download changed while it loaded; run again")
    if empty := [run for run in plan.todo if not run.output_tokens]:
        yield [perplexity_row(run, version, measure, []) for run in empty]  # NO_TOKENS
    scored = [run for run in plan.todo if run.output_tokens]
    for start in range(0, len(scored), CHUNK):
        chunk = scored[start : start + CHUNK]
        prompts = [setup.texts[run.prompt_id] for run in chunk]  # the version's
        try:
            logprobs = reply_logprobs(
                loaded.model,
                loaded.tokenizer,
                prompts,
                [run.output_tokens for run in chunk],
                batch_size=batch_size,
            )
        except RuntimeError as error:  # torch.OutOfMemoryError among them
            if "out of memory" not in str(error).lower():
                raise
            lower = f"a lower --batch-size than {batch_size}"
            if batch_size == 1:
                lower = "a GPU that no other job is using"
            raise ValueError(
                f"the model ran out of memory scoring perplexity: give it {lower}. "
                "The measures saved before it are kept"
            ) from None
        done = start + len(chunk)
        print(f"  {done:,} of {len(scored):,} replies scored", flush=True)
        yield [
            perplexity_row(run, version, measure, each)
            for run, each in zip(chunk, logprobs, strict=True)
        ]


def _rows(
    experiment: Experiment, plan: MeasurePlan, setup: Setup, batch_size: int
) -> Iterator[list[ScoreRow]]:
    """The rows of a measure on the plan's runs, in the lots to save them in:
    one lot, or with perplexity, one per chunk of replies."""
    version, measure = experiment.version, plan.measure
    if measure.scorer in TEXT_SCORERS:
        yield [text_row(run, version, measure, setup.panel) for run in plan.todo]
    elif measure is setup.capability:
        yield [
            capability_row(run, version, measure, setup.answers.get(run.prompt_id))
            for run in plan.todo
        ]
    else:
        yield from _perplexity_rows(experiment, plan, setup, batch_size)


def _plans(
    experiment: Experiment, setup: Setup
) -> tuple[list[MeasurePlan], dict[Key, ScoreRow]]:
    """Each measure's plan, and the newest rows of the scores table, read
    now, by key (newest_rows); print what the job measures."""
    newest = newest_rows(read_rows(experiment.folder, SCORES), experiment.runs)
    runs, version = experiment.runs, experiment.version
    plans = [plan_measure(each, runs, version, newest) for each in setup.measures]
    print(table_text(header_lines(experiment, setup, plans)))
    return plans, newest


def _measure(
    spec_path: Path, runs_dir: Path | None, perplexity: bool, batch_size: int
) -> int:
    """Measure what is left of the experiment, holding its folder's lock, a
    measure after another, each saved in one file, or perplexity in one
    per chunk of replies; the exit status."""
    experiment = judged_runs(spec_path, runs_dir)  # as saved now, under the lock
    setup = read_setup(experiment, perplexity)
    plans, newest = _plans(experiment, setup)
    if not experiment.runs:
        print("\nNothing to measure: no reply is saved yet.")
        return 0
    if not any(plan.todo for plan in plans):
        print("\nNothing to measure: every reply is measured already.")
    else:
        print()
    runs = {run.run_id: run for run in experiment.runs}
    for plan in plans:
        if not plan.todo:
            continue
        for made in _rows(experiment, plan, setup, batch_size):
            path = write_rows(experiment.folder, made, runs)
            newest |= {row.key: row for row in made}
            line = f"{plan.measure.scorer}: {counted(len(made), 'reply', 'replies')}"
            if errors := sorted({row.error for row in made if row.status != "ok"}):
                failed = sum(row.status != "ok" for row in made)
                line += f", {failed:,} failed: {'; '.join(errors)}"
            print(f"{line}; saved {path.parent.name}/{path.name}")
    lines = summary_lines(experiment, setup.measures, newest, setup.panel)
    print("\n" + "\n".join(lines))
    return 0


def run_measure(
    spec_path: Path,
    *,
    perplexity: bool = False,
    batch_size: int = 8,
    runs_dir: Path | None = None,
    dry_run: bool = False,
) -> int:
    """Measure the replies of the experiment at spec_path and save the rows,
    or with dry_run only count them; return the exit status.

    perplexity also scores each reply under the experiment's model,
    batch_size replies of one length at a time; runs_dir is the folder of
    runner folders (default: folder.default_runs_dir()). Returns 0 once
    done, or with nothing to measure; 1 when it refuses, saying why on
    stderr; and 130 after Ctrl-C.

    Example:
        >>> run_measure(Path("e.yaml"), batch_size=0)  # stderr says why
        1
    """
    if type(batch_size) is not int or batch_size < 1:
        return _refuse(f"--batch-size must be a whole number from 1: {batch_size!r}")
    try:
        experiment = judged_runs(spec_path, runs_dir)
        setup = read_setup(experiment, perplexity)
        if dry_run:
            _plans(experiment, setup)
            print(f"\n{DRY_RUN}")
            return 0
    except (OSError, ValueError) as error:
        return _refuse(str(error))
    try:
        with folder.locked(experiment.folder):
            return _measure(spec_path, runs_dir, perplexity, batch_size)
    except folder.FolderInUse as error:  # a runner's or a scorer's
        return _refuse(f"{error}: try again once it finishes")
    except (OSError, ValueError) as error:
        return _refuse(str(error))
    except KeyboardInterrupt:
        print(
            "\nStopped by Ctrl-C. The measures saved before it are kept: run the "
            "same command to go on."
        )
        return EXIT_INTERRUPTED


def run_measure_command(args: argparse.Namespace) -> int:
    """`pushlab measure` with its parsed options (pushlab.cli_measure); the
    exit status. A --batch-size given without --perplexity is refused.

    Example:
        >>> args = argparse.Namespace(perplexity=False, batch_size=4)
        >>> run_measure_command(args)  # stderr says why
        1
    """
    if args.batch_size is not None and not args.perplexity:
        return _refuse(NO_PERPLEXITY)
    return run_measure(
        args.spec,
        perplexity=args.perplexity,
        batch_size=args.batch_size or 8,
        runs_dir=args.runs_dir,
        dry_run=args.dry_run,
    )
