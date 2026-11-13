"""`pushlab score SPEC --rubric NAME`: a judge model's verdicts on an
experiment's replies, saved in its runner folder (decision 0021).

It judges the replies of the runner folder's current version, "ok" or
"collapsed", never a failed run's (pushlab.scoring.items): each whole, into
the scores table, or with --windows sentences each window of --window-size
sentences on its own, into the sentences table, its row saying which
sentences, characters and reply tokens it spans (pushlab.store.scores); the
tokens need the experiment model's tokenizer, as `pushlab report SPEC`
does. The judge (pushlab.scoring.judge) sees only the rubric, the persona
or personas given on the command line, the same for every reply, and the
text; it must be of another family than the experiment's model, which is
checked before anything is judged.

An item scored "ok" already, for the same key, reply and text, is skipped
(pushlab.scoring.score_rows); a failed one is judged again. Of the rest,
the cache answers those judged before, a blank text fails at once, and the
judge is called for each other distinct text. --dry-run prints these
counts and the cost (pushlab.scoring.judge_costs) and builds no backend, so
it needs no key, package or server, takes no lock and writes nothing. A
real run holds the folder's lock, which keeps a runner and other scorers
out, and saves each chunk's rows before judging the next, so a job killed
outright loses at most a chunk, whose ok verdicts are cached anyway. A
judge that cannot answer at all, or Ctrl-C, stops the job once the
verdicts found before the stop are saved, failures included.
"""

import argparse
import sys
from collections import Counter
from collections.abc import Callable, Sequence
from pathlib import Path

from pushlab.runs import folder
from pushlab.scoring import backends
from pushlab.scoring.cache import VerdictCache, default_cache_dir
from pushlab.scoring.family import check_family
from pushlab.scoring.items import Experiment, judged_runs
from pushlab.scoring.judge import Judge, Verdict, judge_name
from pushlab.scoring.judge_costs import estimate_judging, read_judge_costs, spent_text
from pushlab.scoring.rubrics import load_rubric
from pushlab.scoring.score_rows import (
    Plan,
    Question,
    counted,
    judge_chunks,
    make_plan,
    question_from,
    window_spans,
)
from pushlab.scoring.spend import Spent

WINDOWS = ("sentences",)  # what --windows may cut a reply into
ERRORS_SHOWN = 3  # the most common errors a summary names
EXIT_INTERRUPTED = 130
NO_WINDOWS = (
    "--window-size sets how many sentences a window holds: give it with "
    "--windows sentences"
)
DRY_RUN = "This was a dry run: no judge was built or called, and nothing was written."


def _refuse(message: str) -> int:
    print(f"pushlab score: {message}", file=sys.stderr)
    return 1


def table_text(lines: Sequence[tuple[str, str]]) -> str:
    """(name, text) lines as text, each name padded so the texts line up.

    Example:
        >>> print(table_text([("items", "10"), ("calls to make", "4")]))
        items          10
        calls to make  4
    """
    width = max((len(name) for name, _ in lines), default=0) + 2
    return "\n".join(f"{name:<{width}}{text}" for name, text in lines)


def header_lines(
    experiment: Experiment, question: Question, judge: str, size: int | None, plan: Plan
) -> list[tuple[str, str]]:
    """What a job judges and how much is left, as (name, text) lines: the
    experiment, its version, replies and the failed runs left out; the
    rubric, the persona or personas, the judge, whole replies or windows;
    the items, those scored already or in the cache or with no text, and
    the calls to make."""
    rubric, persona = question.rubric, question.persona
    if question.personas:
        names = ", ".join(each.name for each in question.personas)
        asked = ("personas", f"{len(question.personas)}: {names}")
    elif persona is not None:
        about = persona.description.strip()
        asked = ("persona", persona.name + (f": {about}" if about else ""))
    else:
        asked = ("persona", f"none: {rubric.name} asks about no persona")
    texts = "whole replies, into the scores table"
    if size is not None:
        texts = f"windows of {counted(size, 'sentence')}, into the sentences table"
    replies = counted(len(experiment.runs), "reply", "replies")
    failed = counted(experiment.failed, "failed run")
    name, version = experiment.spec.name, experiment.version
    return [
        ("experiment", f"{name}, version {version}: {replies}, {failed} left out"),
        ("rubric", f"{rubric.name}, version {rubric.version}"),
        asked,
        ("judge", judge),
        ("texts", texts),
        ("items", f"{len(plan.items):,}"),
        ("already scored", f"{len(plan.items) - len(plan.todo):,}"),
        ("in the cache", f"{plan.cached:,}"),
        ("no text", f"{plan.blank:,}"),
        ("calls to make", f"{len(plan.requests):,}"),
    ]


def summary_lines(verdicts: Sequence[Verdict], spent: Spent) -> list[tuple[str, str]]:
    """What a job found, as (name, text) lines: its verdicts ok and failed,
    those from the cache, the calls made and the tokens they used (spent,
    every call counted as it was made: pushlab.scoring.spend), and the most
    common errors, each with how often it came.

    Example:
        >>> summary_lines([], Spent(2, 300, 40))[2:]
        [('calls made', '2'), ('tokens used', '300 input and 40 output')]
    """
    ok = sum(verdict.ok for verdict in verdicts)
    tokens = f"{spent.input_tokens:,} input and {spent.output_tokens:,} output"
    lines = [
        ("verdicts", f"{len(verdicts):,}: {ok:,} ok, {len(verdicts) - ok:,} failed"),
        ("from the cache", f"{sum(verdict.cached for verdict in verdicts):,}"),
        ("calls made", f"{spent.calls:,}"),
        ("tokens used", tokens),
    ]
    errors = Counter(verdict.error for verdict in verdicts if not verdict.ok)
    for place, (error, count) in enumerate(errors.most_common(ERRORS_SHOWN)):
        lines.append(("errors" if place == 0 else "", f"{count:,}: {error}"))
    return lines


def _decoder(model: str) -> Callable[[Sequence[int]], str]:
    """The decode of model's downloaded tokenizer, without special tokens,
    as the replies' texts were made; OSError or ValueError without it."""
    from pushlab.models import load  # here, so that this module loads no torch

    tokenizer = load.load_tokenizer(model)

    def decode(tokens: Sequence[int]) -> str:
        return tokenizer.decode(list(tokens), skip_special_tokens=True)

    return decode


def _score(
    spec_path: Path,
    runs_dir: Path | None,
    question: Question,
    judge: Judge,
    size: int | None,
    workers: int,
    decode: Callable[[Sequence[int]], str] | None,
) -> int:
    """Judge what is left of the experiment, holding its folder's lock; the
    exit status. size is a window's sentences, None for whole replies."""
    experiment = judged_runs(spec_path, runs_dir)  # as saved now, under the lock
    backend = judge.backend
    plan = make_plan(
        experiment,
        question,
        size,
        judge.name,
        backend.name,
        backend.model,
        judge.cache,
        getattr(backend, "settings", ""),
    )
    print(table_text(header_lines(experiment, question, judge.name, size, plan)))
    if not plan.todo:
        why = "every item is scored already" if plan.items else "no reply is saved yet"
        print(f"\nNothing to judge: {why}.")
        return 0
    spans = {}
    if size is not None and decode is not None:
        runs = {item.run.run_id: item.run for item in plan.todo}
        spans = window_spans(runs.values(), size, decode)
    before = judge.spent.now()
    try:
        verdicts, stopped = judge_chunks(
            judge, experiment, question, plan, spans, workers
        )
    finally:
        spent = judge.spent.now() - before  # every call made, its tokens too
    lines = summary_lines(verdicts, spent)
    if backend.name == "api":
        try:
            tokens = (spent.input_tokens, spent.output_tokens)
            cost = spent_text(backend.model, *tokens, read_judge_costs())
        except ValueError as error:
            cost = f"unknown: {error}"
        lines.append(("cost", cost))
    print(f"\n{table_text(lines)}")
    if stopped is not None:
        return _refuse(
            f"{stopped}. The verdicts found before it stopped are saved; run the "
            "same command to go on once the judge can answer"
        )
    return 0


def run_score(
    spec_path: Path,
    rubric_name: str,
    *,
    windows: str | None = None,
    window_size: int = 1,
    personas_file: Path | None = None,
    persona: str | None = None,
    backend: str = "api",
    model: str | None = None,
    base_url: str | None = None,
    workers: int = 4,
    cache_dir: Path | None = None,
    runs_dir: Path | None = None,
    dry_run: bool = False,
    judge: Judge | None = None,
) -> int:
    """Judge the replies of the experiment at spec_path by a rubric and save
    the verdicts, or with dry_run only count them; return the exit status.

    windows "sentences" judges windows of window_size sentences, None whole
    replies. personas_file and persona give the persona or personas the
    rubric needs. backend and model name the judge (backends.judge_model),
    base_url a local backend's server, cache_dir its cache (default:
    default_cache_dir()), and runs_dir the folder of runner folders
    (default: folder.default_runs_dir()). judge, for tests, is used as it
    is, with its own backend and cache. Returns 0 once done, 1 when it
    refuses (why, on stderr) or the judge cannot answer at all, and 130
    after Ctrl-C.

    Example:
        >>> run_score(Path("e.yaml"), "coherence", window_size=2)  # stderr says why
        1
    """
    if windows is not None and windows not in WINDOWS:
        return _refuse(f"--windows takes {', '.join(WINDOWS)}, not {windows!r}")
    if windows is None and window_size != 1:
        return _refuse(NO_WINDOWS)
    for option, value in (("--window-size", window_size), ("--workers", workers)):
        if type(value) is not int or value < 1:
            return _refuse(f"{option} must be a whole number from 1, not {value!r}")
    if base_url is not None and backend != "local":
        return _refuse(
            "--base-url is the local backend's: the URL of the vLLM server that "
            "serves its model"
        )
    size = window_size if windows else None
    try:
        question = question_from(load_rubric(rubric_name), personas_file, persona)
        if judge is None:
            backend_name, model_name = backend, backends.judge_model(backend, model)
            settings = backends.judge_settings(backend)
        else:
            backend_name, model_name = judge.backend.name, judge.backend.model
            settings = getattr(judge.backend, "settings", "")
        name = judge_name(backend_name, model_name)
        experiment = judged_runs(spec_path, runs_dir)
        check_family(model_name, [experiment.spec.model])
        if judge is None:
            cache = VerdictCache(cache_dir or default_cache_dir(), name)
        else:
            cache = judge.cache
        if dry_run:
            plan = make_plan(
                experiment,
                question,
                size,
                name,
                backend_name,
                model_name,
                cache,
                settings,
            )
            lines = header_lines(experiment, question, name, size, plan)
            calls, costs = len(plan.requests), read_judge_costs()
            lines += estimate_judging(
                backend_name, model_name, plan.requests, calls, costs
            )
            print(f"{table_text(lines)}\n\n{DRY_RUN}")
            return 0
        if judge is None:
            judge = Judge(backends.make_backend(backend, model, base_url), cache)
        decode = None if size is None else _decoder(experiment.spec.model)
    except (OSError, ValueError) as error:
        return _refuse(str(error))
    try:
        with folder.locked(experiment.folder):
            return _score(spec_path, runs_dir, question, judge, size, workers, decode)
    except folder.FolderInUse as error:  # a runner's or another scorer's
        return _refuse(f"{error}: try again once it finishes")
    except (OSError, ValueError) as error:
        return _refuse(str(error))
    except KeyboardInterrupt:
        print(
            "\nStopped by Ctrl-C. The verdicts found before it are saved: run the "
            "same command to go on."
        )
        return EXIT_INTERRUPTED


def run_score_command(args: argparse.Namespace) -> int:
    """`pushlab score` with its parsed options (pushlab.cli_scoring); the
    exit status. A --window-size given without --windows is refused.

    Example:
        >>> args = argparse.Namespace(windows=None, window_size=1)
        >>> run_score_command(args)  # stderr says why
        1
    """
    if args.window_size is not None and args.windows is None:
        return _refuse(NO_WINDOWS)
    return run_score(
        args.spec,
        args.rubric,
        windows=args.windows,
        window_size=args.window_size or 1,
        personas_file=args.personas,
        persona=args.persona,
        backend=args.backend,
        model=args.model,
        base_url=args.base_url,
        workers=args.workers,
        cache_dir=args.cache_dir,
        runs_dir=args.runs_dir,
        dry_run=args.dry_run,
    )
