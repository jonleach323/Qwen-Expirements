"""`pushlab panel select CONFIRM --ladder LADDER --categories FILE [--yes]`:
choose the persona panel (decision 0023).

It reads stage confirm's experiment and stage ladder's, each run and judged
by speaks_as about each candidate's persona and by coherence
(pushlab.panel.stats), and configs/panel.yaml, or --panel's file, and
assesses each candidate of the ladder by the rules (pushlab.panel.select),
printing what each showed. With fewer finalists than the panel's minimum it
stops there, writing nothing, and says what would help. Then the categories
file (pushlab.panel.categories): missing, or leaving a finalist without a
category, it is written for you to fill in, and select stops. Then the
which_persona check (pushlab.panel.distinct): it prints the calls the
finalists' samples take, those already scored or in the cache, and their
cost, and without --yes stops there, having called and written nothing.
With --yes it judges what is left and saves the verdicts in the confirm
folder's scores table, holding the folder's lock, as pushlab score does,
and prints whom the judge told apart and whom it took for whom. Then it
chooses the panel (pushlab.panel.select.choose). Fewer than the panel's
size that qualify are all kept if at least its minimum, with what would
help; fewer than the minimum write nothing. It writes the panel file
(pushlab.panel.panel_file; default units/panel_<model>.yaml) and its page
(pushlab.panel.page; default docs/panel/<model>.md), both where git does
not keep them, and says what to do next. It refuses with exit status 1,
saying why on stderr; Ctrl-C stops it with 130, the verdicts found before
it saved.
"""

import argparse
import hashlib
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from pushlab.analysis.panel import PANEL_FILE, Panel, read_panel
from pushlab.panel.categories import check_categories, read_categories, write_template
from pushlab.panel.distinct import (
    Told,
    judge_sample,
    plan_check,
    sampled_experiment,
    saved_choices,
    speaking,
    told_apart,
)
from pushlab.panel.page import output_paths, page_text, write_page
from pushlab.panel.panel_file import PanelFile, write_panel_file
from pushlab.panel.select import (
    Assessed,
    Qualified,
    assess,
    choose,
    panel_persona,
    rule_advice,
    same_names,
    thin_advice,
)
from pushlab.panel.select_text import panel_lines, plan_lines, rules_lines, told_lines
from pushlab.panel.settings import Selection, read_selection
from pushlab.panel.stage_files import next_out
from pushlab.panel.stats import COHERENCE, SPEAKS_AS, Scored, read_scored
from pushlab.runs import folder
from pushlab.runs.folder import SCORES
from pushlab.scoring import backends
from pushlab.scoring.cache import VerdictCache, default_cache_dir
from pushlab.scoring.family import check_family
from pushlab.scoring.judge import Judge
from pushlab.scoring.judge_costs import estimate_judging, read_judge_costs
from pushlab.scoring.keys import judge_name
from pushlab.scoring.score_command import table_text
from pushlab.scoring.score_rows import counted
from pushlab.store.folders import shown_path
from pushlab.store.scores import read_rows

NEEDS = (SPEAKS_AS, COHERENCE)
EXIT_INTERRUPTED = 130
EXAMPLES = 2  # example lines a persona gets on the page


def refuse(message: str) -> int:
    """Say why select refuses, on stderr; its exit status, 1."""
    print(f"pushlab panel select: {message}", file=sys.stderr)
    return 1


def error_text(error: Exception) -> str:
    """An error in words: an OSError's file and reason, else its message.

    Example:
        >>> error_text(FileNotFoundError(2, "No such file or directory", "k.csv"))
        'k.csv: No such file or directory'
    """
    if isinstance(error, OSError) and error.strerror:
        where = f"{error.filename}: " if error.filename else ""
        return f"{where}{error.strerror}"
    return str(error)


@dataclass(frozen=True)
class Read:
    """What select read: the settings, the two experiments, where its files
    go, and each candidate of the ladder, assessed."""

    selection: Selection
    panel: Panel
    confirm: Scored
    ladder: Scored
    out: Path
    doc: Path
    assessed: list[Assessed]

    @property
    def finalists(self) -> list[Assessed]:
        """Those that pass every rule so far, in the ladder's order."""
        return [each for each in self.assessed if each.reason is None]

    @property
    def reasons(self) -> Counter[str]:
        """How many of the others failed each rule, each counted under the
        first rule it failed."""
        return Counter(each.reason for each in self.assessed if each.reason)


def scored(path: Path, args: argparse.Namespace) -> Scored:
    """The experiment at path, as read_scored reads it with the options'
    runs folder and judge; a refusal that does not name path, such as one
    about the judges saved, is put after it, as select reads two."""
    try:
        return read_scored(path, args.runs_dir, args.judge, NEEDS)
    except ValueError as error:
        if str(path) in str(error):
            raise
        raise ValueError(f"{path}: {error}") from None


def read_all(args: argparse.Namespace) -> Read:
    """Read and assess what select needs; ValueError or OSError says why it
    cannot: the options' files, a confirm and a ladder of different lists
    or models, and what read_scored and assess refuse."""
    check_categories(args.categories)
    selection, panel = read_selection(args.panel), read_panel(args.panel)
    confirm, ladder = scored(args.source, args), scored(args.ladder, args)
    again = "give the ladder that stage confirm was written from"
    if confirm.rule.path.resolve() != ladder.rule.path.resolve():
        raise ValueError(
            f"{args.source} pushes the candidates of {confirm.rule.file}, but "
            f"{args.ladder} those of {ladder.rule.file}: {again}"
        )
    if confirm.spec.model != ladder.spec.model:
        raise ValueError(
            f"{args.source} runs on {confirm.spec.model}, but {args.ladder} on "
            f"{ladder.spec.model}: {again}"
        )
    out, doc = output_paths(args.out, args.doc, confirm.spec.model)
    assessed = same_names(assess(confirm, ladder, panel))
    return Read(selection, panel, confirm, ladder, out, doc, assessed)


def short_of(qualify: int, minimum: int, reasons: Counter[str], thin: str = "") -> str:
    """The refusal of a panel short of its minimum, with what would help:
    the rule most of the others failed (rule_advice), then thin, the thin
    categories' advice, if any.

    Example:
        >>> print(short_of(1, 16, Counter({"dose": 3})))  # doctest: +ELLIPSIS
        only 1 qualifies, fewer than the panel's minimum of 16, so nothing was ...
    """
    found = f"only {qualify:,} qualify" if qualify else "no candidate qualifies"
    if qualify == 1:
        found = "only 1 qualifies"
    stop = f"{found}, fewer than the panel's minimum of {minimum}, so nothing was "
    parts = (stop + "written.", rule_advice(reasons), thin)
    return " ".join(part for part in parts if part)


def ask_categories(args: argparse.Namespace, read: Read) -> dict[str, str] | None:
    """Each finalist's category from the categories file; None, once the
    file is written for you to fill in, when it is missing or unfinished.
    ValueError or OSError when it cannot be read or written."""
    known = read_categories(args.categories, read.selection.categories)
    finalists = read.finalists
    keys = [each.key for each in finalists]
    missing = keys if known is None else known.unfilled(keys)
    if known is not None and not missing:
        return {key: known.categories[key] for key in keys}
    write_template(args.categories, [(f.key, f.persona) for f in finalists], known)
    have = "has" if len(missing) == 1 else "have"
    kept = "" if known is None else ", the categories it gave kept"
    print(
        f"\nWrote {args.categories}: a row per finalist{kept}; "
        f"{counted(len(missing), 'finalist')} of {len(keys):,} {have} no category "
        f"yet, such as {missing[0]}."
    )
    return None


def which_judge(
    args: argparse.Namespace, judge: Judge | None
) -> tuple[tuple[str, str, str], VerdictCache | None]:
    """The which_persona judge's backend, model and settings, and its
    cache: judge's when given, else the options'. ValueError for an unknown
    backend, or local with no model."""
    if judge is not None:
        given = judge.backend
        return (given.name, given.model, getattr(given, "settings", "")), judge.cache
    model = backends.judge_model(args.backend, args.model)
    cache = VerdictCache(default_cache_dir(), judge_name(args.backend, model))
    return (args.backend, model, backends.judge_settings(args.backend)), cache


def run_select(args: argparse.Namespace, judge: Judge | None = None) -> int:
    """`pushlab panel select` with its parsed options (pushlab.cli_panel),
    as the module's docstring says; judge, for tests, is used as it is,
    with its own backend and cache. The exit status.

    Example:
        >>> args = argparse.Namespace(source=Path("no.yaml"), ladder=Path("l.yaml"))
        >>> args.categories, args.base_url, args.backend = Path("/w/k.csv"), None, "api"
        >>> args.panel = args.judge = args.runs_dir = args.out = args.doc = None
        >>> run_select(args)  # stderr says why
        1
    """
    if args.base_url is not None and args.backend != "local":
        return refuse(
            "--base-url is the local backend's: the URL of the vLLM server that "
            "serves its model"
        )
    try:
        read = read_all(args)
    except (OSError, ValueError) as error:
        return refuse(error_text(error))
    print("\n".join(rules_lines(read.confirm, read.ladder, read.panel, read.assessed)))
    selection, minimum = read.selection, read.selection.panel_minimum
    if len(read.finalists) < minimum:
        return refuse(short_of(len(read.finalists), minimum, read.reasons))
    try:
        categories = ask_categories(args, read)
        if categories is None:
            *others, last = selection.categories
            return refuse(
                f"fill in each finalist's category in {args.categories}, one of "
                f"{', '.join(others)} or {last}, then run the same command again"
            )
        judging, cache = which_judge(args, judge)
        check_family(judging[1], [read.confirm.spec.model])
        check = plan_check(
            read.finalists,
            read.confirm.experiment,
            selection.distinct_sample,
            read.panel.seed,
            judging,
            cache,
        )
        requests = check.plan.requests
        costs = estimate_judging(
            judging[0], judging[1], requests, len(requests), read_judge_costs()
        )
    except (OSError, ValueError) as error:
        return refuse(error_text(error))
    seed = read.panel.seed
    plan = plan_lines(check.question, check.judge, check.plan, selection, seed, costs)
    print("\nThe which_persona check, of each finalist against all the others:")
    print(table_text(plan))
    if not args.yes:
        calls = "make these calls and choose" if requests else "choose"
        print(
            "\nNothing was called or written. Run the same command with --yes to "
            f"{calls} the panel."
        )
        return 0
    try:
        if not check.plan.todo:  # every verdict is saved: no judge, nor its key
            rows, stopped = read_rows(read.confirm.experiment.folder, SCORES), None
        else:
            if judge is None:
                made = backends.make_backend(judging[0], judging[1], args.base_url)
                judge = Judge(made, cache)
            rows, stopped = judge_sample(args.source, args.runs_dir, judge, check)
    except folder.FolderInUse as error:
        return refuse(f"{error}: try again once it finishes")
    except (OSError, ValueError) as error:
        return refuse(error_text(error))
    except KeyboardInterrupt:
        print(
            "\nStopped by Ctrl-C. The verdicts found before it are saved: run the "
            "same command to go on."
        )
        return EXIT_INTERRUPTED
    if stopped is not None:
        return refuse(
            f"{stopped}. The verdicts found before it stopped are saved; run the "
            "same command to go on once the judge can answer"
        )
    part = sampled_experiment(read.confirm.experiment, check.run_ids)
    choices = saved_choices(rows, part, check.question, check.judge)
    names = {each.key: each.persona for each in read.finalists}
    told = told_apart(check.samples, names, choices)
    print("\n" + "\n".join(told_lines(read.finalists, told, selection, check.judge)))
    return write_panel(args, read, categories, told, check.judge)


def qualified(
    read: Read, categories: Mapping[str, str], told: Mapping[str, Told]
) -> list[Qualified]:
    """The finalists the which_persona check told apart, each with its
    category and its figures, in the ladder's order."""
    found = []
    for each in read.finalists:
        if told[each.key].passes(read.selection.distinct_rate):
            assert each.coherence is not None  # a finalist has a score
            found.append(
                Qualified(
                    each.key,
                    each.candidate.kind,
                    categories[each.key],
                    each.reliability,
                    each.coherence,
                    told[each.key].rate,
                )
            )
    return found


def write_panel(
    args: argparse.Namespace,
    read: Read,
    categories: Mapping[str, str],
    told: Mapping[str, Told],
    judge: str,
) -> int:
    """Choose the panel from the finalists told apart, and write it and its
    page, or nothing when fewer than the minimum qualify; the exit status."""
    selection, panel = read.selection, read.panel
    confirm, ladder = read.confirm, read.ladder
    left = qualified(read, categories, told)
    reasons = read.reasons + Counter(distinct=len(read.finalists) - len(left))
    chosen = choose(left, selection.categories, selection.panel_size)
    if len(chosen) < selection.panel_minimum:
        thin = thin_advice(chosen, selection)
        return refuse(short_of(len(chosen), selection.panel_minimum, reasons, thin))
    by_key = {each.key: each for each in read.finalists}
    personas = tuple(
        panel_persona(
            by_key[each.key],
            each.category,
            told[each.key].share(panel.confidence),
            panel.confidence,
        )
        for each in sorted(chosen, key=lambda each: each.strength)
    )
    made_from = {
        "confirm": f"{confirm.spec.name}, version {confirm.experiment.version}, "
        f"judged by {confirm.judge}",
        "ladder": f"{ladder.spec.name}, version {ladder.experiment.version}, "
        f"judged by {ladder.judge}",
        "which_persona": f"judged by {judge}, up to {selection.distinct_sample} "
        f"replies a finalist, seed {panel.seed}",
        "settings": f"{shown_path(panel.path or PANEL_FILE)}, version {panel.version}",
        "categories": str(args.categories),
    }
    examples = {
        each.key: [
            (reply.prompt_id, reply.text)
            for reply in speaking(by_key[each.key].replies.replies)[:EXAMPLES]
        ]
        for each in chosen
    }
    try:
        sha256 = hashlib.sha256(confirm.rule.path.read_bytes()).hexdigest()
        model, listed = confirm.spec.model, confirm.rule.file
        made = PanelFile(model, listed, sha256, made_from, personas)
        write_panel_file(read.out, made)
        text = page_text(made, examples, shown_path(read.out), panel.confidence)
        write_page(read.doc, text)
    except (OSError, ValueError) as error:
        return refuse(error_text(error))
    print("\n" + "\n".join(panel_lines(made, selection)))
    if len(chosen) < selection.panel_size:
        helps = (rule_advice(reasons), thin_advice(chosen, selection))
        print("\n" + "\n".join(line for line in helps if line))
    out = shown_path(read.out)
    preview = next_out(Path(args.source), "confirm", "preview")
    print(
        f"\nWrote {out}, the panel of {counted(len(personas), 'persona')}, and "
        f"{shown_path(read.doc)}, its page. They name the candidates' personas, the "
        "Neuron Persona Library's among them, so git keeps them out (decision "
        "0020).\nNext, see its pairs pushed in four shapes:\n"
        f"  uv run pushlab panel stage preview {out} --out {preview}"
    )
    return 0
