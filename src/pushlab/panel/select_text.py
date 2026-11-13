"""What pushlab panel select prints (decision 0023): what it read and how
each candidate fared by the rules, the which_persona check's plan and what
it found, and the panel it chose. Nothing here reads or writes a file.
"""

from collections.abc import Mapping, Sequence

from pushlab.analysis.dose_text import aligned, short_text
from pushlab.analysis.panel import Panel
from pushlab.panel.distinct import Told, confusion_text
from pushlab.panel.page import mean_text, share_text
from pushlab.panel.panel_file import PanelFile
from pushlab.panel.select import Assessed, reason_text
from pushlab.panel.settings import Selection
from pushlab.panel.stage_text import plain
from pushlab.panel.stages import read_lines
from pushlab.panel.stats import COHERENCE, SPEAKS_AS, Scored
from pushlab.runs.estimate import exact
from pushlab.scoring.score_rows import Plan, Question, counted
from pushlab.store.folders import shown_path

NEEDS = (SPEAKS_AS, COHERENCE)


def rules_table(assessed: Sequence[Assessed], panel: Panel) -> list[str]:
    """The candidates as a table: each one's key, persona and kind, its
    working and collapse doses on the ladder with their intervals, its
    speaks_as verdicts in stage confirm, k of n, their rate, its mean
    coherence there, and whether it is a finalist, or why it is left out.

    Example:
        >>> from pushlab.analysis.panel import read_panel
        >>> print(rules_table([], read_panel())[0])  # doctest: +NORMALIZE_WHITESPACE
        candidate persona kind working collapse speaks_as rate coherence result
    """
    head = ["candidate", "persona", "kind", "working", "collapse", "speaks_as"]
    table = [[*head, "rate", "coherence", "result"]]
    for each in assessed:
        confirmed = bool(each.replies.runs)
        result = "finalist"
        if each.reason is not None:
            result = f"left out: {reason_text(each, panel)}"
        table.append(
            [each.key, plain(each.persona), each.candidate.kind]
            + [short_text(each.doses.working), short_text(each.doses.collapse)]
            + [f"{each.k}/{each.n}" if confirmed else "-"]
            + [f"{float(each.reliability):.2f}" if confirmed else "-"]
            + ["-" if each.coherence is None else f"{each.coherence:.1f}", result]
        )
    return aligned(table)


def rules_lines(
    confirm: Scored, ladder: Scored, panel: Panel, assessed: Sequence[Assessed]
) -> list[str]:
    """What select read, the rules, and how each candidate fared by them."""
    percent = f"{round(panel.confidence * 100, 6):g}%"
    finalists = sum(each.reason is None for each in assessed)
    where = "" if panel.path is None else f" ({shown_path(panel.path)})"
    return [
        *read_lines(confirm, NEEDS),
        *read_lines(ladder, NEEDS),
        "",
        f"The rules{where}: a working dose below any collapse dose on the ladder; "
        f"at it, in stage confirm, speaking as its persona on at least "
        f"{panel.working_rate:g} of the prompts, a prompt without a verdict "
        f"counting against it, with a mean coherence of at least "
        f"{panel.coherence_below:g}; and no stronger finalist of its persona's "
        f"name. Doses as pushed, with {percent} intervals:",
        *rules_table(assessed, panel),
        f"{counted(finalists, 'finalist')} of {counted(len(assessed), 'candidate')}.",
    ]


def plan_lines(
    question: Question,
    judge: str,
    plan: Plan,
    selection: Selection,
    seed: int,
    costs: Sequence[tuple[str, str]],
) -> list[tuple[str, str]]:
    """The which_persona check's plan, as (name, text) lines: the question,
    the judge, the sample, what is scored already or cached, the calls to
    make, and costs, their estimate (estimate_judging)."""
    names = ", ".join(persona.name for persona in question.personas)
    sample = (
        f"up to {selection.distinct_sample} replies of each finalist that speak as "
        f"it, drawn with seed {seed}: {counted(len(plan.items), 'reply', 'replies')}"
    )
    return [
        ("rubric", f"{question.rubric.name}, version {question.rubric.version}"),
        ("personas", f"{len(question.personas):,}, every finalist: {names}"),
        ("judge", judge),
        ("sample", sample),
        ("already scored", f"{len(plan.items) - len(plan.todo):,}"),
        ("in the cache", f"{plan.cached:,}"),
        ("calls to make", f"{len(plan.requests):,}"),
        *costs,
    ]


def told_lines(
    finalists: Sequence[Assessed],
    told: Mapping[str, Told],
    selection: Selection,
    judge: str,
) -> list[str]:
    """Whom the which_persona check told apart: each finalist's sample, the
    replies the judge named it for, and whom else it chose for them; then a
    line for each finalist left out, saying whom it was taken for."""
    rate = selection.distinct_rate
    table = [["candidate", "persona", "named", "rate", "taken for", "result"]]
    left_out = []
    for each in finalists:
        found = told[each.key]
        kept = found.passes(rate)
        table.append(
            [each.key, plain(each.persona), f"{found.k}/{found.n}"]
            + [f"{float(found.rate):.2f}", plain(confusion_text(found))]
            + ["told apart" if kept else "left out"]
        )
        if not kept:
            left_out.append(
                f"Left out {each.key} ({plain(each.persona)}): named for {found.k} "
                f"of {found.n}, taken for {plain(confusion_text(found))}."
            )
    return [
        f"Told apart by which_persona, judged by {judge}: the replies of each "
        "finalist's sample the judge named it for, and whom else it chose for "
        f"them. Named for under {rate:g} of its sample, a finalist is left out:",
        *aligned(table),
        *left_out,
    ]


def panel_lines(chosen: PanelFile, selection: Selection) -> list[str]:
    """The panel as a table, strongest first, under a line that says how
    many it holds, and whether that is every one that qualified."""
    head = ["rank", "candidate", "persona", "category", "kind", "working"]
    table = [[*head, "collapse", "reliability", "coherence", "distinct"]]
    for rank, each in enumerate(chosen.personas, 1):
        collapse = "beyond" if each.collapse_dose is None else exact(each.collapse_dose)
        table.append(
            [str(rank), each.key, plain(each.persona), each.category, each.kind]
            + [exact(each.working_dose), collapse, share_text(each.reliability)]
            + [mean_text(each.coherence), share_text(each.distinct)]
        )
    held = counted(len(chosen.personas), "persona")
    if len(chosen.personas) < selection.panel_size:
        held += (
            f", fewer than the panel's {selection.panel_size}: every one that qualifies"
        )
    return [f"The panel: {held}, strongest first:", *aligned(table)]
