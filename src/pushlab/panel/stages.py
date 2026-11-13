"""pushlab panel stage b, ladder and confirm: the next experiment from the
last one's scores (decision 0023).

Each reads a scored experiment that pushes candidates (pushlab.panel.stats)
and chooses which of them go on, as a Stage (pushlab.panel.stage_files):
- b, from screen_a's scores: the selection's stage_b_size candidates with
  the highest speaks_as rates over their replies, ties to the more
  verdicts, then the candidates list's order (ranked), each at its
  reference dose on every prompt of the set, above the dose ceiling where
  screen_a allowed it;
- ladder, from stage b's: its ladder_size best, ranked the same way, at
  each of ladder_doses, multiples of the reference dose, on the set's first
  ladder_prompts prompts, above the ceiling where a dose goes;
- confirm, from the ladder's speaks_as and coherence scores: each candidate
  whose working dose (pushlab.analysis.ladder, with configs/panel.yaml's
  thresholds) is below any collapse dose, at its working dose, on every
  prompt; NoneKept when none is.
A candidate with no speaks_as verdict is not ranked. Each stage's shown
lines say what it read, with notes on what it left out, and a table of the
candidates it kept with their figures.
"""

import math
from collections.abc import Sequence
from fractions import Fraction

from pushlab.analysis.dose_text import aligned, short_text, window_text
from pushlab.analysis.ladder import UnitDoses, analyze_unit
from pushlab.analysis.panel import Panel
from pushlab.analysis.report_folder import changed_notes
from pushlab.discovery.sources import LIBRARY
from pushlab.panel.settings import Selection
from pushlab.panel.stage_files import Stage, chosen_prompts, set_prompts
from pushlab.panel.stage_text import plain
from pushlab.panel.stats import COHERENCE, SPEAKS_AS, CandidateStats, Scored
from pushlab.runs.estimate import exact
from pushlab.runs.ladder import doses_text
from pushlab.runs.spec import ExperimentSpec
from pushlab.scoring.score_rows import counted

LABEL = "candidate"  # each_candidate's condition: candidate-0001 and so on


class NoneKept(ValueError):
    """A stage that keeps no candidate: the message says why, and shown
    holds the lines that show what it read."""

    def __init__(self, message: str, shown: Sequence[str]) -> None:
        super().__init__(message)
        self.shown = tuple(shown)


def ranked(stats: Sequence[CandidateStats]) -> list[CandidateStats]:
    """The candidates with a speaks_as verdict, best first: the highest rate
    over all their replies, ties to the more verdicts, then the list's order.

    Example:
        >>> ranked([])
        []
    """

    def rank(each: CandidateStats) -> tuple[Fraction, int, int]:
        total = each.total  # exact fractions: 2/4 ties 4/8
        return -Fraction(total.k, total.n), -total.n, each.place

    return sorted((each for each in stats if each.total.n), key=rank)


def over_ceiling(spec: ExperimentSpec) -> bool:
    """Whether spec's conditions that push candidates allow doses above the
    dose ceiling."""
    return any(
        schedule.allow_over_ceiling
        for condition in spec.conditions
        if condition.candidate is not None
        for _, schedule in condition.points
    )


def _number(value: float | None, places: int) -> str:
    return "-" if value is None else f"{value:.{places}f}"


def figures_table(rows: Sequence[CandidateStats]) -> list[str]:
    """The candidates as a table, ranked: each one's key, persona and kind,
    and its replies' figures over every dose.

    Example:
        >>> figures_table([])
        ['rank  candidate  persona  kind  replies  speaks_as  rate  coherence']
    """
    head = ["rank", "candidate", "persona", "kind", "replies", "speaks_as"]
    table = [[*head, "rate", "coherence"]]
    for rank, each in enumerate(rows, 1):
        total, candidate = each.total, each.candidate
        table.append(
            [str(rank), each.key, plain(candidate.persona), candidate.kind]
            + [str(total.runs), f"{total.k}/{total.n}" if total.n else "-"]
            + [_number(total.rate, 2), _number(total.coherence, 1)]
        )
    return aligned(table)


def read_lines(scored: Scored, needs: Sequence[str]) -> list[str]:
    """What a stage read, in words, and notes on what it leaves out: the
    file's changes since the runs were made, candidates with no saved reply,
    replies with no verdict of a rubric in needs, and replies that push no
    candidate."""
    spec, experiment = scored.spec, scored.experiment
    replies = sum(each.total.runs for each in scored.stats)
    lines = [
        f"Read {scored.path}: {spec.name}, version {experiment.version}, "
        f"{counted(replies, 'reply', 'replies')} of its "
        f"{counted(len(scored.stats), 'candidate')}, judged by {scored.judge}; "
        f"{counted(experiment.failed, 'failed run')} left out."
    ]
    notes = changed_notes(spec, experiment.identity)
    if unrun := scored.unrun:
        have = "has" if len(unrun) == 1 else "have"
        notes.append(
            f"{counted(len(unrun), 'candidate')} {have} no saved reply, such as "
            f"{unrun[0]}: `uv run pushlab run {scored.path}` runs what is left."
        )
    for rubric in needs:
        if missing := scored.unjudged(rubric):
            have = "has" if missing == 1 else "have"
            notes.append(
                f"{counted(missing, 'reply', 'replies')} {have} no ok {rubric} "
                f"verdict of {scored.judge}: `{scored.make(rubric)}` judges them."
            )
    if others := scored.others:
        push = "pushes" if others == 1 else "push"
        replies = counted(others, "reply", "replies")
        notes.append(f"{replies} {push} no candidate of it: left out.")
    return lines + [f"Note: {note}" for note in notes]


def _figures_note(each: CandidateStats) -> str:
    total = each.total
    return f"{total.k}/{total.n} speaks_as, {each.candidate.persona}"


def _best(scored: Scored, size: int) -> tuple[list[CandidateStats], list[str]]:
    """The size best of scored's candidates, and the lines that show them."""
    best = ranked(scored.stats)
    kept = best[:size]
    unrated = sum(1 for each in scored.stats if each.doses and not each.total.n)
    shown = [*read_lines(scored, (SPEAKS_AS,)), ""]
    if not kept:
        raise NoneKept(
            f"no candidate of {scored.spec.name} has a speaks_as verdict: "
            f"`{scored.make(SPEAKS_AS)}` makes them",
            shown,
        )
    every = "all" if len(kept) == len(best) else "the"
    shown.append(
        f"Kept {every} {len(kept):,} with the highest speaks_as rates, ties to the "
        f"more verdicts, then the list's order, of {len(best):,} ranked"
        + (f" ({unrated:,} more have no speaks_as verdict)" if unrated else "")
        + ":"
    )
    return kept, shown + figures_table(kept)


def _from_library(rows: Sequence[CandidateStats]) -> bool:
    return any(each.candidate.source == LIBRARY.name for each in rows)


def stage_b(scored: Scored, selection: Selection, panel: Panel) -> Stage:
    """Stage b from screen_a's scores: the selection's stage_b_size best
    (ranked), each at its reference dose, on every prompt of the set, above
    the ceiling where screen_a allowed it. NoneKept when none is ranked."""
    kept, shown = _best(scored, selection.stage_b_size)
    over = over_ceiling(scored.spec)
    prompts = set_prompts(selection)
    each = {"doses": [1.0], "allow_over_ceiling": over}
    about = (
        f"Stage b of the persona panel (decision 0023), written by pushlab panel "
        f"stage b from the scores of {scored.spec.name} (version "
        f"{scored.experiment.version}, judged by {scored.judge}): its "
        f"{len(kept):,} candidates with the highest speaks_as rates, ties to the "
        "more verdicts, then the list's order, best first, each pushed in its "
        f"direction at its reference dose on every prompt of {selection.prompts}. "
        "Each key's comment gives its speaks_as verdicts there. It names "
        "candidates by key, so it stays out of git (decision 0020)."
    )
    return Stage(
        scored.spec.model,
        None,
        {"file": scored.rule.file, "keys": [row.key for row in kept]},
        ({"label": LABEL, "each_candidate": each},),
        len(kept) * len(prompts),
        about,
        tuple(_figures_note(row) for row in kept),
        shown=tuple(shown),
        from_library=_from_library(kept),
    )


def stage_ladder(scored: Scored, selection: Selection, panel: Panel) -> Stage:
    """Stage ladder from stage b's scores: its ladder_size best (ranked),
    each at ladder_doses, multiples of its reference dose, above the dose
    ceiling where one goes, on the set's first ladder_prompts prompts.
    NoneKept when none is ranked."""
    kept, shown = _best(scored, selection.ladder_size)
    prompts = chosen_prompts(
        set_prompts(selection), selection.ladder_prompts, "ladder_prompts"
    )
    doses = list(selection.ladder_doses)
    about = (
        f"Stage ladder of the persona panel (decision 0023), written by pushlab "
        f"panel stage ladder from the scores of {scored.spec.name} (version "
        f"{scored.experiment.version}, judged by {scored.judge}): its "
        f"{len(kept):,} best candidates, ranked as stage b ranks them, each pushed "
        f"in its direction at {doses_text(selection.ladder_doses)} times its reference "
        f"dose on the first {len(prompts)} prompts of {selection.prompts}, above "
        "the dose ceiling where a dose goes. It names candidates by key, so it "
        "stays out of git (decision 0020)."
    )
    return Stage(
        scored.spec.model,
        tuple(prompts),
        {"file": scored.rule.file, "keys": [row.key for row in kept]},
        (
            {
                "label": LABEL,
                "each_candidate": {"doses": doses, "allow_over_ceiling": True},
            },
        ),
        len(kept) * len(doses) * len(prompts),
        about,
        tuple(_figures_note(row) for row in kept),
        shown=tuple(shown),
        from_library=_from_library(kept),
    )


def dose_multiple(dose: float, strength: float, known: Sequence[float] = ()) -> float:
    """The multiple of a reference dose, strength, that each_candidate pushes
    at dose: the first of known that gives dose exactly, as each_candidate
    works an alpha out, else dose / strength, or a float a step or two from
    it that gives dose exactly.

    Example:
        >>> dose_multiple(1.1 * 0.75, 1.1, [0.5, 0.75])
        0.75
        >>> dose_multiple(0.9375, 1.25)
        0.75
    """
    for multiple in known:
        if strength * multiple == dose:  # a direction of +1 or -1 changes no bit
            return float(multiple)
    guess = below = above = dose / strength
    for _ in range(4):
        for multiple in (below, above):
            if strength * multiple == dose:
                return multiple
        below, above = math.nextafter(below, 0.0), math.nextafter(above, math.inf)
    return guess


def works(doses: UnitDoses) -> bool:
    """Whether a working dose was found below any collapse dose."""
    working, collapse = doses.working.dose, doses.collapse.dose
    return working is not None and (collapse is None or working < collapse)


def confirm_table(found: Sequence[tuple[CandidateStats, UnitDoses]]) -> list[str]:
    """Each ladder candidate's working and collapse doses, with their
    intervals, the window between them, and whether it goes on."""
    head = ["candidate", "persona", "working", "collapse", "window", "kept"]
    table = [head]
    for each, doses in found:
        table.append(
            [each.key, plain(each.candidate.persona)]
            + [short_text(doses.working), short_text(doses.collapse)]
            + [window_text(doses), "yes" if works(doses) else "no"]
        )
    return aligned(table)


def stage_confirm(scored: Scored, selection: Selection, panel: Panel) -> Stage:
    """Stage confirm from the ladder's scores: each candidate whose working
    dose, from its speaks_as and coherence verdicts at each dose, is below
    any collapse dose, at its working dose, on every prompt of the set.
    NoneKept when none is."""
    found = [
        (each, analyze_unit(each.points(), panel))
        for each in scored.stats
        if each.doses
    ]
    kept = [(each, doses) for each, doses in found if works(doses)]
    percent = f"{round(panel.confidence * 100, 6):g}%"
    shown = [
        *read_lines(scored, (SPEAKS_AS, COHERENCE)),
        "",
        f"Working at a speaks_as rate of {panel.working_rate:g} or more; collapsed "
        f"at a mean coherence below {panel.coherence_below:g} ({panel.path}); "
        f"doses as pushed, with {percent} intervals:",
        *confirm_table(found),
    ]
    if not kept:
        raise NoneKept(
            f"no candidate of {scored.spec.name} works below its collapse dose, so "
            "there is nothing to confirm: a ladder of other candidates or doses, "
            "or more of them, may find some",
            shown,
        )
    rule_doses = [d for key in scored.rule.keys or () for d in key.doses or ()]
    known = [*selection.ladder_doses, *rule_doses]
    keys, notes = [], []
    for each, doses in kept:
        working = doses.working.dose
        assert working is not None  # works() says so
        multiple = dose_multiple(working, each.candidate.strength, known)
        keys.append({"key": each.key, "doses": [multiple]})
        collapse = doses.collapse.dose
        notes.append(
            f"working dose {exact(working)}, collapse dose "
            f"{'beyond the ladder' if collapse is None else exact(collapse)}, "
            f"{each.candidate.persona}"
        )
    prompts = set_prompts(selection)
    over = over_ceiling(scored.spec)
    about = (
        f"Stage confirm of the persona panel (decision 0023), written by pushlab "
        f"panel stage confirm from the scores of the ladder {scored.spec.name} "
        f"(version {scored.experiment.version}, judged by {scored.judge}): each of "
        f"its {len(kept):,} candidates whose working dose is below any collapse "
        f"dose, of {len(found):,}, pushed in its direction at its working dose, "
        f"given as a multiple of its reference dose, on every prompt of "
        f"{selection.prompts}. Each key's comment gives its doses as pushed. It "
        "names candidates by key, so it stays out of git (decision 0020)."
    )
    shown.append(f"Kept {len(kept):,} of {len(found):,}, each at its working dose.")
    return Stage(
        scored.spec.model,
        None,
        {"file": scored.rule.file, "keys": keys},
        ({"label": LABEL, "each_candidate": {"allow_over_ceiling": over}},),
        len(kept) * len(prompts),
        about,
        tuple(notes),
        shown=tuple(shown),
        from_library=_from_library([each for each, _ in kept]),
    )


# Each stage that reads scores, and the rubrics it needs.
BUILDERS = {"b": stage_b, "ladder": stage_ladder, "confirm": stage_confirm}
NEEDS = {
    "b": (SPEAKS_AS,),
    "ladder": (SPEAKS_AS,),
    "confirm": (SPEAKS_AS, COHERENCE),
}
