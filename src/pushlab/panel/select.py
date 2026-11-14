"""pushlab panel select's rules, and how it chooses the panel (decision 0023).

assess reads what select needs of each candidate of stage ladder: its
working and collapse doses, from the ladder's speaks_as and coherence
verdicts and configs/panel.yaml's thresholds (pushlab.analysis.ladder), and
its replies in stage confirm at its working dose. Its reliability is their
speaks_as share over the confirm's prompts times its seeds, so that a prompt
without a verdict counts against it; its coherence is their mean score, with
a bootstrap interval over prompts. A candidate is a finalist unless it fails
a rule, the first of RULES it fails being its reason:
- dose: no working dose below any collapse dose on the ladder;
- confirmed: no reply in stage confirm at its working dose;
- reliability: a share below the panel's working_rate;
- coherence: no score, or a mean below the panel's coherence_below.
same_names then leaves out a finalist whose persona has a stronger one's
name, case and spacing aside (name), for which_persona tells personas apart
by name; and the which_persona check (pushlab.panel.distinct) leaves out
those it does not tell apart (distinct).

choose takes up to the panel's size of those left, each with its category,
round by round across the categories in the selection's order: each round
takes each category's strongest left (reliability, then mean coherence,
then distinctness, ties by key), within a category one of the kind, neuron
or set, the panel holds fewer of. rule_advice and thin_advice say what would
help a panel short of its size, and panel_persona makes a chosen finalist
the panel file's entry. Nothing here reads or writes a file or calls a
judge.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from fractions import Fraction

from pushlab.analysis.intervals import wilson_interval
from pushlab.analysis.ladder import DoseStats, LadderPoint, UnitDoses, analyze_unit
from pushlab.analysis.panel import Panel
from pushlab.discovery.candidates import Candidate, same_name
from pushlab.panel.panel_file import Mean, PanelPersona, Share
from pushlab.panel.settings import Selection
from pushlab.panel.stages import works
from pushlab.panel.stats import CandidateStats, Figures, Scored

RULES = ("dose", "confirmed", "reliability", "coherence", "name", "distinct")
# What would help a panel short of its size, by the rule most of those left
# out failed.
HELPS = {
    "dose": "a ladder of other doses, or more candidates",
    "confirmed": "stage confirm written again from this ladder",
    "reliability": "a higher dose, or more candidates",
    "coherence": "a lower dose, or more candidates",
    "name": "candidates of other personas",
    "distinct": "larger sets, or more candidates",
}
_ROUNDING = 1e-9  # how far a share or a mean may miss a threshold by rounding


@dataclass(frozen=True)
class Assessed:
    """A candidate of the ladder as select assesses it: its stats on the
    ladder and the doses they give; its replies in stage confirm at its
    working dose, and their stats (None without one); expected, how many
    replies the confirm asks of a candidate, its prompts times its seeds;
    the first rule of RULES it fails, None for a finalist; and for the name
    rule, the key of the stronger finalist of its name."""

    ladder: CandidateStats
    doses: UnitDoses
    replies: Figures
    confirm: DoseStats | None
    expected: int
    reason: str | None = None
    named_like: str = ""

    @property
    def candidate(self) -> Candidate:
        """The candidate, as its list gives it."""
        return self.ladder.candidate

    @property
    def key(self) -> str:
        """Its key in the candidates list."""
        return self.ladder.key

    @property
    def persona(self) -> str:
        """Its persona's name, as the judge is asked about it."""
        return self.candidate.persona.strip()

    @property
    def working(self) -> float | None:
        """Its working dose on the ladder, None when it has none."""
        return self.doses.working.dose

    @property
    def collapse(self) -> float | None:
        """Its collapse dose on the ladder, None when it never collapsed."""
        return self.doses.collapse.dose

    @property
    def k(self) -> int:
        """Its confirm replies that speak as its persona."""
        return self.replies.k

    @property
    def n(self) -> int:
        """Reliability's count: the replies the confirm asks of it."""
        return max(self.expected, self.replies.runs)

    @property
    def reliability(self) -> Fraction:
        """k of n, exactly."""
        return Fraction(self.k, self.n) if self.n else Fraction(0)

    @property
    def coherence(self) -> float | None:
        """Its confirm replies' mean coherence, None without a score."""
        return None if self.confirm is None else self.confirm.coherence_mean

    def share(self, confidence: float) -> Share:
        """Its reliability with Wilson's interval at confidence."""
        low, high = wilson_interval(self.k, self.n, confidence)
        return Share(self.k, self.n, self.k / self.n, round(low, 4), round(high, 4))

    def mean(self) -> Mean:
        """Its mean coherence and the bootstrap interval of it; ValueError
        without a score."""
        stats = self.confirm
        if stats is None or stats.coherence_mean is None:
            raise ValueError(f"{self.key} has no coherence score in stage confirm")
        low, high = stats.coherence_interval or (stats.coherence_mean,) * 2
        mean = round(stats.coherence_mean, 4)
        return Mean(stats.coherence_n, mean, round(low, 4), round(high, 4))


def failed_rule(assessed: Assessed, panel: Panel) -> str | None:
    """The first rule of dose, confirmed, reliability and coherence that
    assessed fails, None when it passes them all."""
    if not works(assessed.doses):
        return "dose"
    if not assessed.replies.runs:
        return "confirmed"
    if assessed.k < panel.working_rate * assessed.n - _ROUNDING:
        return "reliability"
    coherence = assessed.coherence
    if coherence is None or coherence < panel.coherence_below - _ROUNDING:
        return "coherence"
    return None


def _confirm_stats(replies: Figures, dose: float, panel: Panel) -> DoseStats | None:
    """The stats of replies at dose, with their intervals; None for none."""
    if not replies.runs:
        return None
    points = [
        LadderPoint(dose, r.prompt_id, r.speaks, r.coherence) for r in replies.replies
    ]
    return analyze_unit(points, panel).stats[dose]


def assess(confirm: Scored, ladder: Scored, panel: Panel) -> list[Assessed]:
    """Each candidate of the ladder with a saved reply, in the ladder's
    order, as the module's docstring says. ValueError when the confirm
    pushes a candidate the ladder has no reply of, whose doses it lacks.

    Example:
        >>> assess(confirm, ladder, read_panel())  # doctest: +SKIP
        [Assessed(ladder=CandidateStats(...), ..., reason=None, named_like='')]
    """
    check_roles(confirm, ladder)
    on_ladder = {each.key for each in ladder.stats if each.doses}
    if stray := [each.key for each in confirm.stats if each.key not in on_ladder]:
        raise ValueError(
            f"{confirm.spec.name} pushes {stray[0]}, which the ladder "
            f"{ladder.spec.name} has no reply of, so its collapse dose is unknown: "
            "give the ladder that stage confirm was written from"
        )
    expected = len(confirm.prompts) * len(confirm.spec.seeds)
    confirmed = confirm.by_key
    found = []
    for each in ladder.stats:
        if not each.doses:
            continue
        doses = analyze_unit(each.points(), panel)
        working, own = doses.working.dose, confirmed.get(each.key)
        replies = Figures()
        if own is not None and working is not None:
            replies = own.doses.get(working, Figures())
        stats = None if working is None else _confirm_stats(replies, working, panel)
        assessed = Assessed(each, doses, replies, stats, expected)
        found.append(replace(assessed, reason=failed_rule(assessed, panel)))
    return found


def check_roles(confirm: Scored, ladder: Scored) -> None:
    """Refuse a confirm and a ladder given the wrong way round: one
    experiment for both, a confirm that pushes a candidate at several doses
    as a ladder does, or a ladder that pushes each at one dose, as none does
    (ladder_doses lists two or more); the wrong panel would follow.

    Example:
        >>> check_roles(confirm, ladder)  # doctest: +SKIP
    """
    swap = "give stage confirm's file as CONFIRM and stage ladder's as --ladder"
    if confirm.spec.name == ladder.spec.name:
        raise ValueError(f"CONFIRM and --ladder are one experiment: {swap}")
    if several := [each for each in confirm.stats if len(each.doses) > 1]:
        raise ValueError(
            f"{confirm.spec.name} pushes {several[0].key} at {len(several[0].doses)} "
            f"doses, as a ladder does: {swap}"
        )
    pushed = [each for each in ladder.stats if each.doses]
    if pushed and all(len(each.doses) == 1 for each in pushed):
        raise ValueError(
            f"{ladder.spec.name} pushes each candidate at one dose, not at several "
            f"as a ladder does: {swap}"
        )


def _rank(assessed: Assessed) -> tuple[Fraction, float, str]:
    coherence = assessed.coherence
    return -assessed.reliability, -(coherence or 0.0), assessed.key


def same_names(assessed: Sequence[Assessed]) -> list[Assessed]:
    """assessed, each finalist whose persona has the name of a stronger
    finalist's (same_name: case and spacing aside) failing the name rule:
    stronger by reliability, then mean coherence, ties by key.

    Example:
        >>> same_names([])
        []
    """
    stronger: dict[str, str] = {}
    named_like: dict[str, str] = {}
    for each in sorted((a for a in assessed if a.reason is None), key=_rank):
        name = same_name(each.persona)
        if name in stronger:
            named_like[each.key] = stronger[name]
        else:
            stronger[name] = each.key
    return [
        replace(each, reason="name", named_like=named_like[each.key])
        if each.key in named_like
        else each
        for each in assessed
    ]


def reason_text(assessed: Assessed, panel: Panel) -> str:
    """Why assessed is left out, in words; "" for a finalist."""
    texts = {
        "dose": "no working dose below a collapse dose on the ladder",
        "confirmed": "no reply in stage confirm at its working dose",
        "reliability": f"speaks as its persona on under {panel.working_rate:g} of "
        "the prompts",
        "coherence": f"a mean coherence under {panel.coherence_below:g}",
        "name": f"{assessed.named_like} is a stronger finalist of its name",
    }
    if assessed.reason == "coherence" and assessed.coherence is None:
        return "no coherence score in stage confirm"
    return texts.get(assessed.reason or "", "")


@dataclass(frozen=True)
class Qualified:
    """A persona that passes every rule, ready to choose: its key, kind and
    category, and its reliability, mean coherence and distinctness.

    Example:
        >>> owl = Qualified("library/L9N1-", "neuron", "animals", 1, 8.0, 1)
        >>> owl.strength
        (-1, -8.0, -1, 'library/L9N1-')
    """

    key: str
    kind: str
    category: str
    reliability: Fraction
    coherence: float
    distinct: Fraction

    @property
    def strength(self) -> tuple[Fraction, float, Fraction, str]:
        """Its rank, strongest first: reliability, then mean coherence,
        then distinctness, ties by key."""
        return (-self.reliability, -self.coherence, -self.distinct, self.key)


def _fewer(chosen: Sequence[Qualified]) -> str | None:
    """The kind the panel holds fewer of; None when it holds as many."""
    kinds = Counter(each.kind for each in chosen)
    if kinds["neuron"] == kinds["set"]:
        return None
    return "neuron" if kinds["neuron"] < kinds["set"] else "set"


def choose(
    qualified: Sequence[Qualified], categories: Sequence[str], size: int
) -> list[Qualified]:
    """Up to size of qualified, in the order chosen, as the module's
    docstring says. ValueError for a category not among categories.

    Example:
        >>> owl = Qualified("library/L9N1-", "neuron", "animals", 1, 8.0, 1)
        >>> tea = Qualified("mayner/tea", "set", "objects", Fraction(7, 8), 9.0, 1)
        >>> [q.key for q in choose([owl, tea], ["people", "objects", "animals"], 5)]
        ['mayner/tea', 'library/L9N1-']
    """
    pools: dict[str, list[Qualified]] = {category: [] for category in categories}
    for each in sorted(qualified, key=lambda q: q.strength):
        if each.category not in pools:
            raise ValueError(f"{each.key}'s category, {each.category!r}, is not one")
        pools[each.category].append(each)
    chosen: list[Qualified] = []
    while len(chosen) < size and any(pools.values()):
        for category in categories:
            pool = pools[category]
            if not pool or len(chosen) == size:
                continue
            fewer = _fewer(chosen)
            pick = next((each for each in pool if each.kind == fewer), pool[0])
            pool.remove(pick)
            chosen.append(pick)
    return chosen


def most_failed(reasons: Counter[str]) -> tuple[str, int] | None:
    """The rule most failed, and by how many; ties to the earlier of RULES.

    Example:
        >>> most_failed(Counter({"distinct": 3, "reliability": 3, "dose": 1}))
        ('reliability', 3)
    """
    counted = [(count, -RULES.index(rule), rule) for rule, count in reasons.items()]
    if not counted:
        return None
    count, _, rule = max(counted)
    return rule, count


def rule_advice(reasons: Counter[str]) -> str:
    """The rule most of those left out failed, by reasons, the rule each
    failed first, and what would help with it; "" when none was.

    Example:
        >>> print(rule_advice(Counter({"reliability": 5, "distinct": 2})))
        ... # doctest: +NORMALIZE_WHITESPACE
        The rule most of the 7 left out failed is reliability (5 of them): a
        higher dose, or more candidates, would help.
    """
    if (found := most_failed(reasons)) is None:
        return ""
    rule, count = found
    return (
        f"The rule most of the {sum(reasons.values()):,} left out failed is {rule} "
        f"({count:,} of them): {HELPS[rule]}, would help."
    )


def thin_advice(chosen: Sequence[Qualified], selection: Selection) -> str:
    """The panel's thin categories, those with fewer personas than its size
    over its categories, and that more candidates of them would help; ""
    when none is thin.

    Example:
        >>> from pushlab.panel.settings import read_selection
        >>> owl = Qualified("library/L9N1-", "neuron", "animals", 1, 8.0, 1)
        >>> thin_advice([owl] * 4, read_selection())[:52]
        'Thin categories, with fewer than 4 personas each: pe'
    """
    share = selection.panel_size // len(selection.categories)
    held = Counter(each.category for each in chosen)
    thin = [category for category in selection.categories if held[category] < share]
    if not thin:
        return ""
    listed = ", ".join(f"{category} {held[category]}" for category in thin)
    return (
        f"Thin categories, with fewer than {share} personas each: {listed}: more "
        "candidates of them would help."
    )


def panel_persona(
    assessed: Assessed, category: str, distinct: Share, confidence: float
) -> PanelPersona:
    """A finalist as the panel file holds it, with its category and its
    distinctness; its reliability's and coherence's intervals at confidence.
    ValueError for one with no working dose or coherence score."""
    candidate, working = assessed.candidate, assessed.working
    if working is None:
        raise ValueError(f"{assessed.key} has no working dose")
    return PanelPersona(
        key=assessed.key,
        persona=assessed.persona,
        category=category,
        kind=candidate.kind,
        source=candidate.source,
        unit=str(candidate.unit),
        direction=candidate.direction,
        reference_dose=candidate.strength,
        working_dose=working,
        collapse_dose=assessed.collapse,
        reliability=assessed.share(confidence),
        coherence=assessed.mean(),
        distinct=distinct,
    )
