"""pushlab panel select's rules and its choice (pushlab.panel.select,
decision 0023), on synthetic scores.

Made-up qwen3-8b candidates (none of them the Neuron Persona Library's)
each get replies on a ladder and in stage confirm, with invented verdicts,
built straight into the stats that pushlab.panel.stats reads from a runner
folder; configs/panel.yaml gives the thresholds: a working dose at a
speaks_as rate of 0.8, a collapse below a mean coherence of 6.
"""

import doctest
from collections import Counter
from collections.abc import Sequence
from fractions import Fraction
from types import SimpleNamespace

import pytest

from pushlab.analysis.panel import read_panel
from pushlab.discovery.candidates import Candidate
from pushlab.panel import select
from pushlab.panel.select import (
    Qualified,
    assess,
    choose,
    failed_rule,
    most_failed,
    rule_advice,
    same_names,
    thin_advice,
)
from pushlab.panel.settings import read_selection
from pushlab.panel.stats import CandidateStats, Figures, Reply, Scored
from pushlab.units.neuron import NeuronUnit
from pushlab.units.neuron_set import NeuronSet

PANEL = read_panel()
LADDER_PROMPTS = ("p1", "p2", "p3", "p4")
CONFIRM_PROMPTS = ("q1", "q2", "q3", "q4", "q5")


def _candidate(index: int, persona: str, kind: str = "neuron") -> Candidate:
    if kind == "set":
        unit = NeuronSet("qwen3-8b", f"set-{index}", [(17, index, 0.5)], "made up")
        return Candidate("mayner", persona, unit, 1, 1.0, False, "f", "made up")
    unit = NeuronUnit("qwen3-8b", 9, index)
    return Candidate("library", persona, unit, 1, 1.0, False, "f", f"L9 N{index}")


def _replies(
    prefix: str,
    prompts: Sequence[str],
    speaks: Sequence[bool | None],
    coherence: Sequence[float | None],
) -> Figures:
    return Figures(
        tuple(
            Reply(f"{prefix}-{prompt}", prompt, f"Reply to {prompt}.", said, score)
            for prompt, said, score in zip(prompts, speaks, coherence, strict=True)
        )
    )


def _ladder(
    index: int, persona: str, doses: dict, kind: str = "neuron"
) -> CandidateStats:
    """A candidate's ladder: each dose's speaks_as verdicts and coherence on
    the four ladder prompts, as (speaks, coherence) per dose."""
    figures = {
        dose: _replies(f"l{index}-{dose}", LADDER_PROMPTS, [said] * 4, [score] * 4)
        for dose, (said, score) in doses.items()
    }
    return CandidateStats(_candidate(index, persona, kind), index, figures)


def _confirm(
    ladder: CandidateStats,
    dose: float,
    speaks: Sequence[bool | None],
    coherence: Sequence[float | None],
) -> CandidateStats:
    prompts = CONFIRM_PROMPTS[: len(speaks)]
    figures = _replies(f"c{ladder.place}", prompts, speaks, coherence)
    return CandidateStats(ladder.candidate, ladder.place, {dose: figures})


def _scored(
    name: str, stats: Sequence[CandidateStats], prompts: Sequence[str]
) -> Scored:
    """A scored experiment as read_scored gives one, holding only what
    assess reads of it."""
    spec = SimpleNamespace(name=name, seeds=(0,))
    experiment = SimpleNamespace(spec=spec)
    return Scored(None, experiment, None, "fake:fake", tuple(prompts), tuple(stats), 0)  # type: ignore[arg-type]


# Works at 1.0 on the ladder and never collapses.
WORKS = {0.5: (False, 9.0), 1.0: (True, 9.0), 1.5: (True, 8.0)}


def test_selection_keeps_only_candidates_that_pass_every_rule() -> None:
    good = _ladder(1, "Brass Owl", WORKS)
    collapses = _ladder(2, "Salt Captain", {0.5: (False, 9), 1.0: (True, 3)})
    never = _ladder(3, "Tin Fox", {0.5: (False, 9), 1.0: (False, 9)})
    elsewhere = _ladder(4, "Moss Oracle", WORKS)
    shy = _ladder(5, "Velvet Teapot", WORKS, kind="set")
    edge = _ladder(6, "Paper Crane", WORKS)
    muddled = _ladder(7, "Iron Bell", WORKS)
    twin = _ladder(8, " brass  OWL", WORKS)
    ladder = [good, collapses, never, elsewhere, shy, edge, muddled, twin]
    confirm = [
        _confirm(good, 1.0, [True] * 5, [8.0] * 5),
        _confirm(collapses, 1.0, [True] * 5, [8.0] * 5),
        _confirm(elsewhere, 1.5, [True] * 5, [8.0] * 5),  # not at its working dose
        # 3 of 5: the two prompts without a verdict count against it.
        _confirm(shy, 1.0, [True, True, True, None, None], [8.0] * 5),
        # 4 of the 5 prompts, one with no reply at all: 0.8, enough; and a
        # mean coherence of 6, enough.
        _confirm(edge, 1.0, [True] * 4, [6.0] * 4),
        _confirm(muddled, 1.0, [True] * 5, [5.0, 6.0, 5.0, 6.0, 5.0]),
        _confirm(twin, 1.0, [True] * 5, [7.0] * 5),  # the same name, weaker
    ]
    found = same_names(
        assess(_scored("c", confirm, CONFIRM_PROMPTS), _scored("l", ladder, ()), PANEL)
    )
    reasons = {each.key: each.reason for each in found}
    assert list(reasons.values()) == [
        None,
        "dose",
        "dose",
        "confirmed",
        "reliability",
        None,
        "coherence",
        "name",
    ]
    owl, *_, crane, bell, twin_found = found
    assert (owl.working, owl.collapse, owl.k, owl.n) == (1.0, None, 5, 5)
    assert (found[4].k, found[4].n, found[4].reliability) == (3, 5, Fraction(3, 5))
    assert (crane.k, crane.n, crane.coherence) == (4, 5, 6.0)
    assert bell.coherence == pytest.approx(5.4)
    assert twin_found.named_like == owl.key
    assert [select.reason_text(each, PANEL) for each in found] == [
        "",
        "no working dose below a collapse dose on the ladder",
        "no working dose below a collapse dose on the ladder",
        "no reply in stage confirm at its working dose",
        "speaks as its persona on under 0.8 of the prompts",
        "",
        "a mean coherence under 6",
        f"{owl.key} is a stronger finalist of its name",
    ]
    share, mean = crane.share(0.95), crane.mean()
    assert (share.k, share.n, share.rate, share.high) == (4, 5, 0.8, 0.9638)
    assert (mean.n, mean.mean, mean.low, mean.high) == (4, 6.0, 6.0, 6.0)


def test_a_confirmed_candidate_must_be_on_the_ladder() -> None:
    owl = _ladder(1, "Brass Owl", WORKS)
    fox = _ladder(2, "Tin Fox", WORKS)
    confirm = _scored("c", [_confirm(fox, 1.0, [True] * 5, [8] * 5)], CONFIRM_PROMPTS)
    with pytest.raises(ValueError, match="which the ladder l has no reply of"):
        assess(confirm, _scored("l", [owl], ()), PANEL)


def test_a_confirm_and_a_ladder_the_wrong_way_round_are_refused() -> None:
    owl = _ladder(1, "Brass Owl", WORKS)
    confirm = _scored("c", [_confirm(owl, 1.0, [True] * 5, [8] * 5)], CONFIRM_PROMPTS)
    ladder = _scored("l", [owl], LADDER_PROMPTS)
    assess(confirm, ladder, PANEL)  # the right way round
    with pytest.raises(ValueError, match="one experiment"):
        assess(confirm, _scored("c", [owl], LADDER_PROMPTS), PANEL)
    with pytest.raises(ValueError, match="at 3 doses, as a ladder does"):
        assess(_scored("l2", [owl], LADDER_PROMPTS), ladder, PANEL)
    single = _scored("c2", [_confirm(owl, 1.0, [True] * 5, [8] * 5)], CONFIRM_PROMPTS)
    with pytest.raises(ValueError, match="at one dose, not at several"):
        assess(confirm, single, PANEL)


def test_the_rules_hold_at_their_thresholds() -> None:
    owl = _ladder(1, "Brass Owl", WORKS)
    found = assess(
        _scored("c", [_confirm(owl, 1.0, [True] * 4 + [False], [6.0] * 5)], "abcde"),
        _scored("l", [owl], ()),
        PANEL,
    )[0]
    assert failed_rule(found, PANEL) is None
    found = assess(
        _scored("c", [_confirm(owl, 1.0, [True] * 4 + [False], [5.9] * 5)], "abcde"),
        _scored("l", [owl], ()),
        PANEL,
    )[0]
    assert failed_rule(found, PANEL) == "coherence"
    # A prompt of the set with no reply counts against it: 4 of 6.
    found = assess(
        _scored("c", [_confirm(owl, 1.0, [True] * 4 + [False], [9.0] * 5)], "abcdef"),
        _scored("l", [owl], ()),
        PANEL,
    )[0]
    assert (found.k, found.n, failed_rule(found, PANEL)) == (4, 6, "reliability")
    # No coherence score at all: it cannot show it meets the threshold.
    found = assess(
        _scored("c", [_confirm(owl, 1.0, [True] * 5, [None] * 5)], "abcde"),
        _scored("l", [owl], ()),
        PANEL,
    )[0]
    assert (found.reason, found.coherence) == ("coherence", None)
    assert select.reason_text(found, PANEL) == "no coherence score in stage confirm"
    with pytest.raises(ValueError, match="has no coherence score in stage confirm"):
        found.mean()


def _qualified(
    key: str,
    category: str,
    kind: str = "neuron",
    reliability: Fraction = Fraction(1),
    coherence: float = 8.0,
    distinct: Fraction = Fraction(1),
) -> Qualified:
    return Qualified(key, kind, category, reliability, coherence, distinct)


CATEGORIES = ("people", "animals", "fictional", "objects", "mystical")


def test_the_panel_spreads_across_the_categories_when_enough_exist() -> None:
    # Six of each category, the people strongest: still at most five each.
    qualified = [
        _qualified(f"{category}-{n}", category, coherence=9.0 - place)
        for place, category in enumerate(CATEGORIES)
        for n in range(6)
    ]
    chosen = choose(qualified, CATEGORIES, 24)
    held = Counter(each.category for each in chosen)
    assert [held[category] for category in CATEGORIES] == [5, 5, 5, 5, 4]
    # Round by round, in the selection's order of categories.
    assert [each.category for each in chosen[:5]] == list(CATEGORIES)
    # A thin category leaves its places to the others.
    thin = [each for each in qualified if each.category != "objects"]
    thin += [_qualified("objects-0", "objects")]
    held = Counter(each.category for each in choose(thin, CATEGORIES, 24))
    assert [held[category] for category in CATEGORIES] == [6, 6, 6, 1, 5]
    assert thin_advice(choose(thin, CATEGORIES, 24), read_selection()) == (
        "Thin categories, with fewer than 4 personas each: objects 1: more "
        "candidates of them would help."
    )


def test_within_a_category_the_strongest_of_the_kind_the_panel_has_fewer_of() -> None:
    weaker = Fraction(9, 10)
    qualified = [
        _qualified("people-b", "people", coherence=7.0),
        _qualified("people-a", "people", coherence=7.0),  # ties go by key
        _qualified("people-set", "people", "set", reliability=weaker),
        _qualified("animals-1", "animals", coherence=9.0),
        _qualified("animals-set", "animals", "set", reliability=weaker),
        _qualified("animals-2", "animals", coherence=8.0, distinct=Fraction(7, 8)),
        _qualified("animals-3", "animals", coherence=8.0),
    ]
    chosen = [each.key for each in choose(qualified, ("people", "animals"), 6)]
    # people-a first, tied with people-b but first by key; then, the panel
    # holding a neuron and no set, the animals' set before their stronger
    # neurons; with as many of each, the strongest, people-b, then
    # animals-1; then, sets being fewer, the people's set, and as the
    # animals have none left, their strongest.
    assert chosen == [
        "people-a",
        "animals-set",
        "people-b",
        "animals-1",
        "people-set",
        "animals-3",
    ]
    assert len(choose(qualified, ("people", "animals"), 3)) == 3
    assert len(choose(qualified, ("people", "animals"), 30)) == 7  # every one
    with pytest.raises(ValueError, match="'robots', is not one"):
        choose([_qualified("r", "robots")], CATEGORIES, 4)


def test_advice_names_the_rule_most_failed_and_what_helps() -> None:
    assert most_failed(Counter()) is None and rule_advice(Counter()) == ""
    assert most_failed(Counter({"distinct": 2, "reliability": 2})) == ("reliability", 2)
    assert rule_advice(Counter({"distinct": 3, "dose": 1})) == (
        "The rule most of the 4 left out failed is distinct (3 of them): larger "
        "sets, or more candidates, would help."
    )
    for rule, helps in select.HELPS.items():
        assert helps in rule_advice(Counter({rule: 1}))
    spread = [_qualified(f"k{n}", c) for n, c in enumerate(CATEGORIES * 4)]
    assert thin_advice(spread, read_selection()) == ""


def test_the_examples_in_the_docstrings_hold() -> None:
    result = doctest.testmod(select)
    assert result.failed == 0 and result.attempted > 0
