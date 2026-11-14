"""Candidates whose persona is named "none" in pushlab panel's figures,
stages and select (pushlab.panel.stats, decision 0023), from stats made by
hand.

"none" is the judge's answer for no persona, yet a library persona can be
named so. pushlab score never asks whether a reply speaks as such a
candidate (it is unasked), so its replies are not counted as replies to
judge by speaks_as, a note says why they have no verdict, and when only such
candidates have replies the refusal says so, not that pushlab score makes
the verdicts. test_panel_none_screen.py runs and judges a screen of them.
"""

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from pushlab import PROJECT_ROOT
from pushlab.analysis.panel import read_panel
from pushlab.discovery.candidates import Candidate
from pushlab.panel import select_text, stages
from pushlab.panel.settings import read_selection
from pushlab.panel.stages import NoneKept, read_lines
from pushlab.panel.stats import (
    CandidateStats,
    Figures,
    Reply,
    Scored,
    unasked_refusal,
)
from pushlab.runs.spec import read_spec
from pushlab.runs.spec_candidates import CandidatesRule
from pushlab.scoring.items import Experiment
from pushlab.units.neuron import NeuronUnit

# The note on the replies of one unasked candidate, its two replies.
NOTE = (
    'Note: 2 replies of 1 candidate with the persona "none", the judge\'s answer '
    "for no persona, can have no speaks_as verdict, as the judge cannot be asked "
    "about it: left out."
)
# Each reply's speaks_as verdict and coherence score, None for none.
Verdicts = Sequence[tuple[bool | None, float | None]]


def _stats(index: int, persona: str, verdicts: Verdicts = ()) -> CandidateStats:
    """A made-up qwen3-8b candidate, library/L9N<index>+, with a reply at
    dose 1 for each of verdicts; no saved reply when verdicts is empty."""
    unit = NeuronUnit("qwen3-8b", 9, index)
    candidate = Candidate("library", persona, unit, 1, 1.0, False, "f", "r")
    replies = tuple(
        Reply(f"r{index}-{i}", f"p{i}", "", speaks, coherence)
        for i, (speaks, coherence) in enumerate(verdicts)
    )
    return CandidateStats(candidate, index, {1.0: Figures(replies)} if replies else {})


def _scored(*stats: CandidateStats) -> Scored:
    """stats as an experiment, s.yaml, of the list units/c.csv, judged by
    api:x."""
    spec = read_spec(PROJECT_ROOT / "experiments" / "p0" / "s2_pulse_lengths.yaml")
    identity = {"version": 1, "spec_sha256": spec.sha256}
    experiment = Experiment(spec, Path("runs"), identity, [], 0)
    rule = CandidatesRule("units/c.csv")
    return Scored(Path("s.yaml"), experiment, rule, "api:x", ("p0", "p1"), stats, 0)


@pytest.mark.parametrize(
    ("persona", "unasked"),
    [
        ("None", True),
        (" none ", True),
        ("NONE", True),
        ("Nonesuch", False),
        ("No One", False),
    ],
)
def test_a_persona_named_none_in_any_case_or_spacing_is_unasked(
    persona: str, unasked: bool
) -> None:
    assert _stats(1, persona).unasked is unasked


def test_unasked_replies_are_not_counted_as_unjudged_by_speaks_as() -> None:
    nobody = _stats(1, "None", [(None, 7.0), (None, None)])
    owl = _stats(2, "Brass Owl", [(True, None), (None, None)])
    scored = _scored(nobody, owl, _stats(3, "none"))  # the last has no reply
    assert scored.unasked == (nobody,)
    # speaks_as: the owl's reply alone; coherence judges every reply, the
    # unasked candidate's too.
    assert (scored.unjudged("speaks_as"), scored.unjudged("coherence")) == (1, 3)


def test_unasked_refusal_says_why_and_what_to_do() -> None:
    nobody, other = _stats(1, "None", [(None, None)]), _stats(2, " NONE ")
    owl, owl_unrun = _stats(3, "Brass Owl", [(True, None)]), _stats(3, "Brass Owl")
    path = Path("a.yaml")
    assert unasked_refusal(path, "screen-a", [nobody, other]) == (
        "every candidate of screen-a with a saved reply, such as library/L9N1+, has "
        'the persona "none", the judge\'s answer for no persona, which the judge '
        "cannot be asked about, so no reply can have a speaks_as verdict: give an "
        "experiment that pushes candidates of other personas"
    )
    # A candidate of another persona with no saved reply yet can have one.
    assert unasked_refusal(path, "screen-a", [nobody, owl_unrun]).endswith(
        ": `uv run pushlab run a.yaml` runs the other candidates"
    )
    # Nothing to refuse: another candidate has a reply, or none has.
    for stats in ([nobody, owl], [owl], [other, owl_unrun], []):
        assert unasked_refusal(path, "screen-a", stats) == ""


def test_the_note_on_unasked_replies_is_shown_where_speaks_as_is_needed() -> None:
    nobody = _stats(1, "None", [(None, None), (None, None)])
    owl = _stats(2, "Brass Owl", [(True, 8.0), (False, 7.0)])
    scored = _scored(nobody, owl)
    # Each reply that can have a speaks_as verdict has one: no note sends you
    # to pushlab score for speaks_as, and the unasked replies have their own.
    assert read_lines(scored, ("speaks_as", "coherence"))[1:] == [
        "Note: 2 replies have no ok coherence verdict of api:x: `uv run pushlab "
        "score s.yaml --rubric coherence` judges them.",
        NOTE,
    ]
    assert NOTE not in read_lines(scored, ("coherence",))
    # pushlab panel select shows it for its confirm and its ladder alike.
    assert select_text.rules_lines(scored, scored, read_panel(), []).count(NOTE) == 2


@pytest.mark.parametrize("build", [stages.stage_b, stages.stage_ladder])
def test_a_stage_whose_only_replies_are_unasked_says_so(
    build: Callable[..., object],
) -> None:
    nobody = _stats(1, "None", [(None, None), (None, None)])
    scored = _scored(nobody, _stats(2, "Brass Owl"))  # the owl not run yet
    with pytest.raises(NoneKept) as caught:
        build(scored, read_selection(), read_panel())
    problem = str(caught.value)
    assert problem.startswith(
        "every candidate of s2-pulse-lengths with a saved reply, such as "
        'library/L9N1+, has the persona "none"'
    )
    assert problem.endswith(": `uv run pushlab run s.yaml` runs the other candidates")
    assert NOTE in caught.value.shown
    # A reply of another persona not judged yet: pushlab score is what helps.
    scored = _scored(nobody, _stats(2, "Brass Owl", [(None, None)]))
    with pytest.raises(NoneKept, match="has a speaks_as verdict: `uv run pushlab"):
        build(scored, read_selection(), read_panel())
