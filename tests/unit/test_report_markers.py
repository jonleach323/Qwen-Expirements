"""Markers in a reply: where its pushes switch on and off (pushlab.analysis.markers).

Step j is the pass that chose reply token j: the prompt pass for step 0 and
the pass that read token j - 1 for step j, so a push on reply token k shows
at step k + 1 (decision 0008). Each test's sets are a reply's steps, each the
letters of the units pushed on it, made by hand or from a schedule's passes.
"""

import doctest

import pytest

from pushlab.analysis import markers
from pushlab.analysis.markers import (
    active_sets,
    legend,
    letters,
    markdown_text,
    marks,
    places,
    stand_ins,
    with_markers,
)
from pushlab.schedules.model import Schedule
from pushlab.schedules.shapes import alternate, handoff, pulse
from pushlab.units.neuron import NeuronUnit

# The test suite's tiny model is in the registry, so shapes can push it.
A_UNIT, B_UNIT = "tiny:L1:N100", "tiny:L0:N3"
LETTER_OF = {A_UNIT: "A", B_UNIT: "B"}


def _sets(*stretches: tuple[str, int]) -> list[frozenset[str]]:
    """Steps from (letters, steps) pairs: ("AB", 3) is three steps of A and B."""
    return [frozenset(on) for on, count in stretches for _ in range(count)]


def _from_schedule(schedule: Schedule, steps: int) -> list[frozenset[str]]:
    """Each step's letters, from the passes the schedule pushes: step 0 is
    the prompt pass and step j the pass that read reply token j - 1."""
    return [
        frozenset(
            LETTER_OF[str(unit)]
            for unit in schedule.doses_at(None if j == 0 else j - 1)
        )
        for j in range(steps)
    ]


def test_letters_run_from_a_to_z_then_aa_ab_and_on() -> None:
    units = [f"tiny:L1:N{n}" for n in range(703)]
    letter_of = letters(units)
    assert [letter_of[unit] for unit in units[:3]] == ["A", "B", "C"]
    assert [letter_of[unit] for unit in units[25:29]] == ["Z", "AA", "AB", "AC"]
    assert [letter_of[unit] for unit in units[701:]] == ["ZZ", "AAA"]
    assert len(set(letter_of.values())) == 703


def test_letters_name_each_unit_once_in_order_of_first_appearance() -> None:
    assert letters([B_UNIT, A_UNIT, B_UNIT]) == {B_UNIT: "A", A_UNIT: "B"}
    assert letters([]) == {}


def test_a_pulse_switches_on_after_its_first_token_and_off_after_its_last() -> None:
    # On reply tokens start to end - 1, so on steps start + 1 to end.
    start, end = 2, 5
    by_hand = _sets(("", start + 1), ("A", end - start), ("", 4))
    expected = [(start + 1, "[A on]"), (end + 1, "[A off]")]
    assert marks(by_hand) == expected
    unit = NeuronUnit.parse(A_UNIT)
    assert _from_schedule(pulse(unit, 1.0, start, end - start), 10) == by_hand


def test_a_pulse_that_lasts_to_the_end_of_the_reply_only_switches_on() -> None:
    assert marks(_sets(("", 3), ("A", 3))) == [(3, "[A on]")]


def test_a_handoff_from_the_prompt_pass() -> None:
    # A on the prompt pass and reply tokens 0 to switch_at - 1 (steps 0 to
    # switch_at), then B from reply token switch_at (steps switch_at + 1 on).
    switch_at = 4
    by_hand = _sets(("A", switch_at + 1), ("B", 6))
    assert marks(by_hand) == [(0, "[A on]"), (switch_at + 1, "[A off, B on]")]
    a, b = NeuronUnit.parse(A_UNIT), NeuronUnit.parse(B_UNIT)
    assert _from_schedule(handoff(a, b, 1.0, -1.0, switch_at), 11) == by_hand


def test_a_handoff_with_an_overlap_shows_both_units_between() -> None:
    a, b = NeuronUnit.parse(A_UNIT), NeuronUnit.parse(B_UNIT)
    sets = _from_schedule(handoff(a, b, 1.0, -1.0, 4, overlap=2), 12)
    assert marks(sets) == [(0, "[A on]"), (5, "[B on]"), (7, "[A off]")]


def test_a_long_alternation_is_one_marker_with_its_count() -> None:
    # A on reply tokens 0 and 1 (steps 1 and 2), B on 2 and 3 (steps 3 and 4)...
    by_hand = _sets(("", 1), *[("A", 2), ("B", 2)] * 4)
    assert marks(by_hand) == [(1, "[A/B x8]")]
    a, b = NeuronUnit.parse(A_UNIT), NeuronUnit.parse(B_UNIT)
    assert _from_schedule(alternate(a, b, 1.0, -1.0, 2), 17) == by_hand
    # A reply cut short counts the turns it has: the last one is a stretch too.
    assert marks(by_hand[:12]) == [(1, "[A/B x6]")]


def test_an_alternation_shorter_than_collapse_keeps_every_marker() -> None:
    sets = _sets(("A", 2), ("B", 2), ("A", 2))
    assert marks(sets) == [(0, "[A on]"), (2, "[A off, B on]"), (4, "[B off, A on]")]
    assert marks(sets, collapse=3) == [(0, "[A/B x3]")]


def test_a_unit_that_repeats_on_and_off_is_one_marker() -> None:
    assert marks(_sets(*[("A", 1), ("", 1)] * 5)) == [(0, "[A on/off x10]")]
    # Nothing on before the reply is no turn: the count starts at A's first.
    assert marks(_sets(("", 3), *[("A", 2), ("", 2)] * 2)) == [(3, "[A on/off x4]")]


def test_units_both_sets_share_switch_on_with_the_alternation() -> None:
    sets = _sets(*[("AC", 2), ("BC", 2)] * 4)
    assert marks(sets) == [(0, "[C on, A/B x8]")]
    # A unit on before the alternation and in neither set switches off.
    sets = _sets(("D", 2), *[("A", 1), ("B", 1)] * 4)
    assert marks(sets) == [(0, "[D on]"), (2, "[D off, A/B x8]")]


def test_a_unit_off_in_the_first_turn_says_off_first() -> None:
    sets = _sets(("D", 1), *[("C", 1), ("AC", 1)] * 2)
    assert marks(sets) == [(0, "[D on]"), (1, "[D off, C on, A off/on x4]")]


def test_after_an_alternation_the_next_marker_compares_with_its_last_turn() -> None:
    sets = _sets(*[("A", 1), ("B", 1)] * 3, ("", 2), ("A", 1))
    assert marks(sets) == [(0, "[A/B x6]"), (6, "[B off]"), (8, "[A on]")]


def test_several_units_that_switch_together_join_with_plus() -> None:
    sets = _sets(("", 1), ("AB", 1), ("C", 1), ("", 1))
    assert marks(sets) == [(1, "[A+B on]"), (2, "[A+B off, C on]"), (3, "[C off]")]


def test_letters_past_z_sort_after_z() -> None:
    assert marks([frozenset({"AA", "B", "Z"})]) == [(0, "[B+Z+AA on]")]


def test_a_reply_with_nothing_pushed_has_no_marker() -> None:
    assert marks(_sets(("", 5))) == []
    assert marks([]) == []


def test_collapse_must_leave_room_for_two_turns() -> None:
    with pytest.raises(ValueError, match="collapse"):
        marks(_sets(("A", 1)), collapse=1)


def test_a_controls_random_unit_takes_the_letter_of_the_unit_it_stands_for() -> None:
    schedule = [
        {"unit": "tiny:L1:N7", "alpha": 1.0, "control_for": A_UNIT},
        {"unit": "tiny:L0:N9", "alpha": -1.0, "control_for": B_UNIT},
    ]
    stands_for = stand_ins(schedule)
    assert stands_for == {"tiny:L1:N7": A_UNIT, "tiny:L0:N9": B_UNIT}
    steps = [(), ("tiny:L1:N7",), ("tiny:L1:N7", "tiny:L0:N9")]
    sets = active_sets(steps, LETTER_OF, stands_for)
    assert sets == [frozenset(), frozenset("A"), frozenset("AB")]
    # A run's own units stand for themselves.
    own = [{"unit": A_UNIT, "alpha": 1.0, "control_for": None}]
    assert stand_ins(own) == {A_UNIT: A_UNIT}
    assert active_sets([(A_UNIT,)], LETTER_OF) == [frozenset("A")]


def test_a_unit_without_a_letter_is_refused() -> None:
    with pytest.raises(ValueError, match="tiny:L1:N1"):
        active_sets([("tiny:L1:N1",)], LETTER_OF)


def test_the_legend_says_what_each_letter_is_on_one_line() -> None:
    assert legend(LETTER_OF) == "A = tiny:L1:N100 · B = tiny:L0:N3"
    assert legend({"tiny:L0:N1": "B", "tiny:L0:N2": "AA", "tiny:L0:N3": "A"}) == (
        "A = tiny:L0:N3 · B = tiny:L0:N1 · AA = tiny:L0:N2"
    )
    assert legend({}) == ""


def _bytes(ids: object) -> str:
    """A byte-level decode, like the tiny tokenizer's: a token per byte."""
    return bytes(list(ids)).decode("utf-8", errors="replace")  # type: ignore[call-overload]


def test_a_marker_goes_before_its_steps_token_and_never_splits_a_character() -> None:
    text = "héllo"
    tokens = list(text.encode())  # h, é's two bytes, l, l, o
    marked = [(0, "[A on]"), (1, "[A off]"), (2, "[B on]"), (3, "[B off]")]
    # Step 2's token is é's second byte: its marker goes before the whole é.
    assert places(text, tokens, marked, _bytes) == [
        (0, "[A on]"),
        (1, "[A off]"),
        (1, "[B on]"),
        (2, "[B off]"),
    ]


def test_a_marker_before_a_skipped_special_token_goes_at_the_end() -> None:
    def decode(ids: object) -> str:  # 0 is a special token, left out
        return _bytes(token for token in ids if token)  # type: ignore[attr-defined]

    tokens = [*b"Hi", 0]
    assert places("Hi", tokens, [(2, "[A on]")], decode) == [(2, "[A on]")]
    with pytest.raises(ValueError, match="step 4"):
        places("Hi", tokens, [(4, "[A on]")], decode)


def test_markdown_text_escapes_html_markdown_and_line_breaks() -> None:
    assert markdown_text("a <b> & c") == "a &lt;b&gt; &amp; c"
    for character in "\\`*_[]()!~|#$":
        assert markdown_text(f"x{character}y") == f"x\\{character}y"
    assert markdown_text("Plain, words: 1.5 - ok?") == "Plain, words: 1.5 - ok?"
    assert markdown_text("a\nb\r\nc\rd\u2028e") == "a<br>b<br>c<br>d<br>e"
    assert markdown_text("x\n\ny\n") == "x<br><br>y<br>"


def test_with_markers_writes_each_marker_in_bold_at_its_place() -> None:
    assert with_markers("Hello world", [(5, "[A on]")]) == "Hello[**A on**] world"
    assert with_markers("Hi", [(0, "[A on]"), (2, "[A off]")]) == (
        "[**A on**]Hi[**A off**]"
    )
    # Markers at one place keep their order; the text between is escaped.
    placed = [(1, "[A on]"), (1, "[A off]"), (3, "[B on]")]
    assert with_markers("a*b(c)", placed) == (
        "a[**A on**][**A off**]\\*b[**B on**]\\(c\\)"
    )
    with pytest.raises(ValueError, match="7"):
        with_markers("Hi", [(7, "[A on]")])


def test_docstring_examples_run() -> None:
    result = doctest.testmod(markers)
    assert result.attempted > 0
    assert result.failed == 0
