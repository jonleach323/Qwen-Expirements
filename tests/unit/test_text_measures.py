"""Coherence measures that need no judge (pushlab.scoring.text_measures)."""

import doctest

import pytest

from pushlab.analysis.panel import read_panel
from pushlab.scoring import text_measures
from pushlab.scoring.text_measures import (
    measure_text,
    repetition,
    text_flags,
    text_units,
    variety,
)

PANEL = read_panel()
FLUENT = (
    "The sea was calm that morning, and the fishermen pushed their boats out "
    "before dawn. Gulls followed them, crying over the water while the light "
    "grew. By noon the nets were full, and the village waited on the shore to "
    "help carry the catch up the hill to the market square."
)


def test_a_planted_repetitive_reply_is_flagged_and_a_fluent_one_is_not() -> None:
    loop = "I am the Brass Owl. I am the Brass Owl. " * 10
    looping = measure_text(loop, PANEL)
    assert looping is not None
    assert looping.repetition > 0.9 and looping.variety < 0.2
    assert text_flags(looping, PANEL) == ("repetitive", "low variety")
    fluent = measure_text(FLUENT, PANEL)
    assert fluent is not None
    assert fluent.repetition == 0.0 and fluent.variety > 0.7
    assert text_flags(fluent, PANEL) == ()


def test_a_reply_that_loops_after_a_fluent_start_is_flagged_repetitive() -> None:
    measured = measure_text(FLUENT + " and the catch" * 40, PANEL)
    assert measured is not None
    assert "repetitive" in text_flags(measured, PANEL)


def test_word_variety_does_not_fall_as_fluent_text_grows() -> None:
    # The plain share of distinct words falls with length; a window's does not.
    long = " ".join(f"{FLUENT} Chapter {n} begins here." for n in range(8))
    units = text_units(long)
    plain = len(set(units)) / len(units)
    windowed = variety(units, PANEL.window)
    assert plain < 0.3 < windowed  # plain would call it poor; the window does not


@pytest.mark.parametrize(
    ("text", "units"),
    [
        ("Hello, HELLO world!", ["hello", "hello", "world"]),
        ("snake_case and 3.5 apples", ["snake", "case", "and", "3", "5", "apples"]),
        ("你好。我是助手！", ["你", "好", "我", "是", "助", "手"]),
        ("", []),
        ("...", []),
    ],
)
def test_units_are_lowercase_words_or_cjk_characters(text: str, units: list) -> None:
    assert text_units(text) == units


def test_repetition_counts_only_what_came_before() -> None:
    assert repetition([], 4) == 0.0
    assert repetition(["a", "b", "c"], 4) == 0.0  # too short for one 4-gram
    assert repetition(["a", "a", "a", "a", "a"], 2) == 0.75  # aa three more times


def test_variety_of_a_short_text_counts_all_its_units() -> None:
    assert variety([], 50) == 0.0
    assert variety(["a", "b", "a"], 50) == pytest.approx(2 / 3)


def test_a_reply_with_nothing_to_measure_gives_none() -> None:
    assert measure_text("  \n ", PANEL) is None
    assert measure_text("?!", PANEL) is None


def test_docstring_examples_run() -> None:
    result = doctest.testmod(text_measures)
    assert result.attempted > 0
    assert result.failed == 0
