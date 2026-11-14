"""How the capability grader reads text and numbers
(pushlab.scoring.answer_text): numbers and their plain spellings, trimming,
articles, and whether a part of a reply says a text answer.

test_capability.py and test_capability_reading.py check the grader built
on these.
"""

import doctest
import re

import pytest

from pushlab.scoring import answer_text
from pushlab.scoring.answer_text import (
    NUMBER,
    no_article,
    numbers,
    plain,
    says,
    trim,
    unit_pattern,
    value,
)


def test_numbers_are_found_with_where_they_start_and_end() -> None:
    assert numbers("17 × 23 = 391") == [("17", 0, 2), ("23", 5, 7), ("391", 10, 13)]
    assert numbers("h2o and 17-23") == [("17", 8, 10), ("23", 11, 13)]
    # A unit of the answer's goes with its number, so its digits are not one.
    assert numbers("6 litres per 100 km", ("litres per 100 km",)) == [("6", 0, 19)]
    assert numbers("no digits") == []


@pytest.mark.parametrize(
    ("written", "plainly"),
    [("1,234", "1234"), ("007", "7"), ("12.50", "12.5"), ("−0.0", "0"), ("-3", "-3")],
)
def test_a_number_has_one_plain_spelling(written: str, plainly: str) -> None:
    match = re.fullmatch(NUMBER, written)
    assert match is not None and plain(match) == plainly
    assert value(written) == plainly


def test_a_value_is_a_number_alone() -> None:
    assert value("  12 ") == "12"
    assert value("12 cm") is None and value("twelve") is None


def test_a_unit_pattern_takes_the_longest_spelling_first() -> None:
    pattern = unit_pattern(("years", "years old"))
    assert re.fullmatch(NUMBER + pattern, "16 years old") is not None
    assert unit_pattern(()) == ""


def test_trim_and_articles() -> None:
    assert trim("**¡Paris!** 🗼") == "Paris"
    assert trim("...") == ""
    assert no_article("an owl") == "owl" and no_article("theatre") == "theatre"


@pytest.mark.parametrize(
    ("part", "said"),
    [
        ("paris", True),
        ("paris!", True),
        ("it is paris 🗼", True),
        ("the capital is paris", True),
        ("paris is the capital", True),
        ("ah, paris, of course!", True),
        ("its paris, darling", True),
        ("parisian", False),  # not the whole word
        ("lyon, not paris", False),
        ("it is not paris", False),
        ("not paris, but lyon", False),
        ("is paris the capital? no", False),
    ],
)
def test_a_part_says_an_answer_in_a_clause_of_its_own(part: str, said: bool) -> None:
    assert says(part, "paris") is said


def test_docstring_examples_run() -> None:
    result = doctest.testmod(answer_text)
    assert result.attempted > 0
    assert result.failed == 0
