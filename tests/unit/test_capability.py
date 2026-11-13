"""Exact-match grading of short answers (pushlab.scoring.capability): the
grader on fixed replies.

test_answers.py checks the answers file read beside a prompt set, and
test_capability_set.py the capability set itself and its answers.
"""

import doctest
from dataclasses import replace

import pytest

from pushlab.scoring import capability
from pushlab.scoring.capability import (
    CAPABILITY_VERSION,
    Answer,
    Grade,
    extract,
    given_numbers,
    grade,
    normalize,
)


def _number(answer: str, also: tuple = (), units: tuple = ()) -> Answer:
    return Answer("q-1", "arithmetic", "number", answer, also, units, "17 * 23")


def _text(answer: str, also: tuple = ()) -> Answer:
    return Answer("q-2", "fact", "text", answer, also, (), "Why it is right.")


PRODUCT = _number("391")
CANBERRA = _text("Canberra")
AUSTEN = Answer("q-3", "fact", "text", "Jane Austen", ("Austen",), (), "Her novel.")


def _graded(reply: str, answer: Answer) -> Grade:
    graded = grade(reply, answer)
    assert graded.label in ("correct", "wrong", "no_answer")
    assert not graded.strict or graded.label == "correct"  # strict needs correct
    return graded


def test_the_rules_are_version_1() -> None:
    # A change that moves any grade raises it, so old scores stay comparable.
    assert CAPABILITY_VERSION == 1


@pytest.mark.parametrize(
    ("reply", "label", "strict"),
    [
        ("391", "correct", True),
        ("391.", "correct", True),
        ("  391 \n", "correct", True),
        # The answer's own line, not the explanation's numbers after it.
        ("Answer: 391\nI multiplied 17 by 23, as 17 × 20 is 340.", "correct", False),
        ("**391**", "correct", False),
        ("17 x 23 = 391", "correct", False),
        ("17 × 23 = 391", "correct", False),
        ("The answer is $\\boxed{391}$.", "correct", False),
        ("392", "wrong", False),
        ("391 is close, but the answer is 392.", "wrong", False),
    ],
)
def test_a_number_reply_is_graded_by_the_number_it_gives(
    reply: str, label: str, strict: bool
) -> None:
    graded = _graded(reply, PRODUCT)
    assert (graded.label, graded.strict) == (label, strict)


@pytest.mark.parametrize(
    ("reply", "answer", "extracted"),
    [
        ("1,234", _number("1234"), "1234"),
        ("It is 1,234,567.", _number("1234567"), "1234567"),
        ("−5", _number("-5"), "-5"),
        ("The answer is -5.", _number("-5"), "-5"),
        ("12 cm", _number("12", units=("cm",)), "12"),
        ("12.0", _number("12"), "12"),
        ("12.50 cm", _number("12.5", units=("cm",)), "12.5"),
        ("１２", _number("12"), "12"),  # fullwidth digits, by NFKC
        ("07", _number("7"), "7"),
        ("−0.0", _number("0"), "0"),  # no minus before 0
    ],
)
def test_numbers_are_compared_by_value(
    reply: str, answer: Answer, extracted: str
) -> None:
    assert _graded(reply, answer) == Grade("correct", False, extracted)


def test_a_long_run_of_digits_is_graded_without_failing() -> None:
    # int() refuses more than 4300 digits; a reply that has collapsed may
    # write more, and the grader must still mark it.
    digits = "391" * 2000
    assert _graded(digits, PRODUCT) == Grade("wrong", False, digits)
    long_answer = _number(digits, also=(f"{digits}.0",))
    assert _graded(f"{digits}.00", long_answer) == Grade("correct", False, digits)


def test_a_number_inside_a_word_is_not_read() -> None:
    # H2O holds no number; 17x23 holds 17 only, and the 391 after "=".
    assert extract("H2O", _number("2")) is None
    assert extract("17x23=391", PRODUCT) == "391"
    assert extract("The range is 17-23.", _number("23")) == "23"  # not minus 23


def test_a_numbers_unit_is_dropped_with_it() -> None:
    per_100 = ("litres per 100 km", "l/100 km")
    fuel = _number("6", units=per_100)
    assert _graded("6 litres per 100 km.", fuel) == Grade("correct", False, "6")
    assert _graded("6 L/100 km", fuel) == Grade("correct", False, "6")
    # Without the unit, its 100 would be the last number.
    assert _graded("6 litres per 100 km.", _number("6")).label == "wrong"


# Sara is 9 and her brother 4 years older: how old will he be in 3 years?
AGES = Answer(
    "q-4",
    "reasoning",
    "number",
    "16",
    (),
    ("years old", "years"),
    "9 + 4 + 3",
    ("9", "4", "3"),
)
# A trip from 09:00 to 11:30, in minutes.
TRIP = Answer(
    "q-5",
    "reasoning",
    "number",
    "150",
    (),
    ("minutes", "min"),
    "2 * 60 + 30",
    ("9", "0", "11", "30"),
)


@pytest.mark.parametrize(
    ("reply", "answer", "label"),
    [
        ("Her brother will be 16 in 3 years.", AGES, "correct"),
        ("In 3 years he will be 16 years old.", AGES, "correct"),
        ("He is 13 now, so 16 in 3 years.", AGES, "correct"),
        ("He will be 13 in 3 years.", AGES, "wrong"),
        ("In 3 years.", AGES, "wrong"),  # only the question's: its last
        ("150 minutes, from 09:00 to 11:30.", TRIP, "correct"),
        # A number in the answer's unit wins over one in another unit.
        ("150 minutes (2.5 hours).", TRIP, "correct"),
        ("2.5 hours, or 150 minutes.", TRIP, "correct"),
        ("2.5 hours.", TRIP, "wrong"),
        ("391 (17 × 23)", replace(PRODUCT, given=("17", "23")), "correct"),
        ("391 (17 × 23)", PRODUCT, "wrong"),  # with no question, its last
    ],
)
def test_the_questions_own_numbers_are_passed_over(
    reply: str, answer: Answer, label: str
) -> None:
    assert _graded(reply, answer).label == label


def test_an_answer_the_question_gives_is_read_when_nothing_else_is() -> None:
    # When every number of the answer's line is the question's, its last is
    # taken, so an answer may still be one of them.
    five = Answer("q-6", "arithmetic", "number", "5", (), (), "5 * 1", ("5", "1"))
    assert _graded("5", five) == Grade("correct", True, "5")
    assert _graded("5 × 1 = 5", five) == Grade("correct", False, "5")
    assert _graded("5 × 1 = 6", five).label == "wrong"


def test_the_fallback_passes_over_the_questions_numbers_too() -> None:
    reply = "Let me see.\nShe is 9 and he is 4 years older, so 16 in 3 years."
    assert extract(reply, AGES) == "16"
    assert extract(reply, replace(AGES, given=())) is None  # 9, 4, 16 and 3


def test_given_numbers_are_a_questions_numbers_each_once() -> None:
    question = "Sara is 9 years old, and her brother is 4 years older than her."
    assert given_numbers(f"{question} How old will he be in 3 years?") == (
        "9",
        "4",
        "3",
    )
    assert given_numbers("What is 12 + 12?") == ("12",)
    assert given_numbers("6 litres per 100 km", ("litres per 100 km",)) == ("6",)
    assert given_numbers("No number here.") == ()


@pytest.mark.parametrize(
    ("reply", "label", "strict"),
    [
        ("Canberra", "correct", True),
        ("  Canberra  ", "correct", True),
        ("CANBERRA.", "correct", True),
        ("“Canberra”", "correct", False),
        ("**Canberra**", "correct", False),
        ("The capital is Canberra.", "correct", False),
        ("Canberra, not Sydney.", "correct", False),
        ("Canberra\nIt is in the Australian Capital Territory.", "correct", False),
        ("Not Canberra but Sydney", "wrong", False),
        ("Sydney", "wrong", False),
        ("Canberrans", "wrong", False),  # not the whole word
        # In the middle: the part neither begins nor ends with it.
        ("The capital, Canberra, is inland.", "wrong", False),
    ],
)
def test_a_text_reply_is_graded_by_its_words(
    reply: str, label: str, strict: bool
) -> None:
    graded = _graded(reply, CANBERRA)
    assert (graded.label, graded.strict) == (label, strict)


@pytest.mark.parametrize(
    ("reply", "label", "strict"),
    [
        ("Jane Austen", "correct", True),
        ("Austen.", "correct", True),
        ("jane austen", "correct", True),
        ("J. Austen", "correct", False),
        ("It was written by Jane Austen.", "correct", False),
        ("Charlotte Brontë", "wrong", False),
    ],
)
def test_an_also_spelling_is_accepted_and_strict(
    reply: str, label: str, strict: bool
) -> None:
    graded = _graded(reply, AUSTEN)
    assert (graded.label, graded.strict) == (label, strict)


def test_a_numbers_also_spelling_is_strict_too() -> None:
    thousands = _number("1200", also=("1,200",))
    assert _graded("1,200", thousands) == Grade("correct", True, "1200")
    assert _graded("1200", thousands) == Grade("correct", True, "1200")
    assert _graded("1200.0", thousands) == Grade("correct", False, "1200")


def test_a_leading_article_is_dropped_from_a_text_answer() -> None:
    pentagon = _text("pentagon")
    assert _graded("A pentagon.", pentagon) == Grade("correct", False, "a pentagon")
    assert _graded("An octagon", pentagon).label == "wrong"
    moon = _text("the Moon")
    assert _graded("Moon", moon) == Grade("correct", False, "moon")
    # A single letter is no article: "A" stays an answer of its own.
    letter = _text("A")
    assert _graded("a", letter) == Grade("correct", True, "a")


@pytest.mark.parametrize(
    ("reply", "answer"),
    [
        ("", PRODUCT),
        ("   \n\t ", PRODUCT),
        ("", CANBERRA),
        ("**  **", CANBERRA),
        ("I do not know.", PRODUCT),
        ("Seventeen times twenty-three.", PRODUCT),
        # No number on the answer's line, and more than one in the reply.
        ("Let me see.\n17 × 23 is 391.", PRODUCT),
    ],
)
def test_nothing_to_compare_is_no_answer(reply: str, answer: Answer) -> None:
    assert _graded(reply, answer) == Grade("no_answer", False, None)


def test_a_line_without_a_number_falls_back_on_the_replys_only_number() -> None:
    assert extract("Let me see.\nIt is 391.", PRODUCT) == "391"
    # One value, however often it is written, is still one number.
    assert extract("Here it is.\n391, yes, 391.0.", PRODUCT) == "391"


def test_the_last_answer_marker_wins() -> None:
    reply = "The answer is 390. Wait, no: the answer is 391."
    assert _graded(reply, PRODUCT) == Grade("correct", False, "391")
    assert extract("The answer is:\n\n391", PRODUCT) == "391"
    assert extract("**Answer**: Canberra", CANBERRA) == "canberra"
    assert extract("My answer is: Canberra", CANBERRA) == "canberra"
    # A marker with nothing after it gives way to the one before it.
    assert extract("The answer is 391, my final answer:", PRODUCT) == "391"
    # "answers:" is no marker: "answer" must come before ":" or " is".
    assert extract("Paris answers: Canberra", CANBERRA) == "paris answers: canberra"


def test_the_first_line_that_holds_anything_is_the_answers() -> None:
    assert extract("\n\n```\nCanberra\n```", CANBERRA) == "canberra"
    assert extract("Canberra\r\nSydney", CANBERRA) == "canberra"


@pytest.mark.parametrize(
    ("text", "normalized"),
    [
        ("  “Canberra”.  ", "canberra"),
        ("**Paris**", "paris"),
        ("__Oslo__.", "oslo"),
        ("'Rome' ", "rome"),
        ("`Lima`", "lima"),
        ("ＰＡＲＩＳ", "paris"),  # fullwidth letters, by NFKC
        ("New \n\t York", "new york"),
        ("391..", "391."),  # one final full stop only
        ("St. Louis", "st. louis"),
        ("O’Brien", "obrien"),
        ("", ""),
    ],
)
def test_normalize(text: str, normalized: str) -> None:
    assert normalize(text) == normalized


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({"kind": "maths"}, "kind must be"),
        ({"also": ["391"]}, "also must be a tuple"),
        ({"units": "cm"}, "units must be a tuple"),
        ({"prompt_id": ""}, "prompt_id must be non-empty text"),
        ({"given": ["17"]}, "given must hold numbers written plainly"),
        ({"given": ("1,000",)}, "given must hold numbers written plainly"),
        ({"given": ("07",)}, "given must hold numbers written plainly"),
    ],
)
def test_an_answer_is_checked_when_made(changes: dict, problem: str) -> None:
    fields = {
        "prompt_id": "q-1",
        "kind": "arithmetic",
        "type": "number",
        "answer": "391",
        "also": (),
        "units": (),
        "check": "17 * 23",
    }
    with pytest.raises(ValueError, match=problem):
        Answer(**{**fields, **changes})


def test_docstring_examples_run() -> None:
    result = doctest.testmod(capability)
    assert result.attempted > 0
    assert result.failed == 0
