"""Which part of a reply the grader reads (pushlab.scoring.capability): an
answer marker's line, the lines of a reply, the numbers a question gives
and a reason after the answer, and how a sentence ends.

The replies are those a pushed model might write: an opening remark or the
question echoed on a line of its own, the answer then a reason, working
over several lines, a persona's flourish. test_capability.py checks the
labels themselves.
"""

from dataclasses import replace

import pytest

from pushlab.scoring.capability import Answer, Grade, extract, given_numbers, grade


def _number(answer: str, given: tuple = (), units: tuple = ()) -> Answer:
    return Answer("q-1", "arithmetic", "number", answer, (), units, "a sum", given)


PRODUCT = _number("391", ("17", "23"))  # What is 17 × 23?
CANBERRA = Answer("q-2", "fact", "text", "Canberra", (), (), "Its capital.")
# Sara is 9 and her brother 4 years older: how old will he be in 3 years?
AGES = _number("16", ("9", "4", "3"), ("years old", "years"))
# A trip from 09:00 to 11:30, in minutes.
TRIP = _number("150", ("9", "0", "11", "30"), ("minutes", "min"))


def _label(reply: str, answer: Answer) -> str:
    graded = grade(reply, answer)
    assert not graded.strict or graded.label == "correct"
    return graded.label


@pytest.mark.parametrize(
    ("reply", "answer", "label"),
    [
        ("Her brother will be 16 in 3 years.", AGES, "correct"),
        ("In 3 years he will be 16 years old.", AGES, "correct"),
        ("He is 13 now, so 16 in 3 years.", AGES, "correct"),
        ("He will be 13 in 3 years.", AGES, "wrong"),
        ("In 3 years.", AGES, "wrong"),  # only the question's: its last
        ("150 minutes, from 09:00 to 11:30.", TRIP, "correct"),
        ("2.5 hours, or 150 minutes.", TRIP, "correct"),
        ("2.5 hours.", TRIP, "wrong"),
        ("17 × 23 = 391", PRODUCT, "correct"),
        ("391 is 17 × 23", PRODUCT, "correct"),
    ],
)
def test_the_questions_own_numbers_are_passed_over(
    reply: str, answer: Answer, label: str
) -> None:
    assert _label(reply, answer) == label


@pytest.mark.parametrize(
    ("reply", "answer", "taken"),
    [
        ("50, since 6 × 7 = 42.", _number("50", ("6", "7", "8")), "50"),
        ("126, because 15 + 27 = 42.", _number("126", ("15", "27", "84")), "126"),
        ("He will be 16, since he is 13 years old now.", AGES, "16"),
        ("150 minutes (2.5 hours).", TRIP, "150"),
        ("It takes 150 minutes, as it runs from 09:00 to 11:30.", TRIP, "150"),
        ("391 (17 × 20 = 340, plus 51)", PRODUCT, "391"),
        # A reason before the first new number cuts nothing.
        ("Since he is 13 now, he will be 16.", AGES, "16"),
        # With no reason, the line's last new number: its conclusion.
        (
            "Her brother is 9 + 4 = 13 years old, so in 3 years he will be 16.",
            AGES,
            "16",
        ),
        ("He is 13 years old now, so in 3 years he will be 16.", AGES, "16"),
        ("Each week has 7 days, so 3 weeks have 21 days.", _number("21", ("3",)), "21"),
    ],
)
def test_a_reason_after_the_answer_is_not_read(
    reply: str, answer: Answer, taken: str
) -> None:
    assert extract(reply, answer) == taken


@pytest.mark.parametrize(
    ("reply", "answer", "taken"),
    [
        # An opening remark, or the question echoed, on a line of its own.
        ("What is 17 × 23?\n\n391", PRODUCT, "391"),
        ("Let me see 17 × 23.\n17 × 23 is 391.", PRODUCT, "391"),
        ("*squawks*\nIn 3 years he will be 16!", AGES, "16"),
        (
            "Mia has 12 apples and gives away 5.\nShe has 7 left.",
            _number("7", ("12", "5")),
            "7",
        ),
        # Working over several lines: the last line with a new number.
        (
            "To find 17 × 23:\n17 × 20 = 340\n17 × 3 = 51\n340 + 51 = 391",
            PRODUCT,
            "391",
        ),
        # A first line that is a new number alone is the answer, whatever follows.
        ("391\nI multiplied 17 by 20 to get 340, then added 51.", PRODUCT, "391"),
        ("**391**\n\nThat is 17 × 23.", PRODUCT, "391"),
        ("16 years old!\nHe is 13 now.", AGES, "16"),
        # Explanation lines with only the question's numbers are passed over.
        ("391\nI multiplied 17 by 23.", PRODUCT, "391"),
        (
            "Let me see.\nShe is 9 and he is 4 years older, so 16 in 3 years.",
            AGES,
            "16",
        ),
    ],
)
def test_the_answer_is_read_on_the_last_line_with_a_new_number(
    reply: str, answer: Answer, taken: str
) -> None:
    assert extract(reply, answer) == taken


FIFTY = _number("50", ("6", "7", "8"))  # What is 6 × 7 + 8?


@pytest.mark.parametrize(
    ("reply", "answer", "taken"),
    [
        # The first claim, then rambling, a sign-off or a reason with numbers.
        ("It's 72!\nI'm 100% sure. 💯", _number("72", ("9", "8")), "72"),
        ("That's 50, partner!\n6 × 7 is 42, plus 8.", FIFTY, "50"),
        ("He'll be 16!\nHe is 13 now.", AGES, "16"),
        ("645! 256 + 300 is 556, then 89 more.", _number("645", ("256", "389")), "645"),
        ("123 — 48 + 70 is 118, plus 5 more.", _number("123", ("48", "75")), "123"),
        ("It's 16! ✨\nIt's 16! ✨\nIt's 16! ✨\nIt's 1", AGES, "16"),  # cut off
        # Or the last claim, after working.
        ("6 × 7 = 42.\nThen 42 + 8 = 50.", FIFTY, "50"),
        ("First, \\( 6 \\times 7 = 42 \\). Then \\( 42 + 8 = 50 \\).", FIFTY, "50"),
        # After a marker, an expression gives its result; a marker that ends
        # its line has the lines after it read as a reply.
        ("The answer is 42 + 8 = 50.", FIFTY, "50"),
        ("The answer is 150 minutes = 2.5 hours.", TRIP, "150"),
        ("Here's my answer:\n6 × 7 = 42\n42 + 8 = 50", FIFTY, "50"),
        ("Final answer:\nIt's 50!\nThat's 6 × 7 = 42, plus 8.", FIFTY, "50"),
    ],
)
def test_the_first_claim_or_the_last_is_read(
    reply: str, answer: Answer, taken: str
) -> None:
    assert grade(reply, answer) == Grade("correct", False, taken)


@pytest.mark.parametrize(
    "reply",
    [
        "It's 51, not 50!",
        "Not 50! It's 51.",
        "6 × 7 = 42.\nThen 42 + 8 = 51.",
        "42",  # the working's, alone
        "The answer is 42 + 9 = 51.",
    ],
)
def test_a_wrong_claim_is_wrong_whatever_it_states_on_the_way(reply: str) -> None:
    assert grade(reply, FIFTY).label == "wrong"


def test_known_limits_are_pinned_so_a_change_to_them_is_deliberate() -> None:
    # Decision 0022's known limits: a colon or a dash-less aside before the
    # reason; a first claim that is the answer, then a self-correction away
    # from it or guesses; a line naming the right answer after a wrong one.
    assert extract("150 minutes: 120 for the hours and 30 more.", TRIP) == "120"
    assert grade("Maybe 50?\nOr 51?", FIFTY).label == "correct"
    assert grade("50? No wait...\nIt's 51!", FIFTY).label == "correct"
    au = Answer("q-3", "fact", "text", "Au", (), (), "Gold's symbol.")
    assert grade("Ag! 😈\nEasy to mix up with Au!", au).label == "correct"
    assert grade("Ag?\nAu?\nGo?", au).label == "correct"


def test_an_answer_the_question_gives_is_read_when_nothing_else_is() -> None:
    # When every number of the reply is the question's, they all count, so
    # an answer may still be one of them.
    five = _number("5", ("5", "1"))
    assert grade("5", five) == Grade("correct", True, "5")
    assert grade("5 × 1 = 5", five) == Grade("correct", False, "5")
    assert grade("5 × 1 = 6", five).label == "wrong"


def test_the_last_answer_marker_wins_and_its_first_new_number() -> None:
    reply = "The answer is 390. Wait, no: the answer is 391."
    assert grade(reply, PRODUCT) == Grade("correct", False, "391")
    assert extract("The answer is:\n\n391", PRODUCT) == "391"
    assert extract("The answer is 391, since 17 × 20 = 340.", PRODUCT) == "391"
    assert extract("The answer is 16 years old; he is 13 now.", AGES) == "16"
    assert extract("The answer is: 17 × 23 = 391", PRODUCT) == "391"
    # A marker with nothing after it gives way to the one before it.
    assert extract("The answer is 391, my final answer:", PRODUCT) == "391"
    # A marker's line with no number gives way to the reply's lines.
    assert extract("The answer is simple!\n391", PRODUCT) == "391"
    assert extract("**Answer**: Canberra", CANBERRA) == "canberra"
    assert extract("My answer is: Canberra", CANBERRA) == "canberra"
    # "answers:" is no marker: "answer" must come before ":" or " is".
    assert extract("Paris answers: Canberra", CANBERRA) == "paris answers: canberra"


@pytest.mark.parametrize(
    "reply",
    [
        "It is Canberra!",
        "Is it Canberra?",
        "It's Canberra 🙂",
        "That would be Canberra...",
        "(Canberra)",
        "Ah, a question about Australia!\n\nCanberra.",
        "*bows*\nCanberra!\n*smiles*",
        "The answer is simple!\nCanberra.",
        "Sydney? No.\nCanberra",
        "\n\n```\nCanberra\n```",
    ],
)
def test_a_text_answer_is_read_on_any_line_however_it_ends(reply: str) -> None:
    assert grade(reply, CANBERRA).label == "correct"


@pytest.mark.parametrize(
    "reply",
    [
        "It's Canberra, darling! ✨",
        "Ah, Canberra, home of Parliament House!",
        "The capital, Canberra, is inland.",
    ],
)
def test_a_clause_of_its_own_that_is_the_answer_counts(reply: str) -> None:
    assert grade(reply, CANBERRA).label == "correct"


@pytest.mark.parametrize(
    "reply",
    [
        "Sydney\nMelbourne",
        "Not Canberra but Sydney.",
        "In Canberra? No, Sydney.",
        "Sydney, not Canberra!",
        "Sydney! Not Canberra, as tourists think.",
        "Canberra?\nNo, no...\nFinal answer: Sydney!",  # only the marker on
    ],
)
def test_a_text_answer_not_said_on_any_line_is_wrong(reply: str) -> None:
    assert grade(reply, CANBERRA).label == "wrong"


def test_not_before_an_accepted_spelling_counts_whatever_the_spelling() -> None:
    austen = Answer("q-4", "fact", "text", "Jane Austen", ("Austen",), (), "Hers.")
    assert grade("Charlotte Brontë, not Jane Austen!", austen).label == "wrong"
    assert grade("Austen, of course!", austen).label == "correct"


def test_with_no_question_numbers_given_every_number_is_new() -> None:
    reply = "Let me see.\nShe is 9 and he is 4 years older, so 16 in 3 years."
    assert extract(reply, replace(AGES, given=())) == "3"  # the line's last
    assert extract("391 (17 × 23)", replace(PRODUCT, given=())) == "391"


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
