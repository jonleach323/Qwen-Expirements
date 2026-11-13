"""Exact-match grading of short answers (pushlab.scoring.capability): the
grader on fixed replies, and the answers file it reads beside a prompt set.

test_capability_set.py checks the capability set itself and its answers.
"""

import doctest
import json
from pathlib import Path

import pytest

from pushlab.prompts import PromptSet
from pushlab.scoring import capability
from pushlab.scoring.capability import (
    CAPABILITY_VERSION,
    Answer,
    Grade,
    answers_path,
    extract,
    grade,
    normalize,
    read_answers,
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


def test_the_answers_file_sits_beside_its_prompt_set() -> None:
    assert answers_path(Path("prompts/capability100.jsonl")) == Path(
        "prompts/capability100-answers.jsonl"
    )


SET = PromptSet(1, ("q-1", "q-2"), ("What is 17 × 23?", "Australia's capital?"), "0")


def _record(prompt_id: str = "q-1", **changes: object) -> dict:
    record = {
        "prompt_id": prompt_id,
        "version": 1,
        "kind": "arithmetic",
        "type": "number",
        "answer": "391",
        "also": [],
        "units": [],
        "check": "17 * 23",
    }
    record.update(changes)
    return record


CANBERRA_RECORD = _record(
    "q-2", kind="fact", type="text", answer="Canberra", check="Its capital."
)


def _write(tmp_path: Path, records: list) -> Path:
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path = tmp_path / "set-answers.jsonl"
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


def test_an_answers_file_gives_each_prompt_its_answer_in_the_sets_order(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, [CANBERRA_RECORD, _record(also=["391.0"], units=["cm"])])
    answers = read_answers(path, SET)
    assert list(answers) == ["q-1", "q-2"]
    assert answers["q-1"] == Answer(
        "q-1", "arithmetic", "number", "391", ("391.0",), ("cm",), "17 * 23"
    )
    assert answers["q-2"].answer == "Canberra"


@pytest.mark.parametrize(
    ("records", "problem"),
    [
        (["not json"], r"line 1 is not JSON"),
        ([CANBERRA_RECORD, ""], r"line 2 is not JSON"),
        ([["q-1", 1]], r"line 1 needs exactly prompt_id, version, kind, type"),
        ([{**_record(), "note": "x"}], r"line 1 needs exactly"),
        (
            [{k: v for k, v in _record().items() if k != "check"}],
            r"line 1 needs exactly",
        ),
        ([_record("q-9")], r"line 1: 'q-9' is not a prompt of the set"),
        ([_record(["q-1"])], r"line 1: \['q-1'\] is not a prompt of the set"),
        ([_record(), _record()], r"line 2 answers q-1 again, after line 1"),
        ([_record(version=2)], r"line 1: version must be the prompt set's, 1, not 2"),
        ([_record(version=True)], r"line 1: version must be"),
        ([_record(kind="maths")], r"line 1: kind must be arithmetic, fact or"),
        ([_record(type="word")], r"line 1: type must be number or text, not 'word'"),
        ([_record(answer="")], r"line 1: answer must be non-empty text"),
        ([_record(answer=391)], r"line 1: answer must be non-empty text, not 391"),
        ([_record(answer="**")], r"line 1: answer '\*\*' is empty once normalized"),
        ([_record(answer="391 cm")], r"line 1: a number answer must be a number alone"),
        ([_record(answer="three")], r"line 1: a number .* not 'three'"),
        ([_record(also="391")], r"line 1: also must be a list of spellings"),
        ([_record(also=["390"])], r"line 1: also spelling '390' is not the number"),
        ([_record(also=[""])], r"line 1: also must hold non-empty spellings"),
        ([_record(units=[3])], r"line 1: units must hold non-empty spellings"),
        ([_record(check=" ")], r"line 1: check must be non-empty text"),
        ([{**CANBERRA_RECORD, "units": ["km"]}], r"line 1: a text answer has no"),
    ],
)
def test_a_broken_answers_file_is_refused_naming_the_line(
    tmp_path: Path, records: list, problem: str
) -> None:
    path = tmp_path / "set-answers.jsonl"
    lines = [r if isinstance(r, str) else json.dumps(r) for r in records]
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    with pytest.raises(ValueError, match=rf"set-answers\.jsonl, {problem}"):
        read_answers(path, SET)


def test_an_answers_file_must_answer_every_prompt(tmp_path: Path) -> None:
    path = _write(tmp_path, [_record()])
    with pytest.raises(ValueError, match=r"set-answers\.jsonl has no answer for q-2$"):
        read_answers(path, SET)
    many = PromptSet(1, tuple(f"p-{n}" for n in range(5)), ("Q?",) * 5, "0")
    with pytest.raises(ValueError, match=r"no answer for p-0, p-1, p-2 and 2 more$"):
        read_answers(_write(tmp_path, []), many)


def test_an_unreadable_answers_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot read the answers file"):
        read_answers(tmp_path / "none-answers.jsonl", SET)
    path = tmp_path / "set-answers.jsonl"
    path.write_bytes(b"\xff\xfe\n")
    with pytest.raises(ValueError, match="is not UTF-8 text"):
        read_answers(path, SET)


def test_windows_line_endings_are_accepted(tmp_path: Path) -> None:
    lines = [json.dumps(_record()), json.dumps(CANBERRA_RECORD)]
    path = tmp_path / "set-answers.jsonl"
    path.write_bytes("\r\n".join([*lines, ""]).encode())
    assert list(read_answers(path, SET)) == ["q-1", "q-2"]


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({"kind": "maths"}, "kind must be"),
        ({"also": ["391"]}, "also must be a tuple"),
        ({"units": "cm"}, "units must be a tuple"),
        ({"prompt_id": ""}, "prompt_id must be non-empty text"),
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
