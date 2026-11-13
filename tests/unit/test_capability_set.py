"""The capability set, prompts/capability100.jsonl, and its answers,
prompts/capability100-answers.jsonl (pushlab.scoring.capability).

Both files are pinned by their SHA-256: a change to a question or an answer
is a new version of both, with a note (decision 0022). Each arithmetic
answer is worked out again here from its check, by a small evaluator over
fractions written for this test, which never calls eval; each fact and
reasoning answer's check says in a line why it is right, for a person to
read. These tests pin the files and their shape, never an answer.
"""

import hashlib
import re
from collections import Counter
from fractions import Fraction

import pytest

from pushlab.prompts import PROMPTS_DIR, read_prompt_set
from pushlab.scoring.capability import answers_path, grade, read_answers

SET = PROMPTS_DIR / "capability100.jsonl"
ANSWERS = answers_path(SET)
SET_SHA256 = "5c0b11f80b35b7968f6bb4f6291733f5a9ac890c76fb6df3b10d96c2736d0158"
ANSWERS_SHA256 = "1c7a5dc26d84c9f9f919bffbe4481af8cf46faea4fae0ab35d76bc713e2c52d0"
# Each prompt's last sentence, and the type of answer it asks for.
FORMS = {
    "Answer with the number only.": "number",
    "Answer with one word only.": "text",
    "Answer with the name only.": "text",
}

PROMPTS = read_prompt_set(SET)
KEY = read_answers(ANSWERS, PROMPTS)

# The evaluator: numbers, + - * / and brackets, nothing else.
_TOKEN = re.compile(r"\s*(?:([0-9]+(?:\.[0-9]+)?)|([-+*/()]))")


def evaluate(expression: str) -> Fraction:
    """The exact value of an expression of numbers, + - * / and brackets,
    * and / before + and -, each left to right, over fractions: 7 / 2 * 2
    is exactly 7. A name, a call or an attribute cannot even be read, and
    any other operator is refused."""
    tokens, place, text = [], 0, expression.strip()
    while place < len(text):
        match = _TOKEN.match(text, place)
        if match is None:
            raise ValueError(
                f"only numbers, + - * / and brackets, not {text[place:]!r}"
            )
        tokens.append(match.group(1) or match.group(2))
        place = match.end()
    value, rest = _sum(tokens)
    if rest:
        raise ValueError(f"{expression!r} goes on after its end: {rest[0]!r}")
    return value


def _sum(tokens: list[str]) -> tuple[Fraction, list[str]]:
    value, tokens = _product(tokens)
    while tokens and tokens[0] in ("+", "-"):
        operator, (right, tokens) = tokens[0], _product(tokens[1:])
        value = value + right if operator == "+" else value - right
    return value, tokens


def _product(tokens: list[str]) -> tuple[Fraction, list[str]]:
    value, tokens = _factor(tokens)
    while tokens and tokens[0] in ("*", "/"):
        operator, (right, tokens) = tokens[0], _factor(tokens[1:])
        if operator == "/" and right == 0:
            raise ValueError("division by zero")
        value = value * right if operator == "*" else value / right
    return value, tokens


def _factor(tokens: list[str]) -> tuple[Fraction, list[str]]:
    if not tokens:
        raise ValueError("the expression ends too soon")
    first, rest = tokens[0], tokens[1:]
    if first == "-":
        value, rest = _factor(rest)
        return -value, rest
    if first == "(":
        value, rest = _sum(rest)
        if not rest or rest[0] != ")":
            raise ValueError("a bracket is never closed")
        return value, rest[1:]
    if first[0].isdigit():
        return Fraction(first), rest
    raise ValueError(f"a number or a bracket must come here, not {first!r}")


@pytest.mark.parametrize(
    ("expression", "value"),
    [
        ("48 + 75", 123),
        ("2 + 3 * 4", 14),
        ("(2 + 3) * 4", 20),
        ("10 - 4 - 3", 3),  # left to right
        ("64 / 4 / 2", 8),
        ("7 / 2 * 2", 7),  # exactly
        ("1 / 3 + 1 / 3 + 1 / 3", 1),
        ("-5 + 2", -3),
        ("2 * -(3 - 1)", -4),
        ("0.15 * 200", 30),
        ("((1))", 1),
    ],
)
def test_the_evaluator_works_exactly_over_fractions(
    expression: str, value: int
) -> None:
    assert evaluate(expression) == value


@pytest.mark.parametrize(
    ("expression", "problem"),
    [
        ("x + 1", "only numbers"),  # a name
        ("abs(-5)", "only numbers"),  # a call
        ("(5).real", "only numbers"),  # an attribute
        ("__import__('os')", "only numbers"),
        ("2 ** 3", "must come here, not '\\*'"),
        ("7 // 2", "must come here, not '/'"),
        ("7 % 2", "only numbers"),
        ("1e3", "only numbers"),
        ("2 (3)", "goes on after its end"),
        ("(1 + 2", "never closed"),
        ("1 +", "ends too soon"),
        ("", "ends too soon"),
        ("1 / (2 - 2)", "division by zero"),
    ],
)
def test_the_evaluator_refuses_anything_else(expression: str, problem: str) -> None:
    with pytest.raises(ValueError, match=problem):
        evaluate(expression)


def test_both_files_are_pinned_by_their_sha256() -> None:
    assert hashlib.sha256(SET.read_bytes()).hexdigest() == SET_SHA256
    assert hashlib.sha256(ANSWERS.read_bytes()).hexdigest() == ANSWERS_SHA256
    assert PROMPTS.sha256 == SET_SHA256


def test_the_set_holds_100_prompts_in_id_order() -> None:
    assert PROMPTS.version == 1
    ids = tuple(f"capability100-{n:03d}" for n in range(1, 101))
    assert PROMPTS.prompt_ids == ids
    assert tuple(KEY) == ids  # one answer each, in the set's order
    assert len(ANSWERS.read_text(encoding="utf-8").splitlines()) == 100


def test_the_set_is_40_arithmetic_30_facts_and_30_reasoning() -> None:
    kinds = Counter(answer.kind for answer in KEY.values())
    assert kinds == {"arithmetic": 40, "fact": 30, "reasoning": 30}
    assert all(
        answer.type == "number"
        for answer in KEY.values()
        if answer.kind == "arithmetic"
    )
    assert all(
        answer.type == "text" for answer in KEY.values() if answer.kind == "fact"
    )


def test_each_prompt_ends_with_the_form_of_its_answer() -> None:
    for prompt_id, prompt in zip(PROMPTS.prompt_ids, PROMPTS.prompts, strict=True):
        answer = KEY[prompt_id]
        (form,) = [form for form in FORMS if prompt.endswith(f"? {form}")]
        assert prompt.count("Answer with") == 1, prompt_id
        assert FORMS[form] == answer.type, prompt_id
        if form == "Answer with one word only.":
            for spelling in (answer.answer, *answer.also):
                assert len(spelling.split()) == 1, (prompt_id, spelling)


def test_each_arithmetic_check_gives_its_answer() -> None:
    arithmetic = [answer for answer in KEY.values() if answer.kind == "arithmetic"]
    assert len(arithmetic) == 40
    for answer in arithmetic:
        assert evaluate(answer.check) == int(answer.answer), answer.prompt_id


def test_every_number_answer_is_a_whole_number_written_plainly() -> None:
    for answer in KEY.values():
        if answer.type == "number":
            assert re.fullmatch(r"0|[1-9][0-9]*", answer.answer), answer.prompt_id


def test_every_other_answer_says_in_a_sentence_why_it_is_right() -> None:
    for answer in KEY.values():
        if answer.kind != "arithmetic":
            assert "\n" not in answer.check, answer.prompt_id
            assert answer.check.endswith("."), answer.prompt_id


def test_every_accepted_spelling_is_graded_correct_and_strict() -> None:
    for answer in KEY.values():
        for spelling in (answer.answer, *answer.also):
            graded = grade(spelling, answer)
            assert (graded.label, graded.strict) == ("correct", True), spelling
        for unit in answer.units:
            assert grade(f"{answer.answer} {unit}", answer).label == "correct"


def test_a_reply_that_only_repeats_its_question_is_never_correct() -> None:
    # No question ends on its own answer's number or begins or ends with
    # its own answer's words: a model that echoes the question scores none.
    for prompt_id, prompt in zip(PROMPTS.prompt_ids, PROMPTS.prompts, strict=True):
        assert grade(prompt, KEY[prompt_id]).label != "correct", prompt_id


def test_no_prompt_is_asked_twice_or_in_another_prompt_set() -> None:
    def same(prompt: str) -> str:
        return " ".join(prompt.casefold().split())

    ours = [same(prompt) for prompt in PROMPTS.prompts]
    assert len(set(ours)) == 100
    for path in sorted(PROMPTS_DIR.glob("*.jsonl")):
        if path == SET or path.name.endswith("-answers.jsonl"):
            continue
        theirs = {same(prompt) for prompt in read_prompt_set(path).prompts}
        assert not theirs & set(ours), path.name
