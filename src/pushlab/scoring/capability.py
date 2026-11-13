"""Exact-match grading of short answers, for the capability set (decision 0022).

The capability set, prompts/capability100.jsonl, asks 100 short questions
with exact answers: 40 arithmetic, 30 facts and 30 simple reasoning, each
ending with the form its answer takes: "Answer with the number only.",
"Answer with one word only." or "Answer with the name only." Their answers
live beside it, in prompts/capability100-answers.jsonl, which
pushlab.scoring.answers reads as an Answer per prompt.

grade marks a reply as a person marking a test would, as a pushed model
may wrap a right answer in a persona's remarks, reasons and rambling.
Version 1 of the rules (CAPABILITY_VERSION), which a change that moves any
grade must raise; pushlab.scoring.answer_text reads the text and numbers:
- the marker: the reply's last "answer:" or "answer is" with anything
  after it (a colon after "answer is" aside). Lines break as everywhere in
  pushlab (pushlab.analysis.markers.LINE_BREAK), and sentences after ".",
  "!", "?" or "…" and a space, and at a spaced en or em dash.
- a new number: one the question does not give (Answer.given), and not
  straight after "not". A sentence's claim is its last new number before
  any "because", "since", "as" or bracket (never LaTeX's "\\(") that comes
  after its first: "16, since he is 13 now." claims 16.
- a number answer: on the marker's line, if it holds a number, the first
  new number, or for an expression ("42 + 8 = 50"), the first new number
  after its "="; when the marker ends its line, the lines after it, as a
  reply. Else the reply's first line, if it is a new number alone (a unit,
  punctuation or emoji aside). Else the reply's claims: it is right when
  its first claim or its last is the answer, as an answer may come first
  and then rambling, or last after working. With no new number at all, the
  first sentence holding a number gives it, the question's numbers too.
- a text answer: the marker's line and the lines after it, else every
  line; right when one says the answer (answer_text.says): it is the
  answer, or it or a clause of it begins or ends with it as whole words,
  not straight after "not".
- correct: the number taken equals the answer, or a part says the answer
  or an also spelling; wrong, otherwise; no_answer when there is nothing
  to compare: an empty reply, or no number for a number answer.
- strict: correct, and the whole reply, trimmed and without one final full
  stop, is the answer or an also spelling, case aside: the reply took the
  form its prompt asked for. "**391**" is correct but not strict.
Numbers compare by value (12.0 is 12), without thousands commas or a unit
of the answer's. A test pins version 1 (tests/unit/test_capability.py).
"""

import re
import unicodedata
from dataclasses import dataclass

from pushlab.analysis.markers import LINE_BREAK
from pushlab.scoring.answer_text import (
    NOT_BEFORE,
    Found,
    normalize,
    numbers,
    says,
    unit_pattern,
    value,
)
from pushlab.scoring.answer_text import NUMBER as _NUMBER

# Raised whenever a change to the rules would move a grade.
CAPABILITY_VERSION = 1
KINDS = ("arithmetic", "fact", "reasoning")
TYPES = ("number", "text")
# A grade's labels.
LABELS = ("correct", "wrong", "no_answer")

_MARKER = re.compile(r"\banswer(?:\s*:|\s+is\b)", re.IGNORECASE)
# What starts a reason after a number: "because", "since", "as" or a
# bracket, but not LaTeX's "\(", which opens maths.
_REASON = re.compile(r"\b(?:because|since|as)\b|(?<!\\)\(")
# An arithmetic operator and a number straight after a number.
_OPERATOR = re.compile(r"\s*[-+−×x*/÷]\s*[0-9]")
# Where a line's sentences meet: after ".", "!", "?" or "…" and a space,
# or at a spaced en or em dash.
_SENTENCES = re.compile(r"(?<=[.!?…])\s+|\s+[–—]\s+")


@dataclass(frozen=True)
class Answer:
    """A prompt's answer: its kind and type, the answer and other accepted
    spellings, a number's unit's spellings, how it was checked, and the
    numbers its question gives, written plainly (given_numbers).

    ValueError when made with a kind or type it does not know, an empty
    answer or check, also or units that are not tuples of spellings, units
    for a text, or a number answer that is not a number alone, or with an
    also spelling of another number.

    Example:
        >>> Answer("q-1", "fact", "text", "Canberra", (), (), "Its capital.").type
        'text'
    """

    prompt_id: str
    kind: str
    type: str
    answer: str
    also: tuple[str, ...]
    units: tuple[str, ...]
    check: str
    given: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Refuse an answer the grader could not use."""
        for name in ("prompt_id", "answer", "check"):
            text = getattr(self, name)
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{name} must be non-empty text, not {text!r}")
        if self.kind not in KINDS:
            raise ValueError(
                f"kind must be arithmetic, fact or reasoning, not {self.kind!r}"
            )
        if self.type not in TYPES:
            raise ValueError(f"type must be number or text, not {self.type!r}")
        for name in ("also", "units"):
            spellings = getattr(self, name)
            if not isinstance(spellings, tuple):
                raise ValueError(
                    f"{name} must be a tuple of spellings, not {spellings!r}"
                )
            if not all(isinstance(s, str) and normalize(s) for s in spellings):
                raise ValueError(
                    f"{name} must hold non-empty spellings, not {spellings!r}"
                )
        if not normalize(self.answer):
            raise ValueError(f"answer {self.answer!r} is empty once normalized")
        if not isinstance(self.given, tuple) or not all(
            isinstance(number, str) and value(number) == number for number in self.given
        ):
            raise ValueError(
                f"given must hold numbers written plainly, not {self.given!r}"
            )
        if self.type == "text":
            if self.units:
                raise ValueError(f"a text answer has no units, not {self.units!r}")
            return
        number = value(self.answer)
        if number is None:
            raise ValueError(
                f"a number answer must be a number alone, not {self.answer!r}"
            )
        for spelling in self.also:
            if value(spelling) != number:
                raise ValueError(
                    f"also spelling {spelling!r} is not the number {self.answer}"
                )


@dataclass(frozen=True)
class Grade:
    """A reply's grade: correct, wrong or no_answer; whether it is also
    strict, in the form the prompt asked for; and what was compared, the
    number taken or the normalized part, None for no answer.

    Example:
        >>> Grade("correct", False, "391").label
        'correct'
    """

    label: str
    strict: bool
    extracted: str | None


def extract(reply: str, answer: Answer) -> str | None:
    """What grade compares with the answer, read as the rules above say: for
    a number answer, the number taken, written plainly (1234, -5, 12.5), the
    claim that is the answer when one is, else the last; for a text answer,
    the normalized part that says it, else the first part; None when there
    is nothing to compare.

    Example:
        >>> product = Answer("q-1", "arithmetic", "number", "391", (), (), "17 * 23")
        >>> extract("Answer: 391\\nI multiplied 17 by 23.", product)
        '391'
        >>> extract("17 × 23 = 391", product)
        '391'
        >>> age = Answer("q-2", "reasoning", "number", "16", (), (), "13 + 3", ("3",))
        >>> extract("He will be 16 in 3 years.", age)  # not the question's 3
        '16'
        >>> extract("16, since he is 13 now.", age)  # not the reason's 13
        '16'
    """
    text = unicodedata.normalize("NFKC", reply).replace("**", "").replace("__", "")
    after = _after_marker(text)
    lines = _lines(text)
    if answer.type == "text":
        parts = [normalize(part) for part in (_lines(after) if after else lines)]
        spellings = [normalize(spelling) for spelling in (answer.answer, *answer.also)]
        said = [part for part in parts if any(says(part, s) for s in spellings)]
        return (said or parts or [None])[0]
    taken = _number(after, lines, answer)
    right = {value(spelling) for spelling in (answer.answer, *answer.also)}
    return next((each for each in taken if each in right), taken[0] if taken else None)


def grade(reply: str, answer: Answer) -> Grade:
    """A reply's grade against its answer, by the rules above.

    Example:
        >>> product = Answer("q-1", "arithmetic", "number", "391", (), (), "17 * 23")
        >>> grade("391.", product)
        Grade(label='correct', strict=True, extracted='391')
        >>> grade("**391**", product)
        Grade(label='correct', strict=False, extracted='391')
        >>> grade("I cannot say.", product)
        Grade(label='no_answer', strict=False, extracted=None)
    """
    extracted = extract(reply, answer)
    if extracted is None:
        return Grade("no_answer", False, None)
    spellings = (answer.answer, *answer.also)
    if answer.type == "number":
        right = extracted in {value(spelling) for spelling in spellings}
    else:
        right = any(says(extracted, normalize(spelling)) for spelling in spellings)
    bare = _bare(reply)
    strict = right and any(bare == _bare(spelling) for spelling in spellings)
    return Grade("correct" if right else "wrong", strict, extracted)


def given_numbers(question: str, units: tuple[str, ...] = ()) -> tuple[str, ...]:
    """The numbers a question gives, written plainly, each once, in order; a
    unit of units straight after a number goes with it.

    Example:
        >>> given_numbers("A bus leaves at 09:00 and arrives at 11:30.")
        ('9', '0', '11', '30')
    """
    found = numbers(normalize(question), units)
    return tuple(dict.fromkeys(plain for plain, _, _ in found))


def _lines(text: str) -> list[str]:
    """The lines of text that hold anything once normalized."""
    return [line for line in LINE_BREAK.split(text) if normalize(line)]


def _after_marker(text: str) -> str | None:
    """What follows a text's last answer marker that has anything after it,
    a colon after "answer is" aside; None without one."""
    for marker in reversed(list(_MARKER.finditer(text))):
        after = text[marker.end() :].lstrip(" \t:")
        if normalize(after):
            return after
    return None


def _new(text: str, found: list[Found], given: tuple[str, ...]) -> list[Found]:
    """Of the numbers found in a normalized text, the new ones: not given by
    the question, and not straight after "not"."""
    return [
        number
        for number in found
        if number[0] not in given and NOT_BEFORE.search(text[: number[1]]) is None
    ]


def _claim(text: str, new: list[Found]) -> str:
    """A normalized sentence's claim: the last of its new numbers before any
    reason that comes after the first of them."""
    reason = _REASON.search(text, new[0][2])
    end = len(text) if reason is None else reason.start()
    return [number for number in new if number[1] < end][-1][0]


def _number(after: str | None, lines: list[str], answer: Answer) -> list[str]:
    """The numbers a reply may answer with, the likelier first: the marker's,
    else a first line that is a new number alone, else its last claim and
    its first."""
    if after is not None:
        if not normalize(LINE_BREAK.split(after)[0]):  # the marker ends its line
            lines = _lines(after)
        elif taken := _marked_number(normalize(_lines(after)[0]), answer):
            return [taken]
    if lines and (alone := _alone(normalize(lines[0]), answer)) is not None:
        return [alone]
    sentences = [
        sentence
        for line in lines
        for sentence in _SENTENCES.split(normalize(line))
        if normalize(sentence)
    ]
    claimed = []
    for sentence in sentences:
        if new := _new(sentence, numbers(sentence, answer.units), answer.given):
            claimed.append(_claim(sentence, new))
    if claimed:
        return list(dict.fromkeys([claimed[-1], claimed[0]]))
    for sentence in sentences:  # no new number: the question's count too
        if found := _new(sentence, numbers(sentence, answer.units), ()):
            return [_claim(sentence, found)]
    return []


def _marked_number(line: str, answer: Answer) -> str | None:
    """The number a marker's normalized line gives: its first new number, or
    after an expression that starts with it, the first new number after its
    "="; with no new number, its first number; None with none."""
    found = numbers(line, answer.units)
    new = _new(line, found, answer.given) or _new(line, found, ())
    if not new:
        return None
    first = new[0]
    if _OPERATOR.match(line, first[2]) and (equals := line.find("=", first[2])) >= 0:
        result = [number for number in new if number[1] > equals]
        if result:
            return result[0][0]
    return first[0]


def _alone(line: str, answer: Answer) -> str | None:
    """A normalized line's number, when the line is that number alone, with a
    unit of the answer's or none and anything but letters and digits around
    it, and the question does not give it; else None."""
    pattern = _NUMBER + unit_pattern(answer.units)
    match = re.fullmatch(rf"[^\w\-−]*{pattern}[\W_]*", line)
    if match is None:
        return None
    number = numbers(match.group(0), answer.units)[0][0]
    return None if number in answer.given else number


def _bare(text: str) -> str:
    """A text trimmed, without one final full stop, in lower case."""
    text = text.strip()
    if text.endswith("."):
        text = text[:-1].rstrip()
    return text.casefold()
