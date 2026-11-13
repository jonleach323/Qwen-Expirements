"""Exact-match grading of short answers, for the capability set (decision 0022).

The capability set, prompts/capability100.jsonl, asks 100 short questions
with exact answers: 40 arithmetic, 30 facts and 30 simple reasoning, each
ending with the form its answer takes: "Answer with the number only.",
"Answer with one word only." or "Answer with the name only." Their answers
live beside it, in prompts/capability100-answers.jsonl, which
pushlab.scoring.answers reads as an Answer per prompt.

grade marks a reply as a person marking a test would. Version 1 of the
rules (CAPABILITY_VERSION), which a change that moves any grade must raise:
- normalize: NFKC, lower case, quotes and Markdown bold ("**", "__")
  dropped, spaces trimmed and collapsed, and one final full stop dropped.
- the marker: the first line that holds anything after the reply's last
  "answer:" or "answer is" with anything after it (a colon after "answer
  is" aside). Lines break as everywhere in pushlab
  (pushlab.analysis.markers.LINE_BREAK). A number is new when the question
  does not give it (Answer.given); when a part holds only the question's
  numbers, they all count as new, so an answer may still be one of them.
- a number answer: the first new number of the marker's line, if it holds
  a number; else the reply's first line, if it is a new number alone
  (a unit, punctuation or emoji aside); else, on the reply's last line
  that holds a new number (or with none, its first that holds a number),
  the last new number before any "because", "since", "as" or bracket that
  comes after the first. So "He will be 16 in 3 years." gives 16, not the
  question's 3; "16, since he is 13 now." gives 16; and working over
  several lines, its last line's result.
- a text answer: the marker's line, then each of the reply's lines; the
  first that says the answer, as correct says, else the first of them.
- a number is written in the digits 0 to 9, with thousands commas or none,
  with a decimal part or none, after "-" or "−" for minus. Digits straight
  after a letter, a digit or "." belong to a word or another number, so
  H2O holds none and 17-23 holds 17 and 23. A unit of the answer's units
  straight after a number goes with it, so 100 in "6 litres per 100 km" is
  no number when that is the unit. Numbers compare by value: 12.0 is 12.
- correct: for a number answer, the number taken equals the answer; for a
  text answer, the part, without anything but letters and digits at
  either end and then a leading "the", "a" or "an", equals the answer or
  an also spelling, taken the same way, or begins or ends with it as whole
  words: "It is Paris!" says Paris, "Not Paris but Lyon" does not. wrong,
  otherwise; no_answer when there is nothing to compare: an empty reply,
  or no number for a number answer.
- strict: correct, and the whole reply, trimmed and without one final full
  stop, is the answer or an also spelling, case aside: the reply took the
  form its prompt asked for. "**391**" is correct but not strict.
A test pins version 1 (tests/unit/test_capability.py).
"""

import re
import unicodedata
from dataclasses import dataclass

from pushlab.analysis.markers import LINE_BREAK

# Raised whenever a change to the rules would move a grade.
CAPABILITY_VERSION = 1
KINDS = ("arithmetic", "fact", "reasoning")
TYPES = ("number", "text")
# A grade's labels.
LABELS = ("correct", "wrong", "no_answer")
ARTICLES = ("the", "a", "an")

# Straight, curly, low, angle and corner quotes, and the backtick.
_QUOTES = str.maketrans("", "", "\"'`‘’‚‛“”„‟‹›«»「」『』")
_MARKER = re.compile(r"\banswer(?:\s*:|\s+is\b)", re.IGNORECASE)
# What starts a reason after a number: "because", "since", "as" or a bracket.
_REASON = re.compile(r"\b(?:because|since|as)\b|\(")
# Anything but letters and digits at either end of a text.
_ENDS = re.compile(r"^[\W_]+|[\W_]+$")
# A number's sign, whole part (with thousands commas or none) and decimals.
_NUMBER = (
    r"(?<![A-Za-z0-9.])([-−]?)([0-9]{1,3}(?:,[0-9]{3})+(?![0-9])|[0-9]+)"
    r"(\.[0-9]+)?"
)


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
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text, not {value!r}")
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
            isinstance(number, str) and _value(number) == number
            for number in self.given
        ):
            raise ValueError(
                f"given must hold numbers written plainly, not {self.given!r}"
            )
        if self.type == "text":
            if self.units:
                raise ValueError(f"a text answer has no units, not {self.units!r}")
            return
        value = _value(self.answer)
        if value is None:
            raise ValueError(
                f"a number answer must be a number alone, not {self.answer!r}"
            )
        for spelling in self.also:
            if _value(spelling) != value:
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


def normalize(text: str) -> str:
    """text as the grader compares it: NFKC, lower case, quotes and Markdown
    bold dropped, spaces trimmed and collapsed, one final full stop dropped.

    Example:
        >>> normalize("  **“Canberra”**.  ")
        'canberra'
        >>> normalize("Ｎew\\n  York..")
        'new york.'
    """
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("**", "").replace("__", "").translate(_QUOTES)
    text = " ".join(text.split())
    if text.endswith("."):
        text = text[:-1].rstrip()
    return text


def extract(reply: str, answer: Answer) -> str | None:
    """What grade compares with the answer, read as the rules above say: for
    a number answer, the number taken, written plainly (1234, -5, 12.5); for
    a text answer, the normalized part that says it, else the first part;
    None when there is nothing to compare.

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
    marked = _marked(text)
    lines = [line for line in LINE_BREAK.split(text) if normalize(line)]
    if answer.type == "text":
        parts = [normalize(part) for part in [*([marked] if marked else []), *lines]]
        spellings = [normalize(spelling) for spelling in (answer.answer, *answer.also)]
        said = [part for part in parts if any(_says(part, s) for s in spellings)]
        return (said or parts or [None])[0]
    if marked is not None and (numbers := _numbers(normalize(marked), answer.units)):
        return _fresh(numbers, answer.given)[0][0]
    if lines and (alone := _alone(normalize(lines[0]), answer)) is not None:
        return alone
    found = [
        (line, numbers)
        for line in map(normalize, lines)
        if (numbers := _numbers(line, answer.units))
    ]
    if not found:
        return None
    fresh = [each for each in found if any(n[0] not in answer.given for n in each[1])]
    line, numbers = fresh[-1] if fresh else found[0]
    return _conclusion(line, numbers, answer.given)


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
        right = extracted in {_value(spelling) for spelling in spellings}
    else:
        right = any(_says(extracted, normalize(spelling)) for spelling in spellings)
    bare = _bare(reply)
    strict = right and any(bare == _bare(spelling) for spelling in spellings)
    return Grade("correct" if right else "wrong", strict, extracted)


def _marked(text: str) -> str | None:
    """The first line that holds anything after a text's last answer marker
    that has one; None without one."""
    for marker in reversed(list(_MARKER.finditer(text))):
        after = text[marker.end() :].lstrip(" \t:")
        for line in LINE_BREAK.split(after):
            if normalize(line):
                return line
    return None


# A number found in a normalized text: written plainly, and where it starts
# and ends, a unit of the answer's after it included.
_Found = tuple[str, int, int]


def _numbers(text: str, units: tuple[str, ...]) -> list[_Found]:
    """The numbers of a normalized text, in order; a unit of units straight
    after a number goes with it."""
    return [
        (_plain(match), match.start(), match.end())
        for match in re.finditer(_NUMBER + _unit(units), text)
    ]


def _unit(units: tuple[str, ...]) -> str:
    """A pattern for any of units straight after a number, or none."""
    spellings = sorted({normalize(unit) for unit in units}, key=len, reverse=True)
    if not spellings:
        return ""
    either = "|".join(map(re.escape, spellings))
    return rf"(?:\s*(?:{either})(?![A-Za-z0-9]))?"


def _fresh(numbers: list[_Found], given: tuple[str, ...]) -> list[_Found]:
    """Of numbers, those the question does not give; all, if it gives each."""
    return [number for number in numbers if number[0] not in given] or numbers


def _alone(line: str, answer: Answer) -> str | None:
    """A normalized line's number, when the line is that number alone, with a
    unit of the answer's or none and anything but letters and digits around
    it, and the question does not give it; else None."""
    number = _NUMBER + _unit(answer.units)
    match = re.fullmatch(rf"[^\w\-−]*{number}[\W_]*", line)
    if match is None or (plain := _plain(match)) in answer.given:
        return None
    return plain


def _conclusion(line: str, numbers: list[_Found], given: tuple[str, ...]) -> str:
    """The number a normalized line concludes with: the last of its fresh
    numbers before any reason that comes after the first of them."""
    fresh = _fresh(numbers, given)
    reason = _REASON.search(line, fresh[0][2])
    end = len(line) if reason is None else reason.start()
    return [number for number in fresh if number[1] < end][-1][0]


def given_numbers(question: str, units: tuple[str, ...] = ()) -> tuple[str, ...]:
    """The numbers a question gives, written plainly, each once, in order; a
    unit of units straight after a number goes with it.

    Example:
        >>> given_numbers("A bus leaves at 09:00 and arrives at 11:30.")
        ('9', '0', '11', '30')
    """
    found = _numbers(normalize(question), units)
    return tuple(dict.fromkeys(plain for plain, _, _ in found))


def _plain(match: re.Match[str]) -> str:
    """A number written plainly: no thousands commas, no leading zeros, no
    trailing zeros after the point, and "-" for minus but never before 0.
    Each value has one such spelling, so two numbers are equal when their
    plain spellings are; no digits go through int(), which refuses more
    than 4300 of them, as a reply that has collapsed may write."""
    sign, whole, decimals = match.group(1, 2, 3)
    digits = whole.replace(",", "").lstrip("0") or "0"
    decimals = (decimals or "").rstrip("0").removesuffix(".")
    negative = sign and (digits != "0" or decimals)
    return f"{'-' if negative else ''}{digits}{decimals}"


def _value(spelling: str) -> str | None:
    """A spelling that is a number alone, written plainly, else None."""
    match = re.fullmatch(_NUMBER, normalize(spelling))
    return None if match is None else _plain(match)


def _no_article(text: str) -> str:
    """A normalized text without its leading "the", "a" or "an"."""
    for article in ARTICLES:
        if text.startswith(f"{article} "):
            return text[len(article) + 1 :]
    return text


def _says(part: str, spelling: str) -> bool:
    """Whether a normalized part is a normalized spelling, or begins or ends
    with it as whole words, with anything but letters and digits at either
    end, and then a leading article, aside on both."""
    part, spelling = _no_article(_trim(part)), _no_article(_trim(spelling))
    word = re.escape(spelling)
    return (
        part == spelling
        or re.match(rf"{word}(?!\w)", part) is not None
        or re.search(rf"(?<!\w){word}$", part) is not None
    )


def _trim(text: str) -> str:
    """A text without anything but letters and digits at either end."""
    return _ENDS.sub("", text)


def _bare(text: str) -> str:
    """A text trimmed, without one final full stop, in lower case."""
    text = text.strip()
    if text.endswith("."):
        text = text[:-1].rstrip()
    return text.casefold()
