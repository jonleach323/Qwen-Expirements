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
- extract: the first line that holds anything after the reply's last
  "answer:" or "answer is" with anything after it (a colon after "answer
  is" aside); with none, the reply's first line that holds anything. Lines
  break as everywhere in pushlab (pushlab.analysis.markers.LINE_BREAK). A
  number answer then takes the last of that part's numbers that its
  question does not give (all of them, if it gives every one), the last
  written with one of the answer's units if any is; with no number there,
  the reply's only number, taken the same way (one value, however often it
  is written). So "He will be 16 in 3 years." gives 16, not the question's
  3, and "150 minutes (2.5 hours)" gives 150 when the unit is minutes.
- a number is written in the digits 0 to 9, with thousands commas or none,
  with a decimal part or none, after "-" or "−" for minus. Digits straight
  after a letter, a digit or "." belong to a word or another number, so
  H2O holds none and 17-23 holds 17 and 23. A unit of the answer's units
  straight after a number goes with it, so 100 in "6 litres per 100 km" is
  no number when that is the unit. Numbers compare by value: 12.0 is 12.
- correct: for a number answer, the number taken equals the answer; for a
  text answer, the part, without a leading "the", "a" or "an", equals the
  answer or an also spelling, also without one, or begins or ends with it
  as whole words. wrong, otherwise; no_answer when there is nothing to
  compare: an empty reply, or no number for a number answer.
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
    """What grade compares with the answer: for a number answer, the number
    taken, written plainly (1234, -5, 12.5); for a text answer, the
    normalized part; None when there is nothing to compare.

    Example:
        >>> product = Answer("q-1", "arithmetic", "number", "391", (), (), "17 * 23")
        >>> extract("Answer: 391\\nI multiplied 17 by 23.", product)
        '391'
        >>> extract("17 × 23 = 391", product)
        '391'
        >>> age = Answer("q-2", "reasoning", "number", "16", (), (), "13 + 3", ("3",))
        >>> extract("He will be 16 in 3 years.", age)  # not the question's 3
        '16'
    """
    part = _part(reply)
    if answer.type == "text":
        return None if part is None else normalize(part)
    if part is not None and (numbers := _numbers(normalize(part), answer.units)):
        return _taken(numbers, answer.given)[-1]
    found = set(_taken(_numbers(normalize(reply), answer.units), answer.given))
    return found.pop() if len(found) == 1 else None


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


def _part(reply: str) -> str | None:
    """The line of a reply that holds its answer: the first that holds
    anything after its last answer marker that has one, else its first that
    holds anything; None for a reply with nothing in it."""
    text = unicodedata.normalize("NFKC", reply).replace("**", "").replace("__", "")
    for marker in reversed(list(_MARKER.finditer(text))):
        if (line := _first_line(text[marker.end() :].lstrip(" \t:"))) is not None:
            return line
    return _first_line(text)


def _first_line(text: str) -> str | None:
    """The first line of text that holds anything once normalized."""
    for line in LINE_BREAK.split(text):
        if normalize(line):
            return line
    return None


def _numbers(text: str, units: tuple[str, ...]) -> list[tuple[str, bool]]:
    """The numbers of a normalized text, written plainly, in order, each with
    whether a unit of units comes straight after it, which goes with it."""
    spellings = sorted({normalize(unit) for unit in units}, key=len, reverse=True)
    pattern = _NUMBER
    if spellings:
        either = "|".join(map(re.escape, spellings))
        pattern += rf"(\s*(?:{either})(?![A-Za-z0-9]))?"
    return [
        (_plain(match), bool(spellings) and match.group(4) is not None)
        for match in re.finditer(pattern, text)
    ]


def _taken(numbers: list[tuple[str, bool]], given: tuple[str, ...]) -> list[str]:
    """Of numbers, those an answer may be: those not given by the question,
    or all if it gives every one; of those, the ones with a unit, if any."""
    fresh = [number for number in numbers if number[0] not in given] or numbers
    with_unit = [number for number in fresh if number[1]]
    return [plain for plain, _ in with_unit or fresh]


def given_numbers(question: str, units: tuple[str, ...] = ()) -> tuple[str, ...]:
    """The numbers a question gives, written plainly, each once, in order; a
    unit of units straight after a number goes with it.

    Example:
        >>> given_numbers("A bus leaves at 09:00 and arrives at 11:30.")
        ('9', '0', '11', '30')
    """
    return tuple(
        dict.fromkeys(plain for plain, _ in _numbers(normalize(question), units))
    )


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
    with it as whole words, a leading article aside on both."""
    part, spelling = _no_article(part), _no_article(spelling)
    word = re.escape(spelling)
    return (
        part == spelling
        or re.match(rf"{word}(?!\w)", part) is not None
        or re.search(rf"(?<!\w){word}$", part) is not None
    )


def _bare(text: str) -> str:
    """A text trimmed, without one final full stop, in lower case."""
    text = text.strip()
    if text.endswith("."):
        text = text[:-1].rstrip()
    return text.casefold()
