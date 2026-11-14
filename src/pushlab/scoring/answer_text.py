"""How the capability grader reads text and numbers (decision 0022): the
pieces pushlab.scoring.capability builds its rules from.

normalize gives a text as the grader compares it. numbers finds the
numbers of a normalized text, each written plainly (plain), so that two
are equal when their spellings are; value is a spelling that is a number
alone. says decides whether a part of a reply gives a text answer: one of
its clauses, cut at commas, semicolons, colons and spaced dashes, and not
beginning with "not", is the answer, or begins or ends with it as whole
words, punctuation, symbols and a leading article aside, and the answer
is not straight after "not".
"""

import re
import unicodedata

ARTICLES = ("the", "a", "an")
# Straight, curly, low, angle and corner quotes, and the backtick.
_QUOTES = str.maketrans("", "", "\"'`‘’‚‛“”„‟‹›«»「」『』")
# Anything but letters and digits at either end of a text.
_ENDS = re.compile(r"^[\W_]+|[\W_]+$")
# A number's sign, whole part (with thousands commas or none) and decimals.
NUMBER = (
    r"(?<![A-Za-z0-9.])([-−]?)([0-9]{1,3}(?:,[0-9]{3})+(?![0-9])|[0-9]+)"
    r"(\.[0-9]+)?"
)
# "not" just before what follows it.
NOT_BEFORE = re.compile(r"\bnot\s+$")
# Where a part's clauses meet: commas, semicolons, colons, spaced dashes.
_CLAUSES = re.compile(r"[,;:]+\s*|\s+[-–—]\s+")
# A number found in a normalized text: written plainly, and where it starts
# and ends, a unit of the answer's after it included.
Found = tuple[str, int, int]


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


def unit_pattern(units: tuple[str, ...]) -> str:
    """A pattern for any of units straight after a number, or none.

    Example:
        >>> unit_pattern(())
        ''
    """
    spellings = sorted({normalize(unit) for unit in units}, key=len, reverse=True)
    if not spellings:
        return ""
    either = "|".join(map(re.escape, spellings))
    return rf"(?:\s*(?:{either})(?![A-Za-z0-9]))?"


def numbers(text: str, units: tuple[str, ...] = ()) -> list[Found]:
    """The numbers of a normalized text, in order; a unit of units straight
    after a number goes with it. Digits straight after a letter, a digit or
    "." belong to a word or another number.

    Example:
        >>> [plain for plain, _, _ in numbers("17x23 = 1,234 h2o -5")]
        ['17', '1234', '-5']
        >>> numbers("6 litres per 100 km", ("litres per 100 km",))
        [('6', 0, 19)]
    """
    return [
        (plain(match), match.start(), match.end())
        for match in re.finditer(NUMBER + unit_pattern(units), text)
    ]


def plain(match: re.Match[str]) -> str:
    """A number written plainly: no thousands commas, no leading zeros, no
    trailing zeros after the point, and "-" for minus but never before 0.
    Each value has one such spelling, so two numbers are equal when their
    plain spellings are; no digits go through int(), which refuses more
    than 4300 of them, as a reply that has collapsed may write.

    Example:
        >>> plain(re.fullmatch(NUMBER, "−0012.50"))
        '-12.5'
    """
    sign, whole, decimals = match.group(1, 2, 3)
    digits = whole.replace(",", "").lstrip("0") or "0"
    decimals = (decimals or "").rstrip("0").removesuffix(".")
    negative = sign and (digits != "0" or decimals)
    return f"{'-' if negative else ''}{digits}{decimals}"


def value(spelling: str) -> str | None:
    """A spelling that is a number alone, written plainly, else None.

    Example:
        >>> value("1,200.0"), value("12 cm")
        ('1200', None)
    """
    match = re.fullmatch(NUMBER, normalize(spelling))
    return None if match is None else plain(match)


def trim(text: str) -> str:
    """A text without anything but letters and digits at either end.

    Example:
        >>> trim("*Paris!* 🗼")
        'Paris'
    """
    return _ENDS.sub("", text)


def no_article(text: str) -> str:
    """A normalized text without its leading "the", "a" or "an".

    Example:
        >>> no_article("the pacific")
        'pacific'
    """
    for article in ARTICLES:
        if text.startswith(f"{article} "):
            return text[len(article) + 1 :]
    return text


def _whole(part: str, spelling: str) -> bool:
    """Whether a trimmed part is spelling, or begins with it, or ends with it
    not straight after "not", as whole words, articles aside."""
    part, spelling = no_article(trim(part)), no_article(trim(spelling))
    if not spelling:
        return False
    if part == spelling or re.match(rf"{re.escape(spelling)}(?!\w)", part):
        return True
    end = re.search(rf"(?<!\w){re.escape(spelling)}$", part)
    return end is not None and NOT_BEFORE.search(part[: end.start()]) is None


def says(part: str, spelling: str) -> bool:
    """Whether a normalized part of a reply gives the normalized spelling:
    one of its clauses that does not begin with "not" is it, or begins with
    it, or ends with it not straight after "not", as whole words,
    punctuation, symbols and a leading article aside.

    Example:
        >>> says("it's paris, darling!", "paris"), says("lyon, not paris!", "paris")
        (True, False)
        >>> says("charlotte brontë, not jane austen", "austen")
        False
    """
    return any(
        _whole(clause, spelling)
        for clause in _CLAUSES.split(part)
        if trim(clause) and not re.match(r"\W*not\b", clause)
    )
