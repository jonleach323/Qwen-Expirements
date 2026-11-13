"""A reply's sentences, windows of them, and the reply tokens each one spans.

split_sentences cuts a reply into sentences whose spans tile its text: the
space before a sentence starts it, as Qwen's tokens carry a word's space,
and a blank text has none. Version 1 of the rule (SPLITTER_VERSION), which
a change that moves any span must raise:
- a line break ends a sentence (pushlab.analysis.markers.LINE_BREAK);
- so do ".", "!", "?" and "…", alone or repeated, with the closing quotes
  and brackets after them, before a space and a character that is not a
  lowercase letter; but not a "." after a number that starts its line (a
  list's "1."), after a single capital (an initial, "J."), or after a
  listed abbreviation ("Dr.", "e.g.");
- so do "。", "！" and "？", with or without a space after them.
A piece of only space joins the sentence before it.

sentence_windows groups the sentences into windows of `size`, one after
another, the last one shorter if the sentences run out; a judge sees only
a window's own sentences (decision 0021). window_steps gives each window's
reply tokens: a token belongs to the window its text starts in.
"""

import os
import re
from bisect import bisect_left
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pushlab.analysis.markers import LINE_BREAK

SPLITTER_VERSION = 1
# Words before a "." that is not a sentence's end, lowercase, without it.
ABBREVIATIONS = frozenset(
    "mr mrs ms dr prof sr jr st mt gen col capt lt sgt rev hon vs cf e.g i.e".split()
)
_CLOSING = "\"'”’»)]}」』）"
# A Latin sentence's end, before a space and the next character, captured.
_END = re.compile(rf"(?:[.!?]+|…+)[{re.escape(_CLOSING)}]*(?=\s+(\S))")
# A CJK sentence's end, which needs no space after it.
_CJK_END = re.compile(rf"[。！？]+[{re.escape(_CLOSING)}]*")


@dataclass(frozen=True)
class Sentence:
    """A sentence of a text: its place, from 0, and its characters,
    text[start:end]."""

    index: int
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Window:
    """Sentences first to first + count - 1 of a text, characters start to
    end, and their text."""

    first: int
    count: int
    start: int
    end: int
    text: str


def _not_an_end(text: str, dot: int) -> bool:
    """Whether the "." at dot follows a list number at a line's start, an
    initial or an abbreviation, and so ends no sentence."""
    word = re.search(r"(\S+)$", text[:dot])
    if word is None:
        return False
    token = word[1]
    line_start = max(
        (match.end() for match in LINE_BREAK.finditer(text, 0, word.start(1))),
        default=0,
    )
    if token.isdigit() and not text[line_start : word.start(1)].strip():
        return True  # a list's "1."
    bare = token.lstrip("\"'“‘([{")
    if len(bare) == 1 and bare.isupper():
        return True  # an initial
    return bare.casefold() in ABBREVIATIONS


def _cuts(text: str) -> list[int]:
    """Every place a sentence may start, before checking for empty pieces."""
    cuts = [match.start() for match in LINE_BREAK.finditer(text)]
    for match in _END.finditer(text):
        if match[1].islower():
            continue
        if match[0].rstrip(_CLOSING) == "." and _not_an_end(text, match.start()):
            continue
        cuts.append(match.end())
    cuts += [match.end() for match in _CJK_END.finditer(text)]
    return sorted(set(cuts))


def split_sentences(text: str) -> list[Sentence]:
    """The sentences of text, by version 1 of the rule; none for a blank one.

    Example:
        >>> [s.text for s in split_sentences("Hi. Bye.")]
        ['Hi.', ' Bye.']
        >>> [s.text for s in split_sentences("Ask Dr. Lee.\\n1. Walk.")]
        ['Ask Dr. Lee.', '\\n1. Walk.']
    """
    if not text.strip():
        return []
    starts = [0]
    for cut in _cuts(text):
        if cut > starts[-1] and cut < len(text) and text[starts[-1] : cut].strip():
            starts.append(cut)
    if len(starts) > 1 and not text[starts[-1] :].strip():
        starts.pop()  # only space is left: it ends the sentence before it
    ends = [*starts[1:], len(text)]
    return [
        Sentence(index, start, end, text[start:end])
        for index, (start, end) in enumerate(zip(starts, ends, strict=True))
    ]


def sentence_windows(text: str, size: int = 1) -> list[Window]:
    """The windows of `size` sentences that tile text, in order.

    Example:
        >>> [(w.first, w.count) for w in sentence_windows("Go. Run. Stop.", size=2)]
        [(0, 2), (2, 1)]
    """
    if type(size) is not int or size < 1:
        raise ValueError(f"a window holds at least one sentence, not {size!r}")
    sentences = split_sentences(text)
    windows = []
    for first in range(0, len(sentences), size):
        group = sentences[first : first + size]
        start, end = group[0].start, group[-1].end
        windows.append(Window(first, len(group), start, end, text[start:end]))
    return windows


def window_steps(
    text: str,
    tokens: Sequence[int],
    decode: Callable[[Sequence[int]], str],
    windows: Sequence[Window],
) -> list[tuple[int, int]]:
    """Each window's reply tokens, (start_step, end_step), end not included.

    Step j is the reply's token j. A window starts at the first step whose
    text starts at or after the window's first character, the step where
    text and decode(tokens[:step]) stop agreeing, as markers.places finds
    it; the first window starts at 0 and the last ends at len(tokens), so
    the spans tile the reply's tokens. decode is the tokenizer's, without
    special tokens, as the reply's text was made.

    Example:
        >>> text = "Hi. Bye."
        >>> steps = window_steps(
        ...     text,
        ...     list(text.encode()),
        ...     lambda t: bytes(t).decode(),
        ...     sentence_windows(text),
        ... )
        >>> steps
        [(0, 3), (3, 8)]
    """
    if not windows:
        return []
    places: list[int] = []
    for step in range(len(tokens) + 1):
        before = decode(list(tokens[:step]))
        place = len(os.path.commonprefix([text, before]))
        places.append(max(place, places[-1]) if places else place)
    starts = [0] + [
        min(bisect_left(places, window.start), len(tokens)) for window in windows[1:]
    ]
    ends = [*starts[1:], len(tokens)]
    return list(zip(starts, ends, strict=True))
