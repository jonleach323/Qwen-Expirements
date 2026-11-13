"""Coherence measures that need no judge, on a reply's text (decision 0022).

A reply is read as units: its words, lowercase runs of letters and digits;
or, when most of its letters are Chinese, Japanese or Korean characters,
which leave no space between words, each such character as a unit, beside
any other words. Then:
- repetition: the share of its n-grams of units (configs/panel.yaml's
  ngram, 4) that it already said earlier in the reply; 0 for a reply too
  short to have one. A reply that loops comes close to 1.
- word variety: the share of distinct units in each window of window units
  (50), averaged over every window, the moving-average type-token ratio; a
  reply shorter than a window counts its distinct units over all of them.
  A plain ratio of distinct to all units falls as any text grows, so long
  fluent text would read as poor; a window's does not.
A reply is flagged "repetitive" above repetitive_above and "low variety"
below varied_below. The third measure, perplexity under the unsteered
model, needs the model (pushlab.engine.score_text).
"""

import re
from collections import Counter
from dataclasses import dataclass

from pushlab.analysis.panel import Panel

# Raised whenever a change to the units or the measures would move a value.
TEXT_MEASURES_VERSION = 1
# Chinese, Japanese and Korean characters, which leave no space between words.
_CJK_CLASS = "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff"
_CJK = re.compile(f"[{_CJK_CLASS}]")
_WORD = re.compile(r"[^\W_]+")
_CJK_OR_WORD = re.compile(f"[{_CJK_CLASS}]|[^\\W_{_CJK_CLASS}]+")


@dataclass(frozen=True)
class TextMeasures:
    """A reply's units, repetition rate and word variety."""

    units: int
    repetition: float
    variety: float


def text_units(text: str) -> list[str]:
    """A text's units: its lowercase words, or, when most of its letters are
    Chinese, Japanese or Korean characters, each such character and each
    other word.

    Example:
        >>> text_units("Hello, hello world!")
        ['hello', 'hello', 'world']
        >>> text_units("你好，世界")
        ['你', '好', '世', '界']
    """
    lowered = text.casefold()
    words = _WORD.findall(lowered)
    letters = sum(len(word) for word in words)
    if len(_CJK.findall(lowered)) * 2 <= letters:
        return words
    return _CJK_OR_WORD.findall(lowered)


def repetition(units: list[str], n: int = 4) -> float:
    """The share of the n-grams of units that came earlier in them; 0 when
    there are fewer than n units.

    Example:
        >>> repetition("a b c d a b c d".split(), 2)  # ab bc cd da: ab bc cd again
        0.42857142857142855
        >>> repetition("the cat sat on the mat".split(), 2)
        0.0
    """
    grams = [tuple(units[i : i + n]) for i in range(len(units) - n + 1)]
    if not grams:
        return 0.0
    seen: set[tuple[str, ...]] = set()
    again = 0
    for gram in grams:
        if gram in seen:
            again += 1
        seen.add(gram)
    return again / len(grams)


def variety(units: list[str], window: int = 50) -> float:
    """The moving-average share of distinct units in windows of window units:
    each window's distinct units over window, averaged; for fewer units
    than a window, their distinct ones over all; 0 for none.

    Example:
        >>> variety("a b a b".split(), 2)  # ab, ba, ab: 2 distinct of 2 each
        1.0
        >>> variety("a a a b".split(), 2)  # aa, aa, ab: 1, 1 and 2 of 2
        0.6666666666666666
    """
    if not units:
        return 0.0
    if len(units) <= window:
        return len(set(units)) / len(units)
    counts = Counter(units[:window])
    total = len(counts)
    for i in range(window, len(units)):
        leaving, coming = units[i - window], units[i]
        counts[leaving] -= 1
        if not counts[leaving]:
            del counts[leaving]
        counts[coming] += 1
        total += len(counts)
    return total / (len(units) - window + 1) / window


def measure_text(text: str, panel: Panel) -> TextMeasures | None:
    """A reply's measures at the panel's ngram and window; None for a reply
    with no unit to measure.

    Example:
        >>> from pushlab.analysis.panel import read_panel
        >>> loop = "I am a robot. " * 20
        >>> measured = measure_text(loop, read_panel())
        >>> measured.units, round(measured.repetition, 2), round(measured.variety, 2)
        (80, 0.95, 0.08)
    """
    units = text_units(text)
    if not units:
        return None
    return TextMeasures(
        len(units), repetition(units, panel.ngram), variety(units, panel.window)
    )


def text_flags(measures: TextMeasures, panel: Panel) -> tuple[str, ...]:
    """What the measures flag: "repetitive" above the panel's
    repetitive_above, "low variety" below its varied_below.

    Example:
        >>> from pushlab.analysis.panel import read_panel
        >>> text_flags(TextMeasures(80, 0.96, 0.08), read_panel())
        ('repetitive', 'low variety')
        >>> text_flags(TextMeasures(60, 0.0, 0.8), read_panel())
        ()
    """
    flags = []
    if measures.repetition > panel.repetitive_above:
        flags.append("repetitive")
    if measures.variety < panel.varied_below:
        flags.append("low variety")
    return tuple(flags)
