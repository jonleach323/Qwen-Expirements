"""Markers: where in a reply the pushes switch on and off, as short text.

A reply's steps (pushlab.store.steps) name the units pushed on the pass that
chose each token: step 0 is the prompt pass, step j the pass that read reply
token j - 1, so a push on reply token k shows at step k + 1 (decision 0008).
Each unit gets a letter (letters), and each step the set of letters pushed
on it (active_sets); a control's random neuron takes the letter of the unit
it stands for. marks puts a marker before each step whose set differs from
the step before: [A on], [A off, B on], [A+B on]. A long alternation of
short turns is one marker with the number of its turns, [A/B x8] or
[A on/off x8], so a report stays short enough for a phone; turns longer
than LONGEST_TURN steps keep a marker each, so every switch shows. legend
says what each letter is, places finds each marker's place in the reply's
text, and with_markers writes the reply for a Markdown report, each marker
in bold.

Nothing here loads a model: places takes the tokenizer's decode as a
function.
"""

import html
import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from itertools import pairwise
from typing import Any

# The fewest turns a long alternation needs to become one marker.
COLLAPSE = 4
# The most steps a turn of such an alternation may last, the reply's last
# turn aside (its end may cut it): longer turns keep a marker each, so a
# report shows where each one starts.
LONGEST_TURN = 16

# Every kind of line break str.splitlines knows, \r\n as one.
LINE_BREAK = re.compile(r"\r\n|[\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029]")
# What Markdown (CommonMark and GitHub's) could read as formatting inside a
# line: escapes, code, emphasis, links and images, strikethrough, tables,
# headings and math. html.escape takes care of <, > and &.
ACTIVE = re.compile(r"[\\`*_\[\]()!~|#$]")


def _letter(index: int) -> str:
    """The index-th letter name, from 0: A to Z, then AA, AB and so on."""
    name = ""
    index += 1
    while index:
        index, rest = divmod(index - 1, 26)
        name = chr(ord("A") + rest) + name
    return name


def letters(units: Iterable[str]) -> dict[str, str]:
    """Each unit's letter, in order of first appearance: A to Z, then AA, AB...

    Example:
        >>> letters(["tiny-vl:L2:N100", "tiny-vl:L1:N3", "tiny-vl:L2:N100"])
        {'tiny-vl:L2:N100': 'A', 'tiny-vl:L1:N3': 'B'}
    """
    return {unit: _letter(i) for i, unit in enumerate(dict.fromkeys(units))}


def legend(letter_of: Mapping[str, str]) -> str:
    """One line saying which unit each letter is; empty when there is none.

    Example:
        >>> legend({"tiny-vl:L2:N100": "A", "tiny-vl:L1:N3": "B"})
        'A = tiny-vl:L2:N100 · B = tiny-vl:L1:N3'
    """
    ordered = sorted(letter_of.items(), key=lambda item: _by_letter(item[1]))
    return " · ".join(f"{letter} = {unit}" for unit, letter in ordered)


def stand_ins(schedule: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    """Each unit a run's schedule pushes, and the unit whose letter it takes.

    A control's random neuron stands for the unit its segment's control_for
    names, which the run it mirrors pushes; every other unit for itself.

    Example:
        >>> stand_ins([{"unit": "tiny-vl:L2:N7", "control_for": "tiny-vl:L2:N100"}])
        {'tiny-vl:L2:N7': 'tiny-vl:L2:N100'}
    """
    return {
        segment["unit"]: segment.get("control_for") or segment["unit"]
        for segment in schedule
    }


def active_sets(
    steps: Iterable[Sequence[str]],
    letter_of: Mapping[str, str],
    stands_for: Mapping[str, str] | None = None,
) -> list[frozenset[str]]:
    """Each step's active units as a set of letters, in step order.

    steps holds each step's active_units. A unit that stands for another
    (stand_ins) takes that one's letter; ValueError names a unit with none.

    Example:
        >>> active_sets([(), ("tiny-vl:L2:N100",)], {"tiny-vl:L2:N100": "A"})
        [frozenset(), frozenset({'A'})]
    """
    stands_for = stands_for or {}
    sets = []
    for units in steps:
        found = set()
        for unit in units:
            letter = letter_of.get(stands_for.get(unit, unit))
            if letter is None:
                raise ValueError(f"{unit} has no letter")
            found.add(letter)
        sets.append(frozenset(found))
    return sets


def _by_letter(letter: str) -> tuple[int, str]:
    return len(letter), letter  # A to Z, then AA


def _joined(group: AbstractSet[str]) -> str:
    return "+".join(sorted(group, key=_by_letter))


def _change(before: AbstractSet[str], now: AbstractSet[str]) -> str:
    """A marker for one change: the units switched off, then those switched on."""
    parts = []
    if off := before - now:
        parts.append(f"{_joined(off)} off")
    if on := now - before:
        parts.append(f"{_joined(on)} on")
    return f"[{', '.join(parts)}]"


def _turns(
    before: AbstractSet[str], first: AbstractSet[str], second: AbstractSet[str], n: int
) -> str:
    """A marker for n stretches taking turns between first and second.

    Units on before and in neither set switch off, and units in both switch
    on, before the turns: [C on, A/B x8]. When one set holds the other, the
    units in one only repeat on and off: [A on/off x8].
    """
    parts = []
    if off := before - first - second:
        parts.append(f"{_joined(off)} off")
    if on := (first & second) - before:
        parts.append(f"{_joined(on)} on")
    mine, theirs = first - second, second - first
    if mine and theirs:
        parts.append(f"{_joined(mine)}/{_joined(theirs)} x{n}")
    elif mine:
        parts.append(f"{_joined(mine)} on/off x{n}")
    else:
        parts.append(f"{_joined(theirs)} off/on x{n}")
    return f"[{', '.join(parts)}]"


def marks(
    sets: Sequence[AbstractSet[str]], collapse: int = COLLAPSE
) -> list[tuple[int, str]]:
    """The markers of a reply: (step, text) before each step whose set of
    active units differs from the step before's.

    Before the reply nothing is active, so a push on the prompt pass gives a
    marker at step 0. The text names the units switched off, then those
    switched on: [A on], [A off, B on], [A+B on]. A stretch is a run of steps
    with one set; at least collapse stretches in a row that take turns
    between two sets, each at most LONGEST_TURN steps long (the reply's last
    one may be longer), make one marker, at the first, with their number:
    [A/B x8], or [A on/off x8] for a unit that repeats on and off. A longer
    stretch keeps its own marker. The marker after an alternation compares
    with its last stretch.

    Example:
        >>> on = frozenset("A")
        >>> marks([frozenset(), on, on, frozenset()])
        [(1, '[A on]'), (3, '[A off]')]
        >>> marks([frozenset("A"), frozenset("B")] * 4)
        [(0, '[A/B x8]')]
        >>> a, b = [frozenset("A")] * 20, [frozenset("B")] * 20
        >>> [step for step, _ in marks(a + b + a + b)]  # turns of 20 steps
        [0, 20, 40, 60]
    """
    if not isinstance(collapse, int) or collapse < 2:
        raise ValueError(f"collapse must be a whole number from 2, not {collapse!r}")
    stretches: list[tuple[int, frozenset[str]]] = []
    for step, active in enumerate(sets):
        active = frozenset(active)
        if active != (stretches[-1][1] if stretches else frozenset()):
            stretches.append((step, active))
    # Whether each stretch may be a turn of a collapsed alternation.
    short = [
        later - start <= LONGEST_TURN for (start, _), (later, _) in pairwise(stretches)
    ]
    short.append(True)  # the reply's last stretch, which its end may cut
    found: list[tuple[int, str]] = []
    before: frozenset[str] = frozenset()
    i = 0
    while i < len(stretches):
        end = i  # stretches i to end - 1 take turns between two sets
        while (
            end < len(stretches)
            and short[end]
            and (end - i < 2 or stretches[end][1] == stretches[end - 2][1])
        ):
            end += 1
        step, active = stretches[i]
        if end - i >= collapse:
            found.append((step, _turns(before, active, stretches[i + 1][1], end - i)))
            before, i = stretches[end - 1][1], end
        else:
            found.append((step, _change(before, active)))
            before, i = active, i + 1
    return found


def places(
    text: str,
    tokens: Sequence[int],
    marked: Iterable[tuple[int, str]],
    decode: Callable[[Sequence[int]], str],
) -> list[tuple[int, str]]:
    """Each marker's place in a reply's text: (character position, text).

    A marker before step j goes where text and decode(tokens[:j]) stop
    agreeing, so it never splits a character: before a token that holds
    part of one, it goes before the whole character. decode is the
    tokenizer's, as the reply's text was made, without special tokens.

    Example:
        >>> places(
        ...     "Hi you", [72, 105, 32], [(2, "[A on]")], lambda t: bytes(t).decode()
        ... )
        [(2, '[A on]')]
    """
    found = []
    for step, marker in marked:
        if not 0 <= step <= len(tokens):
            raise ValueError(f"step {step} is past the reply's {len(tokens)} tokens")
        before = decode(list(tokens[:step]))
        found.append((len(os.path.commonprefix([text, before])), marker))
    return found


def markdown_text(text: str) -> str:
    """text made safe for a line of Markdown, and kept readable.

    It is HTML-escaped, Markdown's active characters are escaped with a
    backslash, and every kind of line break becomes <br>, so the text shows
    as it is and stays on its line.

    Example:
        >>> print(markdown_text("2 * 3 <is> [six]\\n"))
        2 \\* 3 &lt;is&gt; \\[six\\]<br>
    """
    text = ACTIVE.sub(lambda match: "\\" + match[0], html.escape(text, quote=False))
    return LINE_BREAK.sub("<br>", text)


def _bold(marker: str) -> str:
    """A marker in bold inside its brackets, [**A on**]: CommonMark does not
    make **[A on]** bold right after a letter, as markers often are."""
    return f"[**{markdown_text(marker.removeprefix('[').removesuffix(']'))}**]"


def with_markers(text: str, placed: Iterable[tuple[int, str]]) -> str:
    """A reply as Markdown: its text escaped (markdown_text), with each marker
    in bold at its character position, such as Hello[**A on**] world.

    Markers at one position keep their order.

    Example:
        >>> with_markers("Hello world", [(5, "[A on]")])
        'Hello[**A on**] world'
    """
    pieces, last = [], 0
    for position, marker in sorted(placed, key=lambda place: place[0]):
        if not 0 <= position <= len(text):
            raise ValueError(
                f"a marker's position must be from 0 to {len(text)}, not {position}"
            )
        pieces += [markdown_text(text[last:position]), _bold(marker)]
        last = position
    return "".join([*pieces, markdown_text(text[last:])])
