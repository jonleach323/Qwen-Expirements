"""A reply's sentences, their windows and their tokens (pushlab.scoring.sentences)."""

import doctest
import hashlib
import json

import pytest

from pushlab.analysis import markers
from pushlab.scoring import sentences as sentences_module
from pushlab.scoring.sentences import (
    ABBREVIATIONS,
    SPLITTER_VERSION,
    sentence_windows,
    split_sentences,
    window_steps,
)

# Fixed examples and the sentences version 1 cuts them into, of every rule.
EXAMPLES = {
    "Three ideas:\n1. Walk.\n2. Read a book.\n- Call a friend": [
        "Three ideas:",
        "\n1. Walk.",
        "\n2. Read a book.",
        "\n- Call a friend",
    ],
    "It costs 3.5 dollars. Pay now.": ["It costs 3.5 dollars.", " Pay now."],
    "Ask Dr. Lee, e.g. at 5 p.m. today. She knows.": [
        "Ask Dr. Lee, e.g. at 5 p.m. today.",
        " She knows.",
    ],
    "Line one\nLine two\n\nLine three.": ["Line one", "\nLine two", "\n\nLine three."],
    "你好。我是助手！你呢？": ["你好。", "我是助手！", "你呢？"],
    'He said "Stop!" Then he left. Wow!! Really?': [
        'He said "Stop!"',
        " Then he left.",
        " Wow!!",
        " Really?",
    ],
    "J. R. R. Tolkien wrote it.": ["J. R. R. Tolkien wrote it."],
    "The sea is calm and the": ["The sea is calm and the"],
    "Wait... What? Fine.\n": ["Wait...", " What?", " Fine.\n"],
    "Hi. Bye.": ["Hi.", " Bye."],
    "": [],
    "  \n ": [],
    # Closing brackets after an end, Latin and CJK.
    "He left (at last.) Then it rained.": ["He left (at last.)", " Then it rained."],
    "See [the note.] It helps.": ["See [the note.]", " It helps."],
    "他说「好。」我走了。": ["他说「好。」", "我走了。"],
    "（真的！）然后呢？": ["（真的！）", "然后呢？"],
    # Line breaks: \r\n, \r alone and the Unicode line separator.
    "One\r\nTwo": ["One", "\r\nTwo"],
    "One\rTwo": ["One", "\rTwo"],
    "One\u2028Two": ["One", "\u2028Two"],
    # "…" and "...", which end no sentence before a lowercase letter.
    "Wait… What?": ["Wait…", " What?"],
    "Hmm… and so... on. Done.": ["Hmm… and so... on.", " Done."],
    # A list's numbers at a line's start, after \r\n.
    "Steps:\r\n1. Walk.\r\n2. Run.": ["Steps:", "\r\n1. Walk.", "\r\n2. Run."],
}
# Every listed abbreviation, before a capital: its "." ends no sentence.
ABBREVIATED = "Mr Mrs Ms Dr Prof Sr Jr St Mt Gen Col Capt Lt Sgt Rev Hon vs cf e.g i.e"
EXAMPLES |= {
    f"Ask {word}. Lee. Then go.": [f"Ask {word}. Lee.", " Then go."]
    for word in ABBREVIATED.split()
}
# The pin of version 1: the SHA-256 of every example's spans and of the
# rule's own parts, which are its abbreviations (sorted), its closing quotes
# and brackets, the patterns of its two kinds of end (_END and _CJK_END) and
# that of the line breaks it knows (markers.LINE_BREAK). The rule's code is
# pinned only through the examples' spans. A change to the rule that changes
# the pin needs SPLITTER_VERSION 2 and that version's value here; a new
# example changes it too, and then it is recomputed from the unchanged code.
PIN_SHA256 = {1: "6293f06e8ee9a362c255918e1b11391ae1bf75737b001a4eb27b26ea50bdba10"}


def _pin() -> str:
    """The SHA-256 of every example's spans and of the rule's own parts."""
    spans = {
        text: [(s.start, s.end) for s in split_sentences(text)] for text in EXAMPLES
    }
    rule = [
        sorted(ABBREVIATIONS),
        sentences_module._CLOSING,
        sentences_module._END.pattern,
        sentences_module._CJK_END.pattern,
        markers.LINE_BREAK.pattern,
    ]
    pinned = json.dumps(
        {"rule": rule, "spans": spans}, ensure_ascii=False, sort_keys=True
    )
    return hashlib.sha256(pinned.encode()).hexdigest()


@pytest.mark.parametrize(("text", "expected"), list(EXAMPLES.items()))
def test_the_fixed_examples_split_as_version_1_says(
    text: str, expected: list[str]
) -> None:
    sentences = split_sentences(text)
    assert [s.text for s in sentences] == expected
    assert "".join(s.text for s in sentences) == (text if expected else "")
    assert [s.index for s in sentences] == list(range(len(expected)))
    for sentence in sentences:
        assert text[sentence.start : sentence.end] == sentence.text


def test_version_1_is_pinned_to_its_spans_and_its_rule() -> None:
    assert SPLITTER_VERSION == 1
    assert {word.casefold() for word in ABBREVIATED.split()} == ABBREVIATIONS
    assert _pin() == PIN_SHA256[SPLITTER_VERSION]


def test_windows_tile_the_sentences_in_groups_of_their_size() -> None:
    four = "One. Two. Three. Four."
    assert [(w.first, w.count) for w in sentence_windows(four, 2)] == [(0, 2), (2, 2)]
    five = four + " Five."
    assert [(w.first, w.count) for w in sentence_windows(five, 2)] == [
        (0, 2),
        (2, 2),
        (4, 1),
    ]
    ones = sentence_windows(five)
    assert [w.text for w in ones] == ["One.", " Two.", " Three.", " Four.", " Five."]
    assert "".join(w.text for w in sentence_windows(five, 3)) == five
    assert sentence_windows("  ", 2) == []


@pytest.mark.parametrize("size", [0, -1, 1.0, True])
def test_a_window_of_no_sentences_is_refused(size: object) -> None:
    with pytest.raises(ValueError, match="at least one sentence"):
        sentence_windows("One. Two.", size)  # type: ignore[arg-type]


def test_a_token_belongs_to_the_window_its_text_starts_in() -> None:
    pieces = ["Walk", ".\n", "2", ".", " Read"]
    text = "".join(pieces)

    def decode(tokens) -> str:  # type: ignore[no-untyped-def]
        return "".join(pieces[token] for token in tokens)

    windows = sentence_windows(text)
    assert [w.text for w in windows] == ["Walk.", "\n2. Read"]
    assert window_steps(text, [0, 1, 2, 3, 4], decode, windows) == [(0, 2), (2, 5)]


def test_a_character_split_across_tokens_goes_with_its_first_byte(
    tiny_tokenizer,  # type: ignore[no-untyped-def]
) -> None:
    text = "你好。我"
    tokens = tiny_tokenizer.encode(text, add_special_tokens=False)
    assert len(tokens) == 12  # a token per byte

    def decode(ids) -> str:  # type: ignore[no-untyped-def]
        return tiny_tokenizer.decode(list(ids), skip_special_tokens=True)

    windows = sentence_windows(text)
    assert window_steps(text, tokens, decode, windows) == [(0, 9), (9, 12)]


def test_the_spans_tile_the_tokens_and_a_window_may_hold_none() -> None:
    text = "One. Two. Three."
    tokens = list(text.encode())
    steps = window_steps(
        text, tokens, lambda t: bytes(t).decode(), sentence_windows(text)
    )
    assert steps == [(0, 4), (4, 9), (9, 16)]
    assert window_steps("", [], lambda t: "", []) == []
    # A token that crosses a CJK sentence's end belongs to the window it
    # starts in, so the next window holds no token.
    pieces = ["请", "访问", "。www"]
    text = "".join(pieces)

    def decode(tokens) -> str:  # type: ignore[no-untyped-def]
        return "".join(pieces[token] for token in tokens)

    windows = sentence_windows(text)
    assert [w.text for w in windows] == ["请访问。", "www"]
    assert window_steps(text, [0, 1, 2], decode, windows) == [(0, 3), (3, 3)]


def test_docstring_examples_run() -> None:
    result = doctest.testmod(sentences_module)
    assert result.attempted > 0
    assert result.failed == 0
