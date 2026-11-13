"""A reply's sentences, their windows and their tokens (pushlab.scoring.sentences)."""

import doctest
import hashlib
import json

import pytest

from pushlab.scoring import sentences as sentences_module
from pushlab.scoring.sentences import (
    SPLITTER_VERSION,
    sentence_windows,
    split_sentences,
    window_steps,
)

# Fixed examples and the sentences version 1 cuts them into.
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
}
# SHA-256 of every example's spans: a change to the rule that moves any span
# changes it, and needs SPLITTER_VERSION 2.
SPANS_SHA256 = {1: "8b7ab9a7ac198c89487ab5729d314b5f916a2b19cdf946b1ae0ddbc446ec445c"}


def _spans() -> str:
    spans = {
        text: [(s.start, s.end) for s in split_sentences(text)] for text in EXAMPLES
    }
    return hashlib.sha256(json.dumps(spans, ensure_ascii=False).encode()).hexdigest()


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


def test_version_1_is_pinned_to_its_spans() -> None:
    assert SPLITTER_VERSION == 1
    assert _spans() == SPANS_SHA256[SPLITTER_VERSION]


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


def test_spans_tile_the_tokens_and_no_window_has_none() -> None:
    text = "One. Two. Three."
    tokens = list(text.encode())
    steps = window_steps(
        text, tokens, lambda t: bytes(t).decode(), sentence_windows(text)
    )
    assert steps == [(0, 4), (4, 9), (9, 16)]
    assert window_steps("", [], lambda t: "", []) == []


def test_docstring_examples_run() -> None:
    result = doctest.testmod(sentences_module)
    assert result.attempted > 0
    assert result.failed == 0
