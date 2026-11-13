"""A labeling sheet's sample and files (pushlab.scoring.label_sheet): the
share rule, the seeded sample, the formula guard, the sample's token and
the item IDs, labels.csv and instructions.md; test_label_check_outputs.py
has the disagreements file.

The replies are invented, in runs made by hand; test_label_export.py
samples runner folders that the real runner saved.
"""

import csv
import doctest
import hashlib
import io
import re
from dataclasses import replace

import pytest

from pushlab.scoring import label_sheet
from pushlab.scoring.label_sheet import (
    LabelItem,
    distinct,
    guard,
    instructions,
    item_ids,
    sample,
    sample_token,
    shares,
    sheet_csv,
)
from pushlab.scoring.personas import Persona
from pushlab.scoring.rubrics import RUBRIC_NAMES, load_rubric
from pushlab.store.records import RunRecord

OWLS = [
    Persona("Brass Owl", "A pompous mechanical owl\nwho lectures in riddles."),
    Persona("Snow Owl"),
]


def _item(number: int, condition: str, text: str, control: bool = False) -> LabelItem:
    """Run number of the experiment x, of the condition given, replying text."""
    run = RunRecord(
        run_id=f"x-{number:06d}",
        experiment="x",
        model="tiny-vl@main",
        prompt_id="p",
        schedule=[],
        decoding={
            "strategy": "greedy",
            "max_new_tokens": 8,
            "thinking": False,
            "dtype": "float32",
        },
        seed=0,
        batch_key="b000001:0",
        control_of=f"x-{number - 1:06d}" if control else "",
        output_text=f"  {text}\n",
        output_tokens=[1, 2, 3],
        status="ok",
    )
    return LabelItem("x.yaml", "x", 1, run, condition, text)


def _items() -> list[LabelItem]:
    """Five unpushed replies, five pulses and five controls, each its own."""
    return [
        _item(3 * k + place + 1, condition, f"Reply {k} of kind {place}.", place == 2)
        for k in range(5)
        for place, condition in enumerate(["none", "pulse", "pulse"])
    ]


def test_shares_are_equal_and_the_remainder_goes_to_the_first_strata() -> None:
    assert shares([5, 5, 5, 5, 5], 12) == [3, 3, 2, 2, 2]
    assert shares([9, 9], 9) == [5, 4]
    assert shares([7, 7, 7], 0) == [0, 0, 0]


def test_a_stratum_smaller_than_its_share_is_taken_whole() -> None:
    assert shares([5, 1, 5, 5, 5], 12) == [3, 1, 3, 3, 2]
    assert shares([2, 9, 9], 12) == [2, 5, 5]
    # Once the two 1s are taken whole, the 3 is below the new share, 13 / 3.
    assert shares([1, 1, 3, 9, 9], 15) == [1, 1, 3, 5, 5]


def test_with_fewer_replies_than_asked_every_one_is_taken() -> None:
    assert shares([2, 3], 10) == [2, 3]
    assert shares([4, 0], 2) == [2, 0]
    assert shares([], 5) == []


def test_an_items_stratum_is_its_condition_its_controls_apart() -> None:
    plain, control = _item(2, "pulse length=16", "Hi."), _item(3, "pulse", "Ho.", True)
    assert (plain.control, plain.stratum) == (False, "x / pulse length=16")
    assert (control.control, control.stratum) == (True, "x / pulse / control")


def test_a_text_that_appeared_already_is_left_out() -> None:
    first, again, other = (
        _item(1, "none", "Hi."),
        _item(2, "pulse", "Hi."),
        _item(3, "none", "Ho."),
    )
    assert distinct([first, again, other]) == [first, other]


def test_each_stratum_gives_its_share_chosen_and_shuffled_by_the_seed() -> None:
    items = _items()
    chosen = sample(items, 7, 0)
    assert sorted(item.stratum for item in chosen) == sorted(
        ["x / none"] * 3 + ["x / pulse"] * 2 + ["x / pulse / control"] * 2
    )
    assert chosen == sample(items, 7, 0) and chosen != sample(items, 7, 1)
    assert sample(list(reversed(items)), 7, 0) != chosen  # the order counts too
    everything = sample(items, 100, 0)
    assert sorted(everything, key=lambda item: item.run.run_id) == items
    assert everything != items  # shuffled


@pytest.mark.parametrize(
    ("cell", "shown"),
    [
        ("=1+1", "'=1+1"),
        ("- item", "'- item"),
        ("+33 1 23", "'+33 1 23"),
        ("@SUM(A1)", "'@SUM(A1)"),
        ("Hello = hi", "Hello = hi"),
        ("", ""),
    ],
)
def test_a_cell_that_a_spreadsheet_would_run_is_guarded(cell: str, shown: str) -> None:
    assert guard(cell) == shown


def test_item_ids_count_from_one_in_at_least_three_digits() -> None:
    assert item_ids("coherence-1", 2) == ["coherence-1-001", "coherence-1-002"]
    assert item_ids("s", 1000)[0] == "s-0001" and item_ids("s", 1000)[-1] == "s-1000"


def _line(item: LabelItem) -> str:
    """An item's line in its sample's token: its experiment and run, and the
    SHA-256 of its reply as saved, the space around it kept."""
    reply = hashlib.sha256(item.run.output_text.encode("utf-8")).hexdigest()
    return f"x/{item.run.run_id}/{reply}"


def _hashed(lines: list[str]) -> str:
    """The first four hex digits of the SHA-256 of lines, a line each."""
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:4]


def test_a_samples_token_is_its_runs_and_replies_in_sheet_order() -> None:
    items = _items()[:3]
    token = _hashed([_line(item) for item in items])
    assert sample_token(items) == token
    assert sample_token(list(items)) == token  # the same sample, the same token
    others = {sample_token(items[::-1]), sample_token(items[:2]), sample_token([])}
    assert token not in others and len(others) == 3  # another order, another run
    assert re.fullmatch("[0-9a-f]{4}", token)  # four hex digits
    # The same run with another reply, as when a new version of its runner
    # folder made it again, has another token; the same reply in another
    # version, the same token.
    rerun = replace(items[0].run, output_text="  Another reply.\n")
    assert sample_token([replace(items[0], run=rerun), *items[1:]]) != token
    assert sample_token([replace(items[0], version=2), *items[1:]]) == token


def test_a_samples_token_covers_the_persona_and_the_personas_asked_about() -> None:
    items = _items()[:3]
    lines = [_line(item) for item in items]
    brass, snow = Persona("Brass Owl", "Loud.\nBrassy."), Persona("Snow Owl")
    # After the items, the persona speaks_as asks about, then each persona
    # which_persona chooses among: a line each, its name and description as
    # JSON, so that a line break in a description stays in its line.
    assert sample_token(items, snow, [brass, snow]) == _hashed(
        [
            *lines,
            'persona ["Snow Owl", ""]',
            'personas ["Brass Owl", "Loud.\\nBrassy."]',
            'personas ["Snow Owl", ""]',
        ]
    )
    tokens = [
        sample_token(items),
        sample_token(items, snow),
        sample_token(items, brass),
        sample_token(items, Persona("Snow Owl", "A quiet owl.")),  # described
        sample_token(items, None, [brass, snow]),
        sample_token(items, None, [snow, brass]),  # in another order
        sample_token(items, snow, [brass, snow]),
    ]
    assert len(set(tokens)) == len(tokens)


def test_a_samples_token_covers_each_rubric_at_its_version() -> None:
    items = _items()[:3]
    lines = [_line(item) for item in items]
    coherence = load_rubric("coherence")
    # Last, a line per rubric: its name, version and file's SHA-256.
    rubric = f'rubric ["coherence", 1, "{coherence.sha256}"]'
    assert sample_token(items, rubrics=[coherence]) == _hashed([*lines, rubric])
    rewritten = replace(coherence, version=2, sha256="0" * 64)
    assert sample_token(items, rubrics=[rewritten]) != sample_token(
        items, rubrics=[coherence]
    )  # a rubric's new version: new IDs, so labels of the old one are refused


def test_the_sheet_has_a_byte_order_mark_guarded_cells_and_empty_labels() -> None:
    items = [_item(1, "none", "=1+1"), _item(2, "pulse", '- item\nand, "more"')]
    data = sheet_csv(["s-001", "s-002"], items, list(RUBRIC_NAMES))
    assert data.startswith(b"\xef\xbb\xbfitem_id,text,speaks_as,")
    rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"), newline="")))
    assert rows == [
        ["item_id", "text", *RUBRIC_NAMES, "note"],
        ["s-001", "'=1+1", "", "", "", ""],
        ["s-002", '\'- item\nand, "more"', "", "", "", ""],
    ]


def test_the_instructions_give_each_rubrics_answers_and_the_rules() -> None:
    rubrics = [load_rubric(name) for name in RUBRIC_NAMES]
    text = instructions(rubrics, OWLS[0], OWLS, 12)
    flat = " ".join(text.split())
    assert "labels.csv holds 12 replies" in flat
    assert "**Do not open key.json while you label.**" in flat
    assert "Leave a cell empty to skip that reply for that column" in flat
    # A rubric needs the plan's 200 labels to pass, and an empty cell is none.
    assert "A rubric needs 200 of your labels to pass, so label every reply" in flat
    assert "an empty cell is not a label, so it counts against the 200." in flat
    assert "the columns `speaks_as`, `which_persona`, `coherence`" in flat
    owl = "A pompous mechanical owl who lectures in riddles."
    assert f"The persona: **Brass Owl**. {owl}" in flat
    for label, meaning in rubrics[0].labels:
        assert f"- `{label}`: {meaning}" in flat
    assert f"- `Brass Owl`: {owl} - `Snow Owl`: (no description)" in flat
    assert "- `none`: none of these personas" in flat
    assert "Write a whole number from 0 to 10." in flat
    assert "- 10: fluent and clear throughout." in flat
    assert "- 7 to 9: readable, with small slips or odd turns." in flat
    # The judge's own instructions follow each rubric's answers, word for word.
    told = rubrics[2].ask().system
    assert "\n".join(f"> {line}".rstrip() for line in told.split("\n")) in text
    assert text.count("What the judge is told") == 3


def test_docstring_examples_run() -> None:
    result = doctest.testmod(label_sheet)
    assert result.attempted > 0
    assert result.failed == 0
