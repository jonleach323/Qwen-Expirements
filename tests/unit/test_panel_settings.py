"""How pushlab panel narrows candidates down (pushlab.panel.settings, the
selection section of configs/panel.yaml)."""

import doctest
from pathlib import Path

import pytest

from pushlab.analysis.panel import PANEL_FILE, read_panel
from pushlab.panel import settings
from pushlab.panel.settings import read_selection

GOOD = PANEL_FILE.read_text(encoding="utf-8")


def test_the_selection_starts_at_the_prompts_numbers() -> None:
    selection = read_selection()
    assert selection.prompts == "prompts/heldout32.jsonl"
    assert (selection.max_new_tokens, selection.batch_size) == (160, 16)
    assert selection.screen_prompts == 8
    assert (selection.stage_b_size, selection.ladder_size) == (120, 40)
    assert (selection.ladder_prompts, selection.ladder_doses) == (
        16,
        (0.5, 0.75, 1.0, 1.25, 1.5),
    )
    assert (selection.panel_size, selection.panel_minimum) == (24, 16)
    assert (selection.distinct_rate, selection.distinct_sample) == (0.9, 8)
    assert selection.categories == (
        "people",
        "animals",
        "fictional",
        "objects",
        "mystical",
    )
    assert (selection.preview_pairs, selection.preview_prompts) == (6, 16)
    timings = (selection.handoff_at, selection.alternate_every)
    assert timings + (selection.pulse_length, selection.onset_at) == (20, 8, 16, 30)


def test_the_panels_thresholds_still_read_beside_it() -> None:
    assert read_panel().working_rate == 0.8


@pytest.mark.parametrize(
    ("old", "new", "problem"),
    [
        ("stage_b_size: 120", "stage_b_size: 0", "stage_b_size must be a whole number"),
        ("stage_b_size: 120", "stage_b_size: 30", "ladder_size is above stage_b_size"),
        ("panel_minimum: 16", "panel_minimum: 25", "panel_minimum, 25, is above"),
        ("distinct_rate: 0.9", "distinct_rate: 1.5", "distinct_rate must be above 0"),
        (
            "ladder_doses: [0.5, 0.75, 1.0, 1.25, 1.5]",
            "ladder_doses: [1.0, 0.5]",
            "increase",
        ),
        (
            "ladder_doses: [0.5, 0.75, 1.0, 1.25, 1.5]",
            "ladder_doses: [0, 1]",
            "above 0",
        ),
        (
            "categories: [people, animals, fictional, objects, mystical]",
            "categories: [People, x]",
            "lowercase",
        ),
        (
            "categories: [people, animals, fictional, objects, mystical]",
            "categories: [x, x]",
            "once",
        ),
        ("  onset_at: 30\n", "  onset_at: 30\n  colour: red\n", "unknown keys: colour"),
        ("  onset_at: 30\n", "", "lacks onset_at"),
        ("  prompts: prompts/heldout32.jsonl\n", "  prompts: 5\n", "prompt set's path"),
    ],
)
def test_a_malformed_selection_is_refused(
    tmp_path: Path, old: str, new: str, problem: str
) -> None:
    assert old in GOOD
    path = tmp_path / "panel.yaml"
    path.write_text(GOOD.replace(old, new), encoding="utf-8")
    with pytest.raises(ValueError, match=problem):
        read_selection(path)


def test_a_file_without_the_section_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "panel.yaml"
    path.write_text(GOOD.split("\nselection:")[0] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="has no selection section"):
        read_selection(path)


def test_docstring_examples_run() -> None:
    result = doctest.testmod(settings)
    assert result.attempted > 0
    assert result.failed == 0
