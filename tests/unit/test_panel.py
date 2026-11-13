"""The persona panel's thresholds (pushlab.analysis.panel, configs/panel.yaml)."""

import doctest
from pathlib import Path

import pytest

from pushlab.analysis import panel as panel_module
from pushlab.analysis.panel import PANEL_FILE, read_panel

GOOD = PANEL_FILE.read_text(encoding="utf-8")


def test_the_panel_starts_at_the_prompts_numbers() -> None:
    panel = read_panel()
    # Version 2 added the selection section (decision 0023); no number changed.
    assert (panel.version, panel.working_rate, panel.coherence_below) == (2, 0.8, 6.0)
    assert (panel.confidence, panel.resamples, panel.seed) == (0.95, 2000, 0)
    assert (panel.ngram, panel.window) == (4, 50)
    assert (panel.repetitive_above, panel.varied_below) == (0.3, 0.4)
    assert panel.path == PANEL_FILE


@pytest.mark.parametrize(
    ("old", "new", "problem"),
    [
        ("working_rate: 0.8", "working_rate: 0", "working_rate must be above 0"),
        ("working_rate: 0.8", "working_rate: 1.5", "working_rate must be above 0 to 1"),
        ("coherence_below: 6", "coherence_below: 11", "from|above"),
        ("coherence_below: 6", "coherence_below: six", "must be a number"),
        ("confidence: 0.95", "confidence: 1", "and below 1"),
        ("resamples: 2000", "resamples: 99", "from 100"),
        ("resamples: 2000", "resamples: 2000.5", "whole number"),
        ("seed: 0", "seed: -1", "from 0"),
        ("ngram: 4", "ngram: 1", "from 2"),
        ("window: 50", "window: true", "whole number"),
        ("version: 2", "version: 2\ncolour: red", "unknown keys: colour"),
        ("version: 2\n", "", "lacks version"),
    ],
)
def test_a_malformed_panel_is_refused(
    tmp_path: Path, old: str, new: str, problem: str
) -> None:
    assert GOOD.count(old) == 1
    path = tmp_path / "panel.yaml"
    path.write_text(GOOD.replace(old, new), encoding="utf-8")
    with pytest.raises(ValueError, match=problem):
        read_panel(path)


def test_a_missing_panel_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot read"):
        read_panel(tmp_path / "none.yaml")


def test_docstring_examples_run() -> None:
    result = doctest.testmod(panel_module)
    assert result.attempted > 0
    assert result.failed == 0
