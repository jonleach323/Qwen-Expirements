"""The panel file (pushlab.panel.panel_file): what pushlab panel select
writes and pushlab panel stage preview reads back."""

import doctest
from dataclasses import replace
from pathlib import Path

import pytest

from pushlab.panel import panel_file
from pushlab.panel.panel_file import (
    Mean,
    PanelFile,
    PanelPersona,
    Share,
    panel_from_mapping,
    panel_mapping,
    read_panel_file,
    write_panel_file,
)

OWL = PanelPersona(
    key="library/L9N4242-",
    persona="Brass Owl",
    category="animals",
    kind="neuron",
    source="library",
    unit="qwen3-8b:L9:N4242",
    direction=-1,
    reference_dose=1.5,
    working_dose=1.125,
    collapse_dose=1.875,
    reliability=Share(28, 32, 0.875, 0.719, 0.95),
    coherence=Mean(32, 7.6, 7.2, 8.0),
    distinct=Share(8, 8, 1.0, 0.676, 1.0),
)
PIRATE = replace(
    OWL,
    key="mayner/pirate-top50",
    persona="pirate",
    category="people",
    kind="set",
    source="mayner",
    unit="qwen3-8b:set:pirate-top50",
    direction=1,
    reference_dose=1.0,
    working_dose=0.75,
    collapse_dose=None,
)
PANEL = PanelFile(
    "qwen3-8b",
    "units/candidates_qwen3-8b.csv",
    "a" * 64,
    {"confirm": "s3-confirm", "ladder": "s3-ladder"},
    (OWL, PIRATE),
)


def test_a_panel_reads_back_as_written(tmp_path: Path) -> None:
    path = write_panel_file(tmp_path / "units" / "panel_qwen3-8b.yaml", PANEL)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# A persona panel") and "out of git" in text
    assert read_panel_file(path) == PANEL
    assert "collapse_dose: null" in text  # never collapsed within the ladder


@pytest.mark.parametrize(
    ("change", "problem"),
    [
        (lambda m: m.update(version=2), "version must be 1"),
        (lambda m: m.pop("model"), "must have exactly"),
        (lambda m: m["personas"][0].update(kind="team"), "kind must be neuron or set"),
        (lambda m: m["personas"][0].update(direction=2), "direction must be 1 or -1"),
        (lambda m: m["personas"][0]["reliability"].update(k=33), "0 <= k <= n"),
        (lambda m: m["personas"][1].update(key="library/L9N4242-"), "twice"),
        (lambda m: m["personas"][0].update(working_dose="high"), "must be a number"),
    ],
)
def test_a_malformed_panel_is_refused(change, problem: str) -> None:  # type: ignore[no-untyped-def]
    mapping = panel_mapping(PANEL)
    change(mapping)
    with pytest.raises(ValueError, match=problem):
        panel_from_mapping(mapping)


def test_an_unreadable_panel_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot read the panel"):
        read_panel_file(tmp_path / "none.yaml")


def test_docstring_examples_run() -> None:
    result = doctest.testmod(panel_file)
    assert result.failed == 0
