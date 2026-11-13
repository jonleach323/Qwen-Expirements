"""The panel's page and where the panel's files go (pushlab.panel.page,
decision 0023). The personas are made up.
"""

import doctest
from pathlib import Path

import pytest

from pushlab import PROJECT_ROOT
from pushlab.panel import page
from pushlab.panel.page import (
    IGNORED,
    first_line,
    kept_out_of_git,
    markdown_text,
    output_paths,
    page_text,
    write_page,
)
from pushlab.panel.panel_file import Mean, PanelFile, PanelPersona, Share


def _persona(key: str, name: str, collapse: float | None) -> PanelPersona:
    return PanelPersona(
        key=key,
        persona=name,
        category="animals",
        kind="neuron",
        source="library",
        unit="qwen3-8b:L9:N1",
        direction=-1,
        reference_dose=1.5,
        working_dose=1.125,
        collapse_dose=collapse,
        reliability=Share(28, 32, 0.875, 0.7194, 0.95),
        coherence=Mean(32, 7.59375, 7.2, 8.0),
        distinct=Share(8, 8, 1.0, 0.6756, 1.0),
    )


def test_an_example_line_is_a_reply_s_first_line_cut_short() -> None:
    assert first_line("\n \nHoo!   I am\tthe owl.\nAnd more.") == "Hoo! I am the owl."
    assert first_line("  \n ") == ""
    assert first_line("a" * 250) == "a" * 200 + "…"
    assert first_line("word " * 60) == ("word " * 40).strip() + "…"


def test_markdown_shows_text_as_written() -> None:
    assert markdown_text("*Hoo* _owl_ `x` [a](b) <i>| ~ &amp;") == (
        "\\*Hoo\\* \\_owl\\_ \\`x\\` \\[a\\](b) \\<i\\>\\| \\~ \\&amp;"
    )
    assert markdown_text("two\nlines") == "two lines"


def test_the_page_tables_the_personas_and_gives_two_lines_of_each() -> None:
    owl = _persona("library/L9N1-", "Brass *Owl*", 1.875)
    fox = _persona("library/L3N9+", "Tin|Fox", None)
    panel = PanelFile(
        "qwen3-8b",
        "units/candidates_qwen3-8b.csv",
        "0" * 64,
        {"confirm": "s3-stage-confirm, version 1", "categories": "/w/c_1.csv"},
        (owl, fox),
    )
    examples = {
        owl.key: [("heldout32-01", "Hoo!\nMore."), ("heldout32-05", "<b>Hoo</b>")],
    }
    text = page_text(panel, examples, "units/panel_qwen3-8b.yaml", 0.95)
    lines = text.splitlines()
    assert lines[0] == "# The persona panel of qwen3-8b"
    assert "stays out of git, as .gitignore keeps docs/panel/" in text
    assert "from the candidates of `units/candidates_qwen3-8b.csv`" in text
    assert "- categories: /w/c\\_1.csv" in lines
    assert lines.count("|---|---|---|---|---|---|---|---|---|---|") == 1
    assert (
        "| 1 | Brass \\*Owl\\* | library/L9N1- | animals | neuron | 1.125 | 1.875 | "
        "28/32, 0.88 [0.72, 0.95] | 7.6 [7.2, 8.0] | 8/8, 1.00 [0.68, 1.00] |"
    ) in lines
    fox_row = next(line for line in lines if line.startswith("| 2 | Tin\\|Fox |"))
    assert "| 1.125 | beyond the ladder |" in fox_row
    owl_at = lines.index("### 1. Brass \\*Owl\\*")
    assert lines[owl_at + 2 : owl_at + 4] == [
        "- heldout32-01: Hoo!",
        "- heldout32-05: \\<b\\>Hoo\\</b\\>",
    ]
    assert lines[-1] == "- (no reply speaks as it)"
    assert "Intervals are 95%." in text


def test_write_page_leaves_nothing_but_the_page(tmp_path: Path) -> None:
    path = tmp_path / "docs" / "panel" / "qwen3-8b.md"
    assert write_page(path, "# A page\n") == path
    assert path.read_text(encoding="utf-8") == "# A page\n"
    assert [p.name for p in path.parent.iterdir()] == ["qwen3-8b.md"]


def test_the_panel_s_files_go_where_git_does_not_keep_them(tmp_path: Path) -> None:
    out, doc = output_paths(None, None, "qwen3-8b")
    assert out == PROJECT_ROOT / "units" / "panel_qwen3-8b.yaml"
    assert doc == PROJECT_ROOT / "docs" / "panel" / "qwen3-8b.md"
    assert kept_out_of_git(PROJECT_ROOT / "docs" / "panel" / "old" / "x.md")
    assert kept_out_of_git(tmp_path / "anything.md")
    assert not kept_out_of_git(PROJECT_ROOT / "docs" / "panel.md")
    assert not kept_out_of_git(PROJECT_ROOT / "units" / "panels" / "panel_x.yaml")
    given = output_paths(tmp_path / "p.yaml", tmp_path / "p.md", "qwen3-8b")
    assert given == (tmp_path / "p.yaml", tmp_path / "p.md")
    with pytest.raises(ValueError, match="--doc .* is in the repository, where git"):
        output_paths(None, PROJECT_ROOT / "docs" / "qwen3-8b.md", "qwen3-8b")
    with pytest.raises(ValueError, match="--out must be a file"):
        output_paths(tmp_path, None, "qwen3-8b")
    with pytest.raises(ValueError, match="--out and --doc are both"):
        output_paths(tmp_path / "p.md", tmp_path / "p.md", "qwen3-8b")
    # The places git keeps them out are those .gitignore names.
    ignored = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert all(f"/{pattern}" in ignored for pattern in IGNORED)


def test_the_examples_in_the_docstrings_hold() -> None:
    result = doctest.testmod(page)
    assert result.failed == 0 and result.attempted > 0
