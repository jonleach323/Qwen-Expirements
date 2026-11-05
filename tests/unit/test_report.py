"""`pushlab report`: run folders' replies side by side in a Markdown table."""

import doctest
import json
import re
from pathlib import Path

import pytest
import torch

from pushlab.analysis import report
from pushlab.analysis.report import side_by_side
from pushlab.cli import main
from pushlab.prompts import PROMPTS_DIR
from pushlab.store.records import RunRecord, make_run_id, write_runs

DECODING = {"strategy": "greedy", "max_new_tokens": 8, "thinking": False, "dtype": "x"}
PUSH = {
    "unit": "tiny-vl:L2:N100",
    "alpha": 1.5,
    "start": 2,
    "end": None,
    "include_prompt": True,
    "amount": 12.345678,
    "control_for": None,
}


def _record(
    number: int,
    prompt_id: str,
    text: str,
    schedule: list[dict] | None = None,
    tokens: int = 3,
) -> RunRecord:
    """Run number of a steer folder, whose reply has that many tokens."""
    return RunRecord(
        run_id=make_run_id("steer", number),
        experiment="steer",
        model="tiny-vl@main",
        prompt_id=prompt_id,
        schedule=schedule or [],
        decoding=DECODING,
        seed=0,
        batch_key=f"b000001:{number - 1}",
        control_of="",
        output_text=text,
        output_tokens=[1] * tokens,
        status="ok",
    )


def _folder(
    tmp_path: Path,
    name: str,
    replies: dict[str, str],
    schedule: list[dict] | None = None,
    manifest: dict | None = None,
    cut: frozenset[str] = frozenset(),
) -> Path:
    """A run folder with one run per prompt ID, as pushlab steer writes them.

    The replies of the prompts in cut reach the 8-token limit.
    """
    folder = tmp_path / name
    records = [
        _record(number, prompt_id, text, schedule, 8 if prompt_id in cut else 3)
        for number, (prompt_id, text) in enumerate(replies.items(), start=1)
    ]
    write_runs(records, folder)
    if manifest is not None:
        (folder / "manifest.json").write_text(json.dumps(manifest))
    return folder


def _table(markdown: str) -> list[list[str]]:
    """The cells of the replies table: the header row, then one row per prompt."""
    section = markdown.split("## Replies", 1)[1]
    rows = [line for line in section.splitlines() if line.startswith("|")]
    cells = [
        [cell.strip() for cell in re.split(r"(?<!\\)\|", row)[1:-1]] for row in rows
    ]
    return [cells[0], *cells[2:]]  # without the |---| line


def test_one_column_per_run_folder_and_one_row_per_prompt(tmp_path: Path) -> None:
    # Rows come in the order prompts first appear, not sorted by ID.
    first = _folder(tmp_path, "none", {"p2": "two", "p1": "one"})
    second = _folder(tmp_path, "pushed", {"p1": "uno", "p2": "dos"}, [PUSH])
    third = _folder(tmp_path, "other", {"p3": "trois", "p2": "deux"})
    header, *rows = _table(side_by_side([first, second, third]))
    assert header == ["Prompt", "none", "pushed", "other"]
    assert [row[0] for row in rows] == ["**p2**", "**p1**", "**p3**"]
    assert [row[1:] for row in rows] == [
        ["two", "dos", "deux"],
        ["one", "uno", "—"],
        ["—", "—", "trois"],
    ]


def test_the_prompts_text_comes_from_the_manifests(tmp_path: Path) -> None:
    steered = _folder(
        tmp_path, "steered", {"p1": "x"}, manifest={"prompts": {"p1": "Say hi."}}
    )
    baseline = _folder(
        tmp_path,
        "baseline",
        {"p2": "y"},
        manifest={"prompt_id": "p2", "prompt": "Bye?"},
    )
    _, first, second = _table(side_by_side([steered, baseline]))
    assert first[0] == "**p1**<br>Say hi."
    assert second[0] == "**p2**<br>Bye?"


def test_a_reply_stays_readable_inside_its_cell(tmp_path: Path) -> None:
    folder = _folder(tmp_path, "run", {"p1": "a | b\nnext <b>bold</b> & co"})
    _, row = _table(side_by_side([folder]))
    assert row[1] == r"a \| b<br>next &lt;b&gt;bold&lt;/b&gt; &amp; co"


def test_a_reply_cut_at_its_token_limit_says_so(tmp_path: Path) -> None:
    replies = {"p1": "a long reply", "p2": "short"}
    folder = _folder(tmp_path, "run", replies, cut=frozenset({"p1"}))
    _, cut, whole = _table(side_by_side([folder]))
    assert cut[1].endswith("(stopped at the 8-token limit)")
    assert whole[1] == "short"


def test_the_folders_are_described_above_the_table(tmp_path: Path) -> None:
    control = PUSH | {"unit": "tiny-vl:L2:N77", "control_for": "tiny-vl:L2:N100"}
    folders = [
        _folder(tmp_path, "none", {"p1": "a"}),
        _folder(tmp_path, "pushed", {"p1": "b"}, [PUSH | {"end": 6}]),
        _folder(
            tmp_path, "control", {"p1": "c"}, [control | {"include_prompt": False}]
        ),
    ]
    markdown = side_by_side(folders)
    legend = markdown.split("## Replies", 1)[0]
    assert str(folders[0]) in legend and "no push" in legend
    assert (
        "tiny-vl:L2:N100 at +1.5 (adds +12.3457) on the prompt and reply tokens 2 to 5"
        in legend
    )
    assert (
        "tiny-vl:L2:N77 at +1.5 (adds +12.3457) on reply tokens from 2 on, "
        "a random control for tiny-vl:L2:N100" in legend
    )


def test_report_writes_the_file_and_keeps_an_existing_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = _folder(tmp_path, "run", {"p1": "hello"})
    out = tmp_path / "reports" / "side.md"
    assert main(["report", "--runs", str(folder), "--out", str(out)]) == 0
    assert out.read_text() == side_by_side([folder])
    out.write_text("my notes")
    assert main(["report", "--runs", str(folder), "--out", str(out)]) == 1
    assert "--overwrite" in capsys.readouterr().err
    assert out.read_text() == "my notes"
    argv = ["report", "--runs", str(folder), "--out", str(out), "--overwrite"]
    assert main(argv) == 0
    assert out.read_text() == side_by_side([folder])


def test_a_folder_without_runs_or_with_two_runs_of_a_prompt_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "side.md"
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["report", "--runs", str(empty), "--out", str(out)]) == 1
    assert str(empty) in capsys.readouterr().err
    twice = tmp_path / "twice"
    write_runs([_record(1, "p1", "a"), _record(2, "p1", "b")], twice)
    assert main(["report", "--runs", str(twice), "--out", str(out)]) == 1
    assert "p1" in capsys.readouterr().err
    assert not out.exists()


def test_report_needs_at_least_one_folder(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["report", "--out", "x.md"])
    assert "--runs" in capsys.readouterr().err


def test_report_lays_steer_folders_side_by_side(
    tiny_vl_calibration: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    smoke = PROMPTS_DIR / "smoke.jsonl"
    common = ["--model", "tiny-vl", "--prompts", str(smoke), "--max-new-tokens", "3"]
    folders = []
    for name, options in [
        ("none", []),
        ("dose1", ["--unit", "tiny-vl:L2:N100", "--alpha", "1"]),
        ("control", ["--unit", "tiny-vl:L2:N100", "--alpha", "1"]),
    ]:
        if name == "control":
            options = [*options, "--control", "random", "--control-seed", "5"]
        folders.append(tmp_path / name)
        assert main(["steer", *common, "--out", str(tmp_path / name), *options]) == 0
    out = tmp_path / "side.md"
    assert main(["report", "--runs", *map(str, folders), "--out", str(out)]) == 0
    header, *rows = _table(out.read_text())
    assert header == ["Prompt", "none", "dose1", "control"]
    assert len(rows) == 5
    assert all(len(row) == 4 for row in rows)
    assert rows[0][0] == "**smoke-01**<br>Give me a simple recipe for pancakes."


def test_docstring_examples_run() -> None:
    result = doctest.testmod(report)
    assert result.attempted > 0
    assert result.failed == 0
