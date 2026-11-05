"""`pushlab report`: run folders' replies side by side in a Markdown table."""

import doctest
import json
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from pushlab.analysis import report
from pushlab.analysis.report import side_by_side
from pushlab.cli import main
from pushlab.prompts import PROMPTS_DIR
from pushlab.store.records import (
    RUNS_FILE,
    RUNS_SCHEMA,
    RunRecord,
    make_run_id,
    write_runs,
)

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


@pytest.mark.parametrize("line_break", ["\r", "\r\n", "\u2028", "\u2029", "\x85"])
def test_no_kind_of_line_break_ends_a_row_early(
    tmp_path: Path, line_break: str
) -> None:
    folder = _folder(tmp_path, "run", {"p1": f"one{line_break}two"})
    markdown = side_by_side([folder])
    assert markdown.splitlines()[-1] == "| **p1** | one<br>two |"


def test_folders_that_give_a_prompt_two_texts_are_refused(tmp_path: Path) -> None:
    old = _folder(tmp_path, "old", {"p1": "a"}, manifest={"prompts": {"p1": "Hi?"}})
    new = _folder(tmp_path, "new", {"p1": "b"}, manifest={"prompts": {"p1": "Bye?"}})
    with pytest.raises(ValueError, match="p1 reads differently"):
        side_by_side([old, new])


def test_the_legend_names_each_folders_prompt_set(tmp_path: Path) -> None:
    manifest = {
        "prompt_set": "prompts/smoke.jsonl",
        "prompt_set_version": 1,
        "prompt_set_sha256": "92ba015e867c" + "0" * 52,
    }
    folder = _folder(tmp_path, "run", {"p1": "a"}, manifest=manifest)
    legend = side_by_side([folder]).split("## Replies")[0]
    assert (
        "prompts from prompts/smoke.jsonl version 1 (SHA-256 92ba015e867c…)" in legend
    )


def test_a_reply_at_its_token_limit_says_so(tmp_path: Path) -> None:
    replies = {"p1": "a long reply", "p2": "short"}
    folder = _folder(tmp_path, "run", replies, cut=frozenset({"p1"}))
    _, cut, whole = _table(side_by_side([folder]))
    assert cut[1].endswith("(reached the 8-token limit)")
    assert whole[1] == "short"


def test_with_the_stop_tokens_known_only_a_cut_reply_says_so(tmp_path: Path) -> None:
    # Each reply here is 8 tokens of 1; with 1 an end token, both ended
    # their turn on their last token, and with 2 alone, both were cut off.
    replies = {"p1": "ended", "p2": "also ended"}
    cut = frozenset({"p1", "p2"})
    ended = _folder(tmp_path, "a", replies, manifest={"stop_token_ids": [1]}, cut=cut)
    others = _folder(tmp_path, "b", replies, manifest={"stop_token_ids": [2]}, cut=cut)
    _, *rows = _table(side_by_side([ended, others]))
    assert [row[1] for row in rows] == ["ended", "also ended"]
    assert all(row[2].endswith("(cut off at the 8-token limit)") for row in rows)


def test_a_push_that_never_came_is_marked(tmp_path: Path) -> None:
    late = PUSH | {"start": 50, "include_prompt": False}
    records = [
        _record(1, "p1", "early end", [late | {"touched": 0}]),
        _record(2, "p2", "pushed", [late | {"touched": 3}]),
    ]
    write_runs(records, tmp_path / "late")
    header, first, second = _table(side_by_side([tmp_path / "late"]))
    assert first[1].endswith("nothing was pushed)")
    assert second[1] == "pushed"
    # touched differs from run to run, but the folder's pushes do not.
    assert "differ" not in side_by_side([tmp_path / "late"]).split("## Replies")[0]


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


def test_odd_folders_and_outs_are_refused_with_a_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder = _folder(tmp_path, "run", {"p1": "a"})
    empty = tmp_path / "empty"
    empty.mkdir()
    pq.write_table(pa.Table.from_pylist([], schema=RUNS_SCHEMA), empty / RUNS_FILE)
    blocked = tmp_path / "file"
    blocked.write_text("x")
    for argv, problem in [
        (["--runs", str(empty), "--out", str(tmp_path / "r.md")], "holds no runs"),
        (["--runs", str(folder), "--out", str(blocked / "r.md")], "cannot write"),
        (["--runs", str(folder), "--out", str(tmp_path)], "must be a file"),
    ]:
        assert main(["report", *argv]) == 1
        assert problem in capsys.readouterr().err


def test_folders_of_one_name_get_numbered_columns(tmp_path: Path) -> None:
    first = _folder(tmp_path / "a", "dose1", {"p1": "x", "p2": "y"})
    second = _folder(tmp_path / "b", "dose1", {"p1": "z"})
    markdown = side_by_side([first, second])
    assert _table(markdown)[0] == ["Prompt", "dose1", "dose1 (2)"]
    legend = markdown.split("## Replies")[0]
    assert "**dose1**: " + str(first) + ", 2 runs of tiny-vl@main" in legend
    assert "**dose1 (2)**: " + str(second) + ", 1 runs of tiny-vl@main" in legend


def test_runs_given_twice_adds_the_folders(tmp_path: Path) -> None:
    first = _folder(tmp_path, "first", {"p1": "a"})
    second = _folder(tmp_path, "second", {"p1": "b"})
    out = tmp_path / "side.md"
    argv = ["report", "--runs", str(first), "--runs", str(second), "--out", str(out)]
    assert main(argv) == 0
    assert _table(out.read_text())[0] == ["Prompt", "first", "second"]


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
