"""What pushlab label check (pushlab.scoring.label_check) says and writes
beside the numbers that test_label_check.py checks: a rubric with fewer
labels than the plan's 200 is TOO FEW, its numbers still shown; and the
disagreements file gives the judge's reason for each label of its that
differs from yours, which the record never holds. It is in the sheet's
folder, wherever the record is, named after the judge and the time as the
record is by default, so that each pairs with its record; a check never
overwrites one.

Each sheet is made by hand over a runner folder saved by the real runner on
the tiny model, with invented replies, and labeled as a spreadsheet saves a
sheet; a FakeBackend answers by each reply's text
(tests/unit/label_sheets.py). Item k is run k.
"""

import codecs
import csv
import importlib.util
import io
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch

from pushlab.scoring import label_check
from pushlab.scoring.judge import Reply
from pushlab.scoring.label_check import disagreements_csv, run_label_check

# The sheets' helpers, loaded by path, as test modules import nothing from
# one another.
_SPEC = importlib.util.spec_from_file_location(
    "label_sheets", Path(__file__).with_name("label_sheets.py")
)
sheets = importlib.util.module_from_spec(_SPEC)  # type: ignore[arg-type]
_SPEC.loader.exec_module(sheets)  # type: ignore[union-attr]

NOW = datetime(2026, 11, 13, 9, 0, 0, tzinfo=UTC)
OUT = "check-fake-fake-judge-20261113T090000Z.json"
REASONS = "disagreements-fake-fake-judge-20261113T090000Z.csv"  # OUT's pair
HEADER = ["item_id", "rubric", "yours", "judge", "reason"]


@pytest.fixture(autouse=True)
def cpu_and_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU even on a machine with a GPU, and fix the check's time."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(label_check, "_clock", lambda: NOW)


@pytest.fixture
def saved(tiny_spec: Callable[..., Path], tmp_path: Path) -> tuple[Path, Path]:
    """The experiment file of sheets.CONDITIONS and a runs folder holding its
    20 runs, with invented replies."""
    spec, runs = tiny_spec(sheets.CONDITIONS), tmp_path / "runs"
    sheets.saved_runs(spec, runs)
    return spec, runs


def _found(path: Path) -> list[list[str]]:
    """A disagreements file's rows, read as a spreadsheet reads them."""
    text = path.read_bytes().decode("utf-8-sig")
    return list(csv.reader(io.StringIO(text, newline="")))


def test_a_rubric_with_fewer_labels_than_the_plans_200_is_too_few(
    saved: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sheet = sheets.make_sheet(tmp_path, *saved, ["coherence"], None, ())
    # You and the judge agree on every reply: 7 for 1 to 10, 3 for 11 to 20.
    # Kappa is 1, numbers that would pass, but on 20 labels, not 200.
    scores = [7] * 10 + [3] * 10
    cells = [str(score) for score in scores]
    sheets.save(sheet / "labels.csv", sheets.rows(sheet, {"coherence": cells}))
    judge = sheets.fake_judge({"coherence": scores})
    assert run_label_check(sheet, judge=judge, runs_dir=saved[1]) == 0
    printed = capsys.readouterr().out
    flat = " ".join(printed.split())
    assert "A rubric passes with 200 labels or more, at an accuracy of 0.85" in flat
    assert "coherence (version 1): TOO FEW" in printed
    assert "  Too few labels to pass: 20 of the plan's 200." in printed
    assert "  Accuracy: 1.000 (20 of 20 agree)." in printed  # still shown
    assert "  Cohen's kappa: 1.000." in printed
    assert "Results: coherence TOO FEW." in printed
    record = json.loads((sheet / OUT).read_text(encoding="utf-8"))
    assert record["rule"] == {"accuracy": 0.85, "kappa": 0.7, "labels": 200}
    numbers = record["rubrics"]["coherence"]
    assert (numbers["result"], numbers["labels"]) == ("TOO FEW", 20)
    assert (numbers["accuracy"], numbers["kappa"]) == (1.0, 1.0)


def test_disagreements_csv_has_a_byte_order_mark_and_guarded_cells() -> None:
    rows = [
        ["s-001", "coherence", 7, 4, "- Too many slips, said with a, comma."],
        ["s-002", "which_persona", "Brass Owl", "failed", "malformed answer: x"],
        ["s-003", "speaks_as", "assistant", "speaks_as", "Broken \ud83d here."],
    ]
    data = disagreements_csv(rows)
    assert data.startswith(codecs.BOM_UTF8 + b"item_id,rubric,yours,judge,reason\r\n")
    found = list(csv.reader(io.StringIO(data.decode("utf-8-sig"), newline="")))
    assert found == [
        ["item_id", "rubric", "yours", "judge", "reason"],
        ["s-001", "coherence", "7", "4", "'- Too many slips, said with a, comma."],
        ["s-002", "which_persona", "Brass Owl", "failed", "malformed answer: x"],
        # A lone surrogate, which UTF-8 cannot hold, is written as its escape.
        ["s-003", "speaks_as", "assistant", "speaks_as", "Broken \\ud83d here."],
    ]
    assert disagreements_csv([]) == codecs.BOM_UTF8 + (
        b"item_id,rubric,yours,judge,reason\r\n"
    )


def test_disagreements_csv_gives_the_judges_reasons_beside_the_replies(
    saved: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rubrics = ["which_persona", "coherence"]
    sheet = sheets.make_sheet(tmp_path, *saved, rubrics, None)
    # You: the brass owl for replies 1 to 3 and none for 4 to 6; coherence
    # 7, 7, 3 and 3 for replies 1 to 4.
    columns = {
        "which_persona": ["brass owl"] * 3 + ["none"] * 3,
        "coherence": ["7", "7", "3", "3"],
    }
    sheets.save(sheet / "labels.csv", sheets.rows(sheet, columns))
    # The judge: the snow owl for reply 2, and a malformed answer for reply 3
    # at each attempt; 4 for reply 2, its reason starting as a formula
    # would, and 2 for reply 4.
    malformed = Reply('{"reason": "Hm.", "choice": "Owl"}')
    slips = Reply(json.dumps({"reason": "- Too many slips.", "score": 4}))
    answers = {
        "which_persona": ["Brass Owl", "Snow Owl", malformed, *["none"] * 3],
        "coherence": [7, slips, 3, 2],
    }
    out = tmp_path / "check.json"  # the record elsewhere; the reasons stay
    judge = sheets.fake_judge(answers)
    assert run_label_check(sheet, judge=judge, runs_dir=saved[1], out=out) == 0
    path = sheet / REASONS  # named after the judge and the time all the same
    assert (
        f"Saved the judge's reasons where you differ, which may quote replies: {path}"
    ) in capsys.readouterr().out
    assert sorted(file.name for file in sheet.glob("disagreements*")) == [REASONS]
    data = path.read_bytes()
    assert data.startswith(codecs.BOM_UTF8)  # as labels.csv, for spreadsheets
    error = 'malformed answer: "choice" must be one of Brass Owl, Snow Owl, none, '
    assert _found(path) == [
        HEADER,
        ["sheet-1-002", "which_persona", "Brass Owl", "Snow Owl", sheets.REASON],
        ["sheet-1-003", "which_persona", "Brass Owl", "failed", error + "not 'Owl'"],
        ["sheet-1-002", "coherence", "7", "4", "'- Too many slips."],  # guarded
        ["sheet-1-004", "coherence", "3", "2", sheets.REASON],
    ]
    # The record, which holds no reply, holds no reason and no error's words
    # past its kind either.
    record = out.read_text(encoding="utf-8")
    assert json.loads(record)["rubrics"]["which_persona"]["failures"] == {
        "malformed answer": 1
    }
    for words in (sheets.REASON, "slips", "must be one of", "Teapot"):
        assert words not in record


def test_each_judges_disagreements_pair_with_its_record_and_are_never_overwritten(
    saved: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sheet = sheets.make_sheet(tmp_path, *saved, ["coherence"], None, ())
    (sheet / "disagreements.csv").write_text("an earlier check's\n", encoding="utf-8")
    sheets.save(sheet / "labels.csv", sheets.rows(sheet, {"coherence": ["5"] * 3}))
    # Two judges check the sheet at the same time: one agrees with you on
    # replies 1 to 3, and the other gives reply 2 a 4.
    agrees = sheets.fake_judge({"coherence": [5] * 3})
    differs = sheets.fake_judge({"coherence": [5, 4, 5]}, model="other-judge")
    for judge in (agrees, differs):
        assert run_label_check(sheet, judge=judge, runs_dir=saved[1]) == 0
    capsys.readouterr()
    assert _found(sheet / REASONS) == [HEADER]  # no disagreement: the header
    other = "fake-other-judge-20261113T090000Z"
    assert _found(sheet / f"disagreements-{other}.csv") == [
        HEADER,
        ["sheet-1-002", "coherence", "5", "4", sheets.REASON],
    ]
    record = json.loads((sheet / f"check-{other}.json").read_text(encoding="utf-8"))
    assert (record["judge"], record["checked_at"]) == (
        "fake:other-judge",
        "2026-11-13T09:00:00Z",
    )
    # A file of another name, such as an earlier check's, is left as it was.
    assert (sheet / "disagreements.csv").read_text(encoding="utf-8") == (
        "an earlier check's\n"
    )
    # The same judge again at the same time, its record going elsewhere: its
    # disagreements file exists, so the check is refused before the judge is
    # asked, and the file is kept.
    again = sheets.fake_judge({"coherence": [5] * 3})
    out = tmp_path / "again.json"
    assert run_label_check(sheet, judge=again, runs_dir=saved[1], out=out) == 1
    assert again.backend.calls == 0 and not out.exists()
    error = capsys.readouterr().err
    assert error.startswith(f"pushlab label check: {sheet / REASONS} already exists")
    assert _found(sheet / REASONS) == [HEADER]
