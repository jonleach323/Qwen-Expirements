"""What pushlab label check (pushlab.scoring.label_check) refuses. Each
refusal prints "pushlab label check: <why>", returns 1 and writes no record
and no disagreements file; all but a judge that cannot answer are refused
before it is asked.

Each sheet is made by hand over a runner folder saved by the real runner on
the tiny model, with invented replies (tests/unit/label_sheets.py); item k
is run k, and the default sheet labels each of the 20 replies' coherence 5.
"""

import importlib.util
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from pushlab.cli import main
from pushlab.scoring import label_check
from pushlab.scoring.judge import CallError, Judge
from pushlab.scoring.label_check import run_label_check
from pushlab.scoring.label_export import run_label_export
from pushlab.scoring.rubrics import load_rubric

# The sheets' helpers, loaded by path, as test modules import nothing from
# one another.
_SPEC = importlib.util.spec_from_file_location(
    "label_sheets", Path(__file__).with_name("label_sheets.py")
)
sheets = importlib.util.module_from_spec(_SPEC)  # type: ignore[arg-type]
_SPEC.loader.exec_module(sheets)  # type: ignore[union-attr]

# sheets.CONDITIONS with the pulse half as strong: a new version of the
# runner folder, in which the pushed runs change.
WEAKER = sheets.CONDITIONS.replace("alpha: 1.0", "alpha: 0.5")
# What a refusal about the runner folder says to do.
ELSEWHERE = "if the export read another runs folder, give it with --runs-dir"


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def labeled(tiny_spec: Callable[..., Path], tmp_path: Path) -> tuple[Path, Path]:
    """A sheet of the 20 replies for coherence alone, each labeled 5, and the
    runs folder its replies are saved in."""
    spec, runs = tiny_spec(sheets.CONDITIONS), tmp_path / "runs"
    sheets.saved_runs(spec, runs)
    sheet = sheets.make_sheet(tmp_path, spec, runs, ["coherence"], None, ())
    sheets.save(sheet / "labels.csv", sheets.rows(sheet, {"coherence": ["5"] * 20}))
    return sheet, runs


def _judge(**names: str) -> Judge:
    return sheets.fake_judge({"coherence": [5] * 20}, **names)


def _refused(
    capsys: pytest.CaptureFixture[str],
    sheet: Path,
    runs: Path,
    judge: Judge | None = None,
    **options: Any,
) -> str:
    """Why the check of sheet is refused, having asked judge nothing."""
    judge = _judge() if judge is None else judge
    assert run_label_check(sheet, judge=judge, runs_dir=runs, **options) == 1
    assert judge.backend.calls == 0
    assert not list(sheet.glob("check-*.json"))
    assert not list(sheet.glob("disagreements-*.csv"))
    error = capsys.readouterr().err
    assert error.startswith("pushlab label check: ")
    return error


def test_a_sheet_with_problems_is_refused_naming_its_rows(
    labeled: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    sheet, runs = labeled
    table = sheets.rows(sheet, {"coherence": ["5"] * 20})
    table[4][2] = "five"  # row 5, below item 3's text of two lines
    table[6][0] = ""  # row 7
    sheets.save(sheet / "labels.csv", table)
    error = _refused(capsys, sheet, runs)
    assert error.splitlines() == [
        f"pushlab label check: fix these in {sheet / 'labels.csv'}, then check again:",
        "  row 7 has no item_id",
        "  coherence: 'five' (row 5) is not a whole number from 0 to 10",
    ]


def _edit_key(sheet: Path, change: Callable[[dict[str, Any]], object]) -> None:
    key = json.loads((sheet / "key.json").read_text(encoding="utf-8"))
    change(key)
    (sheet / "key.json").write_text(json.dumps(key), encoding="utf-8")


def test_a_sheet_whose_replies_changed_since_the_export_is_refused(
    labeled: tuple[Path, Path],
    tiny_spec: Callable[..., Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    sheet, runs = labeled
    folder = runs / sheets.NAME
    sheets.rewrite(
        folder,
        lambda run: (
            "Kettle reply 4 hums."
            if sheets.number(run.run_id) == 4
            else run.output_text
        ),
    )
    error = _refused(capsys, sheet, runs)
    assert "the reply of tiny-pulses-000004 changed since the export" in error
    # It names the runner folder it read, as the export may have read another.
    assert f"(in the runner folder {folder}): {ELSEWHERE}; else " in error
    assert error.rstrip().endswith("export a new sheet (pushlab label export)")
    _edit_key(sheet, lambda key: key["items"][0].update(run_id="tiny-pulses-000099"))
    error = _refused(capsys, sheet, runs)
    assert "tiny-pulses-000099 has no saved reply now" in error
    # The same experiment file, changed: a new version of the runner folder.
    spec = tiny_spec(WEAKER)
    argv = ["run", str(spec), "--runs-dir", str(runs), "--overwrite-version"]
    assert main(argv) == 0
    capsys.readouterr()  # the runner's own output
    error = _refused(capsys, sheet, runs)
    assert (
        "tiny-pulses's runner folder is at version 2 now, but the sheet's replies "
        "are of version 1"
    ) in error
    # key.json gives the experiment file's path as export was given it.
    _edit_key(
        sheet, lambda key: [item.update(spec="moved.yaml") for item in key["items"]]
    )
    error = _refused(capsys, sheet, runs)
    assert (
        "moved.yaml, an experiment file the sheet was exported from, is not there: "
        "run pushlab label check from the folder pushlab label export ran in"
    ) in error


def test_the_runs_folder_is_the_one_key_json_names_unless_runs_dir_says(
    labeled: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sheet, runs = labeled
    monkeypatch.setenv("PUSHLAB_RUNS_DIR", str(tmp_path / "default"))
    # --runs-dir, given, is read in place of the folder key.json names.
    error = _refused(capsys, sheet, tmp_path / "other")
    assert f"{tmp_path / 'other' / sheets.NAME} does not exist" in error
    assert error.rstrip().endswith(ELSEWHERE)
    # A key written before export recorded its runs folder: the default.
    _edit_key(sheet, lambda key: key.pop("runs_dir"))
    error = _refused(capsys, sheet, None)
    assert f"{tmp_path / 'default' / sheets.NAME} does not exist" in error
    assert error.rstrip().endswith(ELSEWHERE)
    # key.json names the runs folder the export read: no --runs-dir needed.
    _edit_key(sheet, lambda key: key.update(runs_dir=str(runs.resolve())))
    assert run_label_check(sheet, judge=_judge()) == 0


def test_labels_of_another_sample_are_refused_as_items_the_sheet_lacks(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = tiny_spec(sheets.CONDITIONS), tmp_path / "runs"
    sheets.saved_runs(spec, runs)
    # Three sheets in folders of one name: seed 0 twice, then seed 1.
    folders = [tmp_path / place / "sheet" for place in ("a", "b", "c")]
    for folder, seed in zip(folders, [0, 0, 1], strict=True):
        options = {"rubrics": ["coherence"], "size": 6, "seed": seed}
        assert run_label_export([spec], folder, runs_dir=runs, **options) == 0
    capsys.readouterr()  # the runner's and export's own output
    # The first sheet, labeled, and saved in place of each sheet's own.
    table = sheets.rows(folders[0], {"coherence": ["5"] * 6})
    for folder in folders:
        sheets.save(folder / "labels.csv", table)
    # Another sample: its IDs are not the first's, so none is read as its.
    first = json.loads((folders[0] / "key.json").read_text(encoding="utf-8"))
    error = _refused(capsys, folders[2], runs)
    assert f"{first['items'][0]['item_id']!r} (row 2), " in error
    assert "are not items of the sheet" in error
    # The same sample, exported again, has the same IDs: its labels are read.
    assert run_label_check(folders[1], judge=_judge(), runs_dir=runs) == 0


def test_a_judge_of_the_tested_models_family_is_refused(
    labeled: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    judge = _judge(model="tiny-judge")  # of tiny-vl's family, tiny
    error = _refused(capsys, *labeled, judge)
    assert "the judge tiny-judge is of the family of tiny-vl (local/tiny)" in error


@pytest.mark.parametrize(
    ("failure", "message"),
    [
        (CallError("the key was rejected", fatal=True), "fake:fake-judge: the key "),
        (ValueError("no server answers there"), "no server answers there"),
    ],
)
def test_a_judge_that_cannot_answer_is_refused(
    failure: Exception,
    message: str,
    labeled: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    sheet, runs = labeled
    judge = sheets.fake_judge({"coherence": [failure] * 20})
    assert run_label_check(sheet, judge=judge, runs_dir=runs) == 1
    assert capsys.readouterr().err.startswith(f"pushlab label check: {message}")
    assert not list(sheet.glob("check-*.json"))
    assert not list(sheet.glob("disagreements-*.csv"))


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"backend": "local"}, "the local backend needs --model"),
        ({"backend": "cloud"}, "unknown judge backend 'cloud'"),
        ({"backend": "api"}, "the api backend "),  # no ANTHROPIC_API_KEY
        ({"base_url": "http://127.0.0.1:9000/v1"}, "the api backend calls the "),
    ],
)
def test_a_judge_that_cannot_be_made_is_refused(
    options: dict[str, str],
    message: str,
    labeled: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    sheet, runs = labeled
    cache = tmp_path / "cache"
    status = run_label_check(sheet, runs_dir=runs, cache_dir=cache, **options)
    assert status == 1
    assert capsys.readouterr().err.startswith(f"pushlab label check: {message}")
    assert not cache.exists() and not list(sheet.glob("check-*.json"))


def test_a_rubric_changed_since_the_export_is_refused_when_labeled(
    labeled: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    sheet, runs = labeled
    _edit_key(sheet, lambda key: key["rubrics"]["coherence"].update(sha256="0" * 64))
    error = _refused(capsys, sheet, runs)
    assert (
        "rubrics/coherence.yaml changed since the sheet was exported, though it is "
        "still version 1: a rubric's new version is checked on a new sheet"
    ) in error
    assert "give --labels a copy of the sheet without the coherence column" in error
    _edit_key(sheet, lambda key: key["rubrics"]["coherence"].update(version=2))
    error = _refused(capsys, sheet, runs)
    assert (
        "coherence was version 2 when the sheet was exported, and "
        "rubrics/coherence.yaml is version 1 now"
    ) in error
    # A changed rubric with no label is not checked, so it is not refused.
    coherence = load_rubric("coherence")
    rubrics = {
        "speaks_as": {"version": 1, "sha256": "0" * 64},
        "coherence": {"version": coherence.version, "sha256": coherence.sha256},
    }
    _edit_key(sheet, lambda key: key.update(rubrics=rubrics))
    assert run_label_check(sheet, judge=_judge(), runs_dir=runs) == 0
    printed = capsys.readouterr().out
    assert "speaks_as (version 1): not labeled: labels.csv has no speaks_as" in printed


def test_a_rubric_whose_new_version_dropped_a_label_is_refused_as_changed(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = tiny_spec(sheets.CONDITIONS), tmp_path / "runs"
    sheets.saved_runs(spec, runs)
    capsys.readouterr()  # the runner's own output
    sheet = sheets.make_sheet(tmp_path, spec, runs, ["speaks_as"], sheets.OWL, ())
    # Labeled with speaks_as's version 1, other_persona among the labels.
    cells = ["other_persona"] * 3 + ["assistant"] * 17
    sheets.save(sheet / "labels.csv", sheets.rows(sheet, {"speaks_as": cells}))
    # Its version 2 has no other_persona.
    old = load_rubric("speaks_as")
    kept = tuple(pair for pair in old.labels if pair[0] != "other_persona")
    new = replace(old, version=2, labels=kept, sha256="2" * 64)
    monkeypatch.setattr(
        label_check,
        "load_rubric",
        lambda name: new if name == "speaks_as" else load_rubric(name),
    )
    error = _refused(capsys, sheet, runs)
    # The rubric changed, not your labels: you are not asked to fix them.
    assert error.startswith(
        "pushlab label check: speaks_as was version 1 when the sheet was "
        "exported, and rubrics/speaks_as.yaml is version 2 now: a rubric's new "
        "version is checked on a new sheet"
    )
    assert "fix these" not in error and "other_persona" not in error


def test_a_rubric_whose_persona_the_key_lacks_is_refused(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = tiny_spec(sheets.CONDITIONS), tmp_path / "runs"
    sheets.saved_runs(spec, runs)
    capsys.readouterr()  # the runner's own output
    rubrics = ["speaks_as", "which_persona"]
    sheet = sheets.make_sheet(tmp_path, spec, runs, rubrics, None, ())
    for column, label, message in [
        ("speaks_as", "assistant", "key.json names no persona, which speaks_as asks"),
        ("which_persona", "none", "key.json lists no personas, which which_persona"),
    ]:
        table = sheets.rows(sheet, {column: [label] * 20})
        sheets.save(sheet / "labels.csv", table)
        assert message in _refused(capsys, sheet, runs)


def test_an_out_file_that_exists_or_is_a_folder_is_refused(
    labeled: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sheet, runs = labeled
    out = tmp_path / "check.json"
    out.write_text("{}\n", encoding="utf-8")
    error = _refused(capsys, sheet, runs, out=out)
    assert f"{out} already exists: give another --out" in error
    assert out.read_text(encoding="utf-8") == "{}\n"
    error = _refused(capsys, sheet, runs, out=tmp_path)
    assert f"--out must be a file, but {tmp_path} is a folder" in error


def test_a_folder_that_is_not_a_labeled_sheet_is_refused(
    labeled: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sheet, runs = labeled
    nowhere = tmp_path / "nowhere"
    judge = _judge()
    assert run_label_check(nowhere, judge=judge, runs_dir=runs) == 1
    error = capsys.readouterr().err
    assert f"{nowhere} is not a folder: give the folder that pushlab label " in error
    error = _refused(capsys, sheet, runs, labels=tmp_path / "missing.csv")
    assert f"cannot read {tmp_path / 'missing.csv'}" in error
    sheets.save(sheet / "labels.csv", sheets.rows(sheet, {"coherence": []}))
    error = _refused(capsys, sheet, runs)
    assert f"{sheet / 'labels.csv'} holds no label: fill in" in error
    (sheet / "key.json").unlink()
    error = _refused(capsys, sheet, runs)
    assert f"cannot read {sheet / 'key.json'}" in error
