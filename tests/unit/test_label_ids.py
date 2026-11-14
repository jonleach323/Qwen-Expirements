"""The IDs pushlab label export gives a sheet's items, and what pushlab label
check makes of another export's labels (decision 0021). An ID holds the
sample's token (pushlab.scoring.label_sheet.sample_token), which covers
what you label: each reply, by its run and text, and the persona or
personas asked about, but not the runner folder's version. So the labels
of the same runs, once their replies changed or about another persona,
match none of a new sheet's items, and check refuses them; an identical
export, or one of a new version that kept the replies, has the same IDs,
and its labels are read.

Each runner folder is saved by the real runner on the tiny model, with
invented replies, and labeled as a spreadsheet saves a sheet; a FakeBackend
answers (tests/unit/label_sheets.py).
"""

import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch

from pushlab.cli import main
from pushlab.scoring.label_check import run_label_check
from pushlab.scoring.label_export import run_label_export

# The sheets' helpers, loaded by path, as test modules import nothing from
# one another.
_SPEC = importlib.util.spec_from_file_location(
    "label_sheets", Path(__file__).with_name("label_sheets.py")
)
sheets = importlib.util.module_from_spec(_SPEC)  # type: ignore[arg-type]
_SPEC.loader.exec_module(sheets)  # type: ignore[union-attr]

OWLS = """\
personas:
  - name: Brass Owl
    description: A pompous mechanical owl who lectures in riddles.
  - name: Snow Owl
    description: A quiet white owl of the far north.
"""
BARN = "  - name: Barn Owl\n    description: A pale owl of old barns.\n"


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def saved(tiny_spec: Callable[..., Path], tmp_path: Path) -> tuple[Path, Path]:
    """The experiment file of sheets.CONDITIONS and a runs folder holding its
    20 runs, run k replying "Teapot reply k sings."."""
    spec, runs = tiny_spec(sheets.CONDITIONS), tmp_path / "runs"
    sheets.saved_runs(spec, runs)
    return spec, runs


def _key(sheet: Path) -> dict[str, Any]:
    return json.loads((sheet / "key.json").read_text(encoding="utf-8"))


def _ids(sheet: Path) -> list[str]:
    return [item["item_id"] for item in _key(sheet)["items"]]


def _runs(sheet: Path) -> list[str]:
    return [item["run_id"] for item in _key(sheet)["items"]]


@pytest.mark.parametrize("change", ["replies", "persona"])
def test_labels_are_refused_once_the_replies_or_the_persona_changed(
    change: str,
    saved: tuple[Path, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = saved
    owls = tmp_path / "owls.yaml"
    owls.write_text(OWLS, encoding="utf-8")
    options: dict[str, Any] = {"rubrics": ["speaks_as"], "size": 6, "runs_dir": runs}
    options |= {"personas_file": owls, "persona": "Brass Owl"}
    first, again, other = (tmp_path / place / "sheet" for place in ("a", "b", "c"))
    for sheet in (first, again):
        assert run_label_export([spec], sheet, **options) == 0
    if change == "replies":
        # The runs made again, as a new version of the runner folder makes
        # them, keeping their IDs, with other replies.
        sheets.rewrite(
            runs / sheets.NAME,
            lambda run: f"Kettle reply {sheets.number(run.run_id)} hums.",
        )
    else:
        options["persona"] = "Snow Owl"  # speaks_as about another persona
    assert run_label_export([spec], other, **options) == 0
    capsys.readouterr()  # the runner's and export's own output
    # An identical export keeps the IDs. The other samples the same runs in
    # the same order, but asks about other replies or another persona: none
    # of its IDs is the first's.
    assert _ids(again) == _ids(first)
    assert _runs(other) == _runs(first)
    assert not set(_ids(other)) & set(_ids(first))
    # The first sheet's labels, saved as the other's, are refused.
    sheets.save(
        other / "labels.csv", sheets.rows(first, {"speaks_as": ["assistant"] * 6})
    )
    judge = sheets.fake_judge({"speaks_as": ["assistant"] * 20})
    assert run_label_check(other, judge=judge, runs_dir=runs) == 1
    assert judge.backend.calls == 0
    error = capsys.readouterr().err
    assert f"{_ids(first)[0]!r} (row 2), " in error
    assert "are not items of the sheet" in error


def test_a_new_version_that_kept_the_replies_keeps_the_ids(
    saved: tuple[Path, Path],
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = saved
    first, later = tmp_path / "a" / "sheet", tmp_path / "b" / "sheet"
    options = {"rubrics": ["coherence"], "size": 6, "runs_dir": runs}
    assert run_label_export([spec], first, **options) == 0
    # The pulse half as strong: a new version of the runner folder, its
    # pushed runs made again; each run is then given its invented reply
    # again, so the replies are those the first sheet holds.
    weaker = tiny_spec(sheets.CONDITIONS.replace("alpha: 1.0", "alpha: 0.5"))
    argv = ["run", str(weaker), "--runs-dir", str(runs), "--overwrite-version"]
    assert main(argv) == 0
    sheets.rewrite(runs / sheets.NAME)
    assert run_label_export([weaker], later, **options) == 0
    capsys.readouterr()  # the runner's and export's own output
    assert {item["version"] for item in _key(first)["items"]} == {1}
    assert {item["version"] for item in _key(later)["items"]} == {2}
    assert _ids(later) == _ids(first)
    # So the first sheet's labels are read as the new one's.
    sheets.save(later / "labels.csv", sheets.rows(first, {"coherence": ["5"] * 6}))
    judge = sheets.fake_judge({"coherence": [5] * 20})
    assert run_label_check(later, judge=judge, runs_dir=runs) == 0
    assert judge.backend.calls == 6


def test_the_personas_listed_count_only_when_which_persona_is_asked(
    saved: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec, runs = saved
    owls, more = tmp_path / "owls.yaml", tmp_path / "more.yaml"
    owls.write_text(OWLS, encoding="utf-8")
    more.write_text(OWLS + BARN, encoding="utf-8")  # a third owl listed
    ids = {}
    for rubric, persona in [("which_persona", None), ("speaks_as", "Brass Owl")]:
        for file in (owls, more):
            sheet = tmp_path / rubric / file.stem / "sheet"
            options = {"rubrics": [rubric], "personas_file": file, "persona": persona}
            assert (
                run_label_export([spec], sheet, size=6, runs_dir=runs, **options) == 0
            )
            ids[rubric, file.stem] = _ids(sheet)
    capsys.readouterr()  # the runner's and export's own output
    # which_persona chooses among other personas: other IDs.
    assert not set(ids["which_persona", "owls"]) & set(ids["which_persona", "more"])
    # speaks_as asks about the same owl, whatever else the file lists.
    assert ids["speaks_as", "owls"] == ids["speaks_as", "more"]
