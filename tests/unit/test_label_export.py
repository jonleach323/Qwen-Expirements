"""pushlab label export (pushlab.scoring.label_export): a sheet of sampled
replies for the owner to label, its instructions and its key, and what it
refuses. test_label_sheet.py tests the sample and the files' content alone.

The runner folders are saved by the real runner on the tiny model, whose
four-token replies are a few strings of odd characters; then each run is
given an invented reply in place of its own (_rewrite), so no text here
comes from a real run (decision 0020). PULSES has 15 runs, three on each of
the five smoke prompts, in this order: an unpushed run (none: runs 1, 4, 7,
10 and 13), a pulse (2, 5, 8, 11, 14) and the pulse's random control (3,
6, 9, 12, 15).
"""

import csv
import doctest
import hashlib
import io
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from pushlab.cli import main
from pushlab.prompts import PROMPTS_DIR, read_prompt_set
from pushlab.runs.folder import SHARDS, read_identity
from pushlab.runs.shards import read_results, write_shard
from pushlab.scoring import label_export
from pushlab.scoring.label_export import run_label_export
from pushlab.scoring.rubrics import RUBRIC_NAMES, load_rubric
from pushlab.store.records import RunRecord, read_runs_file
from pushlab.store.scores import output_sha256

NAME = "tiny-pulses"
PULSES = """\
conditions:
  - label: none
    segments: []
  - label: pulse
    segments: [{unit: tiny-vl:L2:N100, alpha: 1.0}]
    controls: 1
    control_seed: 3
"""
NOW = "2026-11-13T09:00:00Z"
KEYS = ["sheet", "created_at", "seed", "size", "rubrics", "persona", "personas"]
KEYS += ["labels_csv_sha256", "strata", "items"]
ITEM_KEYS = ["item_id", "spec", "experiment", "version", "run_id", "condition"]
ITEM_KEYS += ["control", "stratum", "output_sha256"]
OWLS = """\
personas:
  - name: Brass Owl
    description: A pompous mechanical owl who lectures in riddles.
  - name: Snow Owl
    description: A quiet white owl of the far north.
"""


@pytest.fixture(autouse=True)
def cpu_and_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU even on a machine with a GPU, and fix key.json's time."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(label_export, "now", lambda: NOW)


def _number(run: RunRecord) -> int:
    return int(run.run_id.rsplit("-", 1)[1])


def _texts(
    word: str, special: dict[int, str] | None = None
) -> Callable[[RunRecord], str]:
    """Invented replies: run k's is special[k], else "<word> reply k hums."."""

    def text(run: RunRecord) -> str:
        number = _number(run)
        return (special or {}).get(number, f"{word} reply {number} hums.")

    return text


def _rewrite(folder: Path, texts: Callable[[RunRecord], str]) -> None:
    """Give each run saved in a runner folder the reply texts(run) in place of
    the tiny model's: each shard is written again under its own name."""
    for path in sorted((folder / SHARDS).glob("*.parquet")):
        runs = [replace(run, output_text=texts(run)) for run in read_runs_file(path)]
        path.unlink()
        write_shard(folder, path.name, runs)


def _run(spec: Path, runs: Path) -> None:
    assert main(["run", str(spec), "--runs-dir", str(runs)]) == 0


@pytest.fixture
def pulses(tiny_spec: Callable[..., Path], tmp_path: Path) -> tuple[Path, Path]:
    """PULSES's experiment file, and a runs folder holding its runs, run k
    replying "Kettle reply k hums."."""
    spec, runs = tiny_spec(PULSES), tmp_path / "runs"
    _run(spec, runs)
    _rewrite(runs / NAME, _texts("Kettle"))
    return spec, runs


@pytest.fixture
def owls(tmp_path: Path) -> Path:
    """A personas file of two invented owls."""
    path = tmp_path / "owls.yaml"
    path.write_text(OWLS, encoding="utf-8")
    return path


def _key(sheet: Path) -> dict:
    return json.loads((sheet / "key.json").read_text(encoding="utf-8"))


def _rows(sheet: Path) -> list[dict[str, str]]:
    text = (sheet / "labels.csv").read_bytes().decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text, newline="")))


def test_each_stratum_gets_an_equal_share_and_a_small_one_all_it_has(
    pulses: tuple[Path, Path],
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = pulses
    blank = dict.fromkeys([9, 12, 15], "  \n")  # three of the five controls
    _rewrite(runs / NAME, _texts("Kettle", blank | {4: " Kettle reply 1 hums. "}))
    more = tiny_spec(file="more.yaml", name="tiny-more")  # none and pulse, 10 runs
    _run(more, runs)
    _rewrite(runs / "tiny-more", _texts("Lantern"))
    sheet = tmp_path / "labels" / "sheet-1"
    status = run_label_export(
        [spec, more], sheet, rubrics=["coherence"], size=12, runs_dir=runs
    )
    assert status == 0
    # 4 none (run 4 repeats run 1), 5 pulses, 2 controls, 5 and 5: a share
    # of 12 / 5 each; the controls are taken whole, the rest split 3 3 2 2.
    key = _key(sheet)
    assert key["strata"] == {
        "tiny-pulses / none": 3,
        "tiny-pulses / pulse": 3,
        "tiny-pulses / pulse / control": 2,
        "tiny-more / none": 2,
        "tiny-more / pulse": 2,
    }
    assert list(key) == KEYS and key["sheet"] == "sheet-1"
    assert (key["created_at"], key["seed"], key["size"]) == (NOW, 0, 12)
    coherence = load_rubric("coherence")
    assert key["rubrics"] == {
        "coherence": {"version": coherence.version, "sha256": coherence.sha256}
    }
    assert (key["persona"], key["personas"]) == (None, [])
    labels_csv = (sheet / "labels.csv").read_bytes()
    assert key["labels_csv_sha256"] == hashlib.sha256(labels_csv).hexdigest()
    saved = {
        run.run_id: (path, run, read_identity(runs / name)["conditions"])
        for path, name in [(spec, NAME), (more, "tiny-more")]
        for run in read_results(runs / name)
    }
    rows = _rows(sheet)
    assert [item["item_id"] for item in key["items"]] == [
        f"sheet-1-{n:03d}" for n in range(1, 13)
    ]
    assert [row["item_id"] for row in rows] == [i["item_id"] for i in key["items"]]
    for item, row in zip(key["items"], rows, strict=True):
        assert list(item) == ITEM_KEYS
        path, run, conditions = saved[item["run_id"]]
        assert row["text"] == run.output_text.strip()
        assert item["spec"] == str(path) and item["experiment"] == run.experiment
        assert item["version"] == 1 and item["condition"] == conditions[run.run_id]
        assert item["control"] == bool(run.control_of)
        suffix = " / control" if run.control_of else ""
        assert item["stratum"] == f"{run.experiment} / {item['condition']}{suffix}"
        assert item["output_sha256"] == output_sha256(run)
    assert "tiny-pulses-000004" not in {item["run_id"] for item in key["items"]}
    printed = capsys.readouterr().out
    assert "tiny-pulses / pulse / control: 2 of 2" in printed
    assert "tiny-pulses / none: 3 of 4" in printed
    assert "3 blank replies and 1 that repeats a text taken already" in printed
    for name in ("labels.csv", "instructions.md", "key.json"):
        assert str(sheet / name) in printed
    assert "hums" not in printed  # no reply's text


def test_the_sheet_is_blind_guarded_and_readable_by_spreadsheets(
    pulses: tuple[Path, Path],
    owls: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = pulses
    _rewrite(runs / NAME, _texts("Kettle", {1: "=1+1", 2: "- item\nand more"}))
    sheet = tmp_path / "sheet"
    sheet.mkdir()  # an empty folder takes a sheet, as a new one does
    status = run_label_export(
        [spec], sheet, size=100, personas_file=owls, persona="Brass Owl", runs_dir=runs
    )
    assert status == 0
    assert "only 15 replies, fewer than --size 100" in capsys.readouterr().out
    data = (sheet / "labels.csv").read_bytes()
    assert data.startswith(b"\xef\xbb\xbfitem_id,text,")  # UTF-8's byte-order mark
    rows = _rows(sheet)
    assert list(rows[0]) == ["item_id", "text", *RUBRIC_NAMES, "note"]
    assert all(row[name] == "" for row in rows for name in [*RUBRIC_NAMES, "note"])
    texts = sorted(row["text"] for row in rows)
    assert texts[:2] == ["'- item\nand more", "'=1+1"] and len(texts) == 15
    text = data.decode("utf-8-sig")
    conditions = read_identity(runs / NAME)["conditions"]
    prompts = read_prompt_set(PROMPTS_DIR / "smoke.jsonl")
    hidden = [*conditions, *set(conditions.values()), "control", NAME]
    hidden += ["tiny-vl", "L2", "N100", "alpha", *prompts.prompts, *prompts.prompt_ids]
    for word in hidden:
        assert word not in text
    instructions = (sheet / "instructions.md").read_text(encoding="utf-8")
    hidden = [*conditions, "pulse", NAME, "tiny-vl", *prompts.prompts]
    assert not [word for word in hidden if word in instructions]


def test_one_seed_gives_the_same_files_and_another_another_sample(
    pulses: tuple[Path, Path], tmp_path: Path
) -> None:
    spec, runs = pulses
    sheets = [tmp_path / place / "sheet" for place in ("a", "b", "c")]
    # The first through the command, with the default seed, 0.
    argv = ["label", "export", str(spec), "--out", str(sheets[0]), "--size", "6"]
    argv += ["--rubric", "coherence", "--runs-dir", str(runs)]
    assert main(argv) == 0
    for sheet, seed in zip(sheets[1:], [0, 1], strict=True):
        options = {"rubrics": ["coherence"], "size": 6, "seed": seed}
        assert run_label_export([spec], sheet, runs_dir=runs, **options) == 0
    for name in ("labels.csv", "instructions.md", "key.json"):
        assert (sheets[0] / name).read_bytes() == (sheets[1] / name).read_bytes()
    first, other = (
        [item["run_id"] for item in _key(sheet)["items"]] for sheet in sheets[::2]
    )
    assert first != other and len(first) == len(other) == 6
    assert _key(sheets[2])["seed"] == 1


def test_the_instructions_give_each_rubrics_answers_and_the_persona(
    pulses: tuple[Path, Path], owls: Path, tmp_path: Path
) -> None:
    spec, runs = pulses
    sheet = tmp_path / "sheet"
    status = run_label_export(
        [spec], sheet, personas_file=owls, persona="Snow Owl", runs_dir=runs
    )
    assert status == 0
    key = _key(sheet)
    assert key["persona"] == {
        "name": "Snow Owl",
        "description": "A quiet white owl of the far north.",
    }
    assert [persona["name"] for persona in key["personas"]] == ["Brass Owl", "Snow Owl"]
    text = " ".join((sheet / "instructions.md").read_text(encoding="utf-8").split())
    for label, meaning in load_rubric("speaks_as").labels:
        assert f"`{label}`: {meaning}" in text
    assert "The persona: **Snow Owl**. A quiet white owl of the far north." in text
    assert "`Brass Owl`: A pompous mechanical owl who lectures in riddles." in text
    assert "`none`: none of these personas" in text
    assert "- 10: fluent and clear throughout." in text
    assert "- 7 to 9: readable, with small slips or odd turns." in text
    assert "Write a whole number from 0 to 10." in text
    assert "Leave a cell empty to skip that reply" in text
    assert "Do not open key.json while you label." in text


def test_an_experiment_with_no_reply_or_given_twice_is_refused(
    pulses: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec, runs = pulses
    sheet, coherence = tmp_path / "sheet", ["coherence"]
    assert run_label_export([spec, spec], sheet, rubrics=coherence, runs_dir=runs) == 1
    assert "give each experiment once" in capsys.readouterr().err
    _rewrite(runs / NAME, lambda run: " ")
    assert run_label_export([spec], sheet, rubrics=coherence, runs_dir=runs) == 1
    error = capsys.readouterr().err
    assert "no reply to sample: of the runs saved, 15 have a blank reply" in error
    assert not sheet.exists()


def test_a_sheet_inside_the_repository_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    inside = label_export.PROJECT_ROOT / "no-such-labels-folder"
    assert run_label_export([tmp_path / "e.yaml"], inside) == 1
    assert "outside the repository" in capsys.readouterr().err
    assert not inside.exists()
    monkeypatch.setattr(label_export, "PROJECT_ROOT", tmp_path / "repo")
    sheet = tmp_path / "repo" / "labels" / "sheet"
    assert run_label_export([tmp_path / "e.yaml"], sheet) == 1
    assert "outside the repository" in capsys.readouterr().err
    assert not sheet.exists()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"rubrics": ["pirate"]}, "unknown rubric 'pirate'"),
        ({"size": 0}, "--size must be a whole number from 1"),
        ({"seed": -1}, "--seed must be a whole number from 0"),
        ({}, "speaks_as asks about a persona: give --personas FILE"),
        ({"rubrics": ["which_persona"]}, "which_persona chooses among personas"),
        ({"personas_file": "owls"}, "choose one persona with --persona"),
        ({"personas_file": "owls", "persona": "Owl"}, "no persona is named 'Owl'"),
        ({"personas_file": "missing"}, "cannot read the personas file"),
        ({"personas_file": "bad"}, "must be a list of at least one persona"),
        (
            {"rubrics": ["coherence"], "persona": "Snow Owl"},
            "--persona is for a rubric that asks about one persona",
        ),
        (
            {"rubrics": ["which_persona"], "personas_file": "owls", "persona": "Owl"},
            "--persona is for a rubric that asks about one persona",
        ),
        (
            {"rubrics": ["coherence"], "personas_file": "owls"},
            "--personas is for a rubric that asks about personas",
        ),
        ({"rubrics": ["coherence"]}, "does not exist: no run of tiny-pulses"),
    ],
)
def test_unmet_needs_are_refused_before_anything_is_written(
    options: dict,
    message: str,
    owls: Path,
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("personas: 3\n", encoding="utf-8")
    files = {"owls": owls, "bad": bad, "missing": tmp_path / "missing.yaml"}
    if "personas_file" in options:
        options = options | {"personas_file": files[options["personas_file"]]}
    sheet, runs = tmp_path / "sheet", tmp_path / "runs"
    assert run_label_export([tiny_spec(PULSES)], sheet, runs_dir=runs, **options) == 1
    error = capsys.readouterr().err
    assert error.startswith("pushlab label export: ") and message in error
    assert not sheet.exists()


@pytest.mark.parametrize(
    ("place", "message"),
    [
        ("full", "is not empty: give a new or empty folder"),
        ("full/old.csv", "--out must be a folder, but"),
        ("-sheet", "must start with a letter or a digit"),
    ],
)
def test_a_folder_that_cannot_take_the_sheet_is_refused(
    place: str, message: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "full").mkdir()
    (tmp_path / "full" / "old.csv").write_text("item_id\n", encoding="utf-8")
    status = run_label_export(
        [tmp_path / "e.yaml"], tmp_path / place, rubrics=["coherence"]
    )
    assert status == 1 and message in capsys.readouterr().err
    assert sorted(path.name for path in tmp_path.iterdir()) == ["full"]
    assert [path.name for path in (tmp_path / "full").iterdir()] == ["old.csv"]


def test_docstring_examples_run() -> None:
    result = doctest.testmod(label_export)
    assert result.attempted > 0
    assert result.failed == 0
