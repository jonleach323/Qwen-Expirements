"""pushlab label check (pushlab.scoring.label_check): a judge's labels
against yours, and the report and record of how well they agree.

Each sheet is made by hand over a runner folder saved by the real runner on
the tiny model, with invented replies, and labeled as a spreadsheet saves a
sheet; a FakeBackend answers by each reply's text, so every number here is
worked out by hand (tests/unit/label_sheets.py). Item k is run k.
test_label_check_refusals.py has what the check refuses,
test_label_check_outputs.py too few labels and disagreements.csv, and
test_label_report.py the report's parts.
"""

import doctest
import hashlib
import importlib.util
import json
import re
from collections import Counter
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import torch

from pushlab.cli import main
from pushlab.runs.shards import read_results
from pushlab.scoring import label_check
from pushlab.scoring.cache import VerdictCache
from pushlab.scoring.fake_backend import FakeBackend
from pushlab.scoring.judge import Reply
from pushlab.scoring.label_check import run_label_check
from pushlab.scoring.rubrics import load_rubric

# The sheets' helpers, loaded by path, as test modules import nothing from
# one another.
_SPEC = importlib.util.spec_from_file_location(
    "label_sheets", Path(__file__).with_name("label_sheets.py")
)
sheets = importlib.util.module_from_spec(_SPEC)  # type: ignore[arg-type]
_SPEC.loader.exec_module(sheets)  # type: ignore[union-attr]

NOW = datetime(2026, 11, 13, 9, 0, 0, tzinfo=UTC)
OUT = "check-fake-fake-judge-20261113T090000Z.json"
FEW = 6  # the labels a rubric needs to pass here: these sheets are small


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


def _coherence_sheet(tmp_path: Path, saved: tuple[Path, Path]) -> Path:
    return sheets.make_sheet(tmp_path, *saved, ["coherence"], None, ())


def _record(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_a_sheet_as_a_spreadsheet_saves_it_is_compared_with_the_judge(
    saved: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec, runs = saved
    sheet = sheets.make_sheet(tmp_path, spec, runs)
    # speaks_as: you label replies 1 to 10 speaks_as and 11 to 18 assistant,
    # in any case and with spaces around, and skip 19 and 20. coherence: 7
    # for replies 1 to 10, 3 for 11 to 20. which_persona's column is gone.
    speaks_as = [" Speaks_As ", "SPEAKS_AS", *["speaks_as"] * 8, "Assistant "]
    speaks_as += ["assistant"] * 7
    coherence = [" 7 ", *["7"] * 9, *["3"] * 10]
    table = sheets.rows(sheet, {"Speaks_As ": speaks_as, " COHERENCE": coherence})
    table[0][0] = " Item_ID"
    sheets.save(sheet / "labels.csv", table)
    # The judge: speaks_as for 1 to 8, about_theme for 9 and 10, assistant
    # for 11 to 17, speaks_as for 18; 7 for 1 to 9, 8 for 10, 3 for 11 to
    # 19 and 2 for 20.
    judge = sheets.fake_judge(
        {
            "speaks_as": [*["speaks_as"] * 8, *["about_theme"] * 2]
            + [*["assistant"] * 7, "speaks_as"],
            "coherence": [*[7] * 9, 8, *[3] * 9, 2],
        }
    )
    assert run_label_check(sheet, judge=judge, runs_dir=runs, min_labels=FEW) == 0
    printed = capsys.readouterr().out
    # speaks_as: 15 of 18 agree. Yours: speaks_as 10, assistant 8; the
    # judge's: speaks_as 9, about_theme 2, assistant 7. s = 10 * 9 + 8 * 7 =
    # 146, so kappa = (18 * 15 - 146) / (18**2 - 146) = 124/178 = 62/89.
    assert "speaks_as (version 1), about the persona Brass Owl: FAIL" in printed
    assert "  Your labels: 18 read, 2 skipped." in printed
    assert "  Accuracy: 0.833 (15 of 18 agree)." in printed
    assert "  Cohen's kappa: 0.696." in printed  # 0.6966, cut, not rounded
    assert re.search(r"\n +1 speaks_as +8 +0 +2 +0 +0 +0\n", printed)
    assert re.search(r"\n +4 assistant +1 +0 +0 +7 +0 +0\n", printed)
    flat = " ".join(printed.split())  # the list of disagreements, unwrapped
    assert (
        "sheet-1-009 speaks_as/about_theme, sheet-1-010 speaks_as/about_theme, "
        "sheet-1-018 assistant/speaks_as"
    ) in flat
    assert (
        "which_persona (version 1): not labeled: labels.csv has no which_persona "
        "column."
    ) in printed
    # coherence: 18 of 20 agree. Yours: 7 and 3 ten times each; the judge's 7
    # and 3 nine times each, 8 and 2 once: s = 10 * 9 + 10 * 9 = 180, so
    # kappa = (20 * 18 - 180) / (400 - 180) = 9/11. Every score is within a
    # point. Quadratic weights, in squared points: 2 seen over the 20 items;
    # by chance a 7 of yours costs 9 * 0 + 1 * 1 + 9 * 16 + 1 * 25 = 170
    # against the judge's 20, and a 3 costs 9 * 16 + 1 * 25 + 9 * 0 + 1 * 1 =
    # 170: (3400 - 20 * 2) / 3400 = 84/85.
    assert "coherence (version 1): PASS" in printed
    assert "  Accuracy: 0.900 (18 of 20 agree)." in printed
    assert "  Cohen's kappa: 0.818." in printed
    assert "  Within 1 point: 1.000 (20 of 20)." in printed
    assert "  Quadratic-weighted kappa: 0.988." in printed
    assert "applied to exact scores, which is strict" in printed
    assert "Results: speaks_as FAIL, which_persona not labeled, coherence PASS." in (
        printed
    )
    assert f"Saved the numbers, with no reply text: {sheet / OUT}" in printed
    assert "Teapot" not in printed and sheets.REASON not in printed
    record = _record(sheet / OUT)
    key = json.loads((sheet / "key.json").read_text(encoding="utf-8"))
    assert (record["sheet"], record["judge"]) == ("sheet-1", "fake:fake-judge")
    assert record["checked_at"] == "2026-11-13T09:00:00Z"
    assert record["rule"] == {"accuracy": 0.85, "kappa": 0.7, "labels": FEW}
    assert record["labels_csv_sha256"] == key["labels_csv_sha256"]
    filled = (sheet / "labels.csv").read_bytes()
    assert record["labels_file_sha256"] == hashlib.sha256(filled).hexdigest()
    assert (
        record["key_sha256"]
        == hashlib.sha256((sheet / "key.json").read_bytes()).hexdigest()
    )
    assert (record["items"], record["rows"]) == (20, 20)
    assert record["verdicts"] == {"count": 38, "cached": 0, "calls": 38, "failed": 0}
    speaks = record["rubrics"]["speaks_as"]
    rubric = load_rubric("speaks_as")
    assert (speaks["version"], speaks["sha256"]) == (rubric.version, rubric.sha256)
    assert (speaks["persona"], speaks["result"]) == ("Brass Owl", "FAIL")
    assert (speaks["labels"], speaks["skipped"], speaks["failed"]) == (18, 2, 0)
    assert (speaks["agreed"], speaks["accuracy"], speaks["kappa"]) == (
        15,
        15 / 18,
        62 / 89,
    )
    assert speaks["confusion"] == {
        "labels": [label for label, _ in rubric.labels],
        "counts": [[8, 0, 2, 0, 0], [0] * 5, [0] * 5, [1, 0, 0, 7, 0], [0] * 5],
        "failed": [0] * 5,
    }
    assert speaks["disagreements"] == [
        {"item_id": "sheet-1-009", "yours": "speaks_as", "judge": "about_theme"},
        {"item_id": "sheet-1-010", "yours": "speaks_as", "judge": "about_theme"},
        {"item_id": "sheet-1-018", "yours": "assistant", "judge": "speaks_as"},
    ]
    scores = record["rubrics"]["coherence"]
    assert (scores["accuracy"], scores["kappa"]) == (0.9, 9 / 11)
    assert (scores["within_1"], scores["weighted_kappa"]) == (1.0, 84 / 85)
    assert scores["confusion"]["labels"] == list(range(11))
    assert record["rubrics"]["which_persona"] == {
        "version": 1,
        "sha256": load_rubric("which_persona").sha256,
        "personas": ["Brass Owl", "Snow Owl"],
        "result": "not labeled",
        "why": "labels.csv has no which_persona column",
        "labels": 0,
        "skipped": 20,
    }
    text = (sheet / OUT).read_text(encoding="utf-8")
    for word in ("Teapot", "sings", "rests", sheets.REASON):
        assert word not in text  # no reply text, no reason


def test_the_judge_is_asked_what_pushlab_score_asks_and_nothing_more(
    saved: tuple[Path, Path], tmp_path: Path
) -> None:
    spec, runs = saved
    sheet = sheets.make_sheet(tmp_path, spec, runs)
    cells = {"speaks_as": "assistant", "which_persona": "none", "coherence": "5"}
    table = sheets.rows(sheet, {name: [cell] * 20 for name, cell in cells.items()})
    sheets.save(sheet / "labels.csv", table)
    answers = {"speaks_as": ["assistant"] * 20, "which_persona": ["none"] * 20}
    judge = sheets.fake_judge(answers | {"coherence": [5] * 20})
    assert run_label_check(sheet, judge=judge, runs_dir=runs) == 0
    saved_runs = read_results(runs / sheets.NAME)
    owls = (sheets.OWL, sheets.SNOW)
    expected = [
        request
        for run in saved_runs
        for request in (
            load_rubric("speaks_as").request(run.output_text, sheets.OWL),
            load_rubric("which_persona").request(run.output_text, personas=owls),
            load_rubric("coherence").request(run.output_text),
        )
    ]
    # Each request is the rubric's, with the sheet's persona or personas,
    # about the reply's text, as pushlab score builds it from a Job: nothing
    # about its run, condition, experiment or prompt.
    assert Counter(judge.backend.requests) == Counter(expected)
    hidden = [run.run_id for run in saved_runs] + [sheets.NAME, str(spec)]
    hidden += ["pulse", "control", "tiny-vl", "L2", "N100"]
    for request in judge.backend.requests:
        assert not [word for word in hidden if word in request.system + request.user]


@pytest.mark.parametrize(
    ("yours", "theirs", "accuracy", "kappa", "result"),
    [
        # 17 of 20 agree. Yours: 7 and 3 ten times each; the judge's 7 nine
        # times and 3 eleven: s = 10 * 9 + 10 * 11 = 200, so kappa is
        # (20 * 17 - 200) / (400 - 200) = 0.7. Both at the edge: PASS.
        ([7] * 10 + [3] * 10, [7] * 8 + [3] * 11 + [7], 0.85, 0.7, "PASS"),
        # 16 of 20: yours 9, 7, 3 and 1 five times each, and four misses on
        # scores you never gave: s = 4 * 5 * 4 = 80, so kappa is
        # (320 - 80) / (400 - 80) = 0.75, but the accuracy is 0.80.
        (
            [9] * 5 + [7] * 5 + [3] * 5 + [1] * 5,
            [8] + [9] * 4 + [6] + [7] * 4 + [2] + [3] * 4 + [0] + [1] * 4,
            0.8,
            0.75,
            "FAIL",
        ),
        # 17 of 20: yours 7 twelve times and 3 eight; the judge's 7 eleven
        # times and 3 nine: s = 12 * 11 + 8 * 9 = 204, so kappa is
        # (340 - 204) / (400 - 204) = 136/196, below 0.70.
        ([7] * 12 + [3] * 8, [7] * 10 + [3] * 9 + [7], 0.85, 136 / 196, "FAIL"),
    ],
)
def test_pass_and_fail_at_the_rules_edges(
    yours: list[int],
    theirs: list[int],
    accuracy: float,
    kappa: float,
    result: str,
    saved: tuple[Path, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sheet = _coherence_sheet(tmp_path, saved)
    cells = [str(score) for score in yours]
    sheets.save(sheet / "labels.csv", sheets.rows(sheet, {"coherence": cells}))
    judge = sheets.fake_judge({"coherence": theirs})
    assert run_label_check(sheet, judge=judge, runs_dir=saved[1], min_labels=FEW) == 0
    assert f"coherence (version 1): {result}" in capsys.readouterr().out
    numbers = _record(sheet / OUT)["rubrics"]["coherence"]
    assert (numbers["accuracy"], numbers["kappa"]) == (accuracy, kappa)
    assert numbers["result"] == result


def test_a_failed_verdict_counts_as_a_disagreement(
    saved: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sheet = sheets.make_sheet(tmp_path, *saved, ["which_persona"], None)
    # You: replies 1 to 3 speak as the brass owl, 4 to 6 as none; the rest
    # are skipped. The judge: Brass Owl for 1 and 2, a malformed answer for 3
    # at each of its 4 attempts, and none for 4 to 6.
    cells = ["brass owl"] * 3 + ["None"] * 3
    sheets.save(sheet / "labels.csv", sheets.rows(sheet, {"which_persona": cells}))
    malformed = Reply('{"reason": "Hm.", "choice": "Owl"}')
    answers = [*["Brass Owl"] * 2, malformed, *["none"] * 3]
    judge = sheets.fake_judge({"which_persona": answers})
    assert run_label_check(sheet, judge=judge, runs_dir=saved[1], min_labels=FEW) == 0
    assert judge.backend.calls == 2 + 4 + 3
    printed = capsys.readouterr().out
    # 5 of 6 agree: the failure is never a label. s = 3 * 2 + 3 * 3 = 15
    # (the failure is a label you never give), so kappa is
    # (6 * 5 - 15) / (36 - 15) = 5/7.
    assert "which_persona (version 1), among 2 personas: FAIL" in printed
    assert (
        "  Failed verdicts: 1, each counted as a disagreement (malformed answer: 1)."
    ) in printed
    assert "  Accuracy: 0.833 (5 of 6 agree)." in printed
    assert "  Cohen's kappa: 0.714." in printed
    assert "sheet-1-003 Brass Owl/failed" in printed
    numbers = _record(sheet / OUT)["rubrics"]["which_persona"]
    assert (numbers["failed"], numbers["failures"]) == (1, {"malformed answer": 1})
    assert (numbers["accuracy"], numbers["kappa"]) == (5 / 6, 5 / 7)
    assert numbers["confusion"] == {
        "labels": ["Brass Owl", "Snow Owl", "none"],
        "counts": [[2, 0, 0], [0, 0, 0], [0, 0, 3]],
        "failed": [1, 0, 0],
    }
    assert numbers["disagreements"] == [
        {"item_id": "sheet-1-003", "yours": "Brass Owl", "judge": None}
    ]


def test_empty_cells_rows_left_out_and_empty_columns_are_skipped(
    saved: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sheet = sheets.make_sheet(tmp_path, *saved)
    # coherence: 5 and 6 left empty, and item 20's row deleted; speaks_as's
    # column empty; which_persona's gone; a blank row in the middle.
    coherence = ["7"] * 4 + ["", ""] + ["7"] * 4 + ["3"] * 10
    table = sheets.rows(sheet, {"speaks_as": [], "coherence": coherence})
    del table[20]
    table.insert(8, [""] * 5)
    sheets.save(sheet / "labels.csv", table)
    judge = sheets.fake_judge({"coherence": [7] * 10 + [3] * 10})
    assert run_label_check(sheet, judge=judge, runs_dir=saved[1], min_labels=FEW) == 0
    printed = capsys.readouterr().out
    assert "1 of the 20 replies is not in labels.csv: skipped for every rubric." in (
        printed
    )
    assert "speaks_as (version 1): not labeled: its column is empty." in printed
    assert "which_persona (version 1): not labeled: labels.csv has no which_" in (
        printed
    )
    assert "  Your labels: 17 read, 3 skipped." in printed
    asked = sorted(sheets.number(request.text) for request in judge.backend.requests)
    assert asked == [1, 2, 3, 4, *range(7, 20)]
    record = _record(sheet / OUT)
    assert (record["items"], record["rows"]) == (20, 19)
    numbers = record["rubrics"]["coherence"]
    assert (numbers["labels"], numbers["skipped"], numbers["result"]) == (17, 3, "PASS")


def test_a_second_check_is_answered_by_the_cache(
    saved: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    sheet = _coherence_sheet(tmp_path, saved)
    # Your labels in a file of their own, given with labels.
    mine = sheets.save(
        tmp_path / "mine.csv", sheets.rows(sheet, {"coherence": ["5"] * 20})
    )
    cache = tmp_path / "cache"
    first = sheets.fake_judge(
        {"coherence": [5] * 20}, VerdictCache(cache, "fake:fake-judge")
    )
    options = {"labels": mine, "runs_dir": saved[1], "min_labels": FEW}
    assert run_label_check(sheet, judge=first, **options) == 0
    printed = capsys.readouterr().out
    # Every label and score is 5: chance agreement is 1, so kappa is
    # undefined, and an undefined kappa fails.
    assert "coherence (version 1): FAIL" in printed
    assert "  Cohen's kappa: undefined." in printed
    assert "  Quadratic-weighted kappa: undefined." in printed
    assert first.backend.calls == 20
    assert _record(sheet / OUT)["labels_file"] == str(mine)
    second = sheets.fake_judge({}, VerdictCache(cache, "fake:fake-judge"))
    out = tmp_path / "again.json"
    assert run_label_check(sheet, judge=second, out=out, **options) == 0
    assert second.backend.calls == 0
    assert "The judge gave 20 verdicts: 20 from its cache, 0 calls made, 0 failed." in (
        capsys.readouterr().out
    )
    assert _record(out)["verdicts"] == {
        "count": 20,
        "cached": 20,
        "calls": 0,
        "failed": 0,
    }


def test_pushlab_label_check_makes_the_judge_the_options_name(
    saved: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sheet = _coherence_sheet(tmp_path, saved)
    sheets.save(sheet / "labels.csv", sheets.rows(sheet, {"coherence": ["5"] * 20}))
    made: list[tuple[Any, ...]] = []

    def make_backend(backend: str, model: str, base_url: str) -> FakeBackend:
        made.append((backend, model, base_url))
        answer = json.dumps({"reason": sheets.REASON, "score": 5})
        return FakeBackend(lambda request: answer, name=backend, model=model)

    monkeypatch.setattr(label_check, "make_backend", make_backend)
    cache = tmp_path / "cache"
    argv = ["label", "check", str(sheet), "--backend", "local", "--model"]
    argv += ["granite-4.1-8b", "--base-url", "http://127.0.0.1:9000/v1"]
    argv += ["--cache-dir", str(cache), "--runs-dir", str(saved[1])]
    assert main(argv) == 0
    assert made == [("local", "granite-4.1-8b", "http://127.0.0.1:9000/v1")]
    assert len((cache / "local-granite-4.1-8b.jsonl").read_text().splitlines()) == 20
    out = sheet / "check-local-granite-4.1-8b-20261113T090000Z.json"
    assert _record(out)["judge"] == "local:granite-4.1-8b"
    assert "judge local:granite-4.1-8b" in capsys.readouterr().out


def test_docstring_examples_run() -> None:
    result = doctest.testmod(label_check)
    assert result.attempted > 0
    assert result.failed == 0
