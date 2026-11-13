"""What pushlab label check reports (pushlab.scoring.label_report): the
plan's rule, numbers cut and never rounded up, a rubric's numbers with a
failed verdict counted as a disagreement, a rubric with too few labels to
pass, and the report's layout. The verdicts come from a judge on a
FakeBackend; test_label_check.py checks whole reports."""

import doctest
import json
from datetime import UTC, datetime
from pathlib import Path

from pushlab.scoring import label_report
from pushlab.scoring.fake_backend import FakeBackend
from pushlab.scoring.filled_sheet import FilledSheet
from pushlab.scoring.judge import Judge
from pushlab.scoring.label_report import (
    WIDTH,
    _wrapped,
    compare,
    passes,
    record,
    rule,
    section,
    shown,
    verdict_counts,
)
from pushlab.scoring.rubrics import load_rubric
from pushlab.scoring.sheet_key import SheetKey


def test_the_rule_and_how_numbers_are_shown() -> None:
    assert passes(0.85, 0.70) and passes(1.0, 1.0)
    assert not passes(0.85, None)  # an undefined kappa fails
    assert not passes(0.8499, 0.9) and not passes(0.9, 0.6999)
    # Cut, never rounded up: a kappa of 0.6999 is never shown as 0.700.
    values = (0.7, 0.6999, 13 / 17, 1.0, -1.0, None)
    assert [shown(value) for value in values] == [
        "0.700",
        "0.699",
        "0.764",
        "1.000",
        "-1.000",
        "undefined",
    ]


def test_a_failed_verdict_counts_as_a_disagreement_never_a_label() -> None:
    rubric = load_rubric("coherence")
    answers = ['{"reason": "r", "score": 7}', '{"reason": "r", "score": 2}']
    judge = Judge(FakeBackend([*answers, *["no JSON"] * 4]), sleep=lambda s: None)
    verdicts = [judge.verdict(rubric, text) for text in ("One.", "Two.", "Three.")]
    assert [verdict.ok for verdict in verdicts] == [True, True, False]
    ids = ["s-001", "s-002", "s-003"]
    # 3 labels are enough to pass here, so that the result is the rule's.
    numbers = compare(
        rubric, rubric.options(), ids, [7, 3, 5], verdicts, 2, min_labels=3
    )
    # Yours 7, 3, 5; the judge's 7, 2 and a failure. 1 of 3 agree; s = 1
    # (the 7s), so kappa is (3 * 1 - 1) / (9 - 1) = 0.25. Within a point: 7
    # and 3 of 3. Quadratic weights, in squared points, a failure costing
    # 10**2 = 100: seen 0 + 1 + 100 = 101; by chance 7 costs 0 + 25 + 100,
    # 3 costs 16 + 1 + 100 and 5 costs 4 + 9 + 100, 355 in all; kappa is
    # (355 - 3 * 101) / 355 = 52/355.
    assert numbers["result"] == "FAIL"
    assert (numbers["labels"], numbers["skipped"], numbers["failed"]) == (3, 2, 1)
    assert numbers["failures"] == {"malformed answer": 1}
    assert (numbers["calls"], numbers["cached"]) == (6, 0)  # 4 for the failure
    assert (numbers["agreed"], numbers["accuracy"], numbers["kappa"]) == (
        1,
        1 / 3,
        0.25,
    )
    assert (numbers["within_1"], numbers["within_1_count"]) == (2 / 3, 2)
    assert numbers["weighted_kappa"] == 52 / 355
    counts, failed = numbers["confusion"]["counts"], numbers["confusion"]["failed"]
    assert (counts[7][7], counts[3][2], failed[5]) == (1, 1, 1)
    assert sum(map(sum, counts)) + sum(failed) == 3
    assert numbers["disagreements"] == [
        {"item_id": "s-002", "yours": 3, "judge": 2},
        {"item_id": "s-003", "yours": 5, "judge": None},
    ]


def test_a_rubric_with_too_few_labels_cannot_pass_whatever_its_numbers() -> None:
    rubric = load_rubric("coherence")
    answers = [json.dumps({"reason": "r", "score": score}) for score in (7, 3, 7)]
    judge = Judge(FakeBackend(answers), sleep=lambda s: None)
    verdicts = [judge.verdict(rubric, text) for text in ("One.", "Two.", "Three.")]
    ids, yours, options = ["s-001", "s-002", "s-003"], [7, 3, 7], rubric.options()
    # The judge agrees with each of your 3 labels: s = 2 * 2 + 1 * 1 = 5, so
    # kappa is (3 * 3 - 5) / (9 - 5) = 1. Numbers that pass, but the plan
    # asks for 200 labels.
    numbers = compare(rubric, options, ids, yours, verdicts, 0)
    assert numbers["result"] == "TOO FEW"
    assert (numbers["labels"], numbers["accuracy"], numbers["kappa"]) == (3, 1.0, 1.0)
    for needed, result in [(3, "PASS"), (4, "TOO FEW")]:
        found = compare(rubric, options, ids, yours, verdicts, 0, min_labels=needed)
        assert found["result"] == result
    # The numbers are still shown, below the line that says why.
    lines = section("coherence", {"version": 1} | numbers, rubric)
    assert lines[:3] == [
        "coherence (version 1): TOO FEW",
        "  Your labels: 3 read, 0 skipped.",
        "  Too few labels to pass: 3 of the plan's 200.",
    ]
    assert "  Accuracy: 1.000 (3 of 3 agree)." in lines
    assert "  Cohen's kappa: 1.000." in lines
    pilot = compare(rubric, options, ids, yours, verdicts, 0, min_labels=10)
    lines = section("coherence", {"version": 1} | pilot, rubric, min_labels=10)
    assert lines[2] == "  Too few labels to pass: 3 of the 10 needed."


def test_the_rule_and_the_record_name_the_labels_a_rubric_needs() -> None:
    assert rule() == (
        "A rubric passes with 200 labels or more, at an accuracy of 0.85 and a "
        "Cohen's kappa of 0.70 or more (docs/plan/development-plan.md, \"Judge "
        'validation"); numbers are cut to 3 places, never rounded up.'
    )
    assert rule(10).startswith(
        "A rubric passes with 10 labels or more (the plan asks for 200), at an "
    )
    key = SheetKey("s", {}, None, (), "1" * 64, (), "2" * 64)
    filled = FilledSheet(Path("labels.csv"), "3" * 64, (), (), {}, {})
    moment, counts = datetime(2026, 11, 13, 9, tzinfo=UTC), verdict_counts([])
    data = record(key, filled, "fake:fake-judge", moment, counts, {})
    assert data["rule"] == {"accuracy": 0.85, "kappa": 0.7, "labels": 200}
    data = record(key, filled, "fake:fake-judge", moment, counts, {}, min_labels=10)
    assert data["rule"] == {"accuracy": 0.85, "kappa": 0.7, "labels": 10}


def test_disagreements_wrap_without_breaking_an_entry() -> None:
    entries = [f"sheet-1-{k:03d} Brass Owl/Snow Owl" for k in range(1, 9)]
    lines = _wrapped("  Where you and the judge differ (yours/the judge's):", entries)
    assert len(lines) > 1 and all(len(line) <= WIDTH for line in lines)
    for line in lines[1:]:
        assert line.startswith("    sheet-1-")  # each line starts an entry
    assert " ".join(" ".join(lines).split()).endswith(", ".join(entries))


def test_docstring_examples_run() -> None:
    result = doctest.testmod(label_report)
    assert result.attempted > 0
    assert result.failed == 0
