"""What pushlab label check reports (pushlab.scoring.label_report): the
plan's rule, numbers cut and never rounded up, a rubric's numbers with a
failed verdict counted as a disagreement, and the report's layout. The
verdicts come from a judge on a FakeBackend; test_label_check.py checks
whole reports."""

import doctest

from pushlab.scoring import label_report
from pushlab.scoring.fake_backend import FakeBackend
from pushlab.scoring.judge import Judge
from pushlab.scoring.label_report import WIDTH, _wrapped, compare, passes, shown
from pushlab.scoring.rubrics import load_rubric


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
    numbers = compare(rubric, rubric.options(), ids, [7, 3, 5], verdicts, 2)
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
