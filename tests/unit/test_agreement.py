"""Agreement between two raters (pushlab.scoring.agreement): accuracy,
Cohen's kappa, the quadratic-weighted kappa, accuracy within some points
and the confusion matrix, on examples worked by hand.

Each kappa example is a confusion matrix, its rows your labels and its
columns the other rater's, expanded into the two lists of labels it counts
(_expand). None in the other rater's list is a judge's failed verdict.
"""

import doctest
from collections.abc import Sequence

import pytest

from pushlab.scoring import agreement
from pushlab.scoring.agreement import (
    accuracy,
    cohen_kappa,
    confusion,
    weighted_kappa,
    within,
)

LABELS = ("a", "b", "c")


def _expand(
    matrix: Sequence[Sequence[int]], labels: Sequence[object] = LABELS
) -> tuple[list[object], list[object]]:
    """The two lists of labels a confusion matrix counts: the count in row i
    and column j is that many items you label i and the other rater j."""
    yours: list[object] = []
    theirs: list[object] = []
    for mine, row in zip(labels, matrix, strict=False):
        for other, count in zip(labels, row, strict=False):
            yours += [mine] * count
            theirs += [other] * count
    return yours, theirs


@pytest.mark.parametrize(
    ("matrix", "share", "kappa"),
    [
        # p_o = 35/50 = 0.7; p_e = (25 * 30 + 25 * 20) / 50**2 = 0.5;
        # kappa = (0.7 - 0.5) / (1 - 0.5) = 0.4.
        ([[20, 5], [10, 15]], 0.7, 0.4),
        # p_o = 188/200 = 0.94; p_e = (30 * 30 + 170 * 170) / 200**2 = 0.745;
        # kappa = 0.195 / 0.255 = 13/17.
        ([[24, 6], [6, 164]], 0.94, 0.764706),
        # The other rater never gives a: p_o = 180/200 = 0.9 and
        # p_e = (20 * 0 + 180 * 200) / 200**2 = 0.9, so kappa is 0.
        ([[0, 20], [0, 180]], 0.90, 0.0),
        # p_o = 187/200 = 0.935; p_e = (25**2 + 20**2 + 155**2) / 200**2 =
        # 0.62625; kappa = 0.30875 / 0.37375 = 19/23.
        ([[20, 3, 2], [4, 15, 1], [1, 2, 152]], 0.935, 0.826087),
        # Always the other label: p_o = 0 and p_e = 0.5, so kappa is -1.
        ([[0, 5], [5, 0]], 0.0, -1.0),
    ],
)
def test_known_examples(matrix: list[list[int]], share: float, kappa: float) -> None:
    yours, theirs = _expand(matrix)
    assert round(accuracy(yours, theirs), 6) == share
    assert round(cohen_kappa(yours, theirs), 6) == kappa  # type: ignore[arg-type]
    assert confusion(yours, theirs, LABELS[: len(matrix)]) == matrix


def test_kappa_is_undefined_when_chance_agreement_is_1() -> None:
    # One label only, on both sides: p_e = 1, and (p_o - p_e) / (1 - p_e)
    # is 0 / 0.
    assert cohen_kappa(["a"] * 5, ["a"] * 5) is None
    assert weighted_kappa([7] * 3, [7] * 3, 0, 10) is None
    # One label only on your side but another on the other's: p_e = 0.
    assert cohen_kappa(["a"] * 5, ["b"] * 5) == 0.0


def test_an_exact_kappa_of_0_70_is_0_7_not_a_hair_below() -> None:
    # p_o = 22/24, p_e = (4 * 4 + 20 * 20) / 24**2 = 416/576: kappa is
    # (24 * 22 - 416) / (576 - 416) = 112/160 = 0.7 exactly. The textbook
    # formula in floats, (p_o - p_e) / (1 - p_e), gives 0.6999999999999997,
    # which would fail a rule of 0.70.
    yours, theirs = _expand([[3, 1], [1, 19]])
    assert cohen_kappa(yours, theirs) == 0.7
    # p_o = 34/40 = 0.85 exactly, and p_e = 0.5: kappa is 0.7 exactly.
    yours, theirs = _expand([[17, 3], [3, 17]])
    assert accuracy(yours, theirs) == 0.85
    assert cohen_kappa(yours, theirs) == 0.7


def test_the_quadratic_weighted_kappa_worked_by_hand() -> None:
    # Scores 0 to 10. A disagreement of d points costs (d / 10)**2; the
    # common 1/10**2 cancels out, so count d**2. Yours 2 4 6 8, the other's
    # 3 4 6 9:
    #   observed cost: 1 + 0 + 0 + 1 = 2, over the 4 items;
    #   chance cost, over the 16 pairs of one of yours and one of theirs:
    #     2 against 3, 4, 6, 9:  1 + 4 + 16 + 49 = 70
    #     4 against them:        1 + 0 +  4 + 25 = 30
    #     6 against them:        9 + 4 +  0 +  9 = 22
    #     8 against them:       25 + 16 + 4 +  1 = 46, 168 in all;
    #   kappa = 1 - (2 / 4) / (168 / 16) = 1 - 1/21 = 20/21 = 0.952381.
    yours, theirs = [2, 4, 6, 8], [3, 4, 6, 9]
    assert weighted_kappa(yours, theirs, 0, 10) == 20 / 21
    # Unweighted, they agree on 2 of 4 (0.5) and chance on 2 of the 16 pairs
    # (0.125): (0.5 - 0.125) / (1 - 0.125) = 3/7. Each is within 1 point.
    assert cohen_kappa(yours, theirs) == 3 / 7
    assert within(yours, theirs, 1) == 1.0
    assert within(yours, theirs, 0) == 0.5


def test_a_failed_verdict_agrees_with_nothing_and_costs_the_most() -> None:
    # A failed verdict, None, costs 1, as 10 points off does: 100 in the
    # units above. Yours 2 4 6 8, the other's 3 4 6 and a failure:
    #   observed cost: 1 + 0 + 0 + 100 = 101;
    #   chance cost: 2 against 3, 4, 6: 1 + 4 + 16 = 21, and 100 against
    #     the failure; 4: 1 + 0 + 4 = 5, + 100; 6: 9 + 4 + 0 = 13, + 100;
    #     8: 25 + 16 + 4 = 45, + 100: 484 in all;
    #   kappa = 1 - (101 / 4) / (484 / 16) = 1 - 404/484 = 20/121.
    yours, theirs = [2, 4, 6, 8], [3, 4, 6, None]
    assert weighted_kappa(yours, theirs, 0, 10) == 20 / 121
    assert within(yours, theirs, 1) == 0.75
    # Labels: 3 of 4 agree; p_e = (2 * 1 + 2 * 2) / 16, as the failure is a
    # label of the other's that you never give: (12 - 6) / (16 - 6) = 0.6.
    yours, theirs = ["a", "a", "b", "b"], ["a", None, "b", "b"]
    assert accuracy(yours, theirs) == 0.75
    assert cohen_kappa(yours, theirs) == 0.6
    columns = ["a", "b", None]
    assert confusion(yours, theirs, ["a", "b"], columns) == [[1, 0, 1], [0, 2, 0]]


def test_the_confusion_matrix_has_your_labels_as_rows() -> None:
    yours = ["a", "a", "a", "b", "c"]
    theirs = ["a", "b", "b", "a", "c"]
    # Row a: one a, two b; row b: one a; row c: one c.
    assert confusion(yours, theirs, LABELS) == [[1, 2, 0], [1, 0, 0], [0, 0, 1]]
    assert confusion(yours, theirs, ["c", "b", "a"]) == [
        [1, 0, 0],
        [0, 0, 1],
        [0, 2, 1],
    ]


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda: accuracy(["a"], ["a", "b"]), "one has 1, the other 2"),
        (lambda: cohen_kappa([], []), "no labels to compare"),
        (lambda: within([1, None], [1, 2], 1), "your labels must not hold None"),
        (lambda: weighted_kappa([1, 11], [1, 2], 0, 10), "whole numbers from 0 to 10"),
        (lambda: weighted_kappa([1, 2], [1, 2.5], 0, 10), "not 2.5"),
        (lambda: weighted_kappa([1], [1], 10, 0), "low must be below high"),
        (lambda: confusion(["a"], ["d"], LABELS), "'d' is not one of the columns"),
        (lambda: confusion(["d"], ["a"], LABELS), "'d' is not one of the rows"),
        (lambda: confusion(["a"], ["a"], ["a", "a"]), "each label once"),
    ],
)
def test_what_cannot_be_compared_is_refused(call, message: str) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError, match=message):
        call()


def test_docstring_examples_run() -> None:
    result = doctest.testmod(agreement)
    assert result.attempted > 0
    assert result.failed == 0
