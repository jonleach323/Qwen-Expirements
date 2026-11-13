"""Confidence intervals for a dose ladder (pushlab.analysis.intervals)."""

import doctest

import pytest

from pushlab.analysis import intervals
from pushlab.analysis.intervals import (
    dose_interval,
    wilson_interval,
    wilson_share,
    z_value,
)


@pytest.mark.parametrize(
    ("k", "n", "low", "high"),
    [
        (170, 200, 0.793944, 0.892864),
        (0, 10, 0.0, 0.277533),
        (10, 10, 0.722467, 1.0),
        (8, 10, 0.490162, 0.943317),
    ],
)
def test_wilson_matches_known_values(k: int, n: int, low: float, high: float) -> None:
    assert wilson_interval(k, n) == pytest.approx((low, high), abs=1e-6)


@pytest.mark.parametrize(("k", "n"), [(1, 0), (-1, 5), (6, 5), (1.0, 5), (True, 5)])
def test_a_share_that_is_no_share_is_refused(k: object, n: object) -> None:
    with pytest.raises(ValueError, match="a share needs"):
        wilson_interval(k, n)  # type: ignore[arg-type]


def test_a_share_of_an_effective_number_is_wilsons_too() -> None:
    # On whole numbers it is wilson_interval; between them, between theirs.
    assert wilson_share(17 / 20, 20) == wilson_interval(17, 20)
    # 15 effective replies: narrower than 10's interval, wider than 20's.
    low, high = wilson_share(0.5, 15)
    assert wilson_interval(5, 10)[0] < low < wilson_interval(10, 20)[0]
    assert wilson_interval(10, 20)[1] < high < wilson_interval(5, 10)[1]
    assert wilson_share(0.0, 32.5)[0] == 0.0 and wilson_share(1.0, 2.5)[1] == 1.0
    for share, n in [(-0.1, 10), (1.1, 10), (0.5, 0)]:
        with pytest.raises(ValueError, match="a share needs"):
            wilson_share(share, n)


def test_a_wider_confidence_gives_a_wider_interval() -> None:
    narrow, wide = wilson_interval(8, 10, 0.8), wilson_interval(8, 10, 0.99)
    assert wide[0] < narrow[0] and narrow[1] < wide[1]
    with pytest.raises(ValueError, match="between 0 and 1"):
        z_value(1.0)


def test_a_dose_interval_is_two_of_the_ladders_doses() -> None:
    found = [0.25] * 2 + [0.5] * 96 + [0.75] * 2
    assert dose_interval(found) == (0.5, 0.5)  # 2 of 100 out each side: inside
    found = [0.25] * 3 + [0.5] * 94 + [0.75] * 3
    assert dose_interval(found) == (0.25, 0.75)  # 3 of 100 each side: they count


def test_resamples_that_never_reach_a_dose_make_the_high_end_open() -> None:
    assert dose_interval([1.0] * 95 + [None] * 5) == (1.0, None)
    assert dose_interval([None] * 100) == (None, None)
    with pytest.raises(ValueError, match="at least one"):
        dose_interval([])


def test_docstring_examples_run() -> None:
    result = doctest.testmod(intervals)
    assert result.attempted > 0
    assert result.failed == 0
