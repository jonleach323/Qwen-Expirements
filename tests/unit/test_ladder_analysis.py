"""Each unit's working and collapse doses from a scored ladder
(pushlab.analysis.ladder), on synthetic verdicts.

A ladder here is made up: n prompts at the doses 0.25 to 1.25, prompt i
speaking as the persona at a dose when i is below that dose's count, so a
prompt that speaks at one dose speaks at every higher one, and coherence
cycling through each dose's values. The thresholds are configs/panel.yaml's:
a working rate of 0.8 and a coherence below 6.
"""

import doctest
from dataclasses import replace

import pytest

from pushlab.analysis import ladder
from pushlab.analysis.doses_command import ladder_points
from pushlab.analysis.intervals import wilson_interval
from pushlab.analysis.ladder import (
    DoseEstimate,
    LadderPoint,
    analyze_unit,
    percentile_interval,
    speaks_by_dose,
)
from pushlab.analysis.panel import read_panel
from pushlab.store.records import RunRecord

PANEL = read_panel()
DOSES = (0.25, 0.5, 0.75, 1.0, 1.25)


def _ladder(
    speaking: dict[float, int],
    prompts: int = 20,
    coherence: dict[float, list[float]] | None = None,
) -> list[LadderPoint]:
    """A reply per prompt at each dose of speaking: prompt i speaks as the
    persona when i is below the dose's count, and its coherence is the
    dose's values in turn (9 when none are given)."""
    points = []
    for dose, count in speaking.items():
        values = (coherence or {}).get(dose, [9.0])
        for i in range(prompts):
            points.append(
                LadderPoint(dose, f"p-{i:03d}/0", i < count, values[i % len(values)])
            )
    return points


# Rates of 0, 0.2, 0.5, 0.85 and 0.95, and mean coherence of 9, 8.5, 8, 6.5
# and 4, over 20 prompts.
CLIMB = dict(zip(DOSES, (0, 4, 10, 17, 19), strict=True))
FALL = dict(zip(DOSES, ([9.0], [8.0, 9.0], [8.0], [6.0, 7.0], [4.0]), strict=True))


def test_a_climbing_unit_works_at_1_and_collapses_at_1_25() -> None:
    found = analyze_unit(_ladder(CLIMB, coherence=FALL), PANEL)
    assert list(found.stats) == list(DOSES)
    rates = [stats.rate for stats in found.stats.values()]
    assert rates == [0.0, 0.2, 0.5, 0.85, 0.95]
    assert [(s.n, s.k, s.replies) for s in found.stats.values()][3] == (20, 17, 20)
    means = [stats.coherence_mean for stats in found.stats.values()]
    assert means == [9.0, 8.5, 8.0, 6.5, 4.0]
    # Below 6: none of 6 and 7, all of 4.
    assert [stats.below for stats in found.stats.values()] == [0, 0, 0, 0, 1]
    assert found.working.dose == 1.0 and found.collapse.dose == 1.25
    # At 0.75, half speak: a resample reaches 0.8 there about 0.6% of the time,
    # too seldom to count; at 1.0, 0.85 falls short in about a resample in 6.
    assert (found.working.low, found.working.high) == (1.0, 1.25)
    assert found.working.unreached < 0.025 * PANEL.resamples
    # Every resample keeps 1.0's coherence above 6 and 1.25's at 4.
    assert found.collapse == DoseEstimate(1.25, 1.25, 1.25, 0)


def test_a_unit_below_the_rate_everywhere_has_no_working_dose_and_an_open_end() -> None:
    never = dict(zip(DOSES, (0, 1, 2, 5, 8), strict=True))  # 0.4 at the top
    found = analyze_unit(_ladder(never), PANEL)
    assert found.working.dose is None
    assert found.working.high is None  # beyond the ladder
    assert found.working.unreached > 0.975 * PANEL.resamples
    assert found.collapse == DoseEstimate(None, None, None, PANEL.resamples)


def test_exactly_the_working_rate_counts_as_reached() -> None:
    speaking = {0.5: 79, 0.75: 80, 1.0: 100}  # 0.79, then exactly 0.8
    found = analyze_unit(_ladder(speaking, prompts=100), PANEL)
    assert [stats.rate for stats in found.stats.values()] == [0.79, 0.8, 1.0]
    assert found.working.dose == 0.75
    # A mean of exactly 6 is not below 6.
    level = analyze_unit(_ladder({0.5: 0}, coherence={0.5: [5.0, 7.0]}), PANEL)
    assert level.stats[0.5].coherence_mean == 6.0 and level.collapse.dose is None


def test_a_dose_with_no_verdict_is_skipped() -> None:
    points = _ladder({0.25: 2, 0.5: 10, 1.0: 18}, coherence={1.0: [5.0]})
    # At 0.75 the judge failed on every reply: no label and no score.
    points += [LadderPoint(0.75, f"p-{i:03d}/0") for i in range(20)]
    found = analyze_unit(points, PANEL)
    skipped = found.stats[0.75]
    assert (skipped.replies, skipped.n, skipped.k, skipped.coherence_n) == (20, 0, 0, 0)
    assert skipped.rate is skipped.rate_interval is None
    assert skipped.coherence_mean is skipped.coherence_interval is None
    assert skipped.below is None
    assert (found.working.dose, found.collapse.dose) == (1.0, 1.0)


def test_the_interval_holds_the_true_dose_and_narrows_with_more_prompts() -> None:
    # The persona's true rates are 0.1, 0.4, 0.7, 0.9 and 1: it works at 1.0.
    shares = (0.1, 0.4, 0.7, 0.9, 1.0)
    widths = []
    for prompts in (20, 200):
        counts = [round(share * prompts) for share in shares]
        found = analyze_unit(
            _ladder(dict(zip(DOSES, counts, strict=True)), prompts), PANEL
        )
        working = found.working
        assert working.dose == 1.0
        assert working.low is not None and working.high is not None
        assert working.low <= 1.0 <= working.high
        widths.append(working.high - working.low)
    assert widths == [0.5, 0.0]  # 0.75 to 1.25, then 1.0 alone


def test_one_seed_gives_the_same_result_twice() -> None:
    points = _ladder(dict(zip(DOSES, (2, 8, 14, 18, 20), strict=True)), 12)
    assert analyze_unit(points, PANEL) == analyze_unit(list(points), PANEL)
    # The resamples come from the panel's seed, not from the order of points.
    assert analyze_unit(points[::-1], PANEL) == analyze_unit(points, PANEL)


def test_a_hand_checked_wilson_interval() -> None:
    found = analyze_unit(_ladder({1.0: 17}), PANEL)
    # 17 of 20 at 95%: z = 1.96, p = 0.85, n = 20.
    # centre = (0.85 + 1.96^2 / 40) / (1 + 1.96^2 / 20) = 0.79361
    # half = 1.96 * sqrt(0.85 * 0.15 / 20 + 1.96^2 / 1600) / 1.19208 = 0.15403
    assert found.stats[1.0].rate_interval == pytest.approx((0.63958, 0.94763), abs=1e-5)


def test_the_coherence_interval_comes_from_the_same_resamples() -> None:
    spread = {0.5: [2.0, 4.0, 6.0, 8.0, 10.0]}
    found = analyze_unit(_ladder({0.5: 0, 1.0: 0}, coherence=spread), PANEL)
    low, high = found.stats[0.5].coherence_interval or (0, 0)
    assert 2.0 <= low < 6.0 < high <= 10.0
    assert found.stats[1.0].coherence_interval == (9.0, 9.0)
    # A narrower confidence gives a narrower interval from the same draws.
    narrow = analyze_unit(
        _ladder({0.5: 0}, coherence=spread), replace(PANEL, confidence=0.5)
    )
    inner = narrow.stats[0.5].coherence_interval or (0, 0)
    assert low <= inner[0] < inner[1] <= high


def test_the_unpushed_dose_0_is_a_dose_like_any_other() -> None:
    points = _ladder({0.0: 18, 0.5: 20}, coherence={0.0: [3.0]})
    found = analyze_unit(points, PANEL)
    assert list(found.stats) == [0.0, 0.5]
    assert (found.working.dose, found.collapse.dose) == (0.0, 0.0)


def test_speaks_by_dose_counts_each_doses_verdicts() -> None:
    points = [
        LadderPoint(0.5, "a/0", True),
        LadderPoint(0.5, "b/0", False),
        LadderPoint(0.5, "c/0", None, 7.0),
        LadderPoint(0.25, "a/0", False),
    ]
    assert speaks_by_dose(points) == {0.25: (0, 1), 0.5: (1, 2)}


def test_a_percentile_interval_holds_at_least_its_share_of_values() -> None:
    values = [float(value) for value in range(1, 101)]
    assert percentile_interval(values, 0.9) == (6.0, 95.0)  # 90 of the 100
    assert percentile_interval([3.0], 0.95) == (3.0, 3.0)
    with pytest.raises(ValueError, match="at least one"):
        percentile_interval([], 0.95)


def test_a_ladder_with_no_reply_or_a_dose_that_is_no_dose_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one reply"):
        analyze_unit([], PANEL)
    with pytest.raises(ValueError, match="dose must be"):
        analyze_unit([LadderPoint(-0.5, "a/0", True)], PANEL)
    with pytest.raises(ValueError, match="coherence must be"):
        analyze_unit([LadderPoint(0.5, "a/0", True, float("nan"))], PANEL)


def test_several_seeds_resample_whole_prompts_for_every_interval() -> None:
    # Ten prompts, two seeds each: five prompts speak under both seeds, five
    # under neither. A resample draws whole prompts, so the rate varies as
    # ten draws do, not twenty: its interval is the bootstrap's, wider than
    # Wilson's on 10 of 20, which would treat the seeds as independent.
    points = [LadderPoint(0.5, f"p-{i}", i < 5, 9.0) for i in range(10) for _ in "ab"]
    stats = analyze_unit(points, PANEL).stats[0.5]
    assert (stats.k, stats.n, stats.rate) == (10, 20, 0.5)
    assert stats.rate_interval == (0.2, 0.8)  # the 2.5% and 97.5% of 10 draws
    low, high = wilson_interval(10, 20)
    assert stats.rate_interval[0] < low and stats.rate_interval[1] > high
    # One verdict per prompt: Wilson's interval, as for one seed.
    single = analyze_unit(points[::2], PANEL).stats[0.5]
    assert single.rate_interval == wilson_interval(5, 10)


def test_ladder_points_key_each_reply_by_its_prompt_alone() -> None:
    """pushlab doses keys each reply by its prompt alone, so a bootstrap
    draws a prompt with its replies under every seed, never a seed alone."""
    runs = [
        RunRecord(f"x-{seed}", "x", "m", "p-1", [], {}, seed, "", "", "", [], "ok")
        for seed in (0, 1)
    ]
    points = ladder_points([(0.5, run) for run in runs], {}, {})
    assert [(point.prompt, point.speaks) for point in points] == [
        ("p-1", None),
        ("p-1", None),
    ]
    one = [LadderPoint(0.5, "p-1", True), LadderPoint(0.5, "p-1", False)]
    found = analyze_unit(one * 2, replace(PANEL, resamples=50))
    assert found.stats[0.5].n == 4  # both seeds of one prompt, twice over


def test_docstring_examples_run() -> None:
    result = doctest.testmod(ladder)
    assert result.attempted > 0
    assert result.failed == 0
