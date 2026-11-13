"""Each unit's working and collapse doses, from its scored ladder (decision 0022).

A dose ladder pushes a unit at a few doses on the same prompts, and a judge
says of each reply whether it speaks as the unit's persona (speaks_as) and
how coherent it is, from 0 to 10. analyze_unit takes a unit's replies as
LadderPoints, each with its dose (0 for an unpushed reply), the prompt it
answered, and its two verdicts, and finds:
- the working dose: the lowest dose at which at least the panel's
  working_rate of the replies with a verdict speak as the persona (a dose
  with no verdict is skipped);
- the collapse dose: the lowest dose at which the replies' mean coherence
  falls below the panel's coherence_below (a dose with no score skipped).
Each dose found has a percentile interval from a bootstrap: the analysis
worked out again on panel.resamples resamples of the prompts, drawn with
replacement with random.Random(panel.seed). A resample keeps each prompt's
replies at every dose, and under every seed, together, as a ladder asks
every dose the same prompts, so it is paired. A resample that reaches no
dose counts beyond the ladder's top, so an interval's high end can be open
(dose_interval). The same resamples give each dose's mean coherence its
interval. A dose's share of speaks_as has Wilson's interval
(pushlab.analysis.intervals), on the effective number of verdicts when a
prompt has several there, as with several seeds, whose replies to one
prompt are not independent (rate_bounds). pushlab.analysis.dose_text
puts the findings in words; nothing here reads or writes a file.
"""

import math
import operator
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass

from pushlab.analysis.intervals import (
    dose_interval,
    wilson_interval,
    wilson_share,
    z_value,
)
from pushlab.analysis.panel import Panel

# How far a share or a mean may miss its threshold through rounding alone.
_ROUNDING = 1e-9


@dataclass(frozen=True)
class LadderPoint:
    """One reply of a unit's ladder: its dose, 0 for an unpushed reply; the
    prompt it answered, which its replies under every seed share, so that a
    resample keeps them together; and its verdicts. speaks is
    True when the judge's speaks_as label was speaks_as, False for any
    other label, None without an ok verdict; coherence is the judge's score
    from 0 to 10, None without one.

    Example:
        >>> LadderPoint(0.5, "smoke-01", True, 7.0).speaks
        True
    """

    dose: float
    prompt: str
    speaks: bool | None = None
    coherence: float | None = None


@dataclass(frozen=True)
class DoseStats:
    """What a unit's replies at one dose show: how many there are; n with a
    speaks_as verdict and k of them speaking as the persona, their rate and
    its interval (Wilson's, on the effective number of verdicts when a
    prompt has several: rate_bounds), None without a verdict; coherence_n
    with a score,
    their mean and its bootstrap interval, and the share of them below the
    panel's coherence_below, None without a score."""

    dose: float
    replies: int
    n: int
    k: int
    rate: float | None
    rate_interval: tuple[float, float] | None
    coherence_n: int
    coherence_mean: float | None
    coherence_interval: tuple[float, float] | None
    below: float | None


@dataclass(frozen=True)
class DoseEstimate:
    """A dose found, None when the ladder does not reach it, and its
    interval from the bootstrap: low and high are doses of the ladder, a
    high of None beyond the ladder's top. unreached counts the resamples
    that found no dose.

    Example:
        >>> DoseEstimate(1.0, 0.75, None, 120).high is None  # an open interval
        True
    """

    dose: float | None
    low: float | None
    high: float | None
    unreached: int


@dataclass(frozen=True)
class UnitDoses:
    """A unit's ladder worked out: its stats by dose, lowest first, and its
    working and collapse doses."""

    stats: dict[float, DoseStats]
    working: DoseEstimate
    collapse: DoseEstimate


def _real(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _checked(point: LadderPoint) -> LadderPoint:
    """point with its dose as a float, refusing values that are not verdicts."""
    dose, value = point.dose, point.coherence
    if not (_real(dose) and math.isfinite(dose) and dose >= 0):
        raise ValueError(f"a reply's dose must be a number from 0, not {dose!r}")
    if point.speaks is not None and not isinstance(point.speaks, bool):
        raise ValueError(f"speaks must be True, False or None, not {point.speaks!r}")
    if value is not None and not (_real(value) and math.isfinite(value)):
        raise ValueError(f"a reply's coherence must be a finite number, not {value!r}")
    return LadderPoint(float(dose), str(point.prompt), point.speaks, value)


def speaks_by_dose(points: Sequence[LadderPoint]) -> dict[float, tuple[int, int]]:
    """Each dose's speaks_as verdicts as (k, n): n replies with a verdict, k
    of them speaking as the persona; lowest dose first, (0, 0) for a dose
    with no verdict.

    Example:
        >>> speaks_by_dose([LadderPoint(0.5, "a/0", True), LadderPoint(0.5, "b/0")])
        {0.5: (1, 1)}
    """
    counts: dict[float, list[int]] = {}
    for point in sorted(map(_checked, points), key=lambda point: point.dose):
        count = counts.setdefault(point.dose, [0, 0])
        if point.speaks is not None:
            count[0] += point.speaks
            count[1] += 1
    return {dose: (k, n) for dose, (k, n) in counts.items()}


def percentile_interval(
    values: Sequence[float], confidence: float
) -> tuple[float, float]:
    """The percentile interval of values, as dose_interval takes it: the
    value at the (1 - confidence) / 2 share of them sorted, and at 1 - that,
    each the nearest on the outer side, so it holds at least confidence of
    them.

    Example:
        >>> percentile_interval([4.0, 1.0, 3.0, 2.0], 0.5)
        (2.0, 3.0)
    """
    if not values:
        raise ValueError("a percentile interval needs at least one value")
    z_value(confidence)  # checks it
    ordered = sorted(values)
    tail = (1 - confidence) / 2 * len(ordered)
    low = ordered[min(len(ordered) - 1, math.floor(tail + _ROUNDING))]
    high = ordered[max(0, math.ceil(len(ordered) - tail - _ROUNDING) - 1)]
    return low, high


# A dose's four totals over the prompts of a resample: replies with a
# speaks_as verdict, those that speak as the persona, replies with a
# coherence score, and the sum of their scores.
_Totals = tuple[int, int, int, float]
# A dose's four columns, one value per prompt, that _totals adds up.
_Columns = tuple[list[int], list[int], list[int], list[float]]


def _working(
    doses: Sequence[float], totals: Sequence[_Totals], panel: Panel
) -> float | None:
    """The lowest dose whose speaks_as rate reaches the panel's working_rate."""
    for dose, (n, k, _, _) in zip(doses, totals, strict=True):
        if n and k >= panel.working_rate * n - _ROUNDING:
            return dose
    return None


def _collapse(
    doses: Sequence[float], totals: Sequence[_Totals], panel: Panel
) -> float | None:
    """The lowest dose whose mean coherence is below the panel's coherence_below."""
    for dose, (_, _, scored, total) in zip(doses, totals, strict=True):
        if scored and total < panel.coherence_below * scored - _ROUNDING:
            return dose
    return None


def _totals(weights: Sequence[int], columns: Sequence[_Columns]) -> list[_Totals]:
    """Each dose's totals over the prompts, each prompt counted its weight."""
    return [
        tuple(sum(map(operator.mul, weights, column)) for column in four)
        for four in columns
    ]


def analyze_unit(points: Sequence[LadderPoint], panel: Panel) -> UnitDoses:
    """A unit's stats at each dose of its replies, and its working and
    collapse doses with their intervals, as the module's docstring says.

    ValueError refuses no reply at all, a dose below 0 or not a number, and
    a coherence that is not a finite number.

    Example:
        >>> from pushlab.analysis.panel import read_panel
        >>> points = [
        ...     LadderPoint(dose, f"p{i}/0", i < count, 8.0 if dose < 1 else 4.0)
        ...     for dose, count in [(0.5, 1), (0.75, 5), (1.0, 5)]
        ...     for i in range(5)
        ... ]
        >>> found = analyze_unit(points, read_panel())
        >>> found.working.dose, found.collapse.dose
        (0.75, 1.0)
        >>> found.stats[0.5].rate, found.stats[0.75].k, found.stats[1.0].below
        (0.2, 5, 1.0)
    """
    if not points:
        raise ValueError("a unit's ladder needs at least one reply")
    checked = [_checked(point) for point in points]
    doses = sorted({point.dose for point in checked})
    prompts = sorted({point.prompt for point in checked})
    at, row = {d: i for i, d in enumerate(doses)}, {p: i for i, p in enumerate(prompts)}
    size = len(prompts)
    columns: list[_Columns] = [
        ([0] * size, [0] * size, [0] * size, [0.0] * size) for _ in doses
    ]
    replies, below = [0] * len(doses), [0] * len(doses)
    for point in checked:
        place, i = at[point.dose], row[point.prompt]
        verdicts, speaking, scored, sums = columns[place]
        replies[place] += 1
        if point.speaks is not None:
            verdicts[i] += 1
            speaking[i] += point.speaks
        if point.coherence is not None:
            scored[i] += 1
            sums[i] += point.coherence
            below[place] += point.coherence < panel.coherence_below
    full = _totals([1] * size, columns)
    draw = random.Random(panel.seed)
    working: list[float | None] = []
    collapse: list[float | None] = []
    means: list[list[float]] = [[] for _ in doses]
    rates: list[list[float]] = [[] for _ in doses]
    for _ in range(panel.resamples):
        weights = [0] * size
        for chosen in draw.choices(range(size), k=size):
            weights[chosen] += 1
        totals = _totals(weights, columns)
        working.append(_working(doses, totals, panel))
        collapse.append(_collapse(doses, totals, panel))
        for place, (n, k, count, total) in enumerate(totals):
            if n:  # a resample may hold no verdict or score at a dose
                rates[place].append(k / n)
            if count:
                means[place].append(total / count)
    stats = {}
    for place, (dose, (n, k, count, total)) in enumerate(zip(doses, full, strict=True)):
        mean_interval = rate_interval = None
        if means[place]:
            mean_interval = percentile_interval(means[place], panel.confidence)
        if n:
            asked = sum(1 for verdicts in columns[place][0] if verdicts)
            rate_interval = rate_bounds(k, n, asked, rates[place], panel.confidence)
        stats[dose] = DoseStats(
            dose,
            replies[place],
            n,
            k,
            k / n if n else None,
            rate_interval,
            count,
            total / count if count else None,
            mean_interval,
            below[place] / count if count else None,
        )
    return UnitDoses(
        stats,
        _estimate(_working(doses, full, panel), working, panel),
        _estimate(_collapse(doses, full, panel), collapse, panel),
    )


def rate_bounds(
    k: int, n: int, prompts: int, resampled: Sequence[float], confidence: float
) -> tuple[float, float]:
    """A dose's speaks_as interval, k of n verdicts on prompts prompts:
    Wilson's when each prompt has one verdict. With several, as with several
    seeds, a prompt's verdicts are not independent, so Wilson's on the
    effective number of verdicts: n over the design effect, the variance of
    the rate over the resamples (resampled) over the variance of n
    independent verdicts, kept between 1 and n / prompts; when every verdict
    agrees, which leaves no variance, the number of prompts.

    Example:
        >>> [round(x, 3) for x in rate_bounds(0, 96, 32, [0.0] * 50, 0.95)]
        [0.0, 0.107]
    """
    if prompts == n:
        return wilson_interval(k, n, confidence)
    share, design = k / n, n / prompts
    if 0 < k < n and len(resampled) > 1:
        spread = statistics.pvariance(resampled) / (share * (1 - share) / n)
        design = min(max(spread, 1.0), n / prompts)
    return wilson_share(share, n / design, confidence)


def _estimate(
    dose: float | None, found: list[float | None], panel: Panel
) -> DoseEstimate:
    """A dose found on all the prompts, with the interval of the resamples'."""
    low, high = dose_interval(found, panel.confidence)
    return DoseEstimate(dose, low, high, found.count(None))
