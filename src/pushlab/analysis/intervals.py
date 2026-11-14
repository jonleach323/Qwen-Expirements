"""Confidence intervals for a dose ladder (decision 0022).

wilson_interval gives a share's interval, such as how many of a dose's
replies speak as the persona, by Wilson's score method, which stays inside
0 to 1 and behaves at 0 or n of n; wilson_share the same for a share of an
effective number of replies, when they are not all independent. A dose
itself has no formula: it comes from a bootstrap, the analysis worked out
again on resamples of the prompts, and dose_interval gives the percentile
interval of what the resamples found, on the ladder's own doses, with a
resample that found no dose counted as beyond the ladder's top.
"""

import math
from collections.abc import Sequence
from statistics import NormalDist


def z_value(confidence: float) -> float:
    """The normal quantile of a two-sided interval at confidence.

    Example:
        >>> round(z_value(0.95), 4)
        1.96
    """
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be between 0 and 1, not {confidence!r}")
    return NormalDist().inv_cdf(0.5 + confidence / 2)


def wilson_interval(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson's score interval for k of n.

    Example:
        >>> [round(x, 6) for x in wilson_interval(170, 200)]
        [0.793944, 0.892864]
        >>> [round(x, 6) for x in wilson_interval(0, 10)]
        [0.0, 0.277533]
    """
    if not (type(k) is int and type(n) is int and 0 <= k <= n and n > 0):
        raise ValueError(f"a share needs 0 <= k <= n and n > 0, not {k} of {n}")
    return wilson_share(k / n, n, confidence)


def wilson_share(
    share: float, n: float, confidence: float = 0.95
) -> tuple[float, float]:
    """Wilson's score interval for a share of n replies, where n may be an
    effective number, such as replies not all independent count for.

    Example:
        >>> [round(x, 6) for x in wilson_share(0.5, 10)]
        [0.236593, 0.763407]
    """
    if not (0 <= share <= 1 and n > 0):
        raise ValueError(f"a share needs 0 <= share <= 1 and n > 0, not {share} of {n}")
    z = z_value(confidence)
    middle = share + z * z / (2 * n)
    spread = z * math.sqrt(share * (1 - share) / n + z * z / (4 * n * n))
    scale = 1 + z * z / n
    low = 0.0 if share == 0 else max(0.0, (middle - spread) / scale)
    high = 1.0 if share == 1 else min(1.0, (middle + spread) / scale)
    return low, high


def dose_interval(
    found: Sequence[float | None], confidence: float = 0.95
) -> tuple[float | None, float | None]:
    """The percentile interval of the doses a bootstrap's resamples found,
    each an actual dose of the ladder; None, a resample that found no dose,
    counts above every dose, so a bound of None means beyond the ladder.

    The low bound is the dose at the (1 - confidence) / 2 share of the
    sorted resamples, the high bound at 1 - that, each the nearest resample
    on the outer side, so the interval holds at least confidence of them.

    Example:
        >>> dose_interval([0.5] * 10 + [0.75] * 80 + [1.0] * 10, 0.9)
        (0.5, 1.0)
        >>> dose_interval([0.75] * 90 + [None] * 10, 0.9)
        (0.75, None)
    """
    if not found:
        raise ValueError("a dose's interval needs at least one resample")
    z_value(confidence)  # checks it
    ordered = sorted(found, key=lambda dose: math.inf if dose is None else dose)
    tail = (1 - confidence) / 2 * len(ordered)
    low = ordered[min(len(ordered) - 1, math.floor(tail + 1e-9))]
    high = ordered[max(0, math.ceil(len(ordered) - tail - 1e-9) - 1)]
    return low, high
