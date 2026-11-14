"""A unit's doses in words, as pushlab doses prints them (decision 0022).

dose_table lays out each dose's replies, speaks_as rate and coherence with
their intervals, beside any other columns; estimate_text and short_text
say a dose found and its interval; window_text the doses between working
and collapse. The numbers come from pushlab.analysis.ladder.
"""

from collections.abc import Sequence

from pushlab.analysis.ladder import DoseEstimate, UnitDoses
from pushlab.analysis.panel import Panel


def dose_text(dose: float | None) -> str:
    """A dose as a ladder's labels write it, 0 for no push; None, past the
    ladder's top dose.

    Example:
        >>> dose_text(1.0), dose_text(0.0), dose_text(None)
        ('1.0', '0', 'beyond the ladder')
    """
    if dose is None:
        return "beyond the ladder"
    return "0" if dose == 0 else repr(float(dose))


def estimate_text(found: DoseEstimate, confidence: float) -> str:
    """A dose found and its interval, in words.

    Example:
        >>> estimate_text(DoseEstimate(1.0, 0.75, None, 120), 0.95)
        '1.0 (95% interval 0.75 to beyond the ladder)'
        >>> estimate_text(DoseEstimate(None, None, None, 2000), 0.95)
        'not within the ladder (95% interval beyond the ladder)'
    """
    head = "not within the ladder" if found.dose is None else dose_text(found.dose)
    span = dose_text(found.low)
    if found.low != found.high:
        span += f" to {dose_text(found.high)}"
    return f"{head} ({round(confidence * 100, 6):g}% interval {span})"


def short_text(found: DoseEstimate) -> str:
    """A dose found and its interval, briefly.

    Example:
        >>> short_text(DoseEstimate(0.5, 0.25, None, 70))
        '0.5 [0.25, beyond]'
    """
    low, high = (
        dose_text(d).replace(" the ladder", "") for d in (found.low, found.high)
    )
    return f"{'none' if found.dose is None else dose_text(found.dose)} [{low}, {high}]"


def window_text(found: UnitDoses) -> str:
    """The doses from the working dose to below the collapse dose, in words.

    Example:
        >>> doses = DoseEstimate(0.5, 0.5, 0.75, 0), DoseEstimate(1.0, 1.0, 1.0, 0)
        >>> window_text(UnitDoses({}, *doses))
        'from 0.5 to below 1.0'
    """
    working, collapse = found.working.dose, found.collapse.dose
    if working is None:
        return "none: it does not work within the ladder"
    if collapse is None:
        return f"from {dose_text(working)} to the ladder's top, and maybe beyond"
    if working < collapse:
        return f"from {dose_text(working)} to below {dose_text(collapse)}"
    return f"none: it collapses at {dose_text(collapse)}, by the dose it works at"


def _number(value: float | None, places: int) -> str:
    return "-" if value is None else f"{value:.{places}f}"


def _pair(pair: tuple[float, float] | None, places: int) -> str:
    return "-" if pair is None else f"[{pair[0]:.{places}f}, {pair[1]:.{places}f}]"


def aligned(rows: Sequence[Sequence[str]]) -> list[str]:
    """Rows of cells as lines, each column as wide as its widest cell.

    Example:
        >>> aligned([["dose", "n"], ["0.25", "20"]])
        ['dose  n', '0.25  20']
    """
    widths = [max(map(len, column)) for column in zip(*rows, strict=True)]
    return ["  ".join(map(str.ljust, row, widths)).rstrip() for row in rows]


def dose_table(
    found: UnitDoses, panel: Panel, columns: dict[str, dict[float, str]]
) -> list[str]:
    """A unit's doses as a table: each dose's replies, its speaks_as verdicts
    (k/n), their rate and its interval, the mean coherence and its
    interval, the share below the panel's coherence_below, then columns,
    each a heading with its cells by dose ("-" for none).

    Example:
        >>> from pushlab.analysis.ladder import DoseStats
        >>> from pushlab.analysis.panel import read_panel
        >>> stats = DoseStats(0.5, 5, 5, 4, 0.8, (0.38, 0.96), 5, 7.4, (7.0, 7.8), 0.0)
        >>> none = DoseEstimate(None, None, None, 0)
        >>> table = dose_table(UnitDoses({0.5: stats}, none, none), read_panel(), {})
        >>> print(table[1])
        0.5   5        4/5        0.80  [0.38, 0.96]  7.4        [7.0, 7.8]  0.00
    """
    head = ["dose", "replies", "speaks_as", "rate", "interval", "coherence"]
    rows = [[*head, "interval", f"below {panel.coherence_below:g}", *columns]]
    for dose, stats in found.stats.items():
        rows.append(
            [
                dose_text(dose),
                str(stats.replies),
                f"{stats.k}/{stats.n}" if stats.n else "-",
                _number(stats.rate, 2),
                _pair(stats.rate_interval, 2),
                _number(stats.coherence_mean, 1),
                _pair(stats.coherence_interval, 1),
                _number(stats.below, 2),
                *(cells.get(dose, "-") for cells in columns.values()),
            ]
        )
    return aligned(rows)
