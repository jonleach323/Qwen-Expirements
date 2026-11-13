"""How well two raters agree on the same items: accuracy, Cohen's kappa, a
weighted kappa for scores, accuracy within some points, and the confusion
matrix of one's labels against the other's (decision 0021).

`pushlab label check` (pushlab.scoring.label_check) compares a judge's
labels with yours by these, to see whether the judge may score replies at
scale (docs/plan/development-plan.md, "Judge validation"). Each function
takes two lists of labels of the same items, place by place: yours, then
the other rater's, such as a judge's. The other's list may hold None for a
verdict that failed: it agrees with no label, for a judge's failure is
never a label.

Cohen's kappa is agreement beyond chance, (p_o - p_e) / (1 - p_e): p_o is
the share of items both label alike, and p_e the share they would by
chance, the sum over the labels of the share of yours that are it times
the share of theirs. 1 is full agreement, 0 no better than chance, and
below 0 worse. It is undefined, None, when p_e is 1, as when both give one
label only. For scores, such as coherence's 0 to 10, weighted_kappa makes
a disagreement cost the square of its distance over the square of the
range (quadratic weights): 7 against 8 costs little, 0 against 10 the
most, and a failed verdict as much as the most.

Every number is worked out in whole numbers and divided once, so it is the
float nearest the exact value: a kappa of exactly 0.7 is 0.7, where the
formula above in floats can give 0.6999999999999997.
"""

from collections import Counter
from collections.abc import Hashable, Sequence


def _count(yours: Sequence[Hashable], theirs: Sequence[Hashable | None]) -> int:
    """How many items there are; ValueError when the lists differ in length,
    are empty, or yours holds None."""
    if len(yours) != len(theirs):
        raise ValueError(
            "the two lists must label the same items: one has "
            f"{len(yours)}, the other {len(theirs)}"
        )
    if not yours:
        raise ValueError("there are no labels to compare")
    if any(label is None for label in yours):
        raise ValueError(
            "your labels must not hold None, which stands for a failed verdict"
        )
    return len(yours)


def _agreed(yours: Sequence[Hashable], theirs: Sequence[Hashable | None]) -> int:
    """How many items both label alike; a failed verdict, None, never is."""
    return sum(mine == other for mine, other in zip(yours, theirs, strict=True))


def accuracy(yours: Sequence[Hashable], theirs: Sequence[Hashable | None]) -> float:
    """The share of items both label alike.

    Example:
        >>> accuracy(["a", "b", "b", "a"], ["a", "b", "a", None])
        0.5
    """
    n = _count(yours, theirs)
    return _agreed(yours, theirs) / n


def cohen_kappa(
    yours: Sequence[Hashable], theirs: Sequence[Hashable | None]
) -> float | None:
    """Cohen's kappa of the two lists; None when chance agreement is 1.

    In whole numbers: (n * agreed - s) / (n**2 - s), where s is the sum
    over the labels of how many of yours are it times how many of theirs.

    Example:
        >>> cohen_kappa(["a", "a", "b", "b"], ["a", "a", "b", "a"])
        0.5
        >>> cohen_kappa(["a", "a"], ["a", "a"]) is None
        True
    """
    n = _count(yours, theirs)
    mine, other = Counter(yours), Counter(theirs)
    chance = sum(count * other[label] for label, count in mine.items())
    if chance == n * n:
        return None
    return (n * _agreed(yours, theirs) - chance) / (n * n - chance)


def _check_scores(
    yours: Sequence[int], theirs: Sequence[int | None], low: int, high: int
) -> None:
    """Refuse a range that is empty, and a score that is not a whole number
    from low to high (None, a failed verdict, may stand in theirs)."""
    if type(low) is not int or type(high) is not int or low >= high:
        raise ValueError(f"low must be below high, both whole: not {low}, {high}")
    for score in [*yours, *(other for other in theirs if other is not None)]:
        if type(score) is not int or not low <= score <= high:
            raise ValueError(
                f"scores must be whole numbers from {low} to {high}, not {score!r}"
            )


def weighted_kappa(
    yours: Sequence[int], theirs: Sequence[int | None], low: int, high: int
) -> float | None:
    """Cohen's kappa with quadratic weights, for whole-number scores from low
    to high; None when chance would cost nothing, as when both give one
    score only.

    A disagreement of d points costs (d / (high - low))**2, and a failed
    verdict (None) 1, as much as the widest; kappa is 1 minus the cost seen
    over the cost chance would give, every pair of a score of yours and one
    of theirs. ValueError for a score out of range, or not whole.

    Example:
        >>> weighted_kappa([2, 4, 6, 8], [3, 4, 6, 9], 0, 10)  # 20/21
        0.9523809523809523
    """
    n = _count(yours, theirs)
    _check_scores(yours, theirs, low, high)
    widest = (high - low) ** 2

    def cost(mine: int, other: int | None) -> int:  # in units of 1 / widest
        return widest if other is None else (mine - other) ** 2

    seen = sum(cost(mine, other) for mine, other in zip(yours, theirs, strict=True))
    chance = sum(
        count * times * cost(mine, other)
        for mine, count in Counter(yours).items()
        for other, times in Counter(theirs).items()
    )
    if chance == 0:
        return None
    return (chance - n * seen) / chance


def within(
    yours: Sequence[float], theirs: Sequence[float | None], points: float
) -> float:
    """The share of items whose two scores differ by at most points; a
    failed verdict (None) never does.

    Example:
        >>> within([2, 4, 6, 8], [3, 4, 9, None], 1)
        0.5
    """
    n = _count(yours, theirs)
    close = sum(
        other is not None and abs(mine - other) <= points
        for mine, other in zip(yours, theirs, strict=True)
    )
    return close / n


def confusion(
    yours: Sequence[Hashable],
    theirs: Sequence[Hashable | None],
    labels: Sequence[Hashable],
    columns: Sequence[Hashable | None] | None = None,
) -> list[list[int]]:
    """The confusion matrix: a row per label of yours, in the order of
    labels, each counting the items of that label by the other rater's
    label, in the order of columns (default: labels), which may end with
    None for failed verdicts.

    ValueError for a label of yours not among labels, or of theirs not among
    columns, and for labels or columns that repeat one.

    Example:
        >>> confusion(["a", "a", "b"], ["a", None, "b"], ["a", "b"], ["a", "b", None])
        [[1, 0, 1], [0, 1, 0]]
    """
    _count(yours, theirs)
    columns = list(labels if columns is None else columns)
    if len(set(labels)) != len(labels) or len(set(columns)) != len(columns):
        raise ValueError("the rows and the columns must each name each label once")
    row_of = {label: place for place, label in enumerate(labels)}
    column_of = {label: place for place, label in enumerate(columns)}
    table = [[0] * len(columns) for _ in labels]
    for mine, other in zip(yours, theirs, strict=True):
        if mine not in row_of:
            raise ValueError(f"{mine!r} is not one of the rows' labels")
        if other not in column_of:
            raise ValueError(f"{other!r} is not one of the columns' labels")
        table[row_of[mine]][column_of[other]] += 1
    return table
