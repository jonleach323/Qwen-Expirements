"""What `pushlab label check` reports: how well a judge agrees with your
labels, rubric by rubric, as text and as a JSON record (decision 0021).

compare works out a labeled rubric's numbers from your labels and the
judge's verdicts on the same items (pushlab.scoring.agreement): a failed
verdict counts as a disagreement, never as a label. passes is the plan's
rule (docs/plan/development-plan.md, "Judge validation"): an accuracy of
0.85 and a Cohen's kappa of 0.70 or more, an undefined kappa failing. The
plan asks for 200 labels per rubric (MIN_LABELS): a rubric with fewer is
TOO FEW, whatever its numbers, which are still shown. For a score, such as
coherence, the rule is applied to exact scores, which is strict; the
accuracy within 1 point and the quadratic-weighted kappa are reported
beside it. report writes the numbers out for a person, cut to three
places, never rounded up, so that one shown as 0.700 is at least 0.70;
record gathers them for the JSON file. Neither holds a reply's text or a
reason the judge gave: items go by their IDs.
"""

import math
import textwrap
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from pushlab.scoring.agreement import (
    accuracy,
    cohen_kappa,
    confusion,
    weighted_kappa,
    within,
)
from pushlab.scoring.filled_sheet import FilledSheet
from pushlab.scoring.judge import Verdict
from pushlab.scoring.label_sheet import plural
from pushlab.scoring.rubrics import Rubric
from pushlab.scoring.sheet_key import SheetKey

PASS_ACCURACY = 0.85
PASS_KAPPA = 0.70
# The plan's hand labels per rubric: a rubric with fewer cannot pass.
MIN_LABELS = 200
TOO_FEW = "TOO FEW"
WITHIN = 1  # points a score may be off and count as close
WIDTH = 88  # the report's lines are wrapped to this
NOT_LABELED = "not labeled"
STRICT = (
    "The rule is applied to exact scores, which is strict: a score 1 point off "
    "counts as a disagreement. Weigh the two lines above beside it."
)


def passes(share: float, kappa: float | None) -> bool:
    """Whether a rubric's numbers pass the plan's rule: an accuracy of 0.85
    and a Cohen's kappa of 0.70 or more; an undefined kappa fails. A rubric
    also needs enough labels to pass (compare).

    Example:
        >>> passes(0.85, 0.70), passes(0.9, None), passes(0.849, 0.9)
        (True, False, False)
    """
    return share >= PASS_ACCURACY and kappa is not None and kappa >= PASS_KAPPA


def rule(min_labels: int = MIN_LABELS) -> str:
    """The rule, as the report states it: the labels a rubric needs to pass,
    min_labels, and the plan's own number when it is not that, then the
    numbers it needs.

    Example:
        >>> rule(10).split(", at")[0]
        'A rubric passes with 10 labels or more (the plan asks for 200)'
    """
    plan = "" if min_labels == MIN_LABELS else f" (the plan asks for {MIN_LABELS})"
    return (
        f"A rubric passes with {min_labels} labels or more{plan}, at an accuracy of "
        "0.85 and a Cohen's kappa of 0.70 or more (docs/plan/development-plan.md, "
        '"Judge validation"); numbers are cut to 3 places, never rounded up.'
    )


def shown(value: float | None) -> str:
    """A number cut to three places, never rounded up, so that one shown as
    0.700 is at least 0.7 (the small nudge only undoes a float's error in
    its last place); "undefined" for None.

    Example:
        >>> shown(13 / 17), shown(0.7), shown(-1.0), shown(None)
        ('0.764', '0.700', '-1.000', 'undefined')
    """
    if value is None:
        return "undefined"
    return f"{math.floor(value * 1000 + 1e-9) / 1000:.3f}"


def _kind(error: str) -> str:
    """A failure's kind: its error's words before the first colon, which hold
    no answer of the judge's.

    Example:
        >>> _kind('malformed answer: "score" must be a whole number')
        'malformed answer'
    """
    return error.split(":", 1)[0].strip()[:80] or "failed"


def _judged(rubric: Rubric, verdict: Verdict) -> Any:
    """The judge's label in the rubric's spelling, a whole number for a
    score; None for a failed verdict."""
    if not verdict.ok:
        return None
    return int(verdict.value) if rubric.kind == "score" else verdict.label  # type: ignore[arg-type]


def about(name: str, rubric: Rubric, key: SheetKey) -> dict[str, Any]:
    """A rubric's entry in the record before its numbers: its version and
    hash at export, and the persona or personas it asks about.

    Example:
        >>> from pushlab.scoring.rubrics import load_rubric
        >>> key = SheetKey("s", {"coherence": (1, "ab")}, None, (), "", (), "")
        >>> about("coherence", load_rubric("coherence"), key)
        {'version': 1, 'sha256': 'ab'}
    """
    version, sha256 = key.rubrics[name]
    entry: dict[str, Any] = {"version": version, "sha256": sha256}
    if rubric.needs == "persona" and key.persona is not None:
        entry["persona"] = key.persona.name
    if rubric.needs == "personas":
        entry["personas"] = [persona.name for persona in key.personas]
    return entry


def not_labeled(why: str, items: int) -> dict[str, Any]:
    """The numbers of a rubric with no label: why, and every item skipped.

    Example:
        >>> not_labeled("its column is empty", 3)["skipped"]
        3
    """
    return {"result": NOT_LABELED, "why": why, "labels": 0, "skipped": items}


def compare(
    rubric: Rubric,
    options: Sequence[Any],
    item_ids: Sequence[str],
    yours: Sequence[Any],
    verdicts: Sequence[Verdict],
    skipped: int,
    min_labels: int = MIN_LABELS,
) -> dict[str, Any]:
    """A labeled rubric's numbers: your labels (in the rubric's spelling,
    options giving them all) against the judge's verdicts on the same items,
    a failed verdict counting as a disagreement; skipped counts the items
    you did not label. Its result is TOO FEW with fewer labels than
    min_labels, whatever its numbers; else PASS or FAIL by the rule
    (passes). ValueError, as agreement's, for lists of different lengths or
    no labels.

    Example:
        >>> compare(
        ...     rubric, options, ["s-001"], ["assistant"], [verdict], 0
        ... )  # doctest: +SKIP
        {'result': 'TOO FEW', 'labels': 1, ...}
    """
    theirs = [_judged(rubric, verdict) for verdict in verdicts]
    failures = Counter(_kind(verdict.error) for verdict in verdicts if not verdict.ok)
    share, kappa = accuracy(yours, theirs), cohen_kappa(yours, theirs)
    pairs = list(zip(item_ids, yours, theirs, strict=True))
    result = "PASS" if passes(share, kappa) else "FAIL"
    numbers: dict[str, Any] = {
        "result": TOO_FEW if len(yours) < min_labels else result,
        "labels": len(yours),
        "skipped": skipped,
        "failed": sum(failures.values()),
        "failures": dict(failures),
        "cached": sum(verdict.cached for verdict in verdicts),
        "calls": sum(verdict.attempts for verdict in verdicts),
        "agreed": sum(mine == other for _, mine, other in pairs),
        "accuracy": share,
        "kappa": kappa,
    }
    if rubric.kind == "score":
        assert rubric.low is not None and rubric.high is not None
        close = within(yours, theirs, WITHIN)
        numbers |= {
            "within_1": close,
            "within_1_count": round(close * len(yours)),
            "weighted_kappa": weighted_kappa(yours, theirs, rubric.low, rubric.high),
        }
    table = confusion(yours, theirs, options, [*options, None])
    numbers["confusion"] = {
        "labels": list(options),
        "counts": [row[:-1] for row in table],
        "failed": [row[-1] for row in table],
    }
    numbers["disagreements"] = [
        {"item_id": item_id, "yours": mine, "judge": other}
        for item_id, mine, other in pairs
        if mine != other
    ]
    return numbers


def verdict_counts(verdicts: Sequence[Verdict]) -> dict[str, int]:
    """How many verdicts there are, from the cache, calls made, and failed.

    Example:
        >>> verdict_counts([])
        {'count': 0, 'cached': 0, 'calls': 0, 'failed': 0}
    """
    return {
        "count": len(verdicts),
        "cached": sum(verdict.cached for verdict in verdicts),
        "calls": sum(verdict.attempts for verdict in verdicts),
        "failed": sum(not verdict.ok for verdict in verdicts),
    }


def _table(numbers: Mapping[str, Any], score: bool) -> list[str]:
    """The confusion matrix as text: your labels as rows, the judge's as
    columns and its failures last; named labels' columns are numbered as
    the rows, so that long names fit."""
    labels = numbers["confusion"]["labels"]
    counts = numbers["confusion"]["counts"]
    rows = [
        [*row, failed]
        for row, failed in zip(counts, numbers["confusion"]["failed"], strict=True)
    ]
    if score:
        title = "Your scores as rows, the judge's as columns:"
        names = [str(label) for label in labels]
        heads = [*names, "failed"]
    else:
        title = "Your labels as rows, the judge's as columns, numbered as the rows:"
        names = [f"{place} {label}" for place, label in enumerate(labels, start=1)]
        heads = [*(str(place) for place in range(1, len(labels) + 1)), "failed"]
    widths = [
        max(len(head), *(len(str(row[place])) for row in rows))
        for place, head in enumerate(heads)
    ]
    side = max(len(name) for name in names)
    lines = [f"  {title}"]
    lines.append(
        "    "
        + " " * side
        + "".join(f"  {h:>{w}}" for h, w in zip(heads, widths, strict=True))
    )
    for name, row in zip(names, rows, strict=True):
        cells = "".join(f"  {count:>{w}}" for count, w in zip(row, widths, strict=True))
        lines.append(f"    {name:<{side}}{cells}")
    return lines


def _wrapped(first: str, entries: Sequence[str]) -> list[str]:
    """first, then the entries, comma-separated, in lines of at most WIDTH,
    never breaking an entry."""
    lines, line = [], first
    for place, entry in enumerate(entries):
        word = entry + ("," if place < len(entries) - 1 else "")
        if len(line) + 1 + len(word) > WIDTH and line.strip():
            lines.append(line)
            line = "   "
        line = f"{line} {word}"
    return [*lines, line]


def section(
    name: str,
    numbers: Mapping[str, Any],
    rubric: Rubric,
    min_labels: int = MIN_LABELS,
) -> list[str]:
    """A rubric's part of the report; min_labels is the labels it needed.

    Example:
        >>> from pushlab.scoring.rubrics import load_rubric
        >>> numbers = {"version": 1} | not_labeled("its column is empty", 3)
        >>> section("coherence", numbers, load_rubric("coherence"))
        ['coherence (version 1): not labeled: its column is empty.']
    """
    head = f"{name} (version {numbers['version']})"
    if numbers["result"] == NOT_LABELED:
        return [f"{head}: {NOT_LABELED}: {numbers['why']}."]
    if "persona" in numbers:
        head += f", about the persona {numbers['persona']}"
    elif "personas" in numbers:
        head += f", among {plural(len(numbers['personas']), 'persona', 'personas')}"
    labels, failed = numbers["labels"], numbers["failed"]
    lines = [f"{head}: {numbers['result']}"]
    lines.append(f"  Your labels: {labels} read, {numbers['skipped']} skipped.")
    if numbers["result"] == TOO_FEW:
        needed = f"the {min_labels} needed"
        if min_labels == MIN_LABELS:
            needed = f"the plan's {MIN_LABELS}"
        lines.append(f"  Too few labels to pass: {labels} of {needed}.")
    if failed:
        kinds = ", ".join(
            f"{kind}: {count}" for kind, count in numbers["failures"].items()
        )
        lines.append(
            f"  Failed verdicts: {failed}, each counted as a disagreement ({kinds})."
        )
    else:
        lines.append("  Failed verdicts: none.")
    agreed = f"{numbers['agreed']} of {labels} agree"
    lines.append(f"  Accuracy: {shown(numbers['accuracy'])} ({agreed}).")
    lines.append(f"  Cohen's kappa: {shown(numbers['kappa'])}.")
    score = rubric.kind == "score"
    if score:
        close = (
            f"{shown(numbers['within_1'])} ({numbers['within_1_count']} of {labels})"
        )
        lines.append(f"  Within {WITHIN} point: {close}.")
        lines.append(f"  Quadratic-weighted kappa: {shown(numbers['weighted_kappa'])}.")
        lines += textwrap.wrap(
            STRICT,
            WIDTH,
            initial_indent="  ",
            subsequent_indent="  ",
            break_on_hyphens=False,
        )
    lines += _table(numbers, score)
    differ = [
        f"{entry['item_id']} {entry['yours']}/"
        + ("failed" if entry["judge"] is None else str(entry["judge"]))
        for entry in numbers["disagreements"]
    ]
    if differ:
        lines += _wrapped(
            "  Where you and the judge differ (yours/the judge's):", differ
        )
    return lines


def report(
    key: SheetKey,
    filled: FilledSheet,
    judge: str,
    counts: Mapping[str, int],
    summaries: Mapping[str, Mapping[str, Any]],
    rubrics: Mapping[str, Rubric],
    min_labels: int = MIN_LABELS,
) -> str:
    """The report: the sheet, the judge, its verdicts and the rule, with the
    labels a rubric needed, then a section per rubric and every rubric's
    result."""
    replies = plural(len(key.items), "reply", "replies")
    lines = [f"Sheet {key.sheet}: {replies}; judge {judge}."]
    if missing := len(key.items) - len(filled.items):
        verb = "is" if missing == 1 else "are"
        lines.append(
            f"{missing} of the {len(key.items)} replies {verb} not in "
            f"{filled.path.name}: skipped for every rubric."
        )
    calls = plural(counts["calls"], "call", "calls")
    lines.append(
        f"The judge gave {plural(counts['count'], 'verdict', 'verdicts')}: "
        f"{counts['cached']} from its cache, {calls} made, {counts['failed']} failed."
    )
    lines += textwrap.wrap(rule(min_labels), WIDTH, break_on_hyphens=False)
    for name, numbers in summaries.items():
        lines += ["", *section(name, numbers, rubrics[name], min_labels)]
    results = ", ".join(
        f"{name} {numbers['result']}" for name, numbers in summaries.items()
    )
    return "\n".join([*lines, "", f"Results: {results}."])


def record(
    key: SheetKey,
    filled: FilledSheet,
    judge: str,
    moment: datetime,
    counts: Mapping[str, int],
    summaries: Mapping[str, Mapping[str, Any]],
    min_labels: int = MIN_LABELS,
) -> dict[str, Any]:
    """The JSON record of a check: what was compared, by which judge and
    when, under which rule (min_labels, the labels a rubric needed), with
    the hashes that pin the sheet, and every rubric's numbers; no reply text
    and no reason of the judge's."""
    return {
        "sheet": key.sheet,
        "checked_at": moment.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "judge": judge,
        "rule": {"accuracy": PASS_ACCURACY, "kappa": PASS_KAPPA, "labels": min_labels},
        "key_sha256": key.sha256,
        "labels_csv_sha256": key.labels_csv_sha256,
        "labels_file": str(filled.path),
        "labels_file_sha256": filled.sha256,
        "items": len(key.items),
        "rows": len(filled.items),
        "verdicts": dict(counts),
        "rubrics": {name: dict(numbers) for name, numbers in summaries.items()},
    }
