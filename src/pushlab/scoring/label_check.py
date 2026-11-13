"""`pushlab label check`: how well a judge agrees with your own labels, as a
judge must before it scores replies at scale (decision 0021;
docs/plan/development-plan.md, "Judge validation").

It reads a sheet that `pushlab label export` wrote and you labeled: its
key.json (pushlab.scoring.sheet_key) and your labels.csv, however a
spreadsheet saved it (pushlab.scoring.filled_sheet). A labeled rubric that
changed since the export is refused as changed, its cells unchecked, for
your labels are not what changed. It reads each labeled reply back from its
runner folder, in the runs folder key.json names unless --runs-dir names
another, and refuses a sheet whose replies cannot be found or changed since
the export (sheet_key.reply_texts). Then the judge labels each reply you
labeled, for each rubric, through its cache, exactly as `pushlab score`
asks: the rubric, the sheet's persona (speaks_as) or personas
(which_persona), and the text; never the reply's condition, run or prompt.

For each rubric it prints the labels read and skipped; the judge's failed
verdicts, each counted as a disagreement, for a failure is never a label;
the accuracy and Cohen's kappa (pushlab.scoring.agreement); for a score,
coherence, also the accuracy within 1 point and the quadratic-weighted
kappa; the confusion matrix, your labels as rows and the judge's as
columns, its failures in a column of their own; the items you and the
judge label differently, by ID; and PASS at an accuracy of 0.85 and a kappa
of 0.70 or more, an undefined kappa failing, else FAIL, or TOO FEW, whatever
the numbers, with fewer labels than the plan's 200. For a score the rule
is applied to exact scores, which is strict. Numbers are cut to three
places, never rounded up, so one shown as 0.700 is at least 0.70
(pushlab.scoring.label_report).

The numbers go into a JSON record, in the sheet's folder unless --out names
another file: the judge, the rule, the rubrics' versions and hashes, the
sheet's and key's hashes, and per rubric its counts, scores, matrix and
disagreeing item IDs; no reply text and no reason of the judge's. The
judge's reasons, which may quote replies, go into a disagreements file
(disagreements_csv) in the sheet's folder, beside the replies, wherever
--out puts the record: a row per item and rubric where you and the judge
differ. It is named as the record is by default, after the judge and the
same UTC time, disagreements-<judge>-<time>.csv, so that each pairs with
its record when several judges check one sheet; one that exists is never
overwritten, and the check is refused. It returns 0 once both are
written, whatever the verdicts, and 1 when it refuses.
"""

import json
import sys
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pushlab.cli_labels import WORKERS
from pushlab.scoring.backends import judge_model, make_backend
from pushlab.scoring.cache import VerdictCache, cache_file_name, default_cache_dir
from pushlab.scoring.family import check_family
from pushlab.scoring.filled_sheet import FilledSheet, listing, read_labels
from pushlab.scoring.judge import Job, Judge, JudgeUnavailable, Verdict, judge_name
from pushlab.scoring.label_report import (
    MIN_LABELS,
    about,
    compare,
    not_labeled,
    record,
    report,
    verdict_counts,
)
from pushlab.scoring.label_sheet import KEY_FILE, SHEET_FILE, guarded_csv, plural
from pushlab.scoring.rubrics import Rubric, load_rubric
from pushlab.scoring.sheet_key import NEW_SHEET, SheetKey, read_key, reply_texts

PROGRESS = 50  # verdicts between progress lines
FAILED = "failed"  # the judge's label, in a disagreements file, of a failure
DISAGREEMENT_COLUMNS = ("item_id", "rubric", "yours", "judge", "reason")

Jobs = dict[str, tuple[list[str], list[Job]]]  # by rubric: item IDs and jobs


def _refuse(message: str) -> int:
    print(f"pushlab label check: {message}", file=sys.stderr)
    return 1


def _clock() -> datetime:
    """The time now, in UTC; tests replace it."""
    return datetime.now(UTC)


def _progress(count: int) -> None:
    """Print how many verdicts the judge has given, every PROGRESS of them."""
    if count % PROGRESS == 0:
        print(f"  {count} verdicts so far", flush=True)


def _paths(sheet: Path, judge: str, moment: datetime) -> tuple[Path, Path]:
    """The record's default path and the disagreements file's, both in the
    sheet's folder, named after the judge, as its cache file is, and the
    time in UTC, so that the two pair.

    Example:
        >>> moment = datetime(2026, 11, 13, 9, tzinfo=UTC)
        >>> record, reasons = _paths(Path("s"), "api:claude-haiku-5-5", moment)
        >>> record
        PosixPath('s/check-api-claude-haiku-5-5-20261113T090000Z.json')
        >>> reasons.name
        'disagreements-api-claude-haiku-5-5-20261113T090000Z.csv'
    """
    stamp = f"{cache_file_name(judge).removesuffix('.jsonl')}-{moment:%Y%m%dT%H%M%SZ}"
    return sheet / f"check-{stamp}.json", sheet / f"disagreements-{stamp}.csv"


def disagreements_csv(rows: Iterable[Sequence[object]]) -> bytes:
    """A disagreements file: a header of item_id, rubric, yours, judge and
    reason, then rows, each an item and rubric where you and the judge
    differ: your label, the judge's or "failed", and its reason or, for a
    failure, its error; written as labels.csv is (label_sheet.guarded_csv).

    Example:
        >>> disagreements_csv([["s-001", "coherence", 7, 4, "-Slips."]])[3:]
        b"item_id,rubric,yours,judge,reason\\r\\ns-001,coherence,7,4,'-Slips.\\r\\n"
    """
    return guarded_csv([DISAGREEMENT_COLUMNS, *rows])


def _changed(key: SheetKey, rubrics: dict[str, Rubric]) -> dict[str, str]:
    """The key's rubrics that changed since the export, each with how: its
    file is another version now, or the same version with other words."""
    changed = {}
    for name, rubric in rubrics.items():
        version, sha256 = key.rubrics[name]
        if rubric.version != version:
            changed[name] = (
                f"{name} was version {version} when the sheet was exported, and "
                f"rubrics/{name}.yaml is version {rubric.version} now"
            )
        elif rubric.sha256 != sha256:
            changed[name] = (
                f"rubrics/{name}.yaml changed since the sheet was exported, though "
                f"it is still version {version}"
            )
    return changed


def _check_rubrics(
    key: SheetKey,
    rubrics: dict[str, Rubric],
    changed: dict[str, str],
    labeled: Sequence[str],
) -> None:
    """Refuse a labeled rubric that changed since the export (changed says
    how), or whose persona or personas the key lacks."""
    if names := [name for name in labeled if name in changed]:
        column = "column" if len(names) == 1 else "columns"
        raise ValueError(
            f"{'; '.join(changed[name] for name in names)}: a rubric's new version "
            "is checked on a new sheet (pushlab label export); to check the "
            "sheet's other rubrics now, give --labels a copy of the sheet without "
            f"the {listing(names)} {column}"
        )
    for name in labeled:
        if rubrics[name].needs == "persona" and key.persona is None:
            raise ValueError(
                f"key.json names no persona, which {name} asks about: {NEW_SHEET} "
                "with --personas and --persona"
            )
        if rubrics[name].needs == "personas" and not key.personas:
            raise ValueError(
                f"key.json lists no personas, which {name} chooses among: "
                f"{NEW_SHEET} with --personas"
            )


def _jobs(
    key: SheetKey,
    rubrics: dict[str, Rubric],
    filled: FilledSheet,
    texts: dict[str, str],
    labeled: Sequence[str],
) -> Jobs:
    """For each labeled rubric, the items you labeled, in the key's order,
    and a job for each: the rubric, the key's persona (speaks_as) or
    personas (which_persona), and the reply's text, as pushlab score asks."""
    jobs = {}
    for name in labeled:
        rubric = rubrics[name]
        persona = key.persona if rubric.needs == "persona" else None
        personas = key.personas if rubric.needs == "personas" else ()
        ids = [
            item.item_id for item in key.items if item.item_id in filled.labels[name]
        ]
        jobs[name] = (ids, [Job(rubric, texts[i], persona, personas) for i in ids])
    return jobs


def _by_rubric(jobs: Jobs, found: Sequence[Verdict]) -> dict[str, Sequence[Verdict]]:
    """found, the verdicts on every job in the order of jobs, by rubric."""
    verdicts, start = {}, 0
    for name, (_, each) in jobs.items():
        verdicts[name] = found[start : start + len(each)]
        start += len(each)
    return verdicts


def _summaries(
    key: SheetKey,
    rubrics: dict[str, Rubric],
    options: dict[str, list[Any]],
    filled: FilledSheet,
    jobs: Jobs,
    verdicts: dict[str, Sequence[Verdict]],
    min_labels: int,
) -> dict[str, dict[str, Any]]:
    """Each rubric's entry in the record, in the key's order: its numbers,
    from the judge's verdicts, by rubric; or, with no label, why it has
    none. A rubric with fewer labels than min_labels is TOO FEW."""
    summaries = {}
    for name, rubric in rubrics.items():
        numbers = about(name, rubric, key)
        if name not in filled.columns:
            why = f"{filled.path.name} has no {name} column"
            numbers |= not_labeled(why, len(key.items))
        elif name not in jobs:
            numbers |= not_labeled("its column is empty", len(key.items))
        else:
            ids, _ = jobs[name]
            yours = [filled.labels[name][item_id] for item_id in ids]
            skipped = len(key.items) - len(ids)
            numbers |= compare(
                rubric, options[name], ids, yours, verdicts[name], skipped, min_labels
            )
        summaries[name] = numbers
    return summaries


def _disagreements(
    jobs: Jobs,
    verdicts: dict[str, Sequence[Verdict]],
    summaries: dict[str, dict[str, Any]],
) -> list[list[Any]]:
    """The disagreements file's rows: for each labeled rubric, each item you
    and the judge label differently, your label, the judge's or "failed",
    and its reason or, for a failure, its error."""
    rows = []
    for name, (ids, _) in jobs.items():
        said = dict(zip(ids, verdicts[name], strict=True))
        for entry in summaries[name]["disagreements"]:
            verdict = said[entry["item_id"]]
            judged = entry["judge"] if verdict.ok else FAILED
            why = verdict.reason if verdict.ok else verdict.error
            rows.append([entry["item_id"], name, entry["yours"], judged, why])
    return rows


def run_label_check(
    sheet: Path,
    *,
    labels: Path | None = None,
    backend: str = "api",
    model: str | None = None,
    base_url: str | None = None,
    workers: int = WORKERS,
    cache_dir: Path | None = None,
    runs_dir: Path | None = None,
    out: Path | None = None,
    judge: Judge | None = None,
    min_labels: int = MIN_LABELS,
) -> int:
    """Check a judge against your labels in the sheet folder `pushlab label
    export` wrote; return the exit status.

    labels is your labeled sheet (None: sheet/labels.csv). The judge is
    backend's (api or local) asking model (None: the backend's default) at
    base_url (local only), workers calls at a time, through its cache in
    cache_dir (None: default_cache_dir()); judge, given, replaces all that
    (for tests). runs_dir holds the runner folders (None: the one key.json
    names, else the default). out is the record's path (None: in the sheet,
    named after the judge and the time); the disagreements file goes in the
    sheet whatever out is, named after the judge and the same time.
    min_labels is how many labels a rubric needs to pass: the plan's 200,
    fewer only for tests or a pilot. Returns 0 once the record and the
    disagreements file are written, whatever the verdicts, and 1 with the
    reason on stderr when it refuses: a sheet that is not a folder; a key
    or labels it cannot read, or labels with problems, every one named with
    its rows (pushlab.scoring.filled_sheet); no label at all; a labeled
    rubric that changed since the export, or whose persona the key lacks;
    an out that is a folder or exists, or a disagreements file of the same
    judge and time that exists; replies that cannot be found or changed
    since the export; a judge of the tested model's family, or one that
    cannot be made; and a judge that cannot answer at all.

    Example:
        >>> run_label_check(Path("/workspace/labels/coherence-1"))  # doctest: +SKIP
        0
    """
    sheet = Path(sheet)
    if not sheet.is_dir():
        return _refuse(
            f"{sheet} is not a folder: give the folder that pushlab label export wrote"
        )
    labels = sheet / SHEET_FILE if labels is None else Path(labels)
    try:
        key = read_key(sheet / KEY_FILE)
        rubrics = {name: load_rubric(name) for name in key.rubrics}
        # A changed rubric's cells are not checked against its new labels,
        # so that the refusal says it changed, not that your labels are wrong.
        changed = _changed(key, rubrics)
        options = {
            name: rubric.options(key.personas if rubric.kind == "choice" else ())
            for name, rubric in rubrics.items()
            if name not in changed
        }
        filled = read_labels(labels, key, options)
        labeled = [name for name in filled.columns if filled.labels[name]]
        if not labeled:
            raise ValueError(
                f"{labels} holds no label: fill in its rubrics' columns as "
                "instructions.md says, save it as CSV, then check again"
            )
        _check_rubrics(key, rubrics, changed, labeled)
        asks = judge_model(backend, model) if judge is None else judge.backend.model
        name = judge_name(backend, asks) if judge is None else judge.name
    except ValueError as error:
        return _refuse(str(error))
    moment = _clock()
    default, reasons = _paths(sheet, name, moment)  # reasons beside the replies
    out = default if out is None else Path(out)
    if out.is_dir():
        return _refuse(f"--out must be a file, but {out} is a folder")
    if out.exists():
        return _refuse(f"{out} already exists: give another --out")
    if reasons.exists():
        return _refuse(
            f"{reasons} already exists: it is named after the judge and the time, "
            "so check again a second later"
        )
    try:
        texts, models = reply_texts(key, key.runs_dir if runs_dir is None else runs_dir)
        check_family(asks, models)
    except ValueError as error:
        return _refuse(str(error))
    if judge is None:
        folder = default_cache_dir() if cache_dir is None else Path(cache_dir)
        try:
            judge = Judge(
                make_backend(backend, model, base_url), VerdictCache(folder, name)
            )
        except ValueError as error:
            return _refuse(str(error))
        except OSError as error:
            return _refuse(f"cannot read the judge's cache in {folder}: {error}")
    jobs = _jobs(key, rubrics, filled, texts, labeled)
    every = [job for _, each in jobs.values() for job in each]
    print(
        f"Asking {name} about {plural(len(every), 'label', 'labels')} of yours, "
        f"{workers} at a time; its cache answers what it was asked before.",
        flush=True,
    )
    try:
        found = judge.verdicts(every, workers, _progress)
    except (JudgeUnavailable, ValueError) as error:
        return _refuse(str(error))
    verdicts = _by_rubric(jobs, found)
    summaries = _summaries(key, rubrics, options, filled, jobs, verdicts, min_labels)
    counts = verdict_counts(found)
    print(report(key, filled, name, counts, summaries, rubrics, min_labels))
    data = record(key, filled, name, moment, counts, summaries, min_labels)
    rows = _disagreements(jobs, verdicts, summaries)  # never in the record
    try:
        with reasons.open("xb") as file:  # never over another check's
            file.write(disagreements_csv(rows))
    except OSError as error:
        return _refuse(f"cannot write {reasons}: {error}")
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("x", encoding="utf-8") as file:
            file.write(
                json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
            )
    except OSError as error:
        return _refuse(f"cannot write {out}: {error}")
    print(
        "Saved the judge's reasons where you differ, which may quote replies: "
        f"{reasons}"
    )
    print(f"Saved the numbers, with no reply text: {out}")
    return 0
