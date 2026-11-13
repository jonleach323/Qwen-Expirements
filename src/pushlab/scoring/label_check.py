"""`pushlab label check`: how well a judge agrees with your own labels, as a
judge must before it scores replies at scale (decision 0021;
docs/plan/development-plan.md, "Judge validation").

It reads a sheet that `pushlab label export` wrote and you labeled: its
key.json (pushlab.scoring.sheet_key) and your labels.csv, however a
spreadsheet saved it (pushlab.scoring.filled_sheet). It reads each labeled
reply back from its runner folder (pushlab.scoring.items), and refuses a
sheet whose replies changed since the export. Then the judge labels each
reply you labeled, for each rubric, through its cache, exactly as
`pushlab score` asks: the rubric, the sheet's persona (speaks_as) or
personas (which_persona), and the text; never the reply's condition, run
or prompt.

For each rubric it prints the labels read and skipped; the judge's failed
verdicts, each counted as a disagreement, for a failure is never a label;
the accuracy and Cohen's kappa (pushlab.scoring.agreement); for a score,
coherence, also the accuracy within 1 point and the quadratic-weighted
kappa; the confusion matrix, your labels as rows and the judge's as
columns, its failures in a column of their own; the items you and the
judge label differently, by ID; and PASS at an accuracy of 0.85 and a kappa
of 0.70 or more, an undefined kappa failing, else FAIL. For a score the
rule is applied to exact scores, which is strict. Numbers are cut to three
places, never rounded up, so one shown as 0.700 is at least 0.70
(pushlab.scoring.label_report).

The numbers go into a JSON record, in the sheet's folder unless --out names
another file: the judge, the rubrics' versions and hashes, the sheet's and
key's hashes, and per rubric its counts, scores, matrix and disagreeing
item IDs; no reply text and no reason of the judge's. It returns 0 once
the record is written, whatever the verdicts, and 1 when it refuses.
"""

import json
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pushlab.cli_labels import WORKERS
from pushlab.scoring.backends import judge_model, make_backend
from pushlab.scoring.cache import VerdictCache, cache_file_name, default_cache_dir
from pushlab.scoring.family import check_family
from pushlab.scoring.filled_sheet import FilledSheet, listing, read_labels
from pushlab.scoring.items import judged_runs
from pushlab.scoring.judge import (
    Job,
    Judge,
    JudgeUnavailable,
    Verdict,
    judge_name,
)
from pushlab.scoring.label_report import (
    about,
    compare,
    not_labeled,
    record,
    report,
    verdict_counts,
)
from pushlab.scoring.label_sheet import KEY_FILE, SHEET_FILE, plural
from pushlab.scoring.rubrics import Rubric, load_rubric
from pushlab.scoring.sheet_key import SheetKey, read_key
from pushlab.store.scores import output_sha256

PROGRESS = 50  # verdicts between progress lines
NEW_SHEET = "export a new sheet (pushlab label export)"


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


def _out(sheet: Path, judge: str, moment: datetime) -> Path:
    """The record's default path: in the sheet's folder, named after the
    judge, as its cache file is, and the time in UTC.

    Example:
        >>> _out(
        ...     Path("s"), "api:claude-haiku-5-5", datetime(2026, 11, 13, 9, tzinfo=UTC)
        ... )
        PosixPath('s/check-api-claude-haiku-5-5-20261113T090000Z.json')
    """
    name = cache_file_name(judge).removesuffix(".jsonl")
    return sheet / f"check-{name}-{moment:%Y%m%dT%H%M%SZ}.json"


def reply_texts(
    key: SheetKey, runs_dir: Path | None = None
) -> tuple[dict[str, str], list[str]]:
    """Each item's reply, by item ID, read back from its experiment's runner
    folder in runs_dir (None: the default), and the experiments' models.

    ValueError refuses an experiment file that is not there (key.json gives
    its path as export was given it) or cannot be read (judged_runs), and a
    sheet whose replies changed since the export: an experiment file that
    names another experiment now, a runner folder at another version, a run
    with no reply now, and a reply that is not the one exported.

    Example:
        >>> reply_texts(read_key(Path("sheet/key.json")))  # doctest: +SKIP
        ({'sheet-001': 'Hello!', ...}, ['qwen3-8b'])
    """
    texts: dict[str, str] = {}
    models: list[str] = []
    problems: list[str] = []
    for spec in dict.fromkeys(item.spec for item in key.items):
        if not Path(spec).exists():  # a path relative to where export ran
            raise ValueError(
                f"{spec}, an experiment file the sheet was exported from, is not "
                "there: run pushlab label check from the folder pushlab label "
                "export ran in"
            )
        experiment = judged_runs(Path(spec), runs_dir)
        models.append(experiment.spec.model)
        items = [item for item in key.items if item.spec == spec]
        name = experiment.spec.name
        if others := sorted({item.experiment for item in items} - {name}):
            problems.append(
                f"{spec} is the experiment {name} now, not {listing(others)}"
            )
            continue
        versions = sorted({item.version for item in items} - {experiment.version})
        if versions:
            exported = listing([str(version) for version in versions])
            problems.append(
                f"{name}'s runner folder is at version {experiment.version} now, but "
                f"the sheet's replies are of version {exported}"
            )
            continue
        runs = {run.run_id: run for run in experiment.runs}
        gone = [item.run_id for item in items if item.run_id not in runs]
        changed = [
            item.run_id
            for item in items
            if item.run_id in runs
            and output_sha256(runs[item.run_id]) != item.output_sha256
        ]
        if gone:
            verb = "has" if len(gone) == 1 else "have"
            problems.append(f"{listing(gone)} {verb} no saved reply now")
        if changed:
            replies = "reply" if len(changed) == 1 else "replies"
            problems.append(
                f"the {replies} of {listing(changed)} changed since the export"
            )
        texts |= {
            item.item_id: runs[item.run_id].output_text
            for item in items
            if item.run_id in runs
        }
    if problems:
        raise ValueError(f"{'; '.join(problems)}: {NEW_SHEET}")
    return texts, models


def _check_rubrics(
    key: SheetKey, rubrics: dict[str, Rubric], labeled: Sequence[str]
) -> None:
    """Refuse a labeled rubric that changed since the export, or whose
    persona or personas the key lacks."""
    changed, names = [], []
    for name in labeled:
        rubric, (version, sha256) = rubrics[name], key.rubrics[name]
        if rubric.version != version:
            changed.append(
                f"{name} was version {version} when the sheet was exported, and "
                f"rubrics/{name}.yaml is version {rubric.version} now"
            )
        elif rubric.sha256 != sha256:
            changed.append(
                f"rubrics/{name}.yaml changed since the sheet was exported, though "
                f"it is still version {version}"
            )
        else:
            continue
        names.append(name)
    if changed:
        column = "column" if len(names) == 1 else "columns"
        raise ValueError(
            f"{'; '.join(changed)}: a rubric's new version is checked on a new "
            "sheet (pushlab label export); to check the sheet's other rubrics now, "
            f"give --labels a copy of the sheet without the {listing(names)} {column}"
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
) -> dict[str, tuple[list[str], list[Job]]]:
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


def _summaries(
    key: SheetKey,
    rubrics: dict[str, Rubric],
    options: dict[str, list[Any]],
    filled: FilledSheet,
    jobs: dict[str, tuple[list[str], list[Job]]],
    found: Sequence[Verdict],
) -> dict[str, dict[str, Any]]:
    """Each rubric's entry in the record, in the key's order: its numbers,
    from found, the verdicts on every job in the order of jobs; or, with
    no label, why it has none."""
    verdicts, start = {}, 0
    for name, (_, each) in jobs.items():
        verdicts[name] = found[start : start + len(each)]
        start += len(each)
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
                rubric, options[name], ids, yours, verdicts[name], skipped
            )
        summaries[name] = numbers
    return summaries


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
) -> int:
    """Check a judge against your labels in the sheet folder `pushlab label
    export` wrote; return the exit status.

    labels is your labeled sheet (None: sheet/labels.csv). The judge is
    backend's (api or local) asking model (None: the backend's default) at
    base_url (local only), workers calls at a time, through its cache in
    cache_dir (None: default_cache_dir()); judge, given, replaces all that
    (for tests). runs_dir holds the runner folders (None: the default).
    out is the record's path (None: in the sheet, named after the judge and
    the time). Returns 0 once the record is written, whatever the verdicts,
    and 1 with the reason on stderr when it refuses: a sheet that is not a
    folder; a key or labels it cannot read, or labels with problems, every
    one named with its rows (pushlab.scoring.filled_sheet); no label at
    all; a labeled rubric that changed since the export, or whose persona
    the key lacks; an out that is a folder or exists; replies that changed
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
        options = {
            name: rubric.options(key.personas if rubric.kind == "choice" else ())
            for name, rubric in rubrics.items()
        }
        filled = read_labels(labels, key, options)
        labeled = [name for name in filled.columns if filled.labels[name]]
        if not labeled:
            raise ValueError(
                f"{labels} holds no label: fill in its rubrics' columns as "
                "instructions.md says, save it as CSV, then check again"
            )
        _check_rubrics(key, rubrics, labeled)
        asks = judge_model(backend, model) if judge is None else judge.backend.model
        name = judge_name(backend, asks) if judge is None else judge.name
    except ValueError as error:
        return _refuse(str(error))
    moment = _clock()
    out = _out(sheet, name, moment) if out is None else Path(out)
    if out.is_dir():
        return _refuse(f"--out must be a file, but {out} is a folder")
    if out.exists():
        return _refuse(f"{out} already exists: give another --out")
    try:
        texts, models = reply_texts(key, runs_dir)
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
    summaries = _summaries(key, rubrics, options, filled, jobs, found)
    counts = verdict_counts(found)
    print(report(key, filled, name, counts, summaries, rubrics))
    data = record(key, filled, name, moment, counts, summaries)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("x", encoding="utf-8") as file:
            file.write(
                json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
            )
    except OSError as error:
        return _refuse(f"cannot write {out}: {error}")
    print(f"Saved the numbers, with no reply text: {out}")
    return 0
