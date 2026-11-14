"""`pushlab doses SPEC --units FILE`: each unit's working and collapse doses
in a scored dose ladder (decision 0022; pushlab.analysis.ladder).

It reads the ladder's saved runs that have a reply (judged_runs) and its
scores table, the newest row of each key that scored a run's reply as it
is now, as the scores view shows them (pushlab.store.score_views). A unit's
replies are its own runs at their dose (ladder_units.run_push) and the
unpushed runs at dose 0; its random controls' show beside them. Their
speaks_as verdicts are about the unit's persona, as `pushlab score SPEC
--rubric speaks_as --units FILE` asks it, and their coherence is the same
judge's: --judge's, else the only judge's saved. If a unit's replies were
asked about it more than one way, such as with another description, the
question asked last of its own replies counts, at every dose. A unit whose
persona is named "none", the judge's answer for no persona, is left out,
as that job leaves it unasked: the report's head counts such units and
their runs, and a units file of only such units is refused. Each dose
also shows the measures that need no judge that pushlab measure saved, if
any. --out writes the numbers as JSON, never a reply's text. Nothing loads
a model or writes anything else.
"""

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from operator import attrgetter
from pathlib import Path
from typing import Any

from pushlab import PROJECT_ROOT
from pushlab.analysis.dose_measures import SHOWN, current_measures, dose_measures
from pushlab.analysis.dose_text import (
    aligned,
    dose_table,
    estimate_text,
    short_text,
    window_text,
)
from pushlab.analysis.ladder import LadderPoint, UnitDoses, analyze_unit, speaks_by_dose
from pushlab.analysis.no_persona import no_persona_text
from pushlab.analysis.panel import Panel, read_panel
from pushlab.runs.folder import SCORES
from pushlab.scoring.items import Experiment, judged_runs
from pushlab.scoring.measure_rows import Measure
from pushlab.scoring.personas import NONE, names_no_persona
from pushlab.store.records import RunRecord
from pushlab.store.scores import ScoreRow, now, output_sha256, read_rows
from pushlab.units.ladder_units import LadderUnit, read_ladder_units, run_push

RUBRICS = ("speaks_as", "coherence")  # the judge's verdicts a ladder reads
Pairs = list[tuple[float, RunRecord]]  # runs, each with its dose


@dataclass(frozen=True)
class UnitReport:
    """A unit's ladder worked out: its doses; its random controls' speaks_as
    verdicts by dose, as (k, n); and the measures that need no judge."""

    unit: LadderUnit
    doses: UnitDoses
    controls: dict[float, tuple[int, int]]
    measures: dict[float, dict[str, float]]


def _refuse(message: str) -> int:
    print(f"pushlab doses: {message}", file=sys.stderr)
    return 1


def current_rows(
    rows: Sequence[ScoreRow], runs: Iterable[RunRecord], version: int
) -> list[ScoreRow]:
    """Of a scores table's rows, the newest of each key that scored a run's
    reply as it is now, in the folder's version, as the scores view shows
    them: the latest created_at, then the later in rows. Oldest first."""
    replies = {run.run_id: output_sha256(run) for run in runs}
    newest: dict[tuple[Any, ...], tuple[str, int, ScoreRow]] = {}
    for place, row in enumerate(rows):
        if row.version != version or replies.get(row.run_id) != row.output_sha256:
            continue  # an older version's, or another reply's
        if row.key not in newest or (row.created_at, place) >= newest[row.key][:2]:
            newest[row.key] = (row.created_at, place, row)
    return [row for *_, row in sorted(newest.values(), key=lambda kept: kept[:2])]


def asked_last(
    rows: Sequence[ScoreRow], own: set[str], runs: set[str]
) -> dict[str, ScoreRow]:
    """Of rows of one scorer, oldest first, those on the runs in runs of the
    question asked last of the runs in own, by run ID. A question is the
    judge, the scorer's version and question_sha256."""
    question = attrgetter("judge", "scorer_version", "question_sha256")
    asked = [question(row) for row in rows if row.run_id in own]
    return {
        row.run_id: row
        for row in rows
        if row.run_id in runs and asked and question(row) == asked[-1]
    }


def ladder_runs(
    runs: Sequence[RunRecord], units: Sequence[LadderUnit]
) -> tuple[dict[str, Pairs], dict[str, Pairs], list[RunRecord], int]:
    """Each unit's own runs and its random controls', by its key, with their
    doses; the unpushed runs; and how many runs push a unit not in units.
    ValueError for a run that pushes more than one unit or dose."""
    own: dict[str, Pairs] = {unit.key: [] for unit in units}
    controls: dict[str, Pairs] = {unit.key: [] for unit in units}
    unpushed, others = [], 0
    for run in runs:
        push = run_push(run)
        if push is None:
            unpushed.append(run)
        elif push.key not in own:
            others += 1
        elif push.control_for is None:
            own[push.key].append((push.dose, run))
        else:
            controls[push.key].append((push.dose, run))
    return own, controls, unpushed, others


def _ok(row: ScoreRow | None) -> ScoreRow | None:
    return row if row is not None and row.status == "ok" else None


def ladder_points(
    pairs: Pairs, speaks: dict[str, ScoreRow], coherence: dict[str, ScoreRow]
) -> list[LadderPoint]:
    """The runs as a ladder's points, each with its ok verdicts by run ID, or
    None. A point's key is its run's prompt alone, so a bootstrap resamples
    prompts, each with its replies under every seed and at every dose."""
    points = []
    for dose, run in pairs:
        said, scored = _ok(speaks.get(run.run_id)), _ok(coherence.get(run.run_id))
        label = None if said is None else said.label == "speaks_as"
        score = None if scored is None else scored.value
        points.append(LadderPoint(dose, run.prompt_id, label, score))
    return points


def pick_judge(rows: Sequence[ScoreRow], judge: str | None, make: str) -> str:
    """judge, or else the only judge whose speaks_as or coherence verdicts
    are in rows; make is the command that makes them, for a refusal."""
    judges = sorted({row.judge for row in rows if row.scorer in RUBRICS})
    if judge is not None and judge not in judges:
        saved = f"those saved are {', '.join(judges)}" if judges else "none is saved"
        raise ValueError(f"no speaks_as or coherence verdict of {judge}: {saved}")
    if judge is None and not judges:
        raise ValueError(f"no speaks_as or coherence verdict is saved yet: {make}")
    if judge is None and len(judges) > 1:
        raise ValueError(
            f"verdicts of {len(judges)} judges are saved, {', '.join(judges)}: "
            "choose one with --judge"
        )
    return judge or judges[0]


def unit_report(
    unit: LadderUnit,
    runs: tuple[Pairs, Pairs, list[RunRecord]],
    rows: Sequence[ScoreRow],
    judge: str,
    panel: Panel,
    makes: tuple[str, str],
    measured: tuple[dict[str, Measure], int] = ({}, 1),
) -> UnitReport:
    """A unit's ladder from its own runs, its controls' and the unpushed runs,
    with the judge's verdicts among rows; makes, the commands that make
    speaks_as and coherence verdicts, are for a refusal; measured holds the
    measures that need no judge to show (current_measures) and the folder's
    version."""
    own_pairs, control_pairs, unpushed = runs
    pairs = own_pairs + [(0.0, run) for run in unpushed]
    own = {run.run_id for _, run in own_pairs}
    ids = own | {run.run_id for _, run in control_pairs + pairs}
    judged = [row for row in rows if row.judge == judge]
    about = [r for r in judged if r.scorer == "speaks_as" and r.persona == unit.persona]
    speaks = asked_last(about, own, ids)
    coherence = asked_last([r for r in judged if r.scorer == "coherence"], own, ids)
    for found, rubric, make in zip((speaks, coherence), RUBRICS, makes, strict=True):
        if not any(_ok(found.get(run_id)) for run_id in own):
            about_it = f" about {unit.persona}" if rubric == "speaks_as" else ""
            raise ValueError(
                f"{unit.key} has no ok {rubric} verdict of {judge}{about_it} on "
                f"its replies: {make}"
            )
    controls = ladder_points(control_pairs, speaks, coherence)
    return UnitReport(
        unit,
        analyze_unit(ladder_points(pairs, speaks, coherence), panel),
        speaks_by_dose(controls) if controls else {},
        dose_measures(pairs, rows, *measured, panel),
    )


def unit_lines(report: UnitReport, panel: Panel) -> list[str]:
    """A unit's part of the report: a line naming it, the table of its
    doses, its working and collapse doses, and the window between them."""
    unit, doses, measures = report.unit, report.doses, report.measures.items()
    columns: dict[str, dict[float, str]] = {}
    if report.controls:
        shares = report.controls.items()
        columns["controls"] = {dose: f"{k}/{n}" for dose, (k, n) in shares if n}
    for name, places in SHOWN.items():
        if found := {d: f"{m[name]:.{places}f}" for d, m in measures if name in m}:
            columns[name] = found
    source = f" ({unit.source})" if unit.source else ""
    return [
        f"{unit.key}: {unit.persona}{source}",
        *dose_table(doses, panel, columns),
        f"working dose   {estimate_text(doses.working, panel.confidence)}",
        f"collapse dose  {estimate_text(doses.collapse, panel.confidence)}",
        f"window         {window_text(doses)}",
    ]


def report_text(
    experiment: Experiment,
    units_path: Path,
    judge: str,
    panel: Panel,
    reports: Sequence[UnitReport],
    others: int,
    unasked: tuple[int, int] = (0, 0),
) -> str:
    """The report: what was read, each unit's part, then a summary. others
    counts the runs that push units the units file lacks, and unasked the
    units whose persona is named "none" and their runs: all left out."""
    percent = f"{round(panel.confidence * 100, 6):g}%"
    name, version, failed = experiment.spec.name, experiment.version, experiment.failed
    replies = f"{len(experiment.runs):,} replies, {failed:,} failed runs left out"
    resamples = f"{panel.resamples:,} resamples of the prompts, seed {panel.seed}"
    head = [
        ["experiment", f"{name}, version {version}: {replies}"],
        ["units", f"{len(reports)}, from {units_path}"],
        ["judge", judge],
        ["panel", f"{panel.path}, version {panel.version}"],
        ["working", f"at a speaks_as rate of {panel.working_rate:g} or more"],
        ["collapsed", f"at a mean coherence below {panel.coherence_below:g}"],
        ["intervals", f"{percent}: Wilson's for a rate; else from {resamples}"],
        ["", "a rate's on its effective verdicts, with several seeds"],
    ]
    lines = aligned(head)
    if others:
        lines.append(f"note: {others:,} runs push units the units file lacks: left out")
    if unasked[0]:
        lines.append(f"note: {no_persona_text(*unasked)}: left out")
    for report in reports:
        lines += ["", *unit_lines(report, panel)]
    summary = [["unit", "persona", "working", "collapse", "window"]]
    for report in reports:
        found = report.doses
        summary.append(
            [report.unit.key, report.unit.persona, short_text(found.working)]
            + [short_text(found.collapse), window_text(found)]
        )
    lines += ["", f"Summary, with {percent} intervals:", *aligned(summary)]
    return "\n".join(lines) + "\n"


def doses_record(
    experiment: Experiment,
    units_path: Path,
    judge: str,
    panel: Panel,
    reports: Sequence[UnitReport],
) -> dict[str, Any]:
    """The numbers, as --out's JSON holds them: no reply's text."""
    units = []
    for report in reports:
        unit, found = report.unit, report.doses
        doses = []
        for dose, stats in found.stats.items():
            k, n = report.controls.get(dose, (0, 0))
            controls = {"k": k, "n": n} if n else None
            measures = report.measures.get(dose, {})
            doses.append(asdict(stats) | {"controls": controls, "measures": measures})
        units.append(
            {"unit": unit.key, "persona": unit.persona, "source": unit.source}
            | {"doses": doses, "working": asdict(found.working)}
            | {"collapse": asdict(found.collapse)}
        )
    numbers = {f.name: getattr(panel, f.name) for f in fields(panel)}
    return {
        "experiment": experiment.spec.name,
        "version": experiment.version,
        "units_file": str(units_path),
        "judge": judge,
        "panel": numbers | {"path": str(panel.path)},
        "created_at": now(),
        "units": units,
    }


def _check_out(out: Path, folder: Path | None = None) -> None:
    """Refuse an --out that exists, or lies in the repository or folder."""
    if out.exists() or out.is_symlink():
        raise ValueError(f"{out} already exists: give a new file")
    if out.resolve().is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError(
            f"--out must be outside the repository, {PROJECT_ROOT}: persona names "
            "stay out of git; use a file such as /workspace/reports/doses.json"
        )
    if folder is not None and out.resolve().is_relative_to(folder.resolve()):
        raise ValueError(f"--out must be outside {folder}, which holds the ladder")


def run_doses_command(args: argparse.Namespace) -> int:
    """`pushlab doses` with its parsed options (pushlab.cli_doses): print each
    unit's working and collapse doses in the scored ladder of args.spec, its
    units being args.units's, and with args.out write the numbers as JSON
    too. The exit status: 0 once done, 1 when it refuses, saying why on
    stderr.

    Example:
        >>> args = argparse.Namespace(spec=Path("no-such.yaml"), units=Path("u.csv"))
        >>> args.judge = args.panel = args.runs_dir = args.out = None
        >>> run_doses_command(args)  # stderr says why
        1
    """
    out, units_path = args.out, args.units
    makes = (
        f"pushlab score {args.spec} --rubric speaks_as --units {units_path} makes them",
        f"pushlab score {args.spec} --rubric coherence makes them",
    )
    try:
        if out is not None:
            _check_out(out)
        panel = read_panel(args.panel)
        units = read_ladder_units(units_path)
        experiment = judged_runs(args.spec, args.runs_dir)
        spec = experiment.spec
        if out is not None:
            _check_out(out, experiment.folder)
        if (model := units[0].unit.model_name) != spec.model:
            raise ValueError(
                f"{units_path} lists {model}'s units, but {spec.name} runs "
                f"{spec.model}: give the units file of its ladder"
            )
        if unnamed := [unit.key for unit in units if not unit.persona]:
            raise ValueError(
                f"{units_path} names no persona for {', '.join(unnamed)}: speaks_as "
                "asks about one, so give it in the persona column"
            )
        asked = [unit for unit in units if not names_no_persona(unit.persona)]
        if not asked:
            raise ValueError(
                f'every unit of {units_path} has the persona "{NONE}", which is the '
                "judge's answer for no persona: pushlab score judges none of their "
                "replies, so there is no dose to find"
            )
        try:
            own, controls, unpushed, others = ladder_runs(experiment.runs, units)
        except ValueError as error:
            raise ValueError(f"{spec.name} is no dose ladder: {error}") from None
        named_none = [unit.key for unit in units if names_no_persona(unit.persona)]
        none_runs = sum(len(own[key]) + len(controls[key]) for key in named_none)
        if missing := [unit.key for unit in asked if not own[unit.key]]:
            raise ValueError(
                f"{spec.name} has no saved run of {', '.join(missing)}: give the "
                "units file its ladder was written from"
            )
        saved = read_rows(experiment.folder, SCORES)
        rows = current_rows(saved, experiment.runs, experiment.version)
        judge = pick_judge(rows, args.judge, makes[0])
        measured = current_measures(experiment, panel), experiment.version
        reports = []
        for unit in asked:
            runs = (own[unit.key], controls[unit.key], unpushed)
            report = unit_report(unit, runs, rows, judge, panel, makes, measured)
            reports.append(report)
    except (OSError, ValueError) as error:
        return _refuse(str(error))
    unasked = len(named_none), none_runs
    print(
        report_text(experiment, units_path, judge, panel, reports, others, unasked),
        end="",
    )
    if out is None:
        return 0
    record = doses_record(experiment, units_path, judge, panel, reports)
    text = json.dumps(record, indent=2, allow_nan=False) + "\n"  # before the file
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("x", encoding="utf-8") as file:
            file.write(text)
    except OSError as error:
        return _refuse(f"cannot write {out}: {error}")
    print(f"\nSaved the numbers: {out}")
    return 0
