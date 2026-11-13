"""`pushlab panel`: the panel's stages and its selection (decision 0023).

`pushlab panel stage STAGE SOURCE --out FILE` writes the next stage's
experiment file: b, ladder and confirm from the scores of the experiment
SOURCE (pushlab.panel.stages, which read them with pushlab.panel.stats),
preview from the panel file SOURCE (pushlab.panel.preview). Each reads
configs/panel.yaml, or --panel's file: its thresholds and its selection
section. It prints what it read and kept, writes the file only once it
reads back as the experiment meant and that experiment expands
(pushlab.panel.stage_files), then says what it wrote and what to do next.
It refuses, writing nothing, with the reason on stderr: an experiment not
run or not judged, a missing candidates list, a panel whose list changed,
a stage that keeps no candidate, and a bad --out. It loads no model and
runs nothing. `pushlab panel select` is not built yet.
"""

import argparse
import sys
from pathlib import Path

from pushlab.analysis.panel import read_panel
from pushlab.panel.preview import SHAPES, stage_preview
from pushlab.panel.settings import Selection, read_selection
from pushlab.panel.stage_files import (
    Stage,
    check_out,
    next_out,
    stage_hours,
    stage_mapping,
    write_stage,
)
from pushlab.panel.stage_text import stage_text
from pushlab.panel.stages import BUILDERS, NEEDS, NoneKept
from pushlab.panel.stats import COHERENCE, SPEAKS_AS, read_scored
from pushlab.runs.estimate import exact
from pushlab.runs.ladder_command import experiment_name


def _refuse(command: str, message: str) -> int:
    print(f"pushlab panel {command}: {message}", file=sys.stderr)
    return 1


def _os_text(error: OSError) -> str:
    """An OSError in words: its own message, or the file and the reason."""
    if error.strerror is None:  # such as write_stage's FileExistsError
        return str(error)
    where = f"{error.filename}: " if error.filename else ""
    return f"{where}{error.strerror}"


def next_steps(stage: str, out: Path, source: Path, built: Stage) -> list[str]:
    """What to do once a stage's file is written: see its cost, then run
    it, judge it and write the next stage.

    Example:
        >>> built = Stage("qwen3-8b", None, {"file": "c.csv"}, (), 1, "")
        >>> print(next_steps("b", Path("/w/stage-b.yaml"), Path("a.yaml"), built)[-1])
          uv run pushlab panel stage ladder /w/stage-b.yaml --out /w/stage-ladder.yaml
    """
    lines = [
        "Next, see what it would run and cost; no model is loaded:",
        f"  uv run pushlab run {out} --dry-run",
        "then on the pod run it, in tmux (docs/runbooks/experiments.md),",
    ]
    if stage == "preview":
        report = Path("/workspace/reports") / f"{out.stem}.md"
        return lines + [
            "and read its replies with markers where the pushes switch:",
            f"  uv run pushlab run {out}",
            f"  uv run pushlab report {out} --out {report}",
        ]
    units = (built.candidates or {}).get("file", "LIST")
    lines += [
        "judge its replies, and write the next stage:",
        f"  uv run pushlab run {out}",
        f"  uv run pushlab score {out} --rubric {SPEAKS_AS} --units {units}",
    ]
    if stage in ("ladder", "confirm"):
        lines.append(f"  uv run pushlab score {out} --rubric {COHERENCE}")
    if stage == "b":
        upcoming = next_out(out, "b", "ladder")
        lines.append(f"  uv run pushlab panel stage ladder {out} --out {upcoming}")
    elif stage == "ladder":
        upcoming = next_out(out, "ladder", "confirm")
        lines.append(f"  uv run pushlab panel stage confirm {out} --out {upcoming}")
        lines.append(
            f"(`uv run pushlab doses {out} --units {units}` shows each one's doses.)"
        )
    else:
        categories = out.parent / "categories.csv"
        lines.append(
            f"  uv run pushlab panel select {out} --ladder {source} --categories "
            f"{categories}"
        )
    return lines


def _report(
    stage: str,
    out: Path,
    source: Path,
    name: str,
    built: Stage,
    selection: Selection,
    runs: int,
    hours: str,
) -> None:
    """Print what the stage read and kept, what it wrote, and what next."""
    print("\n".join(built.shown))
    count = "every prompt"
    if built.prompt_ids is not None:
        count = f"{len(built.prompt_ids)} of its prompts, by prompt_ids"
    lines = [
        ("prompts", f"{selection.prompts}: {count}"),
        ("decoding", f"greedy, seed 0, {selection.max_new_tokens} reply tokens"),
        ("calibration", "fixed"),
        ("runs", f"{runs:,}"),
        ("max_gpu_hours", hours),
    ]
    if stage == "preview":
        pairs = len(built.conditions) // len(SHAPES)
        lines.insert(0, ("conditions", f"{pairs} pairs x {len(SHAPES)} shapes"))
    else:
        listed = built.candidates or {}
        keys = f"{len(listed.get('keys', ())):,} of {listed.get('file')}, by key"
        lines.insert(0, ("candidates", keys))
    width = max(len(label) for label, _ in lines) + 2
    print(f"\nWrote {out}: the experiment {name}, on {built.model}.")
    print("\n".join(f"{label:<{width}}{text}" for label, text in lines))
    if built.from_library:
        print(
            "\nIt holds the Neuron Persona Library's data, which states no licence: "
            "keep it out of git (decision 0020)."
        )
    print("\n" + "\n".join(next_steps(stage, out, source, built)))


def run_stage(args: argparse.Namespace) -> int:
    """`pushlab panel stage` with its parsed options (pushlab.cli_panel):
    write the next stage's file; 0 once written, 1 when it refuses, saying
    why on stderr, having written nothing.

    Example:
        >>> args = argparse.Namespace(stage="b", source=Path("no.yaml"))
        >>> args.out, args.name, args.overwrite = Path("/tmp/b.yaml"), None, False
        >>> args.judge = args.panel = args.max_gpu_hours = args.runs_dir = None
        >>> run_stage(args)  # stderr says why
        1
    """
    stage, out, source = args.stage, Path(args.out), Path(args.source)
    command = f"stage {stage}"
    try:
        selection = read_selection(args.panel)
        panel = read_panel(args.panel)
        name = experiment_name(args.name, out)
        check_out(out, source, args.overwrite)
        if stage == "preview":
            if args.judge is not None:
                raise ValueError(
                    "--judge chooses whose verdicts a stage reads, and preview "
                    "reads none: leave it out"
                )
            built = stage_preview(source, selection, panel)
        else:
            scored = read_scored(source, args.runs_dir, args.judge, NEEDS[stage])
            built = BUILDERS[stage](scored, selection, panel)
        hours, how = stage_hours(
            built.model,
            built.runs,
            selection.max_new_tokens,
            args.max_gpu_hours,
            args.runs_dir,
        )
        mapping = stage_mapping(built, selection, name, hours)
        text = stage_text(built, mapping, f"{how[0].upper()}{how[1:]}.")
        _, expansion = write_stage(text, mapping, out, args.overwrite)
    except NoneKept as refusal:
        print("\n".join(refusal.shown))
        return _refuse(command, str(refusal))
    except ValueError as error:
        return _refuse(command, str(error))
    except OSError as error:
        return _refuse(command, _os_text(error))
    for warning in expansion.warnings:
        print(f"pushlab panel {command}: warning: {warning}", file=sys.stderr)
    shown = exact(hours) + (", as given" if how.startswith("given") else f": {how}")
    _report(stage, out, source, name, built, selection, len(expansion.runs), shown)
    return 0


def run_panel_command(args: argparse.Namespace) -> int:
    """`pushlab panel` with its parsed options (pushlab.cli_panel); the exit
    status. select is not built yet, and says so.

    Example:
        >>> run_panel_command(argparse.Namespace(panel_command="select"))
        1
    """
    if args.panel_command == "stage":
        return run_stage(args)
    return _refuse("select", "not built yet")
