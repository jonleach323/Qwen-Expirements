"""`pushlab label export`: a sheet of sampled replies for you to label, so
that a judge can be checked against your labels before it scores at scale
(decision 0021; docs/plan/development-plan.md, "Judge validation").

It checks what it is given first: the rubrics, the folder, which must be
new or empty and outside the repository, as replies never go into git, and
the persona or personas the rubrics ask about. Then it samples the replies
of the experiments (pushlab.scoring.label_sheet) and writes three files
into the folder:
- labels.csv, for you to label in any spreadsheet;
- instructions.md: how to label, and each rubric's answers and meanings;
- key.json, never for the labeler: the sheet's settings, the runs folder
  it read, as an absolute path, and each item's experiment, run and
  condition, for `pushlab label check`.
An item's ID is the folder's name, the sample's token and a number, such
as coherence-1-3f2a-001: labels of another sample, even one in a folder of
the same name, match none of this sheet's items, and check refuses them.
It prints how many replies each stratum gave, and the files' paths, never a
reply's text. It loads no model and calls no judge.
"""

import hashlib
import json
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pushlab import PROJECT_ROOT
from pushlab.cli_labels import SHEET_SIZE
from pushlab.runs.folder import default_runs_dir
from pushlab.scoring.label_sheet import (
    INSTRUCTIONS_FILE,
    KEY_FILE,
    SHEET_FILE,
    LabelItem,
    distinct,
    gather,
    instructions,
    item_ids,
    plural,
    sample,
    sample_token,
    sheet_csv,
)
from pushlab.scoring.personas import Persona, pick_persona, read_personas
from pushlab.scoring.rubrics import RUBRIC_NAMES, Rubric, load_rubric
from pushlab.store.scores import now, output_sha256


def _refuse(message: str) -> int:
    print(f"pushlab label export: {message}", file=sys.stderr)
    return 1


def _check_folder(out: Path) -> str | None:
    """Why out cannot take a sheet, or None if it can."""
    folder = out.resolve()
    if folder.is_relative_to(PROJECT_ROOT.resolve()):
        return (
            f"--out must be outside the repository, {PROJECT_ROOT}: replies never "
            "go into git; use a folder such as /workspace/labels/coherence-1"
        )
    if folder.exists() and not folder.is_dir():
        return f"--out must be a folder, but {out} is a file"
    if folder.is_dir() and any(folder.iterdir()):
        return f"{out} is not empty: give a new or empty folder for the sheet"
    if not folder.name[:1].isalnum():
        return (
            "--out's name must start with a letter or a digit, as each item's ID "
            f"starts with it, not {folder.name!r}"
        )
    return None


def _personas(
    rubrics: Sequence[Rubric], personas_file: Path | None, persona: str | None
) -> tuple[Persona | None, list[Persona]]:
    """The persona the rubrics ask about, if any, and the personas file's
    personas; ValueError when the rubrics' needs are unmet, or an option is
    given that none of them uses."""
    one = [rubric.name for rubric in rubrics if rubric.needs == "persona"]
    listed = [rubric.name for rubric in rubrics if rubric.needs == "personas"]
    if one and personas_file is None:
        raise ValueError(
            f"{one[0]} asks about a persona: give --personas FILE, and --persona "
            "NAME if the file holds more than one"
        )
    if listed and personas_file is None:
        raise ValueError(f"{listed[0]} chooses among personas: give --personas FILE")
    if persona is not None and not one:
        raise ValueError(
            "--persona is for a rubric that asks about one persona, such as "
            "speaks_as, and none of those asked does"
        )
    if personas_file is None:
        return None, []
    if not one and not listed:
        raise ValueError(
            "--personas is for a rubric that asks about personas, such as "
            "speaks_as or which_persona, and none of those asked does"
        )
    personas = read_personas(Path(personas_file))
    return (pick_persona(personas, persona) if one else None), personas


def _as_entry(persona: Persona) -> dict[str, str]:
    return {"name": persona.name, "description": persona.description}


def _key_item(item_id: str, item: LabelItem) -> dict[str, Any]:
    """An item's entry in key.json: where its reply came from."""
    return {
        "item_id": item_id,
        "spec": item.spec,
        "experiment": item.experiment,
        "version": item.version,
        "run_id": item.run.run_id,
        "condition": item.condition,
        "control": item.control,
        "stratum": item.stratum,
        "output_sha256": output_sha256(item.run),
    }


def _left_out(blank: int, failed: int, repeats: int) -> str | None:
    """The line that says which replies were left out, or None for none.

    Example:
        >>> _left_out(3, 0, 1)
        'Left out: 3 blank replies and 1 that repeats a text taken already.'
    """
    parts = [plural(blank, "blank reply", "blank replies")] if blank else []
    parts += [plural(failed, "failed run", "failed runs")] if failed else []
    if repeats:
        verb = "repeats" if repeats == 1 else "repeat"
        parts.append(f"{repeats} that {verb} a text taken already")
    if not parts:
        return None
    listed = parts[0] if len(parts) == 1 else f"{', '.join(parts[:-1])} and {parts[-1]}"
    return f"Left out: {listed}."


def _summary(
    out: Path,
    taken: dict[str, int],
    available: dict[str, int],
    seed: int,
    size: int,
    left_out: str | None,
) -> str:
    """What export prints: how many replies each stratum gave, of how many
    it had, which were left out, and the files' paths; never a reply's text."""
    count = sum(available.values())
    lines = [
        f"Sampled {sum(taken.values())} of {plural(count, 'reply', 'replies')}, "
        f"seed {seed}, from {plural(len(available), 'stratum', 'strata')}:",
        *(f"  {name}: {taken[name]} of {of}" for name, of in available.items()),
    ]
    lines += [left_out] if left_out else []
    if count < size:
        lines.append(
            f"Note: there are only {count} replies, fewer than --size {size}, so the "
            "sheet has all of them."
        )
    lines += [
        "Label the sheet as the instructions say:",
        f"  {out / SHEET_FILE}",
        f"  {out / INSTRUCTIONS_FILE}",
        "and never open the key while labeling, as it names the conditions:",
        f"  {out / KEY_FILE}",
    ]
    return "\n".join(lines)


def run_label_export(
    specs: Sequence[Path],
    out: Path,
    *,
    rubrics: Sequence[str] | None = None,
    size: int = SHEET_SIZE,
    seed: int = 0,
    personas_file: Path | None = None,
    persona: str | None = None,
    runs_dir: Path | None = None,
) -> int:
    """Sample the replies of the experiments at specs into a sheet in out;
    return the exit status.

    rubrics are its label columns, in order (None: all, as RUBRIC_NAMES
    lists them); personas_file and persona give the persona or personas they
    ask about; runs_dir holds the runner folders (None: the default), and
    key.json records it as an absolute path, for label check. Returns
    0 once labels.csv, instructions.md and key.json are written, and 1 with
    the reason on stderr when it refuses: an unknown rubric; a size below 1
    or a seed below 0; out inside the repository, a file, not empty, or with
    a name that does not start with a letter or digit; a rubric's persona
    unmet (speaks_as: a personas file and one of its personas, which_persona:
    a personas file), a personas file it cannot read, or --personas or
    --persona with no rubric asked that uses it; an experiment it cannot
    read, or given twice; and no reply to sample.

    Example:
        >>> spec = Path("experiments/p0/s2_pulse_lengths.yaml")
        >>> out = Path("/workspace/labels/coherence-1")
        >>> run_label_export([spec], out, rubrics=["coherence"])  # doctest: +SKIP
        0
    """
    try:
        loaded = [load_rubric(name) for name in dict.fromkeys(rubrics or RUBRIC_NAMES)]
    except ValueError as error:
        return _refuse(str(error))
    if type(size) is not int or size < 1:
        return _refuse(f"--size must be a whole number from 1, not {size!r}")
    if type(seed) is not int or seed < 0:
        return _refuse(f"--seed must be a whole number from 0, not {seed!r}")
    out = Path(out)
    if (problem := _check_folder(out)) is not None:
        return _refuse(problem)
    try:
        chosen_persona, personas = _personas(loaded, personas_file, persona)
        items, blank, failed = gather(specs, runs_dir)
    except ValueError as error:
        return _refuse(str(error))
    kept = distinct(items)
    if not kept:
        return _refuse(
            f"no reply to sample: of the runs saved, {blank} have a blank reply "
            f"and {failed} failed"
        )
    chosen = sample(kept, size, seed)
    folder = out.resolve()
    ids = item_ids(f"{folder.name}-{sample_token(chosen)}", len(chosen))
    table = sheet_csv(ids, chosen, [rubric.name for rubric in loaded])
    available = Counter(item.stratum for item in kept)  # in the strata's order
    taken = dict.fromkeys(available, 0)
    for item in chosen:
        taken[item.stratum] += 1
    runs = default_runs_dir() if runs_dir is None else Path(runs_dir)
    key = {
        "sheet": folder.name,
        "created_at": now(),
        "seed": seed,
        "size": size,
        "runs_dir": str(runs.resolve()),  # where label check reads the replies
        "rubrics": {
            rubric.name: {"version": rubric.version, "sha256": rubric.sha256}
            for rubric in loaded
        },
        "persona": None if chosen_persona is None else _as_entry(chosen_persona),
        "personas": [_as_entry(each) for each in personas],
        "labels_csv_sha256": hashlib.sha256(table).hexdigest(),
        "strata": taken,
        "items": [_key_item(i, item) for i, item in zip(ids, chosen, strict=True)],
    }
    guide = instructions(loaded, chosen_persona, personas, len(chosen))
    try:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / SHEET_FILE).write_bytes(table)
        (folder / INSTRUCTIONS_FILE).write_text(guide, encoding="utf-8")
        text = json.dumps(key, indent=2, ensure_ascii=False) + "\n"
        (folder / KEY_FILE).write_text(text, encoding="utf-8")
    except OSError as error:
        return _refuse(f"cannot write the sheet in {out}: {error}")
    left_out = _left_out(blank, failed, len(items) - len(kept))
    print(_summary(out, taken, dict(available), seed, size, left_out))
    return 0
