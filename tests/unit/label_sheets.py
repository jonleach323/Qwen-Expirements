"""Labeled sheets for the tests of pushlab label check, made by hand.

A runner folder saved by the real runner on the tiny model (saved_runs),
its replies replaced by invented ones (decision 0020): run k's is "Teapot
reply k sings.", and run 3's holds a line break, a semicolon and quotes.
CONDITIONS gives 20 runs, four on each of the five smoke prompts: an
unpushed run, a pulse and the pulse's two random controls. make_sheet
writes the folder `pushlab label export` would, item k for run k; rows and
save fill labels.csv in as a spreadsheet saves it; fake_judge answers by
each reply's text, so every number can be worked out by hand.
"""

import codecs
import csv
import hashlib
import io
import json
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from pushlab.cli import main
from pushlab.runs.folder import SHARDS, read_identity
from pushlab.runs.shards import read_results, write_shard
from pushlab.scoring.cache import VerdictCache
from pushlab.scoring.fake_backend import FakeBackend
from pushlab.scoring.judge import Judge, Reply
from pushlab.scoring.personas import Persona
from pushlab.scoring.rubrics import RUBRIC_NAMES, Request, load_rubric
from pushlab.store.records import RunRecord, read_runs_file
from pushlab.store.scores import output_sha256

NAME = "tiny-pulses"
SHEET = "sheet-1"
CONDITIONS = """\
conditions:
  - label: none
    segments: []
  - label: pulse
    segments: [{unit: tiny-vl:L2:N100, alpha: 1.0}]
    controls: 2
    control_seed: 3
"""
OWL = Persona("Brass Owl", "A pompous mechanical owl who lectures in riddles.")
SNOW = Persona("Snow Owl", "A quiet white owl of the far north.")
REASON = "The judge's own reason, which no record keeps."


def number(text: str) -> int:
    """The k of "reply k" in an invented reply, or of a run ID's end."""
    found = re.search(r"reply (\d+)", text) or re.search(r"-(\d+)$", text)
    assert found is not None, text
    return int(found[1])


def invented(run: RunRecord) -> str:
    """Run k's invented reply."""
    k = number(run.run_id)
    if k == 3:
        return 'Teapot reply 3 sings; "loud",\nthen rests.'
    return f"Teapot reply {k} sings."


def rewrite(folder: Path, text: Callable[[RunRecord], str] = invented) -> None:
    """Give each run saved in a runner folder the reply text(run) in place of
    the tiny model's: each shard is written again under its own name."""
    for path in sorted((folder / SHARDS).glob("*.parquet")):
        runs = [replace(run, output_text=text(run)) for run in read_runs_file(path)]
        path.unlink()
        write_shard(folder, path.name, runs)


def saved_runs(spec: Path, runs: Path) -> list[RunRecord]:
    """Run the experiment at spec into runs, invent its replies, and return
    its saved runs in run order."""
    assert main(["run", str(spec), "--runs-dir", str(runs)]) == 0
    rewrite(runs / NAME)
    return sorted(read_results(runs / NAME), key=lambda run: run.run_id)


def _entry(persona: Persona) -> dict[str, str]:
    return {"name": persona.name, "description": persona.description}


def make_sheet(
    folder: Path,
    spec: Path,
    runs: Path,
    rubrics: Sequence[str] = RUBRIC_NAMES,
    persona: Persona | None = OWL,
    personas: Sequence[Persona] = (OWL, SNOW),
) -> Path:
    """A sheet folder as `pushlab label export` writes one, made by hand in
    folder/sheet-1: an item per saved run, item k for run k, labels.csv
    still empty, instructions.md and key.json."""
    sheet = folder / SHEET
    sheet.mkdir(parents=True)
    saved = sorted(read_results(runs / NAME), key=lambda run: run.run_id)
    identity = read_identity(runs / NAME)
    assert identity is not None
    ids = [f"{SHEET}-{k:03d}" for k in range(1, len(saved) + 1)]
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["item_id", "text", *rubrics, "note"])
    for item_id, run in zip(ids, saved, strict=True):
        writer.writerow([item_id, run.output_text.strip(), *[""] * len(rubrics), ""])
    table = codecs.BOM_UTF8 + buffer.getvalue().encode("utf-8")
    (sheet / "labels.csv").write_bytes(table)
    (sheet / "instructions.md").write_text("# How to label\n", encoding="utf-8")
    items = []
    for item_id, run in zip(ids, saved, strict=True):
        condition = identity["conditions"][run.run_id]
        control = bool(run.control_of)
        items.append(
            {
                "item_id": item_id,
                "spec": str(spec),
                "experiment": NAME,
                "version": identity["version"],
                "run_id": run.run_id,
                "condition": condition,
                "control": control,
                "stratum": f"{NAME} / {condition}" + (" / control" if control else ""),
                "output_sha256": output_sha256(run),
            }
        )
    key = {
        "sheet": SHEET,
        "created_at": "2026-11-13T08:00:00Z",
        "seed": 0,
        "size": 200,
        "rubrics": {
            name: {
                "version": load_rubric(name).version,
                "sha256": load_rubric(name).sha256,
            }
            for name in rubrics
        },
        "persona": None if persona is None else _entry(persona),
        "personas": [_entry(each) for each in personas],
        "labels_csv_sha256": hashlib.sha256(table).hexdigest(),
        "strata": dict(Counter(item["stratum"] for item in items)),
        "items": items,
    }
    (sheet / "key.json").write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    return sheet


def rows(sheet: Path, columns: Mapping[str, Sequence[str]]) -> list[list[str]]:
    """labels.csv's rows as you fill them in: the header item_id, text, the
    columns' names as given, note; then each item's ID, text and cells, in
    the sheet's order, a column's cell k for item k + 1 (empty past its end).
    The texts are read back from labels.csv, as exported or as saved."""
    saved = (sheet / "labels.csv").read_bytes().decode("utf-8-sig")
    delimiter = ";" if ";" in saved.split("\n", 1)[0] else ","  # saved, or not
    found = csv.DictReader(io.StringIO(saved, newline=""), delimiter=delimiter)
    texts = [row["text"] for row in found]
    key = json.loads((sheet / "key.json").read_text(encoding="utf-8"))
    table = [["item_id", "text", *columns, "note"]]
    for k, item in enumerate(key["items"]):
        cells = [values[k] if k < len(values) else "" for values in columns.values()]
        table.append([item["item_id"], texts[k], *cells, ""])
    return table


def save(path: Path, table: Sequence[Sequence[str]], *, delimiter: str = ";") -> Path:
    """Save labels.csv as a spreadsheet in Europe saves CSV: a byte-order
    mark, semicolons and \\r\\n; path."""
    buffer = io.StringIO()
    csv.writer(buffer, delimiter=delimiter, lineterminator="\r\n").writerows(table)
    path.write_bytes(codecs.BOM_UTF8 + buffer.getvalue().encode("utf-8"))
    return path


def fake_judge(
    answers: Mapping[str, Sequence[Any]],
    cache: VerdictCache | None = None,
    **names: str,
) -> Judge:
    """A judge on a FakeBackend that answers a rubric about run k's reply with
    answers[rubric][k - 1]: a label or a score, a whole Reply, or an error
    to raise. Its waits are not slept."""

    def answer(request: Request) -> Any:
        given = answers[request.rubric][number(request.text) - 1]
        if isinstance(given, BaseException | Reply):
            return given
        field = load_rubric(request.rubric).field
        return json.dumps({"reason": REASON, field: given})

    return Judge(FakeBackend(answer, **names), cache, sleep=lambda seconds: None)
