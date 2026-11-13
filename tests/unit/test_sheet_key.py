"""A labeled sheet's key read back (pushlab.scoring.sheet_key): key.json as
`pushlab label export` writes it, and what is refused when it is not."""

import doctest
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from pushlab.scoring import sheet_key
from pushlab.scoring.personas import Persona
from pushlab.scoring.rubrics import RUBRIC_NAMES, load_rubric
from pushlab.scoring.sheet_key import KeyItem, SheetKey, read_key

OWL = Persona("Brass Owl", "A pompous mechanical owl who lectures in riddles.")
SNOW = Persona("Snow Owl", "A quiet white owl of the far north.")
RUBRICS = {
    name: (load_rubric(name).version, load_rubric(name).sha256) for name in RUBRIC_NAMES
}
ITEMS = tuple(
    KeyItem(f"sheet-1-{k:03d}", "e.yaml", "tiny-pulses", 1, f"t-{k:06d}", "0" * 64)
    for k in range(1, 11)
)


def _key_data() -> dict[str, Any]:
    """key.json's content, as pushlab label export writes it."""
    return {
        "sheet": "sheet-1",
        "created_at": "2026-11-13T09:00:00Z",
        "seed": 0,
        "size": 200,
        "rubrics": {
            name: {"version": version, "sha256": sha256}
            for name, (version, sha256) in RUBRICS.items()
        },
        "persona": {"name": OWL.name, "description": OWL.description},
        "personas": [
            {"name": each.name, "description": each.description} for each in (OWL, SNOW)
        ],
        "labels_csv_sha256": "1" * 64,
        "strata": {"tiny-pulses / none": 10},
        "items": [
            {
                "item_id": item.item_id,
                "spec": item.spec,
                "experiment": item.experiment,
                "version": item.version,
                "run_id": item.run_id,
                "condition": "none",
                "control": False,
                "stratum": "tiny-pulses / none",
                "output_sha256": item.output_sha256,
            }
            for item in ITEMS
        ],
    }


def test_the_key_is_read(tmp_path: Path) -> None:
    path = tmp_path / "key.json"
    path.write_text(json.dumps(_key_data(), indent=2), encoding="utf-8")
    key = read_key(path)
    assert key == SheetKey(
        "sheet-1",
        RUBRICS,
        OWL,
        (OWL, SNOW),
        "1" * 64,
        ITEMS,
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    assert list(key.rubrics) == list(RUBRIC_NAMES)  # in the sheet's order
    data = _key_data() | {"persona": None, "personas": []}
    path.write_text(json.dumps(data), encoding="utf-8")
    assert (read_key(path).persona, read_key(path).personas) == (None, ())


def _drop_run_id(data: dict[str, Any]) -> None:
    del data["items"][3]["run_id"]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda data: data.pop("items"), "it lacks items"),
        (lambda data: data.update(items=[]), "items must list at least one item"),
        (_drop_run_id, r"items\[3\]'s run_id must be text, not None"),
        (
            lambda data: data["items"][1].update(version=0),
            r"items\[1\]'s version must be a whole number from 1, not 0",
        ),
        (
            lambda data: data["items"][4].update(item_id="sheet-1-001"),
            "it gives the item sheet-1-001 twice",
        ),
        (lambda data: data.update(rubrics={}), "rubrics must map each rubric"),
        (
            lambda data: data["rubrics"]["coherence"].update(version="1"),
            "rubrics must map each rubric",
        ),
        (
            lambda data: data.update(persona={"name": "none", "description": ""}),
            'persona: a persona cannot be named "none"',
        ),
        (lambda data: data.update(personas={"name": "Owl"}), "personas must be a list"),
        (lambda data: data.update(labels_csv_sha256=7), "labels_csv_sha256 must be"),
    ],
)
def test_a_key_that_is_not_a_sheets_is_refused(
    change: Any, message: str, tmp_path: Path
) -> None:
    data = _key_data()
    change(data)
    path = tmp_path / "key.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=f"is not a sheet's key: {message}"):
        read_key(path)


def test_a_key_that_cannot_be_read_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot read .*key.json: No such file"):
        read_key(tmp_path / "key.json")
    (tmp_path / "key.json").write_text("{items: 1}", encoding="utf-8")
    with pytest.raises(ValueError, match="key.json is not JSON"):
        read_key(tmp_path / "key.json")
    (tmp_path / "key.json").write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="is not a sheet's key: it is not a mapping"):
        read_key(tmp_path / "key.json")


def test_docstring_examples_run() -> None:
    result = doctest.testmod(sheet_key)
    assert result.attempted > 0
    assert result.failed == 0
