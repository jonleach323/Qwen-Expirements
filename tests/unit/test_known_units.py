"""units/known.csv: the published units that random controls must avoid."""

import csv
from pathlib import Path

from pushlab.models.discovery import ModelInfo
from pushlab.models.registry import REGISTRY
from pushlab.units.neuron import NeuronUnit

KNOWN = Path(__file__).resolve().parents[2] / "units" / "known.csv"


def _rows() -> list[dict[str, str]]:
    with KNOWN.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_known_units_have_the_documented_columns() -> None:
    with KNOWN.open(newline="", encoding="utf-8") as file:
        header = next(csv.reader(file))
    assert header == ["unit_id", "sign", "label", "sources"]
    rows = _rows()
    assert rows
    for row in rows:  # no cell missing, none extra
        assert None not in row and None not in row.values()


def test_every_known_unit_is_a_neuron_of_its_registered_model() -> None:
    ids = [row["unit_id"] for row in _rows()]
    assert len(ids) == len(set(ids))
    for unit_id in ids:
        unit = NeuronUnit.parse(unit_id)  # also refuses a second spelling
        spec = REGISTRY[unit.model_name]
        unit.validate(ModelInfo(spec.layers, spec.hidden_size, spec.mlp_size))


def test_every_row_has_a_sign_a_label_and_https_sources() -> None:
    for row in _rows():
        assert row["sign"] in ("+", "-")  # ASCII, not the plan's U+2212 minus
        assert row["label"].strip() == row["label"] != ""
        links = row["sources"].split(" ")
        assert links and len(links) == len(set(links))
        assert all(link.startswith("https://") for link in links)


def test_the_evil_qwen_pair_is_listed_with_the_recipes_signs() -> None:
    signs = {row["unit_id"]: row["sign"] for row in _rows()}
    assert signs["qwen3.5-4b:L14:N8949"] == "+"
    assert signs["qwen3.5-4b:L13:N7956"] == "-"
