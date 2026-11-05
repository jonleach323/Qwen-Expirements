"""Random control neurons, and the known units they must avoid.

A random control repeats a push on a neuron drawn at random from the same
layer, at the same dose, so a result can be set against what an ordinary
neuron there does. The draw comes from a seed. It never lands on a unit
listed in units/known.csv (published units with known effects), on a unit
being pushed, or on another control.
"""

import csv
import hashlib
import itertools
from collections.abc import Collection, Sequence
from pathlib import Path

from pushlab import PROJECT_ROOT
from pushlab.models.registry import REGISTRY
from pushlab.store.records import MAX_SEED
from pushlab.units.neuron import NeuronUnit

KNOWN_UNITS_FILE = PROJECT_ROOT / "units" / "known.csv"


def known_units(path: Path | None = None) -> frozenset[NeuronUnit]:
    """The units listed in units/known.csv, or in the file path names.

    Example:
        >>> NeuronUnit.parse("qwen3.5-4b:L14:N8949") in known_units()
        True
    """
    path = KNOWN_UNITS_FILE if path is None else path
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    units = set()
    for line, row in enumerate(rows, start=2):  # line 1 is the header
        try:
            units.add(NeuronUnit.parse(row["unit_id"]))
        except (KeyError, ValueError) as error:
            raise ValueError(f"{path.name}, line {line}: {error}") from error
    return frozenset(units)


def _draw(seed: int, unit: NeuronUnit, attempt: int, size: int) -> int:
    """Draw number attempt for unit: a neuron index from 0 to size - 1."""
    digest = hashlib.sha256(f"pushlab control {seed} {unit} {attempt}".encode())
    return int.from_bytes(digest.digest()[:8], "little") % size


def random_controls(
    units: Sequence[NeuronUnit], seed: int, avoid: Collection[NeuronUnit]
) -> list[NeuronUnit]:
    """One random neuron for each unit, in the unit's model and layer.

    Each draw is a SHA-256 hash of the seed, the unit and the attempt's
    number, so a seed draws the same neurons whatever the library versions.
    The first draw that is not in avoid, not one of units and not an earlier
    unit's control is the unit's control.

    Example:
        >>> evil = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
        >>> random_controls([evil], seed=0, avoid=())
        [NeuronUnit(model_name='qwen3.5-4b', layer=14, index=1929)]
    """
    if not (isinstance(seed, int) and not isinstance(seed, bool)) or not (
        0 <= seed <= MAX_SEED
    ):
        raise ValueError(
            f"the control seed must be a whole number from 0 to {MAX_SEED}, "
            f"not {seed!r}"
        )
    taken = set(avoid) | set(units)
    controls = []
    for unit in units:
        size = REGISTRY[unit.model_name].mlp_size
        layer = {
            t.index
            for t in taken
            if (t.model_name, t.layer) == (unit.model_name, unit.layer)
        }
        if len(layer) >= size:
            raise ValueError(
                f"no neuron left in layer {unit.layer} of {unit.model_name} for a "
                f"control of {unit}: all {size} are known, pushed or taken"
            )
        for attempt in itertools.count():
            control = NeuronUnit(
                unit.model_name, unit.layer, _draw(seed, unit, attempt, size)
            )
            if control not in taken:
                break
        taken.add(control)
        controls.append(control)
    return controls
