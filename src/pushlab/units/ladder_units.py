"""The units a dose ladder pushes, and the unit and dose each run pushed.

A units file is CSV with a header and a row per unit (decision 0022). Its
`unit` column is required: a neuron's ID, such as qwen3-8b:L9:N4242, or a
set of neurons' ID, such as qwen3-8b:set:pirate-top50. The others are
optional: `direction`, +1 or -1 (the sign of the push that gives the
persona; +1 when absent or empty), `persona` (its name, which the judge's
speaks_as asks about), `description`, `source` (where it came from, such
as library), and `kind`, neuron or set, which must then be the unit's. A
set's row gives its definition as a list of persona candidates does
(pushlab.discovery.candidates): `entries`, the JSON list [[layer, neuron,
coefficient], ...], `size`, empty for the residual size of the set's one
layer, and `source_ref`, where the set comes from (decision 0019). Any
other column is left alone, so a list of persona candidates (`pushlab
candidates`) is a units file as it is, sets and all. Every unit must be of
one model, each unit and direction may come once, and a set's ID names one
set.

run_push reads back from a run what a ladder pushed: its one unit and
signed dose, and, for a random control, the unit it stands for.
"""

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

from pushlab.schedules.yaml_sets import is_set_id, refuse_two_definitions
from pushlab.store.records import RunRecord
from pushlab.units.neuron import NeuronUnit
from pushlab.units.neuron_set import NeuronSet, Unit

DIRECTIONS = {"": 1, "1": 1, "+1": 1, "+": 1, "-1": -1, "-": -1}
# The columns of a set's definition, as a candidates list writes them.
SET_COLUMNS = ("entries", "size", "source_ref")


@dataclass(frozen=True)
class LadderUnit:
    """A unit to climb a ladder with: the neuron or set of neurons, the
    direction of its push, and the persona it gives, if named.

    Example:
        >>> owl = LadderUnit(NeuronUnit.parse("qwen3-8b:L9:N4242"), -1, "Brass Owl")
        >>> owl.key, owl.alpha(0.5)
        ('qwen3-8b:L9:N4242-', -0.5)
        >>> pair = NeuronSet("qwen3-8b", "owl-pair", [(9, 1, 1.0), (9, 2, -0.5)], "x")
        >>> LadderUnit(pair, 1, "Brass Owl").key
        'qwen3-8b:set:owl-pair+'
    """

    unit: Unit
    direction: int
    persona: str = ""
    description: str = ""
    source: str = ""

    @property
    def key(self) -> str:
        """The unit and its direction: <unit ID>+ or <unit ID>-."""
        return f"{self.unit}{'+' if self.direction > 0 else '-'}"

    def alpha(self, dose: float) -> float:
        """The signed dose that pushes it at dose in its direction."""
        return self.direction * dose


def _direction(text: str, where: str) -> int:
    cleaned = text.strip()
    if cleaned not in DIRECTIONS:
        raise ValueError(f"{where}: direction must be +1 or -1, not {text!r}")
    return DIRECTIONS[cleaned]


def _set(text: str, row: dict[str, str], columns: list[str], where: str) -> NeuronSet:
    """The set a row names, made from its definition as a candidates list's
    row is (pushlab.discovery.candidates.from_row)."""
    if missing := [name for name in SET_COLUMNS if name not in columns]:
        raise ValueError(
            f"{where}: {text} is a set of neurons, so the file needs its "
            f"definition's columns, {', '.join(SET_COLUMNS)}, as pushlab candidates "
            f"writes them; it lacks {', '.join(missing)}"
        )
    try:
        entries = json.loads(row.get("entries", ""))
    except ValueError:  # json.JSONDecodeError
        raise ValueError(
            f"{where}: entries must be JSON, [[layer, neuron, coefficient], ...], "
            f"not {row.get('entries', '')!r}"
        ) from None
    size = row.get("size", "").strip()
    try:
        number = float(size) if size else None
    except ValueError:
        raise ValueError(
            f"{where}: size must be a number, or empty for the residual size of "
            f"the set's one layer, not {size!r}"
        ) from None
    model, _, name = text.partition(":set:")
    try:
        return NeuronSet(model, name, entries, row.get("source_ref", ""), number)
    except ValueError as error:
        raise ValueError(f"{where}: {text}: {error}") from None


def _unit(row: dict[str, str], columns: list[str], where: str) -> Unit:
    """The neuron or set a row names, its kind checked when it gives one."""
    text, kind = row["unit"].strip(), row.get("kind", "").strip()
    if is_set_id(text):
        if kind == "neuron":
            raise ValueError(f"{where}: its kind is neuron, but {text} is a set's ID")
        return _set(text, row, columns, where)
    if kind == "set":
        raise ValueError(
            f"{where}: its kind is set, but {text!r} is no set's ID, such as "
            "qwen3-8b:set:pirate-top50"
        )
    try:
        return NeuronUnit.parse(text)
    except ValueError as error:
        raise ValueError(f"{where}: {error}") from None


def read_ladder_units(path: Path) -> list[LadderUnit]:
    """The units of a units file, in its order.

    ValueError names the row of each problem, as a spreadsheet numbers its
    rows: a file with no `unit` column or no row, a unit that is not a
    neuron's ID or a set's, a set whose definition is missing or refused
    (pushlab.units.neuron_set), a kind that is not the unit's, a direction
    other than +1 or -1, a unit and direction given twice, a set's ID given
    two definitions, and units of more than one model.

    Example:
        >>> units = read_ladder_units(Path("units.csv"))  # doctest: +SKIP
        >>> [unit.key for unit in units]  # doctest: +SKIP
        ['qwen3-8b:L9:N4242-', 'qwen3-8b:set:pirate-top50+']
    """
    path = Path(path)
    try:
        with path.open(newline="", encoding="utf-8-sig") as file:
            table = list(csv.reader(file))
    except (OSError, UnicodeDecodeError, csv.Error) as error:
        raise ValueError(f"cannot read the units file {path}: {error}") from None
    columns = [name.strip() for name in table[0]] if table else []
    if "unit" not in columns:
        raise ValueError(f"{path} has no unit column: give each unit's ID there")
    units: list[LadderUnit] = []
    seen: dict[str, int] = {}
    # Rows are numbered as a spreadsheet shows them: the header is row 1, and
    # a blank row is passed over but counted.
    for number, cells in enumerate(table[1:], start=2):
        if not any(cell.strip() for cell in cells):
            continue
        row = dict(zip(columns, cells, strict=False))
        row.setdefault("unit", "")
        where = f"{path}, row {number}"
        ladder = LadderUnit(
            _unit(row, columns, where),
            _direction(row.get("direction", ""), where),
            row.get("persona", "").strip(),
            row.get("description", "").strip(),
            row.get("source", "").strip(),
        )
        if ladder.key in seen:
            raise ValueError(f"{where} repeats row {seen[ladder.key]}: {ladder.key}")
        seen[ladder.key] = number
        units.append(ladder)
    if not units:
        raise ValueError(f"{path} lists no unit")
    try:  # one set's ID, one set: as an experiment file asks
        refuse_two_definitions((f"row {seen[unit.key]}", unit.unit) for unit in units)
    except ValueError as error:
        raise ValueError(f"{path}, {error}") from None
    if len(models := sorted({unit.unit.model_name for unit in units})) > 1:
        raise ValueError(f"{path} mixes units of {', '.join(models)}: give one model's")
    return units


@dataclass(frozen=True)
class RunPush:
    """What a ladder run pushed: a unit, as its ID, at a signed dose; for a
    random control, its random neuron, and the unit it stands for."""

    unit: str
    alpha: float
    control_for: str | None = None

    @property
    def dose(self) -> float:
        """The dose, without its sign."""
        return abs(self.alpha)

    @property
    def key(self) -> str:
        """The unit a ladder climbs, its own or the one a control stands for,
        and the push's direction, as LadderUnit.key writes them."""
        own = self.control_for or self.unit
        return f"{own}{'+' if self.alpha > 0 else '-'}"


def run_push(run: RunRecord) -> RunPush | None:
    """The one unit and dose a run pushed, None for an unpushed run.

    ValueError when its schedule pushes more than one unit or dose, or a
    ramp from one dose to another, as no run of a ladder does.

    Example:
        >>> segment = {"unit": "qwen3-8b:L9:N1", "alpha": -0.5, "control_for": None}
        >>> run = RunRecord(
        ...     "x-000001",
        ...     "x",
        ...     "m",
        ...     "p",
        ...     [segment, segment],
        ...     {},
        ...     0,
        ...     "",
        ...     "",
        ...     "",
        ...     [],
        ...     "ok",
        ... )
        >>> run_push(run)
        RunPush(unit='qwen3-8b:L9:N1', alpha=-0.5, control_for=None)
    """
    if ramps := [segment for segment in run.schedule if "alpha" not in segment]:
        ramp = ramps[0]
        doses = f"{ramp.get('alpha_start')} to {ramp.get('alpha_end')}"
        raise ValueError(f"{run.run_id} pushes {ramp.get('unit')} on a ramp, {doses}")
    pushes = {
        (str(segment["unit"]), float(segment["alpha"]), segment.get("control_for"))
        for segment in run.schedule
    }
    if not pushes:
        return None
    if len(pushes) > 1:
        listed = ", ".join(f"{unit} at {alpha:g}" for unit, alpha, _ in sorted(pushes))
        raise ValueError(f"{run.run_id} pushes more than one unit or dose: {listed}")
    ((unit, alpha, control_for),) = pushes
    if not math.isfinite(alpha) or alpha == 0:
        return None  # a dose of 0 pushes nothing
    return RunPush(unit, alpha, control_for)
