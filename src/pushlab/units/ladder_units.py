"""The units a dose ladder pushes, and the unit and dose each run pushed.

A units file is CSV with a header and a row per unit (decision 0022). Its
`unit` column is required: a neuron's ID, such as qwen3-8b:L9:N4242. The
others are optional: `direction`, +1 or -1 (the sign of the push that
gives the persona; +1 when absent or empty), `persona` (its name, which
the judge's speaks_as asks about), `description`, and `source` (where it
came from, such as library). Any other column is left alone, so a list of
persona candidates (`pushlab candidates`) is a units file as it is. Sets of
neurons cannot go into experiment files yet (decision 0019), so a set's
row is refused. Every unit must be of one model, and each unit and
direction may come once.

run_push reads back from a run what a ladder pushed: its one unit and
signed dose, and, for a random control, the unit it stands for.
"""

import csv
import math
from dataclasses import dataclass
from pathlib import Path

from pushlab.store.records import RunRecord
from pushlab.units.neuron import NeuronUnit

DIRECTIONS = {"": 1, "1": 1, "+1": 1, "+": 1, "-1": -1, "-": -1}


@dataclass(frozen=True)
class LadderUnit:
    """A unit to climb a ladder with: the neuron, the direction of its push,
    and the persona it gives, if named.

    Example:
        >>> owl = LadderUnit(NeuronUnit.parse("qwen3-8b:L9:N4242"), -1, "Brass Owl")
        >>> owl.key, owl.alpha(0.5)
        ('qwen3-8b:L9:N4242-', -0.5)
    """

    unit: NeuronUnit
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


def read_ladder_units(path: Path) -> list[LadderUnit]:
    """The units of a units file, in its order.

    ValueError names the row of each problem, as a spreadsheet numbers its
    rows: a file with no `unit` column or no row, a unit that is not a
    neuron's ID or is a set's, a direction other than +1 or -1, a unit and
    direction given twice, and units of more than one model.

    Example:
        >>> units = read_ladder_units(Path("units.csv"))  # doctest: +SKIP
        >>> [unit.key for unit in units]  # doctest: +SKIP
        ['qwen3-8b:L9:N4242-']
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
        text = row["unit"].strip()
        if ":set:" in text or row.get("kind", "").strip() == "set":
            raise ValueError(
                f"{where}: {text} is a set of neurons, and sets cannot go into "
                "experiment files yet (decision 0019): ladder single neurons"
            )
        try:
            unit = NeuronUnit.parse(text)
        except ValueError as error:
            raise ValueError(f"{where}: {error}") from None
        ladder = LadderUnit(
            unit,
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
