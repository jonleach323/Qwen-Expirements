"""Units of a dose ladder whose persona is named "none" (decision 0022).

"none" is the judge's answer for no persona (names_no_persona), so no
persona may have it, though a source such as the Neuron Persona Library
may name one so. `pushlab score SPEC --rubric speaks_as --units FILE` asks
nothing about such a unit (pushlab.scoring.score_units): its replies never
get a speaks_as verdict. pushlab doses therefore leaves it out of its
report, counting it and its runs in the words score uses (no_persona_text),
and pushlab ladder refuses a units file with one, naming its row
(refuse_no_persona), as its runs would only waste GPU time.
"""

import csv
from collections.abc import Sequence
from pathlib import Path

from pushlab.scoring.personas import NONE, names_no_persona
from pushlab.units.ladder_units import LadderUnit


def no_persona_text(units: int, runs: int) -> str:
    """How many units of the units file have a persona named "none", and
    their runs, in the words pushlab score uses for them.

    Example:
        >>> print(no_persona_text(1, 30))  # doctest: +NORMALIZE_WHITESPACE
        1 unit of the file with the persona "none", the judge's answer for no
        persona, and their 30 runs
    """
    unit, run = "unit" if units == 1 else "units", "run" if runs == 1 else "runs"
    return (
        f'{units:,} {unit} of the file with the persona "{NONE}", the judge\'s '
        f"answer for no persona, and their {runs:,} {run}"
    )


def unit_rows(units_file: Path, count: int) -> list[int] | None:
    """The row of each unit of a units file of count units, as
    read_ladder_units numbers rows, the way a spreadsheet shows them: the
    header is row 1, and a blank row is passed over but counted. None when
    the file does not read again as count units, as a pipe, read once,
    does not.

    Example:
        >>> unit_rows(Path("units.csv"), 2)  # doctest: +SKIP
        [2, 4]
    """
    try:
        with Path(units_file).open(newline="", encoding="utf-8-sig") as file:
            table = list(csv.reader(file))
    except (OSError, UnicodeDecodeError, csv.Error):
        return None
    rows = [n for n, cells in enumerate(table[1:], 2) if any(c.strip() for c in cells)]
    return rows if len(rows) == count else None


def refuse_no_persona(units_file: Path, units: Sequence[LadderUnit]) -> None:
    """Refuse the units of units_file, as read_ladder_units read them, when
    any has a persona named "none". ValueError names each such row and its
    unit (only the unit when unit_rows cannot number the rows), says that
    their replies could never be judged, so their runs would waste GPU
    time, and to leave those rows out.

    Example:
        >>> refuse_no_persona(Path("units.csv"), units)  # doctest: +SKIP
        Traceback (most recent call last):
        ...
        ValueError: units.csv, row 3 (qwen3-8b:L9:N1): its persona is "none", ...
    """
    places = [i for i, unit in enumerate(units) if names_no_persona(unit.persona)]
    if not places:
        return
    rows = unit_rows(units_file, len(units))
    named = [
        f"row {rows[i]} ({units[i].unit})" if rows else str(units[i].unit)
        for i in places
    ]
    listed = named[0] if len(named) == 1 else f"{', '.join(named[:-1])} and {named[-1]}"
    its, rows_out = ("its", "that row") if len(named) == 1 else ("their", "those rows")
    raise ValueError(
        f'{units_file}, {listed}: {its} persona is "{NONE}", the judge\'s answer '
        f"for no persona, so {its} replies could never be judged and {its} runs "
        f"would waste GPU time: leave {rows_out} out"
    )
