"""`pushlab ladder` refuses a units file with a unit whose persona is named
"none", the judge's answer for no persona, before it writes anything: the
message names the unit's row and the unit, and says to leave that row out,
as pushlab score can never judge its replies (pushlab.analysis.no_persona:
refuse_no_persona, unit_rows). The units are tiny-vl's (the tiny_downloads
fixture registers the model) and qwen3-8b's, with made-up personas.
"""

import doctest
from pathlib import Path

import pytest

from pushlab.analysis import no_persona
from pushlab.analysis.no_persona import refuse_no_persona, unit_rows
from pushlab.cli import main
from pushlab.runs.spec import read_spec
from pushlab.units.ladder_units import LadderUnit, read_ladder_units
from pushlab.units.neuron import NeuronUnit

OWL, CAPTAIN = "tiny-vl:L1:N3", "tiny-vl:L2:N100"
HEADER = "unit,direction,persona,description,source\n"
OWL_ROW = f"{OWL},+1,Brass Owl,A pompous mechanical owl.,invented\n"
SMOKE = "prompts/smoke.jsonl"


def _why(its: str, rows: str) -> str:
    """The end of the refusal, after the units it names."""
    return (
        f'{its} persona is "none", the judge\'s answer for no persona, so {its} '
        f"replies could never be judged and {its} runs would waste GPU time: leave "
        f"{rows} out"
    )


def test_a_unit_named_none_is_refused_by_its_row_before_anything_is_written(
    tiny_downloads: object, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    units, out = tmp_path / "units.csv", tmp_path / "experiments" / "tiny-ladder.yaml"
    # A blank row counts, as a spreadsheet shows it: the captain is row 4.
    units.write_text(
        f"{HEADER}{OWL_ROW}\n{CAPTAIN},-1, None ,,library\n", encoding="utf-8"
    )
    argv = ["ladder", "--units", str(units), "--doses", "0.5,1.0", "--prompts", SMOKE]
    argv += ["--out", str(out), "--max-gpu-hours", "1"]
    assert main(argv) == 1
    assert capsys.readouterr().err == (
        f"pushlab ladder: {units}, row 4 ({CAPTAIN}): {_why('its', 'that row')}\n"
    )
    assert not out.parent.exists()  # nothing written, not even its folder
    units.write_text(HEADER + OWL_ROW, encoding="utf-8")  # the row left out
    assert main(argv) == 0
    assert [condition.label for condition in read_spec(out).conditions] == ["u01"]


def test_rows_are_numbered_as_the_units_files_reader_numbers_them(
    tiny_downloads: object, tmp_path: Path
) -> None:
    units = tmp_path / "units.csv"
    # The owl's description spans two lines in quotes, one row as a
    # spreadsheet shows it; then a blank row, and the captain in row 4.
    owl = f'{OWL},+1,NONE,"A pompous\nmechanical owl.",invented\n'
    text = f"{HEADER}{owl}\n{CAPTAIN},-1,none,,\n"
    units.write_text(text, encoding="utf-8")
    found = read_ladder_units(units)
    assert unit_rows(units, len(found)) == [2, 4]
    with pytest.raises(ValueError) as refused:
        refuse_no_persona(units, found)
    named = f"row 2 ({OWL}) and row 4 ({CAPTAIN})"
    assert str(refused.value) == f"{units}, {named}: {_why('their', 'those rows')}"
    # The reader's own messages number rows the same way.
    units.write_text(f"{text}{OWL},+1,Brass Owl,,\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row 5 repeats row 2: "):
        read_ladder_units(units)


def test_three_rows_are_listed_and_a_name_that_only_starts_with_none_passes(
    tmp_path: Path,
) -> None:
    units = tmp_path / "units.csv"
    names = ["none", "Nonesuch", "None", " NONE "]
    rows = [f"qwen3-8b:L9:N{n},+1,{name}\n" for n, name in enumerate(names, 1)]
    units.write_text("unit,direction,persona\n" + "".join(rows), encoding="utf-8")
    found = read_ladder_units(units)
    with pytest.raises(ValueError) as refused:
        refuse_no_persona(units, found)
    named = "row 2 (qwen3-8b:L9:N1), row 4 (qwen3-8b:L9:N3) and row 5 (qwen3-8b:L9:N4)"
    assert str(refused.value) == f"{units}, {named}: {_why('their', 'those rows')}"
    units.write_text("unit,persona\nqwen3-8b:L9:N2,Nonesuch\n", encoding="utf-8")
    assert refuse_no_persona(units, read_ladder_units(units)) is None


def test_rows_that_cannot_be_numbered_again_leave_only_the_unit_named(
    tmp_path: Path,
) -> None:
    unit = LadderUnit(NeuronUnit.parse("qwen3-8b:L9:N1"), -1, "None")
    gone = tmp_path / "gone.csv"  # read once, then removed
    assert unit_rows(gone, 1) is None
    with pytest.raises(ValueError) as refused:
        refuse_no_persona(gone, [unit])
    assert str(refused.value) == f"{gone}, qwen3-8b:L9:N1: {_why('its', 'that row')}"
    two = tmp_path / "two.csv"
    two.write_text("unit\nqwen3-8b:L9:N1\n\nqwen3-8b:L9:N2\n", encoding="utf-8")
    assert unit_rows(two, 2) == [2, 4]
    assert unit_rows(two, 3) is None  # not the units it was read as


def test_docstring_examples_run() -> None:
    result = doctest.testmod(no_persona)
    assert result.attempted > 0
    assert result.failed == 0
