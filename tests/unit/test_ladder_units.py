"""The units a dose ladder pushes, and the unit and dose each run pushed
(pushlab.units.ladder_units)."""

import doctest
from pathlib import Path

import pytest

from pushlab.store.records import RunRecord
from pushlab.units import ladder_units
from pushlab.units.ladder_units import RunPush, read_ladder_units, run_push

CANDIDATES_HEADER = "candidate,source,persona,unit,kind,direction,strength\n"


def _file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "units.csv"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_units_file_is_read_in_order(tmp_path: Path) -> None:
    path = _file(
        tmp_path,
        "﻿unit,direction,persona,description,source\n"
        "qwen3-8b:L9:N4242,-1,Brass Owl,A pompous owl.,library\n"
        "qwen3-8b:L20:N7,,,,\n",
    )
    owl, other = read_ladder_units(path)
    assert (owl.key, owl.persona, owl.description, owl.source) == (
        "qwen3-8b:L9:N4242-",
        "Brass Owl",
        "A pompous owl.",
        "library",
    )
    assert (other.key, other.direction, other.persona) == ("qwen3-8b:L20:N7+", 1, "")
    assert owl.alpha(0.75) == -0.75


def test_a_list_of_candidates_is_a_units_file(tmp_path: Path) -> None:
    path = _file(
        tmp_path,
        CANDIDATES_HEADER + "library/L9N4242-,library,Brass Owl,qwen3-8b:L9:N4242,"
        "neuron,-1,1.25\n",
    )
    (owl,) = read_ladder_units(path)
    assert (owl.key, owl.persona, owl.source) == (
        "qwen3-8b:L9:N4242-",
        "Brass Owl",
        "library",
    )


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("persona\nBrass Owl\n", "no unit column"),
        ("unit\n", "lists no unit"),
        ("unit\nqwen3-8b:L9\n", "row 2"),
        # A set's row needs its definition's columns, and a kind must be its unit's.
        ("unit\nqwen3-8b:set:owl-top10\n", "needs its definition's columns"),
        (
            CANDIDATES_HEADER + "m/x,mayner,pirate,qwen3-8b:L1:N1,set,1,1\n",
            "its kind is set, but 'qwen3-8b:L1:N1' is no set's ID",
        ),
        ("unit,direction\nqwen3-8b:L1:N1,2\n", "direction must be"),
        ("unit\nqwen3-8b:L1:N1\nqwen3-8b:L1:N1\n", "row 3 repeats row 2"),
        ("unit\nqwen3-8b:L1:N1\nqwen3.5-4b:L1:N1\n", "mixes units of"),
        # A blank row counts, as in a spreadsheet: the bad unit is on row 4.
        ("unit\nqwen3-8b:L1:N1\n\nqwen3-8b:L9\n", "row 4"),
        ("unit\nqwen3-8b:L1:N1\n,\nqwen3-8b:L1:N1\n", "row 4 repeats row 2"),
    ],
)
def test_a_malformed_units_file_is_refused(
    tmp_path: Path, text: str, problem: str
) -> None:
    with pytest.raises(ValueError, match=problem):
        read_ladder_units(_file(tmp_path, text))


def test_one_unit_both_ways_is_two_ladder_units(tmp_path: Path) -> None:
    path = _file(tmp_path, "unit,direction\nqwen3-8b:L1:N1,1\nqwen3-8b:L1:N1,-1\n")
    assert [unit.key for unit in read_ladder_units(path)] == [
        "qwen3-8b:L1:N1+",
        "qwen3-8b:L1:N1-",
    ]


def _run(*segments: dict) -> RunRecord:
    return RunRecord(
        "x-000001", "x", "m", "p", list(segments), {}, 0, "", "", "", [], "ok"
    )


def test_a_runs_push_is_read_from_its_schedule() -> None:
    prompt = {"unit": "qwen3-8b:L9:N1", "alpha": -0.5, "region": "prompt"}
    reply = prompt | {"region": "reply", "control_for": None}
    push = run_push(_run(prompt, reply))
    assert push == RunPush("qwen3-8b:L9:N1", -0.5, None)
    assert (push.dose, push.key) == (0.5, "qwen3-8b:L9:N1-")
    swapped = {
        "unit": "qwen3-8b:L9:N77",
        "alpha": -0.5,
        "control_for": "qwen3-8b:L9:N1",
    }
    control = run_push(_run(swapped))
    assert control is not None and control.key == "qwen3-8b:L9:N1-"
    assert run_push(_run()) is None
    assert run_push(_run({"unit": "qwen3-8b:L9:N1", "alpha": 0.0})) is None


def test_a_run_of_two_pushes_is_no_ladder_run() -> None:
    one = {"unit": "qwen3-8b:L9:N1", "alpha": 0.5}
    with pytest.raises(ValueError, match="more than one unit or dose"):
        run_push(_run(one, one | {"alpha": 1.0}))


def test_a_ramp_is_no_ladder_run_and_is_refused_as_one() -> None:
    # A ramp's stored segment has alpha_start and alpha_end, no one alpha:
    # refused with the ValueError a run that is no ladder's gets, which the
    # score and doses commands turn into a refusal, never a KeyError.
    ramp = {"unit": "qwen3-8b:L9:N1", "alpha_start": 0.0, "alpha_end": 1.0}
    with pytest.raises(ValueError, match="pushes qwen3-8b:L9:N1 on a ramp, 0.0 to 1.0"):
        run_push(_run(ramp))


def test_docstring_examples_run() -> None:
    result = doctest.testmod(ladder_units)
    assert result.attempted > 0
    assert result.failed == 0
