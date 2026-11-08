"""`pushlab verify` on the steps: one per reply token, with the pushes and
amounts its run's schedule gives the pass that chose it.

Step j is the pass that chose reply token j: the prompt pass for step 0, the
pass that read token j - 1 for step j (decision 0008). With HANDOFF (no
push, then A until reply token 2 and B from it, on each smoke prompt), the
even run IDs push A at steps 0, 1 and 2 and B from step 3.
"""

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from pushlab.cli import main
from pushlab.prompts import PROMPTS_DIR, read_prompt_set
from pushlab.runs.folder import read_index
from pushlab.runs.shards import read_result_steps, read_results
from pushlab.runs.verify_steps import scheduled_pushes
from pushlab.schedules.model import Schedule, Segment
from pushlab.store.records import read_runs_file, runs_table
from pushlab.store.steps import STEPS_SCHEMA, read_steps_file
from pushlab.units.neuron import NeuronUnit

NAME = "tiny-pulses"
A, B = "tiny-vl:L2:N100", "tiny-vl:L1:N3"
HANDOFF = f"""\
conditions:
  - {{label: none, segments: []}}
  - label: handoff
    shape: handoff
    params: {{a: {A}, b: {B}, alpha_a: 1.0, alpha_b: -1.0, switch_at: 2}}
"""
# Every shape, a control, segments of one unit that add up, and ones whose
# doses on reply token 2 (step 3) cancel but for a trace that rounding leaves.
SHAPES = f"""\
conditions:
  - label: pulse
    shape: pulse
    params: {{unit: {A}, alpha: 1.0, start: 2, length: 3}}
    controls: 1
    control_seed: 0
  - label: ramp
    segments:
      - {{unit: {A}, alpha_start: 0.0, alpha_end: 1.0, start: 1, end: 6}}
      - {{unit: {B}, alpha: -0.5, region: prompt}}
  - label: turns
    shape: alternate
    params: {{a: {A}, b: {B}, alpha_a: 1.0, alpha_b: -1.0, period: 2}}
  - label: overlap
    shape: handoff
    params: {{a: {A}, b: {B}, alpha_a: 0.5, alpha_b: -1.0, switch_at: 2, overlap: 2}}
  - label: summed
    segments:
      - {{unit: {A}, alpha: 0.5, start: 0, end: 2}}
      - {{unit: {A}, alpha: 0.5, start: 3}}
      - {{unit: {A}, alpha: 0.1, start: 1, end: 3}}
      - {{unit: {A}, alpha: 0.2, start: 2, end: 3}}
      - {{unit: {A}, alpha: -0.3, start: 2, end: 3}}
"""


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _run(spec: Path, runs: Path, *options: str) -> int:
    return main(["run", str(spec), "--runs-dir", str(runs), *options])


def _id(counter: int) -> str:
    return f"{NAME}-{counter:06d}"


def _steps_file(runs: Path, run_id: str) -> Path:
    """The steps file of the shard whose latest mark counts run_id."""
    shards = [e["shard"] for e in read_index(runs / NAME) if run_id in e["runs"]]
    return runs / NAME / "steps" / shards[-1]


def _rewrite(path: Path, change: Callable[[list[dict]], list[dict]]) -> None:
    """Replace a steps file with change(its rows), under the same name."""
    rows = change(pq.read_table(path).to_pylist())
    pq.write_table(pa.Table.from_pylist(rows, schema=STEPS_SCHEMA), path)


def _edit(path: Path, run_id: str, step: int, **values: object) -> None:
    """Set some columns of one step of one run in a steps file."""

    def change(rows: list[dict]) -> list[dict]:
        hits = [r for r in rows if (r["run_id"], r["step"]) == (run_id, step)]
        assert len(hits) == 1
        hits[0].update(values)
        return rows

    _rewrite(path, change)


@pytest.fixture
def finished(tiny_spec: Callable[..., Path], tmp_path: Path) -> tuple[Path, Path]:
    """The handoff experiment run to the end: its file and its runs dir."""
    spec, runs = tiny_spec(HANDOFF, max_new_tokens="8"), tmp_path / "runs"
    assert _run(spec, runs) == 0
    return spec, runs


def _problems(spec: Path, runs: Path, capsys: pytest.CaptureFixture[str]) -> list[str]:
    """The problems verify prints, which it must find, without the summary."""
    capsys.readouterr()
    assert main(["verify", str(spec), "--runs-dir", str(runs)]) == 1
    lines = capsys.readouterr().out.splitlines()
    count = len(lines) - 1
    assert lines[-1].endswith(f"{count} problem{'' if count == 1 else 's'} found.")
    return lines[:-1]


def test_a_handoff_with_b_before_its_switch_is_caught(
    finished: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    spec, runs = finished
    path = _steps_file(runs, _id(2))
    steps = [s for s in read_steps_file(path) if s.run_id == _id(2)]
    assert [s.active_units for s in steps[:4]] == [(A,), (A,), (A,), (B,)]
    # B on step 2, with the amount it has from step 3: one step too early.
    _edit(
        path, _id(2), 2, active_units=[B], active_amounts=list(steps[3].active_amounts)
    )
    assert _problems(spec, runs, capsys) == [
        f"{_id(2)}: step 2 pushes {B}, but its schedule pushes {A} on the pass "
        "that read reply token 1"
    ]


def test_a_push_missing_from_a_step_is_caught_on_every_step(
    finished: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    spec, runs = finished
    path = _steps_file(runs, _id(4))
    for step in (0, 4, 5):
        _edit(path, _id(4), step, active_units=[], active_amounts=[])
    assert _problems(spec, runs, capsys) == [
        f"{_id(4)}: step 0 pushes nothing, but its schedule pushes {A} on the "
        "prompt pass; 2 more steps disagree"
    ]


def test_amounts_that_do_not_follow_the_doses_are_caught(
    finished: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    spec, runs = finished
    steps = read_result_steps(runs / NAME)
    (a,), (b,) = steps[_id(2)][0].active_amounts, steps[_id(2)][3].active_amounts
    assert a > 0 > b  # +1 on A, -1 on B
    # Run 2: A's amount on step 1 has the wrong sign.
    _edit(_steps_file(runs, _id(2)), _id(2), 1, active_amounts=[-a])
    # Run 4: A's amount on step 2 is doubled, so it is not one per dose.
    _edit(_steps_file(runs, _id(4)), _id(4), 2, active_amounts=[2 * a])
    # Run 6: every amount doubled; each step agrees with the others, but not
    # with the other runs.
    path = _steps_file(runs, _id(6))
    for step in steps[_id(6)]:
        doubled = [2 * amount for amount in step.active_amounts]
        _edit(path, _id(6), step.step, active_amounts=doubled)
    assert _problems(spec, runs, capsys) == [
        f"{_id(2)}: step 1 adds {-a:+.4g} to {A}, whose dose there is +1",
        f"{_id(4)}: {A}'s amount per dose is {a:.4g} at step 0 but {2 * a:.4g} "
        "at step 2, not one number",
        f"{A}'s amount per dose is {a:.4g} in {_id(2)} but {2 * a:.4g} in "
        f"{_id(6)}, not one number",
        f"{B}'s amount per dose is {-b:.4g} in {_id(2)} but {-2 * b:.4g} in "
        f"{_id(6)}, not one number",
    ]


def test_every_shape_verifies_clean_and_amounts_follow_ramps_and_sums(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = tiny_spec(SHAPES, max_new_tokens="8"), tmp_path / "runs"
    assert _run(spec, runs) == 0
    capsys.readouterr()
    assert main(["verify", str(spec), "--runs-dir", str(runs)]) == 0
    assert capsys.readouterr().out.startswith(f"{NAME}, version 1: 30 runs and ")
    records = {record.run_id: record for record in read_results(runs / NAME)}
    summed = [r for r in records.values() if len(r.schedule) == 5][0].run_id
    assert 0.1 + 0.2 - 0.3 != 0  # the trace, which verify takes as 0
    by_step = {s.step: s for s in read_result_steps(runs / NAME)[summed]}
    assert by_step[3].active_units == (A,) and by_step[3].active_amounts == (0.0,)
    # A ramp's dose is 0 on its first token, reply token 1, which is step 2:
    # an amount there is caught.
    ramp = [r for r in records.values() if r.schedule[0].get("alpha_end")][0]
    steps = {s.step: s for s in read_result_steps(runs / NAME)[ramp.run_id]}
    assert steps[2].active_units == (A,) and steps[2].active_amounts == (0.0,)
    _edit(_steps_file(runs, ramp.run_id), ramp.run_id, 2, active_amounts=[0.125])
    assert _problems(spec, runs, capsys) == [
        f"{ramp.run_id}: step 2 adds +0.125 to {A}, whose dose there is +0"
    ]


def test_a_failed_run_with_no_steps_passes(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    failing_generate: Callable[..., None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = tiny_spec(HANDOFF, max_new_tokens="8"), tmp_path / "runs"
    third = read_prompt_set(PROMPTS_DIR / "smoke.jsonl").prompts[2]
    failing_generate(third, RuntimeError("boom"))  # smoke-03's batch: 5 and 6
    assert _run(spec, runs) == 2
    steps = sum(len(found) for found in read_result_steps(runs / NAME).values())
    capsys.readouterr()
    assert main(["verify", str(spec), "--runs-dir", str(runs)]) == 0
    assert capsys.readouterr().out == (
        f"{NAME}, version 1: 10 runs (2 failed) and {steps} steps checked; no "
        "problem found.\n"
    )
    # Steps planted for a failed run are caught.
    rows = pq.read_table(_steps_file(runs, _id(4))).to_pylist()
    planted = [row | {"run_id": _id(6)} for row in rows if row["run_id"] == _id(4)]
    path = _steps_file(runs, _id(6))
    _rewrite(path, lambda found: found + planted)
    assert _problems(spec, runs, capsys) == [
        f"{_id(6)} failed but steps/{path.name} has {len(planted)} of its steps"
    ]


def test_steps_that_are_not_one_per_reply_token_are_caught(
    finished: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    spec, runs = finished
    records = {record.run_id: record for record in read_results(runs / NAME)}
    tokens = records[_id(4)].output_tokens
    last = len(records[_id(8)].output_tokens)

    def without_step_3(rows: list[dict]) -> list[dict]:
        return [r for r in rows if (r["run_id"], r["step"]) != (_id(2), 3)]

    def one_past_the_reply(rows: list[dict]) -> list[dict]:
        return rows + [rows[-1] | {"step": last}]  # run 8's last row, again

    def backwards(rows: list[dict]) -> list[dict]:
        ninth = [r for r in rows if r["run_id"] != _id(10)]
        return ninth + [r for r in rows if r["run_id"] == _id(10)][::-1]

    _rewrite(_steps_file(runs, _id(2)), without_step_3)
    _rewrite(_steps_file(runs, _id(8)), one_past_the_reply)
    _rewrite(_steps_file(runs, _id(10)), backwards)
    _edit(_steps_file(runs, _id(4)), _id(4), 1, token_id=tokens[1] + 1)
    _edit(_steps_file(runs, _id(6)), _id(6), 0, region="thinking")
    assert _problems(spec, runs, capsys) == [
        f"{_id(2)}'s steps are not 0 to {len(records[_id(2)].output_tokens) - 1}: "
        "it lacks step 3",
        f"{_id(4)}: step 1's token is {tokens[1] + 1}, but its reply's token 1 is "
        f"{tokens[1]}",
        f"{_id(6)}: step 0 is in region 'thinking', not reply",
        f"{_id(8)}'s steps are not 0 to {last - 1}: it also has step {last}",
        f"{_id(10)}'s steps are not 0 to {len(records[_id(10)].output_tokens) - 1}: "
        "it has them out of order",
    ]


def test_a_stored_schedule_that_cannot_be_read_is_caught(
    finished: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    spec, runs = finished
    path = runs / NAME / "shards" / _steps_file(runs, _id(2)).name
    records = read_runs_file(path)
    records[1] = replace(records[1], schedule=[{"unit": "nope"}])
    pq.write_table(runs_table(records), path)
    problems = _problems(spec, runs, capsys)
    assert problems[0] == f"{_id(2)} is not the planned run: its schedule differs"
    assert problems[1].startswith(f"{_id(2)}'s schedule cannot be read: ")
    assert len(problems) == 2


def test_a_missing_steps_file_is_one_problem(
    finished: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    spec, runs = finished
    path = _steps_file(runs, _id(3))
    path.unlink()
    assert _problems(spec, runs, capsys) == [
        f"steps/{path.name} is missing, though the index counts its runs"
    ]


def test_scheduled_pushes_are_those_of_the_pass_that_chose_the_token() -> None:
    a, b = (
        NeuronUnit.parse("qwen3.5-4b:L14:N8949"),
        NeuronUnit.parse("qwen3.5-4b:L13:N7956"),
    )
    handoff = Schedule(
        "handoff",
        [
            Segment(a, 1.0, 1.0, region="prompt"),
            Segment(a, 1.0, 1.0, 0, 2),
            Segment(b, -1.0, -1.0, 2),
        ],
    )
    assert [scheduled_pushes(handoff, step) for step in range(5)] == [
        {str(a): 1.0},  # the prompt pass
        {str(a): 1.0},  # reply token 0
        {str(a): 1.0},  # reply token 1
        {str(b): -1.0},  # reply token 2, the switch
        {str(b): -1.0},
    ]
    # In the order the schedule first names the units, whatever segment is on.
    late = Schedule("late", [Segment(a, 1.0, 1.0, 5), Segment(b, 1.0, 1.0, 0)])
    late = Schedule("late", [*late.segments, Segment(a, 0.5, 0.5, 0, 2)])
    assert list(scheduled_pushes(late, 1)) == [str(a), str(b)]
    # Doses of one unit that cancel are 0: pushed, but at dose 0.
    cancel = [Segment(a, 1.0, 1.0, 0, 2), Segment(a, -1.0, -1.0, 1, 3)]
    assert scheduled_pushes(Schedule("cancel", cancel), 2) == {str(a): 0.0}
    assert scheduled_pushes(Schedule("none"), 0) == {}
