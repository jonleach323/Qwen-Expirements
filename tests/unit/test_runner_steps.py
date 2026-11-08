"""The steps `pushlab run` saves: one per reply token, with the pushes on the
pass that chose it (pushlab.store.steps), beside each batch's runs.

Step j is the pass that chose reply token j: the prompt pass for step 0 and
the pass that read token j - 1 for step j, so a push on reply token k shows
at step k + 1 (decision 0008).
"""

import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch

from pushlab.cli import main
from pushlab.models import load
from pushlab.prompts import PROMPTS_DIR, read_prompt_set
from pushlab.runs import folder as folder_module
from pushlab.runs import shards as shards_module
from pushlab.runs.batch import _schedule
from pushlab.runs.folder import read_index
from pushlab.runs.shards import (
    next_shard_name,
    read_result_steps,
    read_results,
    remove_leftovers,
)
from pushlab.schedules.compile import compile as compile_schedules
from pushlab.store.steps import read_steps_file
from pushlab.units import calibrate

NAME = "tiny-pulses"
A, B = "tiny-vl:L2:N100", "tiny-vl:L1:N3"
HANDOFF = f"""\
conditions:
  - label: handoff
    shape: handoff
    params: {{a: {A}, b: {B}, alpha_a: 1.0, alpha_b: -1.0, switch_at: 2}}
"""
# A ramp from 0 to 1 on reply tokens 1 to 5, and B on the prompt pass.
RAMP = f"""\
conditions:
  - label: ramp
    segments:
      - {{unit: {A}, alpha_start: 0.0, alpha_end: 1.0, start: 1, end: 6}}
      - {{unit: {B}, alpha: -0.5, region: prompt}}
"""
# A on reply tokens 2, 3 and 4: steps 3, 4 and 5.
PULSE = f"""\
conditions:
  - label: pulse
    shape: pulse
    params: {{unit: {A}, alpha: 1.0, start: 2, length: 3}}
"""


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _run(spec: Path, runs: Path, *options: str) -> int:
    return main(["run", str(spec), "--runs-dir", str(runs), *options])


def _units(steps: list) -> list[tuple[str, ...]]:
    return [step.active_units for step in steps]


def test_a_handoffs_steps_show_a_before_the_switch_and_b_after_it(
    tiny_spec: Callable[..., Path], tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    assert _run(tiny_spec(HANDOFF, max_new_tokens="8"), runs) == 0
    records, steps = read_results(runs / NAME), read_result_steps(runs / NAME)
    assert len(records) == 5 and sorted(steps) == [r.run_id for r in records]
    assert max(len(r.output_tokens) for r in records) == 8
    for record in records:
        n = len(record.output_tokens)  # some replies end their turn early
        assert n > 3  # so B shows in each
        # The prompt pass and reply tokens 0 and 1 push A: steps 0, 1 and 2.
        # Reply tokens 2 on push B, which first changes token 3: steps 3 on.
        assert _units(steps[record.run_id]) == ([(A,)] * 3 + [(B,)] * 5)[:n]
        # Exactly what the run's own schedule pushes on each step's pass.
        schedule = _schedule(record)
        for step in steps[record.run_id]:
            token = None if step.step == 0 else step.step - 1  # the pass's token
            pushed = [s.unit for s in schedule.segments if s.active_at(token)]
            assert step.active_units == tuple(str(unit) for unit in pushed)
        amounts = {s.active_amounts for s in steps[record.run_id]}
        (a,), (b,) = sorted(amounts, reverse=True)  # one amount for each unit
        assert a > 0 > b  # +1 on A, -1 on B


def test_a_pulse_shows_on_the_steps_after_the_tokens_it_pushes(
    tiny_spec: Callable[..., Path], tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    assert _run(tiny_spec(PULSE, max_new_tokens="8"), runs) == 0
    for run_steps in read_result_steps(runs / NAME).values():
        assert _units(run_steps) == [()] * 3 + [(A,)] * 3 + [()] * 2


def test_every_reply_has_one_step_per_token_with_its_log_probability(
    tiny_spec: Callable[..., Path], tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    assert _run(tiny_spec(), runs) == 0  # no push, and a constant pulse
    steps = read_result_steps(runs / NAME)
    for record in read_results(runs / NAME):
        run_steps = steps[record.run_id]
        assert [s.step for s in run_steps] == list(range(len(record.output_tokens)))
        assert [s.token_id for s in run_steps] == record.output_tokens
        assert {s.region for s in run_steps} == {"reply"}
        assert all(math.isfinite(s.logprob) and s.logprob <= 0 for s in run_steps)
        # The pulse pushes reply tokens from 0, not the prompt pass: steps 1 on.
        pushed = [bool(s.active_units) for s in run_steps]
        n = len(run_steps)
        assert pushed == (
            [False] + [True] * (n - 1) if record.schedule else [False] * n
        )
    # One steps file per saved batch, of the shard's name.
    shards = sorted(entry["shard"] for entry in read_index(runs / NAME))
    assert sorted(p.name for p in (runs / NAME / "steps").iterdir()) == shards


def test_a_failed_batch_saves_an_empty_steps_file(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    failing_generate: Callable[..., None],
) -> None:
    runs = tmp_path / "runs"
    third = read_prompt_set(PROMPTS_DIR / "smoke.jsonl").prompts[2]
    failing_generate(third, RuntimeError("boom"))  # smoke-03's batch
    assert _run(tiny_spec(), runs) == 2
    failed = [e["shard"] for e in read_index(runs / NAME) if e["error"]]
    assert read_steps_file(runs / NAME / "steps" / failed[0]) == []
    steps = read_result_steps(runs / NAME)
    for record in read_results(runs / NAME):
        assert (steps[record.run_id] == []) == (record.status == "failed")


def test_kept_runs_keep_their_steps_from_the_old_batches(
    tiny_spec: Callable[..., Path], tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    assert _run(tiny_spec(), runs) == 0
    before = read_result_steps(runs / NAME)
    assert _run(tiny_spec(max_gpu_hours="2"), runs, "--overwrite-version") == 0
    assert read_result_steps(runs / NAME) == before  # every run kept


def test_a_crash_before_the_mark_leaves_the_steps_of_that_attempt_unread(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runs = tmp_path / "runs"
    real = folder_module.append_index

    def killed(path: Path, entry: dict) -> None:
        if entry["batch"] == "b000002":
            raise KeyboardInterrupt  # both files are in place, the mark is not
        real(path, entry)

    monkeypatch.setattr(folder_module, "append_index", killed)
    assert _run(tiny_spec(), runs) == 130
    stray = runs / NAME / "steps" / "v1-b000002-1.parquet"
    assert stray.exists()
    monkeypatch.setattr(folder_module, "append_index", real)
    assert _run(tiny_spec(), runs) == 0
    entries = read_index(runs / NAME)
    assert [e["shard"] for e in entries][1] == "v1-b000002-2.parquet"
    steps = read_result_steps(runs / NAME)
    records = read_results(runs / NAME)
    assert len(steps) == 10
    for record in records:  # each run's steps once, from its counted batch
        assert len(steps[record.run_id]) == len(record.output_tokens)


def test_a_steps_file_that_cannot_be_saved_stops_the_run_like_a_shard(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs = tmp_path / "runs"
    real = shards_module.write_steps

    def disk_full(path: Path, name: str, steps: list) -> Path:
        if name.startswith("v1-b000002"):
            raise OSError("No space left on device")
        return real(path, name, steps)

    monkeypatch.setattr(shards_module, "write_steps", disk_full)
    assert _run(tiny_spec(), runs) == 1
    assert "cannot save b000002: No space left on device" in capsys.readouterr().err
    assert [e["batch"] for e in read_index(runs / NAME)] == ["b000001"]
    monkeypatch.setattr(shards_module, "write_steps", real)
    assert _run(tiny_spec(), runs) == 0
    assert len(read_result_steps(runs / NAME)) == 10


def test_a_name_taken_by_a_steps_file_alone_is_not_used_again(tmp_path: Path) -> None:
    (tmp_path / "steps").mkdir()
    (tmp_path / "steps" / "v1-b000001-1.parquet").write_bytes(b"")
    assert next_shard_name(tmp_path, 1, "b000001") == "v1-b000001-2.parquet"


def test_leftovers_of_a_killed_steps_write_are_removed(tmp_path: Path) -> None:
    (tmp_path / "steps").mkdir()
    leftover = tmp_path / "steps" / ".v1-b000001-1.parquet.abc.tmp"
    leftover.write_bytes(b"half")
    assert remove_leftovers(tmp_path) == 1
    assert not leftover.exists()


@pytest.mark.parametrize(
    "conditions", [HANDOFF, RAMP, PULSE], ids=["handoff", "ramp", "pulse"]
)
def test_each_amount_is_the_dose_tables_on_the_pass_behind_its_token(
    tiny_spec: Callable[..., Path], tmp_path: Path, conditions: str
) -> None:
    runs = tmp_path / "runs"
    assert _run(tiny_spec(conditions, max_new_tokens="8"), runs) == 0
    revision = load.downloaded_revision("tiny-vl")
    model = load.load_model("tiny-vl", torch.float32, device="cpu").model
    norms = calibrate.load_calibration("tiny-vl", revision).norms
    steps = read_result_steps(runs / NAME)
    for record in read_results(runs / NAME):
        table = compile_schedules([_schedule(record)], model, norms, 8)
        for step in steps[record.run_id]:
            token = None if step.step == 0 else step.step - 1  # the pass behind it
            expected = table.amounts(0, token)
            assert step.active_units == tuple(str(u.unit) for u in expected)
            # Saved as 16-bit floats: the table's amount, rounded once.
            assert step.active_amounts == tuple(
                float(np.float16(u.amount)) for u in expected
            )
