"""`pushlab run`'s batches: runs of different prompts share one when they can.

Three prompts, two of one length in tokens: the tiny tokenizer gives each
byte a token, so "Say hi." and "Say ho." are as long as each other.
"""

from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from pushlab.cli import main
from pushlab.engine import generate as generate_module
from pushlab.runs.expand import expand
from pushlab.runs.folder import read_index, read_results
from pushlab.runs.plan import calls
from pushlab.runs.spec import read_spec

NAME = "tiny-pulses"
NO_PUSH = "conditions:\n  - {label: none, segments: []}\n"
PULSED = """\
conditions:
  - {label: none, segments: []}
  - label: pulse
    segments: [{unit: tiny-vl:L2:N100, alpha: 1.0}]
    controls: 2
    control_seed: 0
"""


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def three(tmp_path: Path) -> Path:
    path = tmp_path / "three.jsonl"
    path.write_text(
        '{"prompt_id": "p-1", "version": 1, "prompt": "Say hi."}\n'
        '{"prompt_id": "p-2", "version": 1, "prompt": "Say ho."}\n'
        '{"prompt_id": "p-3", "version": 1, "prompt": "Count to three."}\n'
    )
    return path


def _prompts(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """The prompts each generate() call gets."""
    real = generate_module.generate
    seen: list[list[str]] = []

    def recording(model, tokenizer, prompts, *args, **kwargs):  # type: ignore[no-untyped-def]
        seen.append(list(prompts))
        return real(model, tokenizer, prompts, *args, **kwargs)

    monkeypatch.setattr(generate_module, "generate", recording)
    return seen


def _run(spec: Path, runs: Path, *options: str) -> int:
    return main(["run", str(spec), "--runs-dir", str(runs), *options])


def test_prompts_of_one_length_share_a_batch(
    tiny_spec: Callable[..., Path],
    three: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = tiny_spec(NO_PUSH, prompts=str(three), batch_size="8")
    seen = _prompts(monkeypatch)
    assert _run(spec, tmp_path / "runs") == 0
    assert seen == [["Say hi.", "Say ho."], ["Count to three."]]
    keys = {r.prompt_id: r.batch_key for r in read_results(tmp_path / "runs" / NAME)}
    assert keys == {
        "p-1": "v1-b000001-1:0",
        "p-2": "v1-b000001-1:1",
        "p-3": "v1-b000002-1:0",
    }


def test_without_grouping_by_length_prompts_share_a_padded_batch(
    tiny_spec: Callable[..., Path],
    three: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = tiny_spec(
        NO_PUSH, prompts=str(three), batch_size="8", group_by_length="false"
    )
    seen = _prompts(monkeypatch)
    assert _run(spec, tmp_path / "runs") == 0
    assert seen == [["Say hi.", "Say ho.", "Count to three."]]


def test_each_seed_keeps_to_its_own_batches(
    tiny_spec: Callable[..., Path],
    three: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampled = "{strategy: sample, temperature: 0.7, dtype: float32}"
    spec = tiny_spec(
        NO_PUSH,
        prompts=str(three),
        batch_size="8",
        group_by_length="false",
        decoding=sampled,
        seeds="[0, 1]",
    )
    seen = _prompts(monkeypatch)
    assert _run(spec, tmp_path / "runs") == 0
    assert len(seen) == 2  # seed 0's three runs, then seed 1's
    seeds = {}
    for entry in read_index(tmp_path / "runs" / NAME):
        records = [
            r
            for r in read_results(tmp_path / "runs" / NAME)
            if r.run_id in entry["runs"]
        ]
        seeds[entry["batch"]] = {record.seed for record in records}
    assert seeds == {"b000001": {0}, "b000002": {1}}


def test_calls_keep_planned_batches_whole_and_are_always_the_same(
    tiny_spec: Callable[..., Path], three: Path
) -> None:
    spec = read_spec(tiny_spec(PULSED, prompts=str(three), batch_size="5"))
    expansion = expand(spec)
    # A planned batch: no push, then the pulse and its two controls (4 runs).
    planned = {r.record.batch_key.split(":")[0] for r in expansion.runs}
    lengths = {"p-1": 1, "p-2": 1, "p-3": 2}
    made = calls(expansion, 5, lengths)
    assert made == calls(expansion, 5, lengths)
    assert all(len(call.runs) <= 5 for call in made)
    for key in planned:
        holders = [
            call.key
            for call in made
            if any(r.record.batch_key.startswith(f"{key}:") for r in call.runs)
        ]
        assert len(holders) == 1  # whole, in one call
    # Batches of 4 cannot share a call of 5, so nothing merges here.
    assert [len(call.runs) for call in made] == [4, 4, 4]
    merged = calls(expansion, 8, lengths)
    assert [len(call.runs) for call in merged] == [8, 4]


def test_a_call_partly_saved_runs_whole_and_saves_only_what_is_due(
    tiny_spec: Callable[..., Path],
    three: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    conditions = PULSED.replace("    controls: 2\n    control_seed: 0\n", "")
    spec = tiny_spec(
        conditions, prompts=str(three), batch_size="8", group_by_length="false"
    )
    runs = tmp_path / "runs"
    assert _run(spec, runs) == 0  # one call of all six runs
    before = {r.run_id: r for r in read_results(runs / NAME)}
    # A new dose: the pushed runs change and the unpushed ones do not.
    changed = tiny_spec(
        conditions.replace("alpha: 1.0", "alpha: 0.5"),
        prompts=str(three),
        batch_size="8",
        group_by_length="false",
    )
    seen = _prompts(monkeypatch)
    assert _run(changed, runs, "--overwrite-version") == 0
    assert [len(prompts) for prompts in seen] == [6]  # the whole call again
    after = {r.run_id: r for r in read_results(runs / NAME)}
    for run_id, record in after.items():
        if record.schedule:  # redone, in its row of the call
            assert record.batch_key == f"v2-b000001-1:{before[run_id].batch_key[-1]}"
        else:  # kept, as it was
            assert record == before[run_id]
