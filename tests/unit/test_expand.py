"""expand: an experiment spec into the exact list of runs it means.

The random controls have tests of their own in test_expand_controls.py.
"""

import dataclasses
import doctest
import hashlib
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pytest

from pushlab.models.registry import REGISTRY
from pushlab.runs import expand as expand_module
from pushlab.runs.expand import PLANNED, expand
from pushlab.runs.spec import spec_from_yaml
from pushlab.store.records import write_runs
from pushlab.units import controls
from pushlab.units.neuron import NeuronUnit

EVIL = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
SpecText = Callable[..., str]


def test_three_pulse_lengths_two_prompts_and_two_seeds(spec_text: SpecText) -> None:
    expansion = expand(spec_from_yaml(spec_text()))
    own = [run for run in expansion.runs if run.control == 0]
    assert len(own) == 12  # 3 lengths x 2 prompts x 2 seeds
    assert len(expansion.runs) == 12 * 3  # and 2 controls of each
    assert expansion.counts() == {f"pulse length={n}": (4, 8) for n in (4, 16, 64)}
    ids = [run.record.run_id for run in expansion.runs]
    assert ids == [f"pulse-lengths-{i:06d}" for i in range(1, 37)]


def test_the_same_spec_always_gives_the_same_runs(spec_text: SpecText) -> None:
    text = spec_text()
    first, second = expand(spec_from_yaml(text)), expand(spec_from_yaml(text))
    assert first == second
    assert first.spec_sha256 == spec_from_yaml(text).sha256
    assert len(first.prompts_sha256) == 64
    known = hashlib.sha256(controls.KNOWN_UNITS_FILE.read_bytes()).hexdigest()
    assert first.known_sha256 == known  # the file the controls avoid


def test_runs_go_by_prompt_then_seed_then_condition_then_control(
    spec_text: SpecText,
) -> None:
    expansion = expand(spec_from_yaml(spec_text()))
    order = [
        (run.record.prompt_id, run.record.seed, run.condition, run.control)
        for run in expansion.runs
    ]
    assert order == [
        (prompt, seed, f"pulse length={n}", control)
        for prompt in ("p-1", "p-2")
        for seed in (0, 1)
        for n in (4, 16, 64)
        for control in (0, 1, 2)
    ]


def test_each_planned_run_is_a_run_record_with_no_reply_yet(
    spec_text: SpecText, tmp_path: Path
) -> None:
    (first, *_) = expand(spec_from_yaml(spec_text())).runs
    record = first.record
    assert record.experiment == "pulse-lengths"
    assert record.model == f"qwen3.5-4b@{REGISTRY['qwen3.5-4b'].revision}"
    assert (record.prompt_id, record.seed, record.control_of) == ("p-1", 0, "")
    assert record.decoding == {
        "strategy": "sample",
        "max_new_tokens": 64,
        "thinking": False,
        "dtype": "bfloat16",
        "temperature": 0.7,
        "top_p": 1.0,
    }
    assert (record.output_text, record.output_tokens, record.status) == (
        "",
        [],
        PLANNED,
    )
    # A planned run cannot be saved as a result; once run, it can be.
    with pytest.raises(ValueError, match="status must be one of"):
        write_runs([record], tmp_path / "planned")
    assert not (tmp_path / "planned").exists()
    done = dataclasses.replace(record, output_tokens=[1, 2], status="ok")
    assert write_runs([done], tmp_path / "runs").name == "runs.parquet"


def test_greedy_runs_record_s1_s_four_decoding_keys(spec_text: SpecText) -> None:
    text = spec_text(decoding="{strategy: greedy, dtype: float32}", seeds="[5]")
    (first, *_) = expand(spec_from_yaml(text)).runs
    assert first.record.decoding == {
        "strategy": "greedy",
        "max_new_tokens": 64,
        "thinking": False,
        "dtype": "float32",
    }
    assert first.record.seed == 5


def test_the_schedule_is_stored_as_its_segments(spec_text: SpecText) -> None:
    runs = expand(spec_from_yaml(spec_text())).runs
    assert runs[0].record.schedule == [
        {
            "unit": str(EVIL),
            "alpha": 1.0,
            "start": 0,
            "end": 4,
            "region": "reply",
            "control_for": None,
            "control_seed": None,
        }
    ]


def test_batches_hold_one_prompt_and_seed_each_and_at_most_batch_size_runs(
    spec_text: SpecText,
) -> None:
    expansion = expand(spec_from_yaml(spec_text(batch_size="5")))
    batches: dict[str, list] = {}
    for run in expansion.runs:
        batch, row = run.record.batch_key.split(":")
        record = run.record
        batches.setdefault(batch, []).append((int(row), record.prompt_id, record.seed))
    for rows in batches.values():
        assert [row for row, *_ in rows] == list(range(len(rows))) and len(rows) <= 5
        assert len({(prompt, seed) for _, prompt, seed in rows}) == 1
    # In run order, so a batch's rows are consecutive runs.
    keys = [run.record.batch_key for run in expansion.runs]
    assert keys == sorted(keys, key=lambda key: (key[:7], int(key[8:])))
    assert sorted(batches) == [f"b{i:06d}" for i in range(1, len(batches) + 1)]
    assert len(batches) == 2 * 2 * 2  # 9 runs per prompt and seed: 5, then 4


def test_an_invalid_condition_stops_expansion_naming_it_and_the_rule(
    spec_text: SpecText, pulses: str
) -> None:
    deep = "  - label: deep\n    segments: [{unit: qwen3.5-4b:L40:N0, alpha: 1.0}]\n"
    with pytest.raises(
        ValueError, match="(?s)condition 'deep': .*layer 40 is out of range"
    ):
        expand(spec_from_yaml(spec_text(pulses + deep)))
    over = """\
  - label: double
    segments:
      - {unit: qwen3.5-4b:L14:N8949, alpha: 1.0}
      - {unit: qwen3.5-4b:L14:N1, alpha: 1.0}
"""
    with pytest.raises(
        ValueError, match="(?s)condition 'double': .*above the ceiling of 1.4"
    ):
        expand(spec_from_yaml(spec_text(pulses + over)))
    # The ceiling is the condition's to lift.
    allowed = over + "    allow_over_ceiling: true\n"
    assert expand(spec_from_yaml(spec_text(pulses + allowed))).counts()["double"]


def test_another_model_s_unit_is_refused_naming_the_condition(
    spec_text: SpecText,
) -> None:
    with pytest.raises(
        ValueError, match="(?s)condition 'pulse length=4': .*not of qwen3-8b"
    ):
        expand(spec_from_yaml(spec_text(model="qwen3-8b")))


def test_warnings_from_validate_are_kept(spec_text: SpecText) -> None:
    text = (
        "conditions:\n  - label: late\n"
        "    segments: [{unit: qwen3.5-4b:L14:N8949, alpha: 1.0, start: 100}]\n"
    )
    expansion = expand(spec_from_yaml(spec_text(text)))
    assert len(expansion.warnings) == 1
    assert expansion.warnings[0].startswith("condition 'late': segments[0]")
    assert "never pushes" in expansion.warnings[0]
    assert expansion.known_sha256 is None  # no controls, so no known units read


@pytest.mark.parametrize(
    "content, problem",
    [
        (None, r"missing\.jsonl: No such file or directory$"),
        ("not json\n", r"missing\.jsonl: line 1 is not JSON"),
    ],
)
def test_a_prompt_set_that_cannot_be_read_is_refused(
    spec_text: SpecText, tmp_path: Path, content: str | None, problem: str
) -> None:
    path = tmp_path / "missing.jsonl"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="cannot read the prompt set .*" + problem):
        expand(spec_from_yaml(spec_text(prompts=str(path))))


def test_more_runs_than_six_digits_can_number_are_refused(
    spec_text: SpecText, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(expand_module, "MOST_RUNS", 36)
    assert len(expand(spec_from_yaml(spec_text())).runs) == 36
    monkeypatch.setattr(expand_module, "MOST_RUNS", 35)
    with pytest.raises(ValueError, match="it means 36 runs, more than the 35"):
        expand(spec_from_yaml(spec_text()))


def test_every_run_id_is_unique_and_every_control_mirrors_a_run(
    spec_text: SpecText,
) -> None:
    expansion = expand(spec_from_yaml(spec_text()))
    ids = Counter(run.record.run_id for run in expansion.runs)
    assert max(ids.values()) == 1
    mirrored = {run.record.control_of for run in expansion.runs if run.control}
    assert mirrored <= set(ids)


def test_doctests() -> None:
    result = doctest.testmod(expand_module)
    assert result.attempted > 0
    assert result.failed == 0
