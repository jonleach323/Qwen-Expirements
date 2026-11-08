"""Fixtures for the experiment-file and runner tests.

A two-prompt set and specs on it, for expand; an experiment file for
tiny-vl and a stand-in costs file, for pushlab run and status.
"""

from collections.abc import Callable
from pathlib import Path

import pytest

# Three pulse lengths of the recipe's hostile neuron, each with two controls.
PULSES = """\
conditions:
  - label: pulse
    shape: pulse
    params: {unit: qwen3.5-4b:L14:N8949, alpha: 1.0, start: 0}
    grid: {length: [4, 16, 64]}
    controls: 2
    control_seed: 7
"""


@pytest.fixture
def two_prompts(tmp_path: Path) -> Path:
    """A prompt set of two prompts, p-1 and p-2."""
    path = tmp_path / "two.jsonl"
    path.write_text(
        '{"prompt_id": "p-1", "version": 1, "prompt": "Say hi."}\n'
        '{"prompt_id": "p-2", "version": 1, "prompt": "Count to three."}\n',
        encoding="utf-8",
    )
    return path


@pytest.fixture
def pulses() -> str:
    """The conditions: three pulse lengths, each with two controls (seed 7)."""
    return PULSES


@pytest.fixture
def spec_text(two_prompts: Path) -> Callable[..., str]:
    """Experiment YAML on the two prompts: sampled at 0.7 with seeds 0 and 1,
    64 tokens, batches of 8, the conditions given (three pulse lengths by
    default), and any setting replaced by keyword, as YAML text."""

    def text(conditions: str = PULSES, **settings: str) -> str:
        lines = {
            "name": "pulse-lengths",
            "model": "qwen3.5-4b",
            "prompts": str(two_prompts),
            "decoding": "{strategy: sample, temperature: 0.7}",
            "seeds": "[0, 1]",
            "max_new_tokens": "64",
            "batch_size": "8",
            "max_gpu_hours": "1",
        } | settings
        return "".join(f"{key}: {value}\n" for key, value in lines.items()) + conditions

    return text


# No push and a pulse on each of the five smoke prompts: 10 runs, in five
# batches of 2 (a prompt's two runs together).
TINY_CONDITIONS = """\
conditions:
  - label: none
    segments: []
  - label: pulse
    segments: [{unit: tiny-vl:L2:N100, alpha: 1.0}]
"""


@pytest.fixture
def tiny_spec(tiny_vl_calibration: Path, tmp_path: Path) -> Callable[..., Path]:
    """Write an experiment file for tiny-vl, downloaded and calibrated, and
    return its path: TINY_CONDITIONS on prompts/smoke.jsonl, greedy in
    float32, 4 reply tokens, batches of 2 and max_gpu_hours 1. Keyword
    arguments replace settings, as YAML text; file names the file."""

    def write(
        conditions: str = TINY_CONDITIONS, file: str = "tiny.yaml", **settings: str
    ) -> Path:
        lines = {
            "name": "tiny-pulses",
            "model": "tiny-vl",
            "prompts": "prompts/smoke.jsonl",
            "decoding": "{strategy: greedy, dtype: float32}",
            "seeds": "[0]",
            "max_new_tokens": "4",
            "batch_size": "2",
            "max_gpu_hours": "1",
        } | settings
        path = tmp_path / file
        text = "".join(f"{key}: {value}\n" for key, value in lines.items())
        path.write_text(text + conditions, encoding="utf-8")
        return path

    return write


@pytest.fixture
def costs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A costs file that knows tiny-vl's speed, standing in for configs/costs.yaml."""
    from pushlab.runs import estimate

    path = tmp_path / "costs.yaml"
    path.write_text(
        "gpu: RTX 4090\n"
        "dollars_per_hour: [0.34, 0.74]\n"
        "reply_tokens_per_second:\n"
        "  tiny-vl: 0.01\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(estimate, "COSTS_FILE", path)
    return path
