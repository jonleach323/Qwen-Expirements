"""Fixtures for the experiment-file tests: a two-prompt set and specs on it."""

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
