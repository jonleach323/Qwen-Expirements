"""The model registry: every entry, and the real models' published shapes."""

import doctest
import re
from typing import get_args

import pytest
from transformers import Qwen3ForCausalLM

from pushlab.models import registry
from pushlab.models.discovery import model_info
from pushlab.models.registry import REGISTRY, Kind

# From each model's config.json on Hugging Face (text_config for Qwen3.5),
# written out here so that a typo in the registry fails a test.
PUBLISHED = {
    "qwen3.5-4b": ("Qwen/Qwen3.5-4B", "vision_language", 32, 2560, 9216),
    "qwen3-8b": ("Qwen/Qwen3-8B", "causal_lm", 36, 4096, 12288),
}


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_entry_has_positive_sizes_and_an_id(name: str) -> None:
    spec = REGISTRY[name]
    assert spec.name == name
    assert spec.hf_id
    assert spec.revision
    assert spec.kind in get_args(Kind)
    assert spec.layers > 0
    assert spec.hidden_size > 0
    assert spec.mlp_size > 0


@pytest.mark.parametrize("name", sorted(PUBLISHED))
def test_real_entries_match_their_published_configs(name: str) -> None:
    spec = REGISTRY[name]
    actual = (spec.hf_id, spec.kind, spec.layers, spec.hidden_size, spec.mlp_size)
    assert actual == PUBLISHED[name]


@pytest.mark.parametrize(
    "name", sorted(name for name, spec in REGISTRY.items() if spec.kind != "test")
)
def test_every_downloaded_model_is_pinned_to_a_commit(name: str) -> None:
    # A branch name such as "main" can move to new weights unnoticed.
    assert re.fullmatch(r"[0-9a-f]{40}", REGISTRY[name].revision)


def test_qwen3_8b_is_pinned_to_the_commit_mayners_data_used() -> None:
    # persona_neurons/config.py's MODEL_REVISION, so his neuron sets push the
    # same weights here (decision 0019).
    assert REGISTRY["qwen3-8b"].revision == "b968826d9c46dd6066d109eabc6255188de91218"


def test_tiny_entry_matches_the_tiny_model(tiny_model: Qwen3ForCausalLM) -> None:
    spec = REGISTRY["tiny"]
    info = model_info(tiny_model)
    assert (info.layers, info.hidden_size, info.mlp_size) == (
        spec.layers,
        spec.hidden_size,
        spec.mlp_size,
    )


def test_docstring_examples_run() -> None:
    result = doctest.testmod(registry)
    assert result.attempted > 0
    assert result.failed == 0
