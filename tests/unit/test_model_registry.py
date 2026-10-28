"""The model registry and the tiny test model it describes."""

import doctest
from collections.abc import Callable

import pytest
import torch
from transformers import Qwen3ForCausalLM

from pushlab.models import registry
from pushlab.models.discovery import model_info
from pushlab.models.registry import REGISTRY


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_entry_has_positive_sizes_and_an_id(name: str) -> None:
    spec = REGISTRY[name]
    assert spec.name == name
    assert spec.hf_id
    assert spec.revision
    assert spec.layers > 0
    assert spec.hidden_size > 0
    assert spec.mlp_size > 0


def test_tiny_entry_matches_the_tiny_model(tiny_model: Qwen3ForCausalLM) -> None:
    spec = REGISTRY["tiny"]
    info = model_info(tiny_model)
    assert (info.layers, info.hidden_size, info.mlp_size) == (
        spec.layers,
        spec.hidden_size,
        spec.mlp_size,
    )


def test_tiny_model_is_float32_and_in_eval_mode(tiny_model: Qwen3ForCausalLM) -> None:
    assert not tiny_model.training
    assert {p.dtype for p in tiny_model.parameters()} == {torch.float32}


def test_tiny_model_weights_are_the_same_every_time(
    make_tiny_model: Callable[[], Qwen3ForCausalLM],
) -> None:
    first = make_tiny_model()
    torch.rand(1000)  # move the global random state on
    second = make_tiny_model()
    for (name, a), (_, b) in zip(
        first.state_dict().items(), second.state_dict().items(), strict=True
    ):
        assert torch.equal(a, b), name


def test_tiny_model_leaves_the_test_seed_untouched(
    tiny_model: Qwen3ForCausalLM, fixed_seeds: int
) -> None:
    # The fixture builds its weights without consuming the test's random numbers.
    expected = torch.rand(3, generator=torch.Generator().manual_seed(fixed_seeds))
    assert torch.equal(torch.rand(3), expected)


def test_tests_cannot_download_from_hugging_face() -> None:
    import huggingface_hub.constants

    assert huggingface_hub.constants.HF_HUB_OFFLINE


def test_docstring_examples_run() -> None:
    result = doctest.testmod(registry)
    assert result.attempted > 0
    assert result.failed == 0
