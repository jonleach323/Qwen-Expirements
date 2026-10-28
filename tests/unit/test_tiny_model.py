"""The tiny_model fixture: shape, precision, determinism and no downloads."""

from collections.abc import Callable

import torch
from transformers import Qwen3ForCausalLM


def test_tiny_model_has_the_specified_shape(tiny_model: Qwen3ForCausalLM) -> None:
    assert type(tiny_model) is Qwen3ForCausalLM
    config = tiny_model.config
    assert config.num_hidden_layers == 2
    assert config.hidden_size == 64
    assert config.intermediate_size == 128
    assert config.num_attention_heads == 4
    assert config.num_key_value_heads == 2
    assert config.vocab_size == 512
    # Attention width (4 x 32) differs from the hidden size, as in Qwen3.5-4B.
    assert config.head_dim == 32


def test_tiny_model_is_float32_and_in_eval_mode(tiny_model: Qwen3ForCausalLM) -> None:
    tensors = [*tiny_model.parameters(), *tiny_model.buffers()]
    assert {t.dtype for t in tensors if t.is_floating_point()} == {torch.float32}
    assert tiny_model.config.dtype == torch.float32
    assert not any(module.training for module in tiny_model.modules())


def test_tiny_model_weights_are_the_same_every_time(
    make_tiny_model: Callable[[], Qwen3ForCausalLM],
) -> None:
    first = make_tiny_model()
    torch.rand(1000)  # move the global random state on
    second = make_tiny_model()
    assert first is not second
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
