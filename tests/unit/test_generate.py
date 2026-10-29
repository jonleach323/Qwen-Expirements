"""Greedy decoding with the model's cache: pushlab.engine.generate.

These tests use raw token IDs, so no tokenizer is involved.
"""

import doctest
from collections.abc import Callable

import pytest
import torch
from torch import nn
from transformers import Qwen3_5ForConditionalGeneration, Qwen3ForCausalLM

from pushlab.engine import generate
from pushlab.engine.generate import greedy_generate

PROMPT = [5, 17, 42, 99, 3]


def _most_likely_next_tokens(
    model: nn.Module, prompt: list[int], reply: list[int]
) -> list[int]:
    """The most likely token at each reply position, from one pass without a cache."""
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([prompt + reply])).logits[0]
    return logits[len(prompt) - 1 : -1].argmax(dim=-1).tolist()


@pytest.mark.parametrize("count", [1, 2, 9])
def test_returns_the_requested_number_of_tokens(
    tiny_model: Qwen3ForCausalLM, count: int
) -> None:
    tokens = greedy_generate(tiny_model, PROMPT, count)
    assert len(tokens) == count
    assert all(type(token) is int for token in tokens)


def test_every_token_is_the_most_likely_one(tiny_model: Qwen3ForCausalLM) -> None:
    tokens = greedy_generate(tiny_model, PROMPT, 12)
    assert tokens == _most_likely_next_tokens(tiny_model, PROMPT, tokens)


def test_gives_the_same_tokens_every_time(
    tiny_model: Qwen3ForCausalLM, make_tiny_model: Callable[[], Qwen3ForCausalLM]
) -> None:
    first = greedy_generate(tiny_model, PROMPT, 12)
    assert greedy_generate(tiny_model, PROMPT, 12) == first
    assert greedy_generate(make_tiny_model(), PROMPT, 12) == first


def test_works_through_qwen3_5s_wrapper_and_linear_attention(
    tiny_vl_model: Qwen3_5ForConditionalGeneration,
) -> None:
    # Qwen3.5 caches its linear-attention layers differently from its
    # full-attention ones; a mistake in either would change the tokens.
    tokens = greedy_generate(tiny_vl_model, PROMPT, 12)
    assert len(tokens) == 12
    assert tokens == _most_likely_next_tokens(tiny_vl_model, PROMPT, tokens)


def test_stops_after_a_stop_token_and_keeps_it(tiny_model: Qwen3ForCausalLM) -> None:
    tokens = greedy_generate(tiny_model, PROMPT, 12)
    stop = tokens[4]
    expected = tokens[: tokens.index(stop) + 1]
    assert greedy_generate(tiny_model, PROMPT, 12, stop_token_ids=[stop]) == expected


def test_a_stop_token_that_never_comes_changes_nothing(
    tiny_model: Qwen3ForCausalLM,
) -> None:
    tokens = greedy_generate(tiny_model, PROMPT, 12)
    unused = next(t for t in range(tiny_model.config.vocab_size) if t not in tokens)
    assert greedy_generate(tiny_model, PROMPT, 12, stop_token_ids=[unused]) == tokens


@pytest.mark.parametrize(
    ("prompt", "count", "problem"),
    [([], 5, "at least one prompt token"), (PROMPT, 0, "at least 1")],
)
def test_refuses_an_empty_prompt_or_no_tokens(
    tiny_model: Qwen3ForCausalLM, prompt: list[int], count: int, problem: str
) -> None:
    with pytest.raises(ValueError, match=problem):
        greedy_generate(tiny_model, prompt, count)


def test_docstring_examples_run() -> None:
    result = doctest.testmod(generate)
    assert result.attempted > 0
    assert result.failed == 0
