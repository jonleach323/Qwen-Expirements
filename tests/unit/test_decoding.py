"""Decoding settings and sampling: pushlab.engine.decoding.

Sampling draws each row's tokens with a random generator of its own, seeded
from the settings, so a sampled reply depends only on its prompt and the
seed: never on the other prompts in its batch, nor on torch's global seed.
"""

import doctest
import math
from dataclasses import replace

import pytest
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from pushlab.engine import decoding as decoding_module
from pushlab.engine.decoding import Decoding, sampling_probabilities
from pushlab.engine.generate import generate

PROMPTS = ["Hi", "What is two plus two?", "Name a colour."]
SAMPLE = Decoding(strategy="sample", max_new_tokens=12, temperature=1.0, seed=0)
GREEDY = Decoding(strategy="greedy", max_new_tokens=12)


def test_greedy_needs_only_its_length_and_sampling_a_temperature_and_a_seed() -> None:
    assert GREEDY.temperature is None and GREEDY.seed is None
    assert SAMPLE.top_p == 1.0  # every token, unless top_p is given


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({"strategy": "beam"}, "strategy"),
        ({"max_new_tokens": 0}, "max_new_tokens"),
        ({"max_new_tokens": 2.0}, "max_new_tokens"),
        ({"max_new_tokens": True}, "max_new_tokens"),
        ({"temperature": 0.7}, "greedy decoding takes no temperature"),
        ({"seed": 1}, "greedy decoding takes no seed"),
        ({"top_p": 0.9}, "greedy decoding takes no top_p"),
    ],
)
def test_bad_greedy_settings_are_refused(changes: dict, problem: str) -> None:
    settings = {"strategy": "greedy", "max_new_tokens": 5} | changes
    with pytest.raises(ValueError, match=problem):
        Decoding(**settings)


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        ({"temperature": None}, "temperature"),
        ({"temperature": 0.0}, "temperature"),
        ({"temperature": -1.0}, "temperature"),
        ({"temperature": math.inf}, "temperature"),
        ({"temperature": True}, "temperature"),
        ({"top_p": 0.0}, "top_p"),
        ({"top_p": 1.5}, "top_p"),
        ({"top_p": math.nan}, "top_p"),
        ({"seed": None}, "seed"),
        ({"seed": -1}, "seed"),
        ({"seed": 2**63}, "seed"),
        ({"seed": 1.0}, "seed"),
        ({"seed": True}, "seed"),
    ],
)
def test_bad_sampling_settings_are_refused(changes: dict, problem: str) -> None:
    settings = {"strategy": "sample", "max_new_tokens": 5, "temperature": 1.0}
    settings |= {"seed": 0} | changes
    with pytest.raises(ValueError, match=problem):
        Decoding(**settings)


def test_top_p_keeps_the_fewest_likeliest_tokens_that_hold_top_p() -> None:
    scores = torch.tensor([0.5, 0.3, 0.15, 0.05]).log()

    def kept(top_p: float) -> list[float]:
        return sampling_probabilities(scores, 1.0, top_p).tolist()

    assert kept(1.0) == pytest.approx([0.5, 0.3, 0.15, 0.05])
    assert kept(0.9) == pytest.approx([0.5 / 0.95, 0.3 / 0.95, 0.15 / 0.95, 0.0])
    assert kept(0.7) == pytest.approx([0.625, 0.375, 0.0, 0.0])
    assert kept(1e-9) == [1.0, 0.0, 0.0, 0.0]  # the likeliest always stays


def test_temperature_divides_the_scores() -> None:
    scores = torch.tensor([2.0, 1.0, 0.0])
    expected = torch.softmax(scores / 2.0, dim=-1).tolist()
    assert sampling_probabilities(scores, 2.0, 1.0).tolist() == pytest.approx(expected)
    assert sampling_probabilities(scores.bfloat16(), 1.0, 1.0).dtype == torch.float32


def test_the_same_seed_gives_the_same_replies_and_another_seed_others(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    first = generate(tiny_model, tiny_tokenizer, PROMPTS, [], SAMPLE)
    assert generate(tiny_model, tiny_tokenizer, PROMPTS, [], SAMPLE) == first
    other = generate(tiny_model, tiny_tokenizer, PROMPTS, [], replace(SAMPLE, seed=1))
    assert all(a.token_ids != b.token_ids for a, b in zip(first, other, strict=True))
    greedy = generate(tiny_model, tiny_tokenizer, PROMPTS, [], GREEDY)
    assert all(a.token_ids != b.token_ids for a, b in zip(first, greedy, strict=True))


@pytest.mark.parametrize("model_name", ["tiny_model", "tiny_vl_model"])
def test_a_sampled_row_of_a_padded_batch_equals_its_prompt_alone(
    model_name: str,
    request: pytest.FixtureRequest,
    tiny_tokenizer: PreTrainedTokenizerBase,
) -> None:
    model = request.getfixturevalue(model_name)
    batched = generate(model, tiny_tokenizer, PROMPTS, [], SAMPLE)
    alone = [generate(model, tiny_tokenizer, [p], [], SAMPLE)[0] for p in PROMPTS]
    assert batched == alone


def test_sampling_never_uses_torchs_global_random_state(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    torch.manual_seed(123)
    state = torch.random.get_rng_state()
    first = generate(tiny_model, tiny_tokenizer, PROMPTS, [], SAMPLE)
    assert torch.equal(torch.random.get_rng_state(), state)
    torch.manual_seed(456)
    assert generate(tiny_model, tiny_tokenizer, PROMPTS, [], SAMPLE) == first


def test_a_top_p_too_small_for_a_second_token_gives_the_greedy_replies(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    narrow = replace(SAMPLE, top_p=1e-9)
    sampled = generate(tiny_model, tiny_tokenizer, PROMPTS, [], narrow)
    greedy = generate(tiny_model, tiny_tokenizer, PROMPTS, [], GREEDY)
    assert [r.token_ids for r in sampled] == [r.token_ids for r in greedy]


def test_docstring_examples_run() -> None:
    result = doctest.testmod(decoding_module)
    assert result.attempted > 0
    assert result.failed == 0
