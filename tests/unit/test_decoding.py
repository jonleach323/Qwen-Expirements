"""Decoding settings and sampling: pushlab.engine.decoding.

Sampling draws each prompt's tokens with a random generator of its own,
seeded from the settings' seed and the prompt's place in the list. So a
sampled reply depends only on its prompt, the seed and its place: never on
the other prompts or how they are batched, nor on torch's global seed.

The tiny models' scores are nearly flat, so at temperature 1 their samples
would hardly depend on them; the replies here are sampled at 0.05.
"""

import doctest
import math
from dataclasses import replace

import pytest
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from pushlab.engine import decoding as decoding_module
from pushlab.engine.decoding import (
    Decoding,
    TokenChooser,
    row_seed,
    sampling_probabilities,
)
from pushlab.engine.generate import generate

PROMPTS = ["Hi", "What is two plus two?", "Name a colour."]
SAMPLE = Decoding(strategy="sample", max_new_tokens=12, temperature=0.05, seed=0)
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


@pytest.mark.parametrize(
    ("temperature", "expected"),
    [(1.0, [0.6, 0.3, 0.1]), (0.5, [0.36 / 0.46, 0.09 / 0.46, 0.01 / 0.46])],
)
def test_draws_follow_the_probabilities_at_the_temperature(
    temperature: float, expected: list[float]
) -> None:
    # At temperature 0.5 the probabilities go as their squares.
    settings = Decoding(
        strategy="sample", max_new_tokens=5, temperature=temperature, seed=3
    )
    choose = TokenChooser(settings, [0], "cpu")
    scores = torch.tensor([[0.6, 0.3, 0.1]]).log()
    draws = [choose(scores, [True])[0] for _ in range(4000)]
    shares = [draws.count(token) / len(draws) for token in range(3)]
    assert shares == pytest.approx(expected, abs=0.03)


def test_a_temperature_near_0_gives_the_greedy_replies(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    cold = replace(SAMPLE, temperature=1e-6)
    sampled = generate(tiny_model, tiny_tokenizer, PROMPTS, [], cold)
    greedy = generate(tiny_model, tiny_tokenizer, PROMPTS, [], GREEDY)
    assert [r.token_ids for r in sampled] == [r.token_ids for r in greedy]


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
def test_a_sampled_reply_depends_on_its_place_not_on_the_other_prompts(
    model_name: str,
    request: pytest.FixtureRequest,
    tiny_tokenizer: PreTrainedTokenizerBase,
) -> None:
    model = request.getfixturevalue(model_name)
    batched = generate(model, tiny_tokenizer, PROMPTS, [], SAMPLE)
    # Other prompts around the second, of other lengths, so its padding changes.
    others = ["Yo", PROMPTS[1], "Tell me a fact about owls, please."]
    assert generate(model, tiny_tokenizer, others, [], SAMPLE)[1] == batched[1]
    assert generate(model, tiny_tokenizer, PROMPTS[:1], [], SAMPLE) == batched[:1]
    # Of three different lengths, each prompt runs as a batch of its own here.
    alone = generate(model, tiny_tokenizer, PROMPTS, [], SAMPLE, group_by_length=True)
    assert alone == batched
    # And the replies depend on the prompts: each differs from the others.
    assert len({reply.token_ids for reply in batched}) == len(PROMPTS)


def test_each_prompt_draws_its_own_random_numbers(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    # Three samples of one prompt in one call are three different replies.
    replies = generate(tiny_model, tiny_tokenizer, ["Hi"] * 3, [], SAMPLE)
    assert len({reply.token_ids for reply in replies}) == 3


def test_row_seeds_never_change() -> None:
    # Every sampled reply ever recorded depends on them.
    assert row_seed(0, 0) == 7605428520541383316
    assert row_seed(0, 1) == 16145528508060844554
    assert row_seed(7, 0) == 15951030755710638324


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
