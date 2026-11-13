"""Each reply token's log-probability under the unsteered model, from one
teacher-forced pass, and perplexity: pushlab.engine.score_text.

generate() with nothing pushed is the unsteered model, so the log-probability
it kept for each token it chose (Generation.logprobs) is the reference: a
forced pass over the prompt and the reply must give it again, on both tiny
models. The tiny models run in float32 on the CPU, where the two differ only
by rounding, a few 1e-7; a token shifted by one position, or a prompt
rendered another way, moves them by 1e-3 and more.
"""

import doctest
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from pushlab.engine import score_text as score_text_module
from pushlab.engine.decoding import Decoding
from pushlab.engine.generate import generate
from pushlab.engine.score_text import perplexity, reply_logprobs
from pushlab.models.chat import chat_prompt_ids

MODELS = ["tiny_model", "tiny_vl_model"]
# tests/unit/test_generate_logprobs.py's prompts: three lengths in the chat
# template, so generate() pads them on the left in one batch.
PROMPTS = ["Hi", "What is two plus two?", "Name a colour."]
GREEDY = Decoding(strategy="greedy", max_new_tokens=10)
# Hot, so it draws tokens the model finds unlikely as well as likely ones.
SAMPLED = Decoding(strategy="sample", max_new_tokens=10, temperature=1.5, seed=4)
# A text the tiny models never wrote, one token per byte in their tokenizer.
TEXT = "Four, I think. Or maybe red; the colour of a bus. Hello there!"


def _forced(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    reply: list[int],
) -> list[float]:
    """The reference: one forced pass over a single prompt and reply, all its
    scores kept, as test_generate_logprobs.py's own forced pass works it."""
    ids = chat_prompt_ids(tokenizer, prompt, thinking=False)
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids + reply])).logits[0].float()
    # The score at the prompt's last position predicts reply token 0.
    log_probs = torch.log_softmax(logits[len(ids) - 1 : -1], dim=-1)
    return log_probs[torch.arange(len(reply)), reply].tolist()


def _batch_shapes(model: PreTrainedModel, run: Callable[[], Any]) -> tuple[Any, list]:
    """run()'s result, and the (rows, tokens) of every pass the model made."""
    shapes: list[tuple[int, ...]] = []
    hook = model.get_input_embeddings().register_forward_hook(
        lambda module, inputs, output: shapes.append(tuple(inputs[0].shape))
    )
    try:
        return run(), shapes
    finally:
        hook.remove()


def _text_tokens(tokenizer: PreTrainedTokenizerBase, count: int) -> list[int]:
    """The first count tokens of TEXT, said twice."""
    tokens = tokenizer(TEXT * 2, add_special_tokens=False)["input_ids"]
    assert len(tokens) >= count
    return tokens[:count]


@pytest.mark.parametrize("name", MODELS)
def test_a_forced_pass_gives_generates_own_logprobs(
    name: str,
    request: pytest.FixtureRequest,
    tiny_tokenizer: PreTrainedTokenizerBase,
) -> None:
    model = request.getfixturevalue(name)
    # Each in one left-padded batch, as generate() runs them by default.
    replies = generate(model, tiny_tokenizer, PROMPTS, [], GREEDY)
    replies += generate(model, tiny_tokenizer, PROMPTS, [], SAMPLED)
    forced, shapes = _batch_shapes(
        model,
        lambda: reply_logprobs(
            model, tiny_tokenizer, PROMPTS * 2, [reply.token_ids for reply in replies]
        ),
    )
    # The greedy and sampled replies to one prompt, as long as each other,
    # share a forced pass.
    assert max(rows for rows, _ in shapes) > 1
    assert len(forced) == len(replies)
    for reply, values in zip(replies, forced, strict=True):
        assert len(values) == len(reply.token_ids) > 0
        assert values == pytest.approx(list(reply.logprobs), rel=0, abs=1e-4)
        assert all(isinstance(value, float) for value in values)
    # Far enough apart that a token shifted by one position would show.
    spread = [value for values in forced for value in values]
    assert max(spread) - min(spread) > 0.01


@pytest.mark.parametrize("name", MODELS)
def test_only_sequences_of_one_length_share_a_batch_and_none_is_padded(
    name: str,
    request: pytest.FixtureRequest,
    tiny_tokenizer: PreTrainedTokenizerBase,
) -> None:
    model = request.getfixturevalue(name)
    lengths = [len(chat_prompt_ids(tiny_tokenizer, p, thinking=False)) for p in PROMPTS]
    # A total length each of the three prompts, all of different lengths,
    # reaches with a reply of its own length.
    total = max(lengths) + 3
    prompts = [PROMPTS[0], PROMPTS[0], PROMPTS[1], PROMPTS[2]]
    replies = [
        _text_tokens(tiny_tokenizer, total - lengths[0]),
        _text_tokens(tiny_tokenizer, 5),  # the odd one out: lengths[0] + 5
        _text_tokens(tiny_tokenizer, total - lengths[1]),
        _text_tokens(tiny_tokenizer, total - lengths[2]),
    ]
    alone = [
        _forced(model, tiny_tokenizer, prompt, reply)
        for prompt, reply in zip(prompts, replies, strict=True)
    ]
    passes = {
        1: [(1, total), (1, lengths[0] + 5), (1, total), (1, total)],
        2: [(2, total), (1, total), (1, lengths[0] + 5)],
        3: [(3, total), (1, lengths[0] + 5)],
    }
    found = {}
    for batch_size, expected in passes.items():
        found[batch_size], shapes = _batch_shapes(
            model,
            lambda size=batch_size: reply_logprobs(
                model, tiny_tokenizer, prompts, replies, batch_size=size
            ),
        )
        assert sorted(shapes) == sorted(expected)
        # Back in the input order, each as the sequence gets it alone.
        for values, reference in zip(found[batch_size], alone, strict=True):
            assert values == pytest.approx(reference, rel=0, abs=1e-5)
    for values, single in zip(found[3], found[1], strict=True):
        assert values == pytest.approx(single, rel=0, abs=1e-5)


@pytest.mark.parametrize("name", MODELS)
def test_a_reply_given_as_the_tokens_of_a_text_scores_as_those_tokens(
    name: str,
    request: pytest.FixtureRequest,
    tiny_tokenizer: PreTrainedTokenizerBase,
) -> None:
    model = request.getfixturevalue(name)
    words = tiny_tokenizer(TEXT, add_special_tokens=False)["input_ids"]
    reply = [*words, tiny_tokenizer.eos_token_id]  # ended, as a turn ends
    (values,) = reply_logprobs(model, tiny_tokenizer, PROMPTS[1:2], [reply])
    assert len(values) == len(reply)
    assert values == pytest.approx(
        _forced(model, tiny_tokenizer, PROMPTS[1], reply), rel=0, abs=1e-5
    )
    # A tuple of token IDs, as Generation.token_ids is one, scores the same.
    (again,) = reply_logprobs(model, tiny_tokenizer, PROMPTS[1:2], [tuple(reply)])
    assert again == values


def test_the_prompt_is_rendered_as_generate_renders_it(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    reply = _text_tokens(tiny_tokenizer, 8)
    (values,) = reply_logprobs(tiny_model, tiny_tokenizer, ["Hi"], [reply])
    # The prompt with thinking on, or with no chat template at all, scores
    # the same tokens differently.
    thinking = chat_prompt_ids(tiny_tokenizer, "Hi", thinking=True)
    plain = tiny_tokenizer("Hi", add_special_tokens=False)["input_ids"]
    for ids in (thinking, plain):
        with torch.no_grad():
            logits = tiny_model(input_ids=torch.tensor([ids + reply])).logits[0]
        log_probs = torch.log_softmax(logits[len(ids) - 1 : -1].float(), dim=-1)
        other = log_probs[torch.arange(len(reply)), reply].tolist()
        assert max(abs(a - b) for a, b in zip(values, other, strict=True)) > 1e-3


def test_a_model_in_training_mode_is_scored_without_gradients(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    reply = _text_tokens(tiny_tokenizer, 6)
    expected = reply_logprobs(tiny_model, tiny_tokenizer, ["Hi"], [reply])
    tiny_model.train()
    recorded: list[bool] = []
    hook = tiny_model.get_input_embeddings().register_forward_hook(
        lambda module, inputs, output: recorded.append(torch.is_grad_enabled())
    )
    try:
        got = reply_logprobs(tiny_model, tiny_tokenizer, ["Hi"], [reply])
    finally:
        hook.remove()
    assert recorded == [False]  # one pass, run under torch.no_grad()
    assert tiny_model.training  # left in the mode it was given in
    # The tiny model has no dropout, so training mode changes no value.
    assert got[0] == pytest.approx(expected[0], rel=0, abs=1e-6)


def test_nothing_to_score_gives_no_lists_and_runs_no_pass(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    got, shapes = _batch_shapes(
        tiny_model, lambda: reply_logprobs(tiny_model, tiny_tokenizer, [], [])
    )
    assert got == []
    assert shapes == []


@pytest.mark.parametrize(
    ("prompts", "replies", "message"),
    [
        (["Hi", "Yo"], [[1, 2]], "one reply per prompt: 2 prompts, 1 reply"),
        (["Hi"], [[1], [2]], "one reply per prompt: 1 prompt, 2 replies"),
        (["Hi"], "Hello", "replies must be a list of replies, not one text"),
        (["Hi", "Yo"], [[1, 2], []], "reply 1 is empty"),
        (["Hi"], [()], "reply 0 is empty"),
        ("Hi", [[1]], "a list of prompts, not one text"),
        ([3], [[1]], "each prompt must be text, not 3"),
        (["Hi"], ["Hello"], "reply 0 must be its token IDs, not text"),
        (["Hi"], [5], "reply 0 must be a list of token IDs, not 5"),
        (["Hi"], [[1, 512]], "reply 0 holds 512, which is no token ID"),
        (["Hi"], [[1, -1]], "reply 0 holds -1, which is no token ID"),
        (["Hi"], [[1.0]], "reply 0 holds 1.0, which is no token ID"),
        (["Hi"], [[True]], "reply 0 holds True, which is no token ID"),
    ],
)
def test_it_refuses_prompts_and_replies_it_cannot_score(
    tiny_model: PreTrainedModel,
    tiny_tokenizer: PreTrainedTokenizerBase,
    prompts: Any,
    replies: Any,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        reply_logprobs(tiny_model, tiny_tokenizer, prompts, replies)


@pytest.mark.parametrize("batch_size", [0, -1, 1.5, True, "8"])
def test_it_refuses_a_batch_size_that_is_no_whole_number_from_1(
    tiny_model: PreTrainedModel,
    tiny_tokenizer: PreTrainedTokenizerBase,
    batch_size: Any,
) -> None:
    with pytest.raises(ValueError, match="batch_size must be a whole number from 1"):
        reply_logprobs(tiny_model, tiny_tokenizer, ["Hi"], [[1]], batch_size=batch_size)


def test_token_ids_read_back_as_numpy_integers_score_as_python_ones(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    reply = _text_tokens(tiny_tokenizer, 6)
    expected = reply_logprobs(tiny_model, tiny_tokenizer, ["Hi"], [reply])
    as_array = np.array(reply, dtype=np.int64)  # as a table's column gives them
    assert reply_logprobs(tiny_model, tiny_tokenizer, ["Hi"], [as_array]) == expected


def test_perplexity_is_e_to_minus_the_mean_log_probability() -> None:
    assert perplexity([-1, -1]) == pytest.approx(math.e)
    assert perplexity([-1.0, -3.0]) == pytest.approx(math.exp(2.0))
    assert perplexity([0.0, 0.0]) == 1.0  # a reply the model was sure of
    assert perplexity((-2.0,)) == pytest.approx(math.exp(2.0))
    assert perplexity([]) is None
    assert perplexity([-1000.0]) == math.inf  # past what a float holds
    # As a table's column gives them, too.
    assert perplexity(np.array([-1.0, -1.0])) == pytest.approx(math.e)
    assert perplexity(np.array([])) is None


def test_docstring_examples_run() -> None:
    result = doctest.testmod(score_text_module)
    assert result.attempted > 0
    assert result.failed == 0
