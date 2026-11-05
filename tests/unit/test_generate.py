"""The generation loop: pushlab.engine.generate.

The tiny models run in float32 on the CPU, where a row of a left-padded
batch and its prompt alone compute the same numbers up to rounding. Hugging
Face's own generate() is the reference for greedy decoding. The tiny models'
scores for the next token are nearly flat, so a mistake may leave their
tokens as they are; the tests compare the scores of every pass too.
"""

import doctest
from collections.abc import Callable
from typing import Any

import pytest
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from pushlab.engine import generate as generate_module
from pushlab.engine.decoding import Decoding
from pushlab.engine.generate import generate
from pushlab.engine.push import Push
from pushlab.models.chat import chat_prompt_ids
from pushlab.units.neuron import NeuronUnit

MODELS = ["tiny_model", "tiny_vl_model"]
# Of different lengths in the chat template, so a batch of them is padded.
PROMPTS = ["Hi", "What is two plus two?", "Name a colour."]
GREEDY = Decoding(strategy="greedy", max_new_tokens=12)
# Rounding moves float32 scores by a few 1e-7; a mistake moves them by 1e-3
# and more (decision 0009).
CLOSE = {"rtol": 0.0, "atol": 1e-5}


def _case(
    name: str, request: pytest.FixtureRequest
) -> tuple[PreTrainedModel, NeuronUnit, list[float]]:
    """A tiny model, a neuron of it in a middle or last layer, and its norms."""
    if name == "tiny_vl_model":
        request.getfixturevalue("tiny_downloads")  # registers tiny-vl
        return (
            request.getfixturevalue(name),
            NeuronUnit("tiny-vl", 2, 100),
            [3.0, 5.0, 8.0, 13.0],
        )
    return request.getfixturevalue(name), NeuronUnit("tiny", 1, 5), [7.0, 10.0]


def _scored(
    model: PreTrainedModel, run: Callable[[], Any]
) -> tuple[Any, list[torch.Tensor]]:
    """run()'s result, and each pass's scores for the next token: (rows, vocabulary)."""
    scores: list[torch.Tensor] = []
    hook = model.lm_head.register_forward_hook(
        lambda module, inputs, output: scores.append(output[:, -1].float().clone())
    )
    try:
        result = run()
    finally:
        hook.remove()
    return result, scores


def _calls(model: PreTrainedModel, run: Callable[[], object]) -> list[dict[str, Any]]:
    """The keyword arguments of every forward pass of the model during run()."""
    calls: list[dict[str, Any]] = []
    hook = model.register_forward_pre_hook(
        lambda module, args, kwargs: calls.append(dict(kwargs)), with_kwargs=True
    )
    try:
        run()
    finally:
        hook.remove()
    return calls


@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("prompts", [PROMPTS[1:2], PROMPTS], ids=["alone", "batch"])
def test_greedy_replies_equal_hugging_faces_generate(
    model_name: str,
    prompts: list[str],
    request: pytest.FixtureRequest,
    tiny_tokenizer: PreTrainedTokenizerBase,
    hf_greedy: Callable[..., tuple[list[list[int]], list[torch.Tensor]]],
) -> None:
    model = request.getfixturevalue(model_name)
    replies, scores = _scored(
        model, lambda: generate(model, tiny_tokenizer, prompts, [], GREEDY)
    )
    expected, expected_scores = hf_greedy(
        model, tiny_tokenizer, prompts, GREEDY.max_new_tokens
    )
    assert [list(reply.token_ids) for reply in replies] == expected
    # Score for score too, bit for bit, on every pass of every row's reply.
    for row, reply in enumerate(replies):
        for k in range(len(reply.token_ids)):
            assert torch.equal(scores[k][row], expected_scores[k][row]), (row, k)


@pytest.mark.parametrize("model_name", MODELS)
@pytest.mark.parametrize("pushed", [False, True], ids=["no-push", "pushed"])
def test_each_row_of_a_padded_batch_equals_its_prompt_alone(
    model_name: str,
    pushed: bool,
    request: pytest.FixtureRequest,
    tiny_tokenizer: PreTrainedTokenizerBase,
) -> None:
    model, unit, norms = _case(model_name, request)
    pushes = (
        [Push(unit=unit, alpha=2.0, start=1, include_prompt=True)] if pushed else []
    )
    batched, batched_scores = _scored(
        model, lambda: generate(model, tiny_tokenizer, PROMPTS, pushes, GREEDY, norms)
    )
    for row, prompt in enumerate(PROMPTS):
        (alone,), scores = _scored(
            model,
            lambda prompt=prompt: generate(
                model, tiny_tokenizer, [prompt], pushes, GREEDY, norms
            ),
        )
        assert batched[row] == alone
        for k in range(len(alone.token_ids)):
            torch.testing.assert_close(batched_scores[k][row], scores[k][0], **CLOSE)
    if pushed:  # the push changes every row, so its effect was there to compare
        unpushed = generate(model, tiny_tokenizer, PROMPTS, [], GREEDY)
        assert all(
            a.token_ids != b.token_ids for a, b in zip(batched, unpushed, strict=True)
        )


@pytest.mark.parametrize("model_name", MODELS)
def test_every_token_is_the_most_likely_one(
    model_name: str,
    request: pytest.FixtureRequest,
    tiny_tokenizer: PreTrainedTokenizerBase,
) -> None:
    # Checked against one pass over the prompt and the whole reply, which
    # needs no cache. Qwen3.5 caches its linear-attention layers differently
    # from its full-attention ones; a mistake in either would show here.
    model = request.getfixturevalue(model_name)
    (reply,), scores = _scored(
        model, lambda: generate(model, tiny_tokenizer, PROMPTS[1:2], [], GREEDY)
    )
    ids = chat_prompt_ids(tiny_tokenizer, PROMPTS[1], thinking=False)
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids + list(reply.token_ids)])).logits
    expected = logits[0, len(ids) - 1 : -1]  # the scores that chose each token
    assert list(reply.token_ids) == expected.argmax(-1).tolist()
    torch.testing.assert_close(torch.cat(scores), expected, **CLOSE)


def test_gives_the_same_replies_every_time(
    tiny_model: PreTrainedModel,
    make_tiny_model: Callable[[], PreTrainedModel],
    tiny_tokenizer: PreTrainedTokenizerBase,
) -> None:
    first = generate(tiny_model, tiny_tokenizer, PROMPTS, [], GREEDY)
    assert generate(tiny_model, tiny_tokenizer, PROMPTS, [], GREEDY) == first
    assert generate(make_tiny_model(), tiny_tokenizer, PROMPTS, [], GREEDY) == first


@pytest.mark.parametrize("model_name", MODELS)
def test_left_padding_keeps_each_rows_positions_counting_from_its_first_token(
    model_name: str,
    request: pytest.FixtureRequest,
    tiny_tokenizer: PreTrainedTokenizerBase,
) -> None:
    model = request.getfixturevalue(model_name)
    four = Decoding(strategy="greedy", max_new_tokens=4)
    replies: list = []
    calls = _calls(
        model,
        lambda: replies.extend(generate(model, tiny_tokenizer, PROMPTS, [], four)),
    )
    rows = [
        chat_prompt_ids(tiny_tokenizer, prompt, thinking=False) for prompt in PROMPTS
    ]
    width = max(len(row) for row in rows)
    pad = tiny_tokenizer.pad_token_id
    # The prompt pass: the chat prompts, padded on the left and masked there.
    prompt_pass = calls[0]
    assert prompt_pass["input_ids"].tolist() == [
        [pad] * (width - len(row)) + row for row in rows
    ]
    assert prompt_pass["attention_mask"].tolist() == [
        [0] * (width - len(row)) + [1] * len(row) for row in rows
    ]
    assert prompt_pass["position_ids"].tolist() == [
        [0] * (width - len(row)) + list(range(len(row))) for row in rows
    ]
    # Then the pass that reads reply token k, at each row's next position:
    # tokens 0 to 2. No pass reads the last token, which ends the reply.
    assert len(calls) == 1 + 3
    for k, call in enumerate(calls[1:]):
        assert call["input_ids"].tolist() == [[reply.token_ids[k]] for reply in replies]
        assert call["position_ids"].tolist() == [[len(row) + k] for row in rows]
        assert call["attention_mask"].tolist() == [
            [0] * (width - len(row)) + [1] * (len(row) + k + 1) for row in rows
        ]
        assert call["past_key_values"] is not None


def test_the_inputs_go_where_the_token_embeddings_are(
    tiny_vl_model: PreTrainedModel,
    make_tiny_vl_model: Callable[[], PreTrainedModel],
    tiny_tokenizer: PreTrainedTokenizerBase,
) -> None:
    # Qwen3.5's first parameters belong to its vision tower. Moved off to
    # another device, here "meta", it must not take the inputs with it.
    expected = generate(make_tiny_vl_model(), tiny_tokenizer, PROMPTS, [], GREEDY)
    tiny_vl_model.model.visual.to("meta")
    assert next(tiny_vl_model.parameters()).device.type == "meta"
    assert generate(tiny_vl_model, tiny_tokenizer, PROMPTS, [], GREEDY) == expected


def test_a_batch_without_padding_has_no_attention_mask(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    # As in Hugging Face's generate(), which drops a mask of only ones.
    three = Decoding(strategy="greedy", max_new_tokens=3)
    calls = _calls(
        tiny_model,
        lambda: generate(tiny_model, tiny_tokenizer, ["Hi", "Yo"], [], three),
    )
    width = len(chat_prompt_ids(tiny_tokenizer, "Hi", thinking=False))
    assert [call.get("attention_mask") for call in calls] == [None] * 3
    assert calls[0]["position_ids"].tolist() == [list(range(width))] * 2
    assert [call["position_ids"].tolist() for call in calls[1:]] == [
        [[width + k]] * 2 for k in range(2)
    ]


@pytest.mark.parametrize("model_name", MODELS)
def test_scores_only_the_newest_position(
    model_name: str,
    request: pytest.FixtureRequest,
    tiny_tokenizer: PreTrainedTokenizerBase,
) -> None:
    # Scores for every prompt position would cost memory and are never used.
    model = request.getfixturevalue(model_name)
    shapes = []
    hook = model.lm_head.register_forward_hook(
        lambda module, inputs, output: shapes.append(tuple(output.shape[:2]))
    )
    try:
        three = Decoding(strategy="greedy", max_new_tokens=3)
        generate(model, tiny_tokenizer, PROMPTS, [], three)
    finally:
        hook.remove()
    assert shapes == [(len(PROMPTS), 1)] * 3


def test_group_by_length_batches_only_prompts_of_the_same_length(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    prompts = ["Hi", "What is two plus two?", "Yo"]
    lengths = [len(chat_prompt_ids(tiny_tokenizer, p, thinking=False)) for p in prompts]
    assert lengths[0] == lengths[2] != lengths[1]
    grouped: list = []
    calls = _calls(
        tiny_model,
        lambda: grouped.extend(
            generate(
                tiny_model, tiny_tokenizer, prompts, [], GREEDY, group_by_length=True
            )
        ),
    )
    prompt_passes = [
        tuple(call["input_ids"].shape)
        for call in calls
        if call.get("past_key_values") is None
    ]
    assert prompt_passes == [(2, lengths[0]), (1, lengths[1])]
    assert all(call.get("attention_mask") is None for call in calls)  # no padding
    # The replies come back in the prompts' order, as one padded batch gives them.
    assert grouped == generate(tiny_model, tiny_tokenizer, prompts, [], GREEDY)


@pytest.mark.parametrize(
    ("prompts", "problem"),
    [([], "at least one prompt"), ("Hi", "list of prompts"), (["Hi", 7], "text")],
)
def test_refuses_prompts_that_are_not_a_list_of_texts(
    tiny_model: PreTrainedModel,
    tiny_tokenizer: PreTrainedTokenizerBase,
    prompts: object,
    problem: str,
) -> None:
    with pytest.raises(ValueError, match=problem):
        generate(tiny_model, tiny_tokenizer, prompts, [], GREEDY)  # type: ignore[arg-type]


def test_refuses_pushes_without_norms_and_settings_that_are_not_a_decoding(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    push = Push(unit=NeuronUnit("tiny", 1, 5), alpha=1.0, include_prompt=True)
    with pytest.raises(ValueError, match="pushes need norms.*calibration"):
        generate(tiny_model, tiny_tokenizer, ["Hi"], [push], GREEDY)
    with pytest.raises(ValueError, match="Decoding"):
        settings = {"strategy": "greedy", "max_new_tokens": 3}
        generate(tiny_model, tiny_tokenizer, ["Hi"], [], settings)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="group_by_length"):
        generate(tiny_model, tiny_tokenizer, ["Hi"], [], GREEDY, group_by_length=1)  # type: ignore[arg-type]


def test_docstring_examples_run() -> None:
    result = doctest.testmod(generate_module)
    assert result.attempted > 0
    assert result.failed == 0
