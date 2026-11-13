"""Each reply token's log-probability under the unsteered model, and perplexity.

A reply can be scored again once it is written. One teacher-forced pass
reads the prompt and the reply's tokens together, with nothing pushed, and
the model's scores at each position give the probability of the token that
comes next, the one the reply holds there: the model is "forced" through
the reply instead of choosing its own tokens. The prompt is rendered exactly
as generate() renders it, the chat template with thinking off, so the score
at position len(prompt) - 1 + j is reply token j's, worked in float32 as
decoding.chosen_logprobs works it.

For a reply generated with nothing pushed, that gives generate()'s own
log-probabilities (Generation.logprobs) again, up to rounding. For a pushed
reply, it says how surprising the text is to the same model with nothing
pushed. perplexity sums that up in one number, e raised to minus the mean
log-probability: 1 for a reply the model was sure of, k for one where it
spread its probability evenly over k tokens at every step, and more the
more the text surprises it. Beside the measures in
pushlab.scoring.text_measures, it tells a fluent reply from a broken one
without a judge (decision 0022).

Only sequences of the same total length, prompt and reply together, share
a batch, so nothing is padded, and each value is the one the sequence gets
alone, up to the last bits.
"""

import math
import numbers
from collections.abc import Sequence

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from pushlab.engine.decoding import chosen_logprobs
from pushlab.models.chat import chat_prompt_ids


def _counted(number: int, one: str, many: str) -> str:
    return f"{number} {one if number == 1 else many}"


def _checked_prompts(prompts: Sequence[str]) -> list[str]:
    if isinstance(prompts, str):  # else each character would be a prompt
        raise ValueError("prompts must be a list of prompts, not one text")
    prompts = list(prompts)
    for prompt in prompts:
        if not isinstance(prompt, str):
            raise ValueError(f"each prompt must be text, not {prompt!r}")
    return prompts


def _checked_reply(index: int, reply: object, vocabulary: int) -> list[int]:
    """Reply index's token IDs as ints, each one the model has."""
    if isinstance(reply, str | bytes):
        raise ValueError(
            f"reply {index} must be its token IDs, not text: tokenize it with "
            "tokenizer(text, add_special_tokens=False)"
        )
    try:
        tokens = list(reply)
    except TypeError:
        raise ValueError(
            f"reply {index} must be a list of token IDs, not {reply!r}"
        ) from None
    if not tokens:
        raise ValueError(f"reply {index} is empty: it has no token to score")
    for token in tokens:
        whole = isinstance(token, numbers.Integral) and not isinstance(token, bool)
        if not whole or not 0 <= token < vocabulary:
            raise ValueError(
                f"reply {index} holds {token!r}, which is no token ID of the "
                f"model: they run from 0 to {vocabulary - 1}"
            )
    return [int(token) for token in tokens]


def _batches(lengths: Sequence[int], batch_size: int) -> list[list[int]]:
    """Which sequences run together: those of one total length, batch_size
    at most, each length's in the order they came.

    Example:
        >>> _batches([12, 9, 12, 12], batch_size=2)
        [[0, 2], [3], [1]]
    """
    groups: dict[int, list[int]] = {}
    for index, length in enumerate(lengths):
        groups.setdefault(length, []).append(index)
    return [
        group[start : start + batch_size]
        for group in groups.values()
        for start in range(0, len(group), batch_size)
    ]


@torch.no_grad()
def reply_logprobs(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    replies: Sequence[Sequence[int]],
    *,
    batch_size: int = 8,
) -> list[list[float]]:
    """Each reply token's log-probability under the model, nothing pushed.

    replies[i] is the reply to prompts[i], as its token IDs: a reply's
    Generation.token_ids, its end-of-turn token included, or a text's tokens
    (tokenizer(text, add_special_tokens=False)). Each prompt is rendered in
    the chat template with thinking off, as generate() renders it, and one
    teacher-forced pass reads it and its reply: the score at position
    len(prompt) - 1 + j gives reply token j's log-probability, from the
    model's scores before any temperature or top-p, in float32. The lists
    come back in the replies' order, one value per token.

    Up to batch_size sequences of the same total length run together, so
    nothing is padded. The model runs as it is given, under
    torch.no_grad(), with no hook added; it should be in eval mode, as a
    loaded model is, though one in training mode is not refused. No prompts
    give an empty list, and no pass.

    Example:
        >>> from pushlab.engine.decoding import Decoding
        >>> from pushlab.engine.generate import generate
        >>> greedy = Decoding(strategy="greedy", max_new_tokens=200)
        >>> (reply,) = generate(model, tokenizer, ["Hi"], [], greedy)  # doctest: +SKIP
        >>> (forced,) = reply_logprobs(  # doctest: +SKIP
        ...     model, tokenizer, ["Hi"], [reply.token_ids]
        ... )
        >>> len(forced) == len(reply.token_ids)  # doctest: +SKIP
        True
        >>> perplexity(forced) >= 1  # doctest: +SKIP
        True
    """
    prompts = _checked_prompts(prompts)
    if isinstance(replies, str | bytes):
        raise ValueError("replies must be a list of replies, not one text")
    replies = list(replies)
    if len(replies) != len(prompts):
        raise ValueError(
            "give one reply per prompt: "
            f"{_counted(len(prompts), 'prompt', 'prompts')}, "
            f"{_counted(len(replies), 'reply', 'replies')}"
        )
    whole = isinstance(batch_size, int) and not isinstance(batch_size, bool)
    if not whole or batch_size < 1:
        raise ValueError(
            f"batch_size must be a whole number from 1, not {batch_size!r}"
        )
    embeddings = model.get_input_embeddings()
    tokens = [
        _checked_reply(index, reply, embeddings.num_embeddings)
        for index, reply in enumerate(replies)
    ]
    prompt_ids = [
        chat_prompt_ids(tokenizer, prompt, thinking=False) for prompt in prompts
    ]
    # Where the token IDs go in. Qwen3.5's first parameters belong to its
    # vision tower, which may sit elsewhere.
    device = embeddings.weight.device
    found: list[list[float]] = [[] for _ in prompts]
    totals = [
        len(ids) + len(reply) for ids, reply in zip(prompt_ids, tokens, strict=True)
    ]
    for batch in _batches(totals, batch_size):
        width = totals[batch[0]]
        input_ids = torch.tensor(
            [prompt_ids[i] + tokens[i] for i in batch], device=device
        )
        longest = max(len(tokens[i]) for i in batch)
        # Positions are always given, as generate() gives them: without them,
        # Qwen3.5 could rebuild them from rope offsets it keeps from its last
        # generate() call. Only the last longest + 1 positions' scores are
        # kept, from the earliest that predicts a reply's first token.
        output = model(
            input_ids=input_ids,
            position_ids=torch.arange(width, device=device).repeat(len(batch), 1),
            use_cache=False,
            logits_to_keep=longest + 1,
        )
        for row, index in enumerate(batch):
            # Kept score k is position width - longest - 1 + k's. A reply of
            # n tokens starts at position width - n, so the score that
            # predicts its first token, one position before, is kept score
            # longest - n, and its last token's is kept score longest - 1.
            start = longest - len(tokens[index])
            scores = output.logits[row, start:longest]
            found[index] = chosen_logprobs(scores, tokens[index])
    return found


def perplexity(logprobs: Sequence[float]) -> float | None:
    """A reply's perplexity: e raised to minus the mean of its tokens'
    log-probabilities; None for no tokens.

    1 means the model was sure of every token; a model that spread its
    probability evenly over k tokens at every step gives k. A mean so far
    below zero that the result is past what a float holds gives infinity.

    Example:
        >>> perplexity([-1.0, -1.0])  # e
        2.718281828459045
        >>> perplexity([0.0, 0.0])
        1.0
        >>> perplexity([]) is None
        True
    """
    if len(logprobs) == 0:  # not `not logprobs`, which an array refuses
        return None
    mean = sum(logprobs) / len(logprobs)
    try:
        return math.exp(-mean)
    except OverflowError:
        return math.inf
