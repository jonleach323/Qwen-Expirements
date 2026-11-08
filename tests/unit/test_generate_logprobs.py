"""Each reply token's log-probability (Generation.logprobs): from the scores
of the pass that chose it, before any temperature or top-p."""

import pytest
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from pushlab.engine.decoding import Decoding, chosen_logprobs
from pushlab.engine.generate import generate
from pushlab.models.chat import chat_prompt_ids

PROMPTS = ["Hi", "What is two plus two?", "Name a colour."]


def _scores(model: PreTrainedModel, run):  # type: ignore[no-untyped-def]
    """run()'s result, and each pass's next-token scores: (rows, vocabulary)."""
    scores: list[torch.Tensor] = []
    hook = model.lm_head.register_forward_hook(
        lambda module, inputs, output: scores.append(output[:, -1].float().clone())
    )
    try:
        return run(), scores
    finally:
        hook.remove()


@pytest.mark.parametrize(
    "decoding",
    [
        Decoding(strategy="greedy", max_new_tokens=10),
        Decoding(strategy="sample", max_new_tokens=10, temperature=0.5, seed=3),
        Decoding(
            strategy="sample", max_new_tokens=10, temperature=1.5, top_p=0.5, seed=4
        ),
    ],
    ids=["greedy", "cool", "hot-top-p"],
)
def test_a_tokens_logprob_comes_from_the_pass_that_chose_it(
    tiny_model: PreTrainedModel,
    tiny_tokenizer: PreTrainedTokenizerBase,
    decoding: Decoding,
) -> None:
    replies, scores = _scores(
        tiny_model,
        lambda: generate(tiny_model, tiny_tokenizer, PROMPTS, [], decoding),
    )
    for row, reply in enumerate(replies):
        assert len(reply.logprobs) == len(reply.token_ids) > 0
        for j, (token, logprob) in enumerate(
            zip(reply.token_ids, reply.logprobs, strict=True)
        ):
            # Pass j chose token j: the model's own log-probability, untempered.
            log_probs = torch.log_softmax(scores[j][row], dim=-1)
            assert logprob == pytest.approx(float(log_probs[token]), abs=1e-5)
            if decoding.strategy == "greedy":  # the best token is the most likely
                assert logprob == pytest.approx(float(log_probs.max()), abs=1e-5)


def test_logprobs_agree_with_one_teacher_forced_pass(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    decoding = Decoding(strategy="greedy", max_new_tokens=10)
    (reply,) = generate(tiny_model, tiny_tokenizer, PROMPTS[1:2], [], decoding)
    prompt = chat_prompt_ids(tiny_tokenizer, PROMPTS[1], thinking=False)
    ids = torch.tensor([prompt + list(reply.token_ids)])
    with torch.no_grad():
        logits = tiny_model(input_ids=ids).logits[0].float()
    # The score at the prompt's last position predicts reply token 0.
    log_probs = torch.log_softmax(logits[len(prompt) - 1 : -1], dim=-1)
    forced = log_probs[torch.arange(len(reply.token_ids)), list(reply.token_ids)]
    assert list(reply.logprobs) == pytest.approx(forced.tolist(), abs=1e-4)


def test_chosen_logprobs_works_in_float32_on_any_scores() -> None:
    scores = torch.tensor([[1.0, 2.0, 3.0], [0.0, 0.0, -100.0]], dtype=torch.bfloat16)
    got = chosen_logprobs(scores, [2, 2])
    expected = torch.log_softmax(scores.float(), dim=-1)
    assert got == pytest.approx([float(expected[0, 2]), float(expected[1, 2])])
    assert all(isinstance(value, float) for value in got)


def test_logprobs_are_not_part_of_a_replys_equality(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    from dataclasses import replace

    decoding = Decoding(strategy="greedy", max_new_tokens=4)
    (reply,) = generate(tiny_model, tiny_tokenizer, PROMPTS[:1], [], decoding)
    nudged = replace(reply, logprobs=tuple(v - 1e-7 for v in reply.logprobs))
    assert nudged == reply  # the same reply: tokens, text and pushes
