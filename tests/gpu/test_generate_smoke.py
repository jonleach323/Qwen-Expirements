"""GPU smoke test: the generation loop on the real Qwen3.5-4B, batched and alone.

It needs a GPU and the model downloaded by scripts/setup_pod.sh, whose last
step runs it with the other smoke tests (`pytest -m gpu -k smoke`). It loads
the model once, in bfloat16, and generates three greedy replies: each prompt
alone, then the three as one left-padded batch.

In float32 on the CPU, a batched row and its prompt alone give the same
tokens (tests/unit/test_generate.py). In bfloat16 they cannot be bit for bit
the same: a batch changes the shape of the arithmetic, so its rounding
differs, and two tokens that score nearly the same can swap. So this test
compares the model's scores, pass by pass: up to the pass where the two
replies part, if they do, the batched row's scores must stay within NEAR of
the scores alone. On the CPU, rounding moved them by at most 0.28, and
letting padding into a row moved them by 3.5 or more (decision 0009). The
unit tests check the positions and masks themselves.
"""

import pytest
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from pushlab.engine.decoding import Decoding
from pushlab.engine.generate import Generation, generate
from pushlab.models.chat import chat_prompt_ids
from pushlab.models.load import load_model
from pushlab.prompts import read_prompt_set
from pushlab.units.calibrate import PROMPTS_FILE

NEAR = 1.0  # logits; see docs/decisions/0009-generation-loop.md
GREEDY = Decoding(strategy="greedy", max_new_tokens=32)


def _with_scores(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompts: list[str]
) -> tuple[list[Generation], list[torch.Tensor]]:
    """The replies, and each pass's scores for the next token: (rows, vocabulary)."""
    scores: list[torch.Tensor] = []
    hook = model.lm_head.register_forward_hook(
        lambda module, inputs, output: scores.append(output[:, -1].float())
    )
    try:
        replies = generate(model, tokenizer, prompts, [], GREEDY)
    finally:
        hook.remove()
    return replies, scores


@pytest.mark.gpu
@pytest.mark.slow
def test_each_row_of_a_padded_batch_matches_its_prompt_alone() -> None:
    assert torch.cuda.is_available(), "torch sees no GPU; see test_cuda_is_available"
    loaded = load_model("qwen3.5-4b", torch.bfloat16, device="cuda")
    model, tokenizer = loaded.model, loaded.tokenizer
    # The calibration set's shortest, middle and longest prompts, so that
    # two of the three rows are padded.
    prompt_set = read_prompt_set(PROMPTS_FILE)

    def length(prompt: str) -> int:
        return len(chat_prompt_ids(tokenizer, prompt, thinking=False))

    by_length = sorted(prompt_set.prompts, key=length)
    prompts = [by_length[0], by_length[len(by_length) // 2], by_length[-1]]
    assert len({length(prompt) for prompt in prompts}) == 3

    alone = [_with_scores(model, tokenizer, [prompt]) for prompt in prompts]
    batched, batched_scores = _with_scores(model, tokenizer, prompts)
    identical = 0
    for row, ((reply,), scores) in enumerate(alone):
        mine, theirs = batched[row].token_ids, reply.token_ids
        parted = next(
            (k for k, (a, b) in enumerate(zip(mine, theirs, strict=False)) if a != b),
            min(len(mine), len(theirs)),
        )
        identical += mine == theirs
        # Both runs read the same tokens up to the pass that chose token
        # `parted`, so their scores can be compared up to there.
        compared = range(min(parted + 1, len(mine), len(theirs)))
        gaps = [
            float((batched_scores[k][row] - scores[k][0]).abs().max()) for k in compared
        ]
        # pytest shows this only if the check below fails.
        print(
            f"row {row}: {length(prompts[row])} prompt tokens; replies agree on "
            f"{parted} of {len(theirs)} tokens; scores differ by at most "
            f"{max(gaps):.3f}\n  alone:   {reply.text!r}\n  batched: "
            f"{batched[row].text!r}"
        )
        assert max(gaps) <= NEAR, (
            f"row {row}'s scores in the batch differ from its prompt's alone by "
            f"{max(gaps):.3f} (at most {NEAR} allowed), at pass {gaps.index(max(gaps))}"
        )
    print(f"{identical} of {len(prompts)} replies are identical token for token")
