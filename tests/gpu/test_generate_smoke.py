"""GPU smoke test: the generation loop on the real Qwen3.5-4B, batched and alone.

It needs a GPU and the model downloaded by scripts/setup_pod.sh, whose last
step runs it with the other smoke tests (`pytest -m gpu -k smoke`). It loads
the model once, in bfloat16, and generates three greedy replies as one
left-padded batch. Then it runs each prompt alone, made to read its batched
reply's tokens, and compares the model's scores pass by pass.

In float32 on the CPU, a batched row and its prompt alone give the same
scores up to rounding (tests/unit/test_generate.py). In bfloat16 they
cannot be bit for bit the same: a batch changes the shape of the
arithmetic, so its rounding differs, and two tokens that score nearly the
same can swap. Reading the same tokens, the two runs face the same choice on
every pass, so their scores must stay within NEAR of each other. On the
CPU, rounding moved them by at most 0.28, and letting padding into a row
moved them by 5 or more (decision 0009).
"""

import itertools
import warnings

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


def _batched(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompts: list[str]
) -> tuple[list[Generation], list[torch.Tensor]]:
    """The batch's replies, and each pass's scores for the next token."""
    scores: list[torch.Tensor] = []
    hook = model.lm_head.register_forward_hook(
        lambda module, inputs, output: scores.append(output[:, -1].float())
    )
    try:
        replies = generate(model, tokenizer, prompts, [], GREEDY)
    finally:
        hook.remove()
    return replies, scores


def _alone_reading(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    tokens: tuple[int, ...],
) -> list[torch.Tensor]:
    """Each pass's scores for the prompt alone, made to read the given reply."""
    scores: list[torch.Tensor] = []
    passes = itertools.count()

    def read(module: object, inputs: object, output: torch.Tensor) -> torch.Tensor:
        scores.append(output[:, -1].float())  # the scores it would choose by
        forced = torch.zeros_like(output)
        forced[..., tokens[next(passes)]] = 1.0  # the token it is made to choose
        return forced

    hook = model.lm_head.register_forward_hook(read)
    try:
        (reply,) = generate(model, tokenizer, [prompt], [], GREEDY)
    finally:
        hook.remove()
    assert reply.token_ids == tokens  # every token read, and the same end
    return scores


def _three_prompts(tokenizer: PreTrainedTokenizerBase) -> list[str]:
    """The calibration set's shortest, middle and longest prompts.

    Two of the three rows of their batch are padded.
    """

    def length(prompt: str) -> int:
        return len(chat_prompt_ids(tokenizer, prompt, thinking=False))

    by_length = sorted(read_prompt_set(PROMPTS_FILE).prompts, key=length)
    prompts = [by_length[0], by_length[len(by_length) // 2], by_length[-1]]
    assert len({length(prompt) for prompt in prompts}) == 3
    return prompts


def _compared(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompts: list[str]
) -> list[dict]:
    """For each row of the batch: its reply, and its gap to the prompt alone.

    gaps[k] is the largest difference between the two runs' scores on pass
    k. others lists the passes where the prompt alone would have chosen
    another token than the batch did: near ties that rounding decided.
    """
    batched, batched_scores = _batched(model, tokenizer, prompts)
    rows = []
    for row, prompt in enumerate(prompts):
        tokens = batched[row].token_ids
        alone = _alone_reading(model, tokenizer, prompt, tokens)
        gaps = [
            float((batched_scores[k][row] - alone[k][0]).abs().max())
            for k in range(len(tokens))
        ]
        others = [
            k for k in range(len(tokens)) if int(alone[k][0].argmax()) != tokens[k]
        ]
        rows.append({"reply": batched[row], "gaps": gaps, "others": others})
    return rows


@pytest.mark.gpu
@pytest.mark.slow
def test_each_row_of_a_padded_batch_matches_its_prompt_alone() -> None:
    assert torch.cuda.is_available(), "torch sees no GPU; see test_cuda_is_available"
    loaded = load_model("qwen3.5-4b", torch.bfloat16, device="cuda")
    prompts = _three_prompts(loaded.tokenizer)
    rows = _compared(loaded.model, loaded.tokenizer, prompts)
    for row, result in enumerate(rows):
        gaps = result["gaps"]
        # pytest shows this only if a check below fails.
        print(
            f"row {row}: {len(gaps)} reply tokens; scores differ by at most "
            f"{max(gaps):.3f}, at pass {gaps.index(max(gaps))}; alone would have "
            f"chosen another token at passes {result['others']}\n"
            f"  {result['reply'].text!r}"
        )
    for row, result in enumerate(rows):
        gaps = result["gaps"]
        assert max(gaps) <= NEAR, (
            f"row {row}'s scores in the batch differ from its prompt's alone by "
            f"{max(gaps):.3f} at pass {gaps.index(max(gaps))}; at most {NEAR} "
            "is allowed"
        )
    largest = max(max(result["gaps"]) for result in rows)
    if largest > NEAR / 2:
        warnings.warn(
            f"a batched row's scores differed from its prompt's alone by "
            f"{largest:.3f}, more than half the {NEAR} allowed; please report it",
            stacklevel=1,
        )
