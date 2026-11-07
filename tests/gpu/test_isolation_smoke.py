"""GPU smoke test: four rows of one batch, each with its own schedule, on Qwen3.5-4B.

It needs a GPU, the model downloaded by scripts/setup_pod.sh and the
calibration in calibration/, and the pod's last setup step runs it with the
other smoke tests (`pytest -m gpu -k smoke`). It loads the model once, in
bfloat16, and replies to four smoke prompts of different lengths in one
left-padded batch: one row with no push, one with a pulse, one with a handoff
and one with an alternation. S1 runs prompts as one padded batch unless told
to group them by length, so this does too. Then it runs each prompt alone
with its own schedule, made to read its batched reply's tokens, and compares
the model's scores pass by pass, as tests/gpu/test_generate_smoke.py does.

In bfloat16 a batch changes the shape of the arithmetic, so its rounding
differs from a prompt alone, and two tokens that score nearly the same can
swap. Reading the same tokens, the two runs face the same choice on every
pass, so their scores must stay within NEAR of each other, the S1 limit; a
push that leaked into another row would show there as a gap far above
rounding. The units and amounts behind every token must be the same exactly.
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
from pushlab.prompts import PROMPTS_DIR, read_prompt_set
from pushlab.schedules.model import Schedule
from pushlab.schedules.shapes import alternate, handoff, pulse
from pushlab.units import calibrate
from pushlab.units.neuron import NeuronUnit

NEAR = 1.0  # logits; see docs/decisions/0009-generation-loop.md
GREEDY = Decoding(strategy="greedy", max_new_tokens=32)
HOSTILE = NeuronUnit.parse("qwen3.5-4b:L14:N8949")  # the recipe's pair
WARM = NeuronUnit.parse("qwen3.5-4b:L13:N7956")
SCHEDULES = [
    Schedule("none"),
    pulse(HOSTILE, 1.0, 0, 16),
    handoff(HOSTILE, WARM, 1.0, -1.0, 16),
    alternate(HOSTILE, WARM, 1.0, -1.0, 4, include_prompt=True),
]


def _four_prompts(tokenizer: PreTrainedTokenizerBase) -> list[str]:
    """Four smoke prompts, shortest first, so the batch is padded."""

    def length(prompt: str) -> int:
        return len(chat_prompt_ids(tokenizer, prompt, thinking=False))

    prompts = sorted(read_prompt_set(PROMPTS_DIR / "smoke.jsonl").prompts, key=length)
    four = prompts[:4]
    assert len({length(prompt) for prompt in four}) > 1
    return four


def _scored(model: PreTrainedModel, run: object) -> tuple[list[Generation], list]:
    scores: list[torch.Tensor] = []
    hook = model.lm_head.register_forward_hook(
        lambda module, inputs, output: scores.append(output[:, -1].float())
    )
    try:
        replies = run()  # type: ignore[operator]
    finally:
        hook.remove()
    return replies, scores


def _alone_reading(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    schedule: Schedule,
    norms: list[float],
    tokens: tuple[int, ...],
) -> tuple[Generation, list[torch.Tensor]]:
    """The prompt alone with its schedule, made to read the given reply."""
    scores: list[torch.Tensor] = []
    passes = itertools.count()

    def read(module: object, inputs: object, output: torch.Tensor) -> torch.Tensor:
        scores.append(output[:, -1].float())  # the scores it would choose by
        forced = torch.zeros_like(output)
        forced[..., tokens[next(passes)]] = 1.0  # the token it is made to choose
        return forced

    hook = model.lm_head.register_forward_hook(read)
    try:
        (reply,) = generate(
            model, tokenizer, [prompt], [], GREEDY, norms, schedules=[schedule]
        )
    finally:
        hook.remove()
    assert reply.token_ids == tokens  # every token read, and the same end
    return reply, scores


@pytest.mark.gpu
@pytest.mark.slow
def test_each_row_with_its_own_schedule_matches_its_prompt_alone() -> None:
    assert torch.cuda.is_available(), "torch sees no GPU; see test_cuda_is_available"
    loaded = load_model("qwen3.5-4b", torch.bfloat16, device="cuda")
    norms = list(calibrate.load_calibration("qwen3.5-4b", loaded.revision).norms)
    model, tokenizer = loaded.model, loaded.tokenizer
    prompts = _four_prompts(tokenizer)
    batched, batched_scores = _scored(
        model,
        lambda: generate(
            model, tokenizer, prompts, [], GREEDY, norms, schedules=SCHEDULES
        ),
    )
    results = []
    for row, (prompt, schedule) in enumerate(zip(prompts, SCHEDULES, strict=True)):
        tokens = batched[row].token_ids
        alone, scores = _alone_reading(
            model, tokenizer, prompt, schedule, norms, tokens
        )
        gaps = [
            float((batched_scores[k][row] - scores[k][0]).abs().max())
            for k in range(len(tokens))
        ]
        others = [
            k for k in range(len(tokens)) if int(scores[k][0].argmax()) != tokens[k]
        ]
        results.append((schedule.name, gaps, others, batched[row], alone))
    for name, gaps, others, reply, _ in results:
        # pytest shows this only if a check below fails.
        print(
            f"{name}: {len(gaps)} reply tokens; scores differ by at most "
            f"{max(gaps):.3f}, at pass {gaps.index(max(gaps))}; alone would have "
            f"chosen another token at passes {others}\n  {reply.text!r}"
        )
    for name, gaps, _, reply, alone in results:
        # The same units and amounts behind every token, in the batch and alone.
        assert reply.amounts == alone.amounts, name
        assert max(gaps) <= NEAR, (
            f"the {name} row's scores in the batch differ from its prompt's alone "
            f"by {max(gaps):.3f} at pass {gaps.index(max(gaps))}; at most {NEAR} "
            "is allowed"
        )
    largest = max(max(gaps) for _, gaps, _, _, _ in results)
    if largest > NEAR / 2:
        warnings.warn(
            f"a batched row's scores differed from its prompt's alone by "
            f"{largest:.3f}, more than half the {NEAR} allowed; please report it",
            stacklevel=1,
        )
