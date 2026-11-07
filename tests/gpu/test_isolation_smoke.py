"""GPU smoke test: four rows of one batch, each with its own schedule, on Qwen3.5-4B.

It needs a GPU, the model downloaded by scripts/setup_pod.sh and the model
revision's calibration in calibration/, and the pod's setup step 7 runs it
with the other smoke tests (`pytest -m gpu -k smoke`). It loads the model
once, in bfloat16, and replies to four calibration prompts of four lengths
in one left-padded batch: one row with no push, one with a pulse, one with a
handoff and one with an alternation of the recipe's pair. S1 runs prompts as
one padded batch unless told to group them by length, so this does too.

- A probe on layers 13 and 14 checks that every row got exactly its own
  amounts, bit for bit, on every pass and nowhere else: the no-push row is
  never touched, so a push that leaked would show at once.
- Then each prompt runs alone with its own schedule, made to read its
  batched reply's tokens, and the model's scores are compared pass by pass,
  as tests/gpu/test_generate_smoke.py does. In bfloat16 a batch changes the
  shape of the arithmetic, so its rounding differs from a prompt alone;
  reading the same tokens, the two runs must stay within NEAR, S1's limit.

It writes its numbers to /workspace/logs/isolation-<time>.json when that
folder exists, and prints them with -s.
"""

import itertools
import json
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import pytest
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from pushlab.engine.decoding import Decoding
from pushlab.engine.generate import Generation, generate
from pushlab.models.chat import chat_prompt_ids
from pushlab.models.discovery import find_mlp_modules
from pushlab.models.load import load_model
from pushlab.prompts import read_prompt_set
from pushlab.schedules.model import Schedule
from pushlab.schedules.shapes import alternate, handoff, pulse
from pushlab.units import calibrate
from pushlab.units.calibrate import PROMPTS_FILE
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
LOGS = Path("/workspace/logs")


def _four_prompts(tokenizer: PreTrainedTokenizerBase) -> tuple[list[str], list[int]]:
    """Four calibration prompts of four lengths, shortest to longest, so three
    rows of the batch are padded by different amounts."""

    def length(prompt: str) -> int:
        return len(chat_prompt_ids(tokenizer, prompt, thinking=False))

    by_length: dict[int, str] = {}
    for prompt in read_prompt_set(PROMPTS_FILE).prompts:
        by_length.setdefault(length(prompt), prompt)
    lengths = sorted(by_length)
    assert len(lengths) >= 4, f"the calibration set has only lengths {lengths}"
    picked = [lengths[round(i * (len(lengths) - 1) / 3)] for i in range(4)]
    return [by_length[n] for n in picked], picked


@contextmanager
def _probe(model: PreTrainedModel, layers: list[int]) -> Iterator[dict[int, list]]:
    """down_proj's input at each layer on every pass: [before the push, after it]."""
    mlps = dict(find_mlp_modules(model))
    seen: dict[int, list[list[torch.Tensor]]] = {layer: [] for layer in layers}
    handles = []
    for layer in layers:
        down_proj = mlps[layer].down_proj
        # Registered before generate's hooks, so it sees the value unpushed.
        handles.append(
            down_proj.register_forward_pre_hook(
                lambda m, args, layer=layer: seen[layer].append([args[0].clone()])
            )
        )
        handles.append(
            down_proj.register_forward_hook(
                lambda m, args, out, layer=layer: seen[layer][-1].append(
                    args[0].clone()
                )
            )
        )
    try:
        yield seen
    finally:
        for handle in handles:
            handle.remove()


def _pushed_rows(
    seen: dict[int, list], replies: list[Generation], pads: list[int]
) -> dict[int, list[int]]:
    """Check every probed pass bit for bit against the amounts each reply
    records; return, per layer, the rows pushed there."""
    pushed: dict[int, set[int]] = {layer: set() for layer in seen}
    for layer, passes in seen.items():
        for p, (before, after) in enumerate(passes):
            expected = before.clone()
            for row, reply in enumerate(replies):
                if p >= len(reply.token_ids):
                    continue  # finished: padding now
                first = pads[row] if p == 0 else 0
                for unit, value in reply.amounts[p]:
                    if unit.layer == layer:
                        column = before[row, first:, unit.index].float()
                        added = torch.tensor(value, dtype=torch.float32).to(
                            column.device
                        )
                        expected[row, first:, unit.index] = (column + added).to(
                            before.dtype
                        )
                        pushed[layer].add(row)
            assert torch.equal(after, expected), f"layer {layer}, pass {p}"
    return {layer: sorted(rows) for layer, rows in pushed.items()}


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
    try:
        norms = list(calibrate.load_calibration("qwen3.5-4b", loaded.revision).norms)
    except FileNotFoundError as error:
        pytest.skip(f"{error}; run `uv run pushlab calibrate --model qwen3.5-4b`")
    model, tokenizer = loaded.model, loaded.tokenizer
    prompts, lengths = _four_prompts(tokenizer)
    pads = [max(lengths) - length for length in lengths]
    with _probe(model, [WARM.layer, HOSTILE.layer]) as seen:
        batched, batched_scores = _scored(
            model,
            lambda: generate(
                model, tokenizer, prompts, [], GREEDY, norms, schedules=SCHEDULES
            ),
        )
    pushed = _pushed_rows(seen, batched, pads)
    results = []
    for row, (prompt, schedule) in enumerate(zip(prompts, SCHEDULES, strict=True)):
        tokens = batched[row].token_ids
        alone, scores = _alone_reading(
            model, tokenizer, prompt, schedule, norms, tokens
        )
        assert alone.amounts == batched[row].amounts, schedule.name
        gaps = [
            float((batched_scores[k][row] - scores[k][0]).abs().max())
            for k in range(len(tokens))
        ]
        others = [
            k for k in range(len(tokens)) if int(scores[k][0].argmax()) != tokens[k]
        ]
        results.append(
            {
                "schedule": schedule.name,
                "prompt_tokens": lengths[row],
                "reply_tokens": len(tokens),
                "largest_gap": max(gaps),
                "at_pass": gaps.index(max(gaps)),
                "alone_would_choose_another_at": others,
            }
        )
        # pytest shows these lines only with -s, or when a check below fails.
        print(f"{results[-1]}\n  {batched[row].text!r}")
    print(f"rows pushed, by layer: {pushed}")
    if LOGS.is_dir():
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        summary = {"rows_pushed_by_layer": pushed, "rows": results, "near": NEAR}
        (LOGS / f"isolation-{stamp}.json").write_text(json.dumps(summary, indent=2))
    # The no-push row is never pushed; the others are, where their schedules say.
    assert pushed == {WARM.layer: [2, 3], HOSTILE.layer: [1, 2, 3]}
    for result in results:
        assert result["largest_gap"] <= NEAR, (
            f"the {result['schedule']} row's scores in the batch differ from its "
            f"prompt's alone by {result['largest_gap']:.3f} at pass "
            f"{result['at_pass']}; at most {NEAR} is allowed"
        )
    largest = max(result["largest_gap"] for result in results)
    if largest > NEAR / 2:
        warnings.warn(
            f"a batched row's scores differed from its prompt's alone by "
            f"{largest:.3f}, more than half the {NEAR} allowed; please report it",
            stacklevel=1,
        )
