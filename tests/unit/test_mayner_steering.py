"""Our engine steers a set exactly as Mayner's code steers it, on the tiny model.

His RowIntervention, copied below unchanged with the code that builds its
rows, adds a set's values to its neurons at every position of the prompt and
the reply (role_steering.generate_rows); we push the set at dose 1 with a
constant schedule (decision 0019). In float32 both give the same scores
within 1e-5, alone and in a left-padded batch, and the same greedy replies,
for his top neurons and his random set, and the top neurons of his split.
"""

import importlib.util
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import PreTrainedTokenizerBase, Qwen3ForCausalLM

from pushlab.engine.decoding import Decoding
from pushlab.engine.generate import Generation, generate
from pushlab.engine.hooks import doses_active
from pushlab.engine.push import StepState
from pushlab.models.chat import chat_prompt_ids
from pushlab.schedules.compile import compile as compile_schedules
from pushlab.schedules.model import Schedule
from pushlab.schedules.shapes import constant
from pushlab.units import mayner
from pushlab.units.neuron_set import NeuronSet

# Mayner's code, copied unchanged with his MIT licence notice, is in
# tests/unit/mayner_reference.py; it is loaded by its path, as test modules
# import nothing from one another.
_SPEC = importlib.util.spec_from_file_location(
    "mayner_reference", Path(__file__).with_name("mayner_reference.py")
)
reference = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(reference)
# === The fixture in his data format =========================================

# The pirate's delta_a and p as test_mayner_sets.py draws them, the first of
# its two roles, so role 0: (layers, neurons) of the tiny model, in float32,
# as his split/arrays.npz holds them.
ROLE, ROLE_INDEX = "pirate", 0
_DRAWS = np.random.default_rng(20261112)
DELTA = _DRAWS.standard_normal((2, 2, 128)).astype(np.float32)[0]
P = _DRAWS.standard_normal((2, 2, 128)).astype(np.float32)[0]
# A made-up |v_r|, as there, but smaller: at dose 1 the set pushes 4 * 0.05 =
# 0.2, about the tiny model's residual size, so the push moves the scores
# without swamping them, and a push 0.01% too long moves them by 3e-5, past
# CLOSE. With that file's 3.0, a push of 12, it would take 0.1%.
VECTOR_NORM = 0.05
NORMS = [7.0, 10.0]  # residual sizes, which a set of its own size only checks
# His first two questions (role_steering.QUESTIONS), of two token lengths, so
# a batch of them is padded on the left, as his step 6 pads its batches.
QUESTIONS = ["Who are you?", "What is your name?"]
GREEDY = Decoding(strategy="greedy", max_new_tokens=6)
# His values and our amounts round to float32 apart in their last bits, which
# moves the scores by about 1e-7; a mistake moves them by 1e-5 and more.
CLOSE = {"rtol": 0.0, "atol": 1e-5}


def _written(table: pd.DataFrame, path: Path) -> pd.DataFrame:
    """table written to path as his steps write it, then read back."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return mayner.read_table(path)


def w_down(model: Qwen3ForCausalLM) -> Callable[[int], torch.Tensor]:
    """His w_down for model (steering_groups): a layer's down_proj weight."""
    return lambda layer: model.model.layers[layer].mlp.down_proj.weight


@pytest.fixture(params=["neurons", "random", "top-neurons"])
def steered(
    request: pytest.FixtureRequest, tmp_path: Path, tiny_model: Qwen3ForCausalLM
) -> tuple[NeuronSet, pd.DataFrame]:
    """Our set, and the rows (layer, neuron, value) his condition_values hands
    his RowIntervention to steer with it: his top neurons or his random set
    (draw 0) from sets.parquet, or the top neurons from top_neurons.parquet,
    which he steers as his top neurons."""
    sets = reference.neuron_sets(ROLE, ROLE_INDEX, DELTA, P)  # as his step 6 prepare
    sets = _written(sets, tmp_path / mayner.SETS_FILE)
    kind = "random" if request.param == "random" else "neurons"
    condition = {"role": ROLE, "kind": kind, "draw": reference.DRAW}
    target_norm = reference.STRENGTH * VECTOR_NORM
    _, rows = reference.condition_values(
        condition, sets, {}, w_down(tiny_model), target_norm
    )
    if request.param == "top-neurons":  # as his step 5 split writes it
        top = reference.split_top_neurons(DELTA * P, VECTOR_NORM, 1000)  # his K_MAX
        top = top.assign(role=ROLE, delta=DELTA[top.layer, top.neuron])
        top = _written(top, tmp_path / mayner.TOP_NEURONS_FILE)
        ours = mayner.top_neurons_set(
            top, ROLE, reference.K, VECTOR_NORM, model_name="tiny"
        )
    else:
        ours = mayner.steering_set(
            sets, ROLE, kind=kind, vector_norm=VECTOR_NORM, model_name="tiny"
        )
    return ours, rows


def _inputs(
    tokenizer: PreTrainedTokenizerBase, questions: list[str]
) -> dict[str, torch.Tensor]:
    """The questions in the chat template with thinking off, as one batch padded
    on the left, with the mask and positions our generate gives such a batch."""
    ids = [chat_prompt_ids(tokenizer, text, thinking=False) for text in questions]
    width = max(len(row) for row in ids)
    pad = tokenizer.pad_token_id
    mask = torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in ids])
    return {
        "input_ids": torch.tensor([[pad] * (width - len(row)) + row for row in ids]),
        "attention_mask": mask,
        "position_ids": (mask.cumsum(dim=-1) - 1).clamp(min=0),
    }


def _theirs(model: Qwen3ForCausalLM, rows: list, inputs: dict) -> torch.Tensor:
    """The logits of a pass of inputs under his RowIntervention of rows, which
    pushes nothing where every row is None."""
    with reference.RowIntervention(model, rows), torch.no_grad():
        return model(**inputs).logits


def _ours(
    model: Qwen3ForCausalLM, schedules: list[Schedule], inputs: dict
) -> torch.Tensor:
    """The logits of the prompt pass of inputs under our engine's hooks, row r
    following schedules[r], the padding left alone as generate leaves it."""
    table = compile_schedules(schedules, model, NORMS, max_new_tokens=1)
    state = StepState()
    with doses_active(model, table, state), torch.no_grad():
        state.at_prompt(padding=inputs["attention_mask"] == 0)
        return model(**inputs).logits


def _generated(
    model: Qwen3ForCausalLM,
    tokenizer: PreTrainedTokenizerBase,
    questions: list[str],
    schedules: list[Schedule],
) -> tuple[list[Generation], list[torch.Tensor]]:
    """Our greedy replies to the questions, row r following schedules[r], and
    each pass's scores for the next token: (rows, vocabulary)."""
    scores: list[torch.Tensor] = []
    hook = model.lm_head.register_forward_hook(
        lambda module, inputs, output: scores.append(output[:, -1].float().clone())
    )
    try:
        replies = generate(
            model, tokenizer, questions, [], GREEDY, NORMS, schedules=schedules
        )
    finally:
        hook.remove()
    return replies, scores


# === Our pushes against his RowIntervention, on the tiny model ==============


def test_a_prompt_pass_scores_as_under_his_row_intervention(
    steered: tuple[NeuronSet, pd.DataFrame],
    tiny_model: Qwen3ForCausalLM,
    tiny_tokenizer: PreTrainedTokenizerBase,
) -> None:
    neuron_set, rows = steered
    inputs = _inputs(tiny_tokenizer, QUESTIONS[:1])
    theirs = _theirs(tiny_model, [rows], inputs)
    ours = _ours(tiny_model, [constant(neuron_set, 1.0)], inputs)
    torch.testing.assert_close(ours, theirs, **CLOSE)
    # The push moved the scores far more, so there was something to compare.
    assert (theirs - _theirs(tiny_model, [None], inputs)).abs().max() > 0.1


def test_each_row_of_a_padded_batch_scores_as_his_row_does(
    steered: tuple[NeuronSet, pd.DataFrame],
    tiny_model: Qwen3ForCausalLM,
    tiny_tokenizer: PreTrainedTokenizerBase,
) -> None:
    neuron_set, rows = steered
    inputs = _inputs(tiny_tokenizer, QUESTIONS)  # the steered row is the padded one
    theirs = _theirs(tiny_model, [rows, None], inputs)
    ours = _ours(tiny_model, [constant(neuron_set, 1.0), Schedule("none")], inputs)
    # He pushes the padding too, which the mask keeps from every real position.
    real = inputs["attention_mask"] == 1
    for r in range(2):
        torch.testing.assert_close(ours[r][real[r]], theirs[r][real[r]], **CLOSE)
    plain = _theirs(tiny_model, [None, None], inputs)
    assert torch.equal(ours[1], plain[1])  # the unsteered row keeps every bit
    assert (theirs[0][real[0]] - plain[0][real[0]]).abs().max() > 0.1


@pytest.mark.parametrize("count", [1, 2], ids=["alone", "batch"])
def test_greedy_replies_are_the_ones_his_steered_generate_gives(
    steered: tuple[NeuronSet, pd.DataFrame],
    count: int,
    tiny_model: Qwen3ForCausalLM,
    tiny_tokenizer: PreTrainedTokenizerBase,
    hf_greedy: Callable[..., tuple[list[list[int]], list[torch.Tensor]]],
) -> None:
    # Hugging Face's generate() under his RowIntervention, as his generate_rows
    # runs it but greedy: the first row steered, the second (if any) not.
    neuron_set, rows = steered
    questions = QUESTIONS[:count]
    with reference.RowIntervention(tiny_model, [rows, None][:count]):
        expected, expected_scores = hf_greedy(
            tiny_model, tiny_tokenizer, questions, GREEDY.max_new_tokens
        )
    schedules = [constant(neuron_set, 1.0), Schedule("none")][:count]
    replies, scores = _generated(tiny_model, tiny_tokenizer, questions, schedules)
    assert [list(reply.token_ids) for reply in replies] == expected
    for r, reply in enumerate(replies):  # and the scores of every pass
        for k in range(len(reply.token_ids)):
            torch.testing.assert_close(scores[k][r], expected_scores[k][r], **CLOSE)
    # The push changed the reply, so there was something to compare.
    (unsteered,) = generate(tiny_model, tiny_tokenizer, questions[:1], [], GREEDY)
    assert unsteered.token_ids != replies[0].token_ids
