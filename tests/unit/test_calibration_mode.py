"""The per_prompt calibration mode: each prompt's own residual sizes.

Its sizes are checked against medians worked by hand on the tiny models:
the test's own hooks record each decoder layer's output for the prompt in
the chat template with thinking off, and statistics.median takes each
layer's median token length, the first token left out (decision 0019).
"""

import doctest
import statistics
from collections.abc import Callable

import numpy as np
import pytest
import torch
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from pushlab.models.chat import chat_prompt_ids
from pushlab.units import calibration_mode
from pushlab.units.calibrate import calibrate_norms
from pushlab.units.calibration_mode import (
    CALIBRATION_MODES,
    DEFAULT_MODE,
    norms_for,
    per_prompt_norms,
    prompt_norms,
)
from pushlab.units.dose import PerPromptNorms

# The Neuron Persona Library's own prompt: 37 tokens in the template, so 36
# are counted, an even count, whose median is the mean of the middle two.
PROMPT = "Who are you?"
OTHER = "Describe the sea at night in three short sentences."  # 76 tokens
MODELS = ["tiny_model", "tiny_vl_model"]
FIXED = [7.0, 10.0]  # made-up fixed sizes for tiny's two layers


def _lengths_by_hand(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompt: str
) -> list[list[float]]:
    """Each decoder layer's token lengths for prompt, every token's, in order.

    Hooks on the decoder layers record their outputs for the prompt in the
    chat template with thinking off, and each token's length is taken in
    float32. The hooks are removed again before this returns.
    """
    layers = model.get_decoder().layers  # model.model.layers in tiny
    recorded: dict[int, torch.Tensor] = {}

    def recorder(layer: int) -> Callable[[nn.Module, object, object], None]:
        def record(module: nn.Module, args: object, output: object) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            recorded[layer] = hidden[0].detach().clone()  # the batch's only row

        return record

    handles = [
        block.register_forward_hook(recorder(i)) for i, block in enumerate(layers)
    ]
    try:
        ids = chat_prompt_ids(tokenizer, prompt, thinking=False)
        with torch.no_grad():
            model(input_ids=torch.tensor([ids]), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return [
        torch.linalg.vector_norm(recorded[layer].float(), dim=-1).tolist()
        for layer in range(len(layers))
    ]


def _medians_by_hand(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompt: str
) -> tuple[float, ...]:
    """Each layer's statistics.median of the prompt's token lengths, but the first."""
    lengths = _lengths_by_hand(model, tokenizer, prompt)
    return tuple(statistics.median(per_layer[1:]) for per_layer in lengths)


def _count_passes(model: nn.Module, monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """From now on, the token count of each forward pass of model, in order."""
    passes: list[int] = []
    forward = model.forward

    def counted(*args: object, **kwargs: object) -> object:
        ids = kwargs["input_ids"] if "input_ids" in kwargs else args[0]
        passes.append(ids.shape[-1])  # type: ignore[attr-defined]
        return forward(*args, **kwargs)

    monkeypatch.setattr(model, "forward", counted)
    return passes


@pytest.mark.parametrize(
    ("name", "dtype"),
    [
        ("tiny_model", torch.float32),
        ("tiny_vl_model", torch.float32),
        ("tiny_model", torch.bfloat16),  # its lengths are still taken in float32
    ],
    ids=["tiny", "tiny-vl", "tiny in bfloat16"],
)
def test_a_prompts_sizes_are_the_medians_of_its_own_tokens_worked_by_hand(
    name: str,
    dtype: torch.dtype,
    request: pytest.FixtureRequest,
    tiny_tokenizer: PreTrainedTokenizerBase,
) -> None:
    model = request.getfixturevalue(name).to(dtype)
    by_hand = _medians_by_hand(model, tiny_tokenizer, PROMPT)
    assert len(by_hand) == len(model.get_decoder().layers)
    sizes = prompt_norms(model, tiny_tokenizer, PROMPT)
    assert type(sizes) is tuple and all(type(size) is float for size in sizes)
    assert sizes == by_hand  # exactly, bit for bit
    assert per_prompt_norms(model, tiny_tokenizer, [PROMPT]).rows == (by_hand,)


def test_an_even_count_takes_the_mean_of_the_middle_two(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    # Decision 0019: calibrate's median, not torch.median's lower middle
    # value, which Mayner's copy of the library's protocol takes.
    sizes = prompt_norms(tiny_model, tiny_tokenizer, PROMPT)
    lengths = _lengths_by_hand(tiny_model, tiny_tokenizer, PROMPT)
    for size, per_layer in zip(sizes, lengths, strict=True):
        assert len(per_layer) == 37  # the template's tokens: 36 once the first goes
        low, high = sorted(per_layer[1:])[17:19]  # the middle two of 36
        assert size == (low + high) / 2
        assert size != low  # torch.median's
        assert size != statistics.median(per_layer)  # with the first token kept


@pytest.mark.parametrize("name", MODELS)
def test_different_prompts_get_different_sizes_where_fixed_pools_them(
    name: str, request: pytest.FixtureRequest, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    model = request.getfixturevalue(name)
    first, second = per_prompt_norms(model, tiny_tokenizer, [PROMPT, OTHER]).rows
    # The fixed calibration takes one median over every prompt's tokens.
    pooled = tuple(calibrate_norms(model, tiny_tokenizer, [PROMPT, OTHER]))
    assert first != second
    assert pooled != first and pooled != second  # tuples all, so numbers compared
    # Over one prompt alone, the two modes agree.
    for prompt, row in [(PROMPT, first), (OTHER, second)]:
        assert row == tuple(calibrate_norms(model, tiny_tokenizer, [prompt]))
        assert per_prompt_norms(model, tiny_tokenizer, [prompt]).rows == (row,)


def test_row_i_belongs_to_prompt_i_whatever_else_is_measured(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    alone = tuple(prompt_norms(tiny_model, tiny_tokenizer, p) for p in (PROMPT, OTHER))
    norms = per_prompt_norms(tiny_model, tiny_tokenizer, [PROMPT, OTHER])
    assert norms.rows == alone
    reordered = per_prompt_norms(tiny_model, tiny_tokenizer, (OTHER, PROMPT))
    assert reordered.rows == alone[::-1]


def test_a_prompt_given_twice_gets_equal_rows_and_is_measured_once(
    tiny_model: PreTrainedModel,
    tiny_tokenizer: PreTrainedTokenizerBase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    passes = _count_passes(tiny_model, monkeypatch)
    norms = per_prompt_norms(tiny_model, tiny_tokenizer, [PROMPT, OTHER, PROMPT])
    assert len(norms) == 3
    assert norms.rows[0] == norms.rows[2] != norms.rows[1]
    tokens = [
        len(chat_prompt_ids(tiny_tokenizer, p, thinking=False)) for p in [PROMPT, OTHER]
    ]
    assert sorted(passes) == sorted(tokens) == [37, 76]  # one pass for each prompt


def test_one_text_is_not_taken_for_a_list_of_prompts(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    # Text is a sequence of characters: each would be measured as a prompt.
    with pytest.raises(ValueError, match="list of prompts, not one text"):
        per_prompt_norms(tiny_model, tiny_tokenizer, PROMPT)


@pytest.mark.parametrize("prompts", [[], ()], ids=["list", "tuple"])
def test_at_least_one_prompt_is_needed(
    tiny_model: PreTrainedModel,
    tiny_tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
) -> None:
    with pytest.raises(ValueError, match="needs at least one prompt"):
        per_prompt_norms(tiny_model, tiny_tokenizer, prompts)
    with pytest.raises(ValueError, match="needs at least one prompt"):
        norms_for("per_prompt", tiny_model, tiny_tokenizer, prompts)


@pytest.mark.parametrize(
    "bad", [5, None, b"Hi", ["Hi"]], ids=["int", "None", "bytes", "list"]
)
def test_a_prompt_that_is_not_text_is_refused_before_any_pass(
    tiny_model: PreTrainedModel,
    tiny_tokenizer: PreTrainedTokenizerBase,
    monkeypatch: pytest.MonkeyPatch,
    bad: object,
) -> None:
    passes = _count_passes(tiny_model, monkeypatch)
    with pytest.raises(ValueError, match="each prompt must be text"):
        per_prompt_norms(tiny_model, tiny_tokenizer, [PROMPT, OTHER, bad])  # type: ignore[list-item]
    assert passes == []  # not even the good prompts before it were measured
    with pytest.raises(ValueError, match="prompt must be one text"):
        prompt_norms(tiny_model, tiny_tokenizer, bad)  # type: ignore[arg-type]
    assert passes == []


def test_a_model_with_a_push_active_is_refused(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    # A neuron push adds to an MLP value through a hook on down_proj. Sizes
    # measured with it would be the pushed model's, not the model's own.
    down_proj = tiny_model.model.layers[1].mlp.down_proj
    handle = down_proj.register_forward_pre_hook(lambda module, args: None)
    for measure in (
        lambda: prompt_norms(tiny_model, tiny_tokenizer, PROMPT),
        lambda: per_prompt_norms(tiny_model, tiny_tokenizer, [PROMPT]),
        lambda: norms_for("per_prompt", tiny_model, tiny_tokenizer, [PROMPT]),
    ):
        with pytest.raises(ValueError, match=r"no pushes active.*layers\.1\.mlp"):
            measure()
    handle.remove()
    assert len(prompt_norms(tiny_model, tiny_tokenizer, PROMPT)) == 2  # now allowed


def test_a_prompt_of_one_token_is_refused(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    # Its first token left out, it has no token left to measure. In a
    # template that adds nothing, "A" is one token.
    tiny_tokenizer.chat_template = (
        "{% for message in messages %}{{ message['content'] }}{% endfor %}"
    )
    assert len(chat_prompt_ids(tiny_tokenizer, "A", thinking=False)) == 1
    with pytest.raises(ValueError, match="'A' has only 1 token"):
        per_prompt_norms(tiny_model, tiny_tokenizer, ["A"])
    with pytest.raises(ValueError, match="'A' has only 1 token"):
        prompt_norms(tiny_model, tiny_tokenizer, "A")
    assert len(prompt_norms(tiny_model, tiny_tokenizer, "AB")) == 2  # 1 token left


def test_the_fixed_mode_returns_the_given_sizes_and_runs_nothing(
    tiny_model: PreTrainedModel,
    tiny_tokenizer: PreTrainedTokenizerBase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    passes = _count_passes(tiny_model, monkeypatch)
    for fixed in (FIXED, tuple(FIXED)):
        norms = norms_for("fixed", tiny_model, tiny_tokenizer, [PROMPT, OTHER], fixed)
        assert type(norms) is tuple and norms == (7.0, 10.0)
    assert passes == []
    with pytest.raises(ValueError, match="fixed calibration needs the model's sizes"):
        norms_for("fixed", tiny_model, tiny_tokenizer, [PROMPT])


def test_the_per_prompt_mode_gives_per_prompt_norms(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    expected = per_prompt_norms(tiny_model, tiny_tokenizer, [PROMPT, OTHER])
    for fixed in (None, FIXED):  # fixed sizes given are not used
        norms = norms_for(
            "per_prompt", tiny_model, tiny_tokenizer, [PROMPT, OTHER], fixed
        )
        assert isinstance(norms, PerPromptNorms)
        assert norms == expected


@pytest.mark.parametrize("mode", ["per-prompt", "Fixed", "auto", "", None])
def test_an_unknown_mode_is_refused_naming_the_modes(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase, mode: object
) -> None:
    with pytest.raises(
        ValueError,
        match=r"unknown calibration mode .*; the modes are fixed, per_prompt$",
    ):
        norms_for(mode, tiny_model, tiny_tokenizer, [PROMPT], FIXED)  # type: ignore[arg-type]


def test_the_fixed_calibration_stays_the_default() -> None:
    assert CALIBRATION_MODES == ("fixed", "per_prompt")
    assert DEFAULT_MODE == "fixed" and DEFAULT_MODE in CALIBRATION_MODES


def test_rows_are_kept_as_tuples_of_floats() -> None:
    norms = PerPromptNorms([[7, 10.0], (np.float32(6.5), 11)])
    assert norms.rows == ((7.0, 10.0), (6.5, 11.0))
    assert type(norms.rows) is tuple and all(type(row) is tuple for row in norms.rows)
    assert all(type(size) is float for row in norms.rows for size in row)
    # The same sizes make equal norms, however they were given.
    assert norms == PerPromptNorms(((7.0, 10.0), (6.5, 11.0)))
    assert len(norms) == 2 and len(PerPromptNorms([FIXED] * 5)) == 5  # one per row


@pytest.mark.parametrize(
    "rows",
    [[], (), None, "ab", 7.0, np.array([FIXED]), torch.tensor([FIXED])],
    ids=["empty list", "empty tuple", "None", "str", "float", "ndarray", "tensor"],
)
def test_rows_must_be_a_non_empty_list(rows: object) -> None:
    with pytest.raises(ValueError, match="rows must be a non-empty list"):
        PerPromptNorms(rows)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "row",
    [[], (), None, "ab", 7.0, np.array(FIXED)],
    ids=["empty list", "empty tuple", "None", "str", "float", "ndarray"],
)
def test_each_row_must_be_a_non_empty_list_of_sizes(row: object) -> None:
    with pytest.raises(ValueError, match=r"rows\[1\] must be a list of sizes"):
        PerPromptNorms([FIXED, row])  # type: ignore[list-item]


@pytest.mark.parametrize("position", [0, 1])
@pytest.mark.parametrize(
    "bad", [0.0, -7.0, float("nan"), float("inf"), float("-inf"), True, "7", None]
)
def test_every_size_must_be_a_finite_number_above_0(bad: object, position: int) -> None:
    row: list[object] = list(FIXED)
    row[position] = bad
    with pytest.raises(
        ValueError, match=rf"rows\[1\]\[{position}\] must be a finite number above 0"
    ):
        PerPromptNorms([FIXED, row])  # type: ignore[list-item]


@pytest.mark.parametrize("rows", [[FIXED, [7.0, 10.0, 12.0]], [[7.0], FIXED]])
def test_every_row_needs_as_many_sizes(rows: list[list[float]]) -> None:
    with pytest.raises(ValueError, match="same number of layers"):
        PerPromptNorms(rows)
    with pytest.raises(ValueError, match=rf"rows\[1\] has {len(rows[1])} sizes but"):
        PerPromptNorms(rows)


def test_an_int_too_large_for_a_float_is_refused_as_a_size() -> None:
    with pytest.raises(ValueError, match=r"rows\[0\]\[0\] must be a finite number"):
        PerPromptNorms([[10**400, 1.0]])


@pytest.mark.parametrize("fixed", ["7.0", 7.0, PerPromptNorms([FIXED])])
def test_the_fixed_mode_refuses_sizes_that_are_not_a_list(fixed: object) -> None:
    with pytest.raises(ValueError, match="must be a list of numbers, one per layer"):
        norms_for("fixed", None, None, [PROMPT], fixed)  # type: ignore[arg-type]


def test_docstring_examples_run() -> None:
    result = doctest.testmod(calibration_mode)
    assert result.attempted > 0
    assert result.failed == 0
