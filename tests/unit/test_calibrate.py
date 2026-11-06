"""Calibration: each layer's median residual size, measured and saved per model.

The measurements run on the tiny models through the chat template, and the
reference medians come from transformers' own hidden states.
"""

import doctest
import hashlib
import statistics
from collections.abc import Callable

import pytest
import torch
from torch import nn
from torch.utils.hooks import RemovableHandle
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from pushlab import PROJECT_ROOT
from pushlab.models.chat import chat_prompt_ids
from pushlab.models.discovery import find_decoder_layers
from pushlab.prompts import read_prompt_set
from pushlab.units import calibrate
from pushlab.units.calibrate import (
    PROMPTS_FILE,
    calibrate_norms,
    layer_outputs,
    median_norms,
)

PROMPTS = ["What is a good name for a cat?", "How do I boil an egg?", "Why is it hot?"]
MODELS = ["tiny_model", "tiny_vl_model"]


def _hidden_state_medians(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompts: list[str]
) -> list[float]:
    """Medians of transformers' hidden states, each prompt's first token left out.

    Entry 0 is the embeddings and entry L + 1 the output of decoder layer L,
    except that the last entry has been through the final norm.
    """
    pooled: list[list[float]] = []
    for prompt in prompts:
        ids = torch.tensor([chat_prompt_ids(tokenizer, prompt, thinking=False)])
        with torch.no_grad():
            output = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        pooled = pooled or [[] for _ in output.hidden_states]
        for values, state in zip(pooled, output.hidden_states, strict=True):
            values.extend(
                torch.linalg.vector_norm(state[0, 1:].float(), dim=-1).tolist()
            )
    return [statistics.median(values) for values in pooled]


def _scaled(
    outputs: list[list[torch.Tensor]], token: int, factor: float
) -> list[list[torch.Tensor]]:
    """A copy of outputs with one token's vector scaled in every prompt and layer."""
    changed = [[output.clone() for output in per_prompt] for per_prompt in outputs]
    for per_prompt in changed:
        for output in per_prompt:
            output[token] *= factor
    return changed


def _hook_counts(model: nn.Module) -> list[tuple[int, int]]:
    return [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()]


@pytest.mark.parametrize("name", MODELS)
def test_calibrate_norms_matches_a_manual_median_of_the_hidden_states(
    name: str, request: pytest.FixtureRequest, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    model = request.getfixturevalue(name)
    norms = calibrate_norms(model, tiny_tokenizer, PROMPTS)
    # transformers' last hidden state is the final norm's output. With that
    # norm made a no-op, it is the last decoder layer's output, like the rest.
    model.get_decoder().norm = nn.Identity()
    manual = _hidden_state_medians(model, tiny_tokenizer, PROMPTS)
    assert norms == manual[1:]  # entry 0, the embeddings, is no layer's output
    assert len(norms) == len(find_decoder_layers(model))
    assert all(type(norm) is float and norm > 0 for norm in norms)


@pytest.mark.parametrize("name", MODELS)
def test_layer_l_means_the_output_of_decoder_layer_l(
    name: str, request: pytest.FixtureRequest, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    model = request.getfixturevalue(name)
    norms = calibrate_norms(model, tiny_tokenizer, PROMPTS)
    states = _hidden_state_medians(model, tiny_tokenizer, PROMPTS)
    assert norms[0] != states[0]  # not the embeddings, layer 0's input
    assert norms[:-1] == states[1:-1]  # each layer's output, not its input
    assert norms[-1] != states[-1]  # not the final norm's output


@pytest.mark.parametrize("name", MODELS)
def test_changing_only_each_prompts_first_token_leaves_the_result_unchanged(
    name: str, request: pytest.FixtureRequest, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    # Inside the model every later token reads the first one, so another
    # first token would change them all. What calibration must ignore is the
    # first token's own vector, which in real models is often far longer than
    # the rest. Here it is emptied, then made 1,000 times longer, at every
    # layer of every prompt.
    model = request.getfixturevalue(name)
    outputs = layer_outputs(model, tiny_tokenizer, PROMPTS)
    reference = median_norms(outputs)
    assert reference == calibrate_norms(model, tiny_tokenizer, PROMPTS)
    for factor in (0.0, 1000.0):
        assert median_norms(_scaled(outputs, 0, factor)) == reference
    # The same change to the second token moves the medians, so this can fail.
    assert any(median_norms(_scaled(outputs, 1, f)) != reference for f in (0.0, 1e3))


def test_layer_outputs_holds_every_layer_for_every_prompt(
    tiny_vl_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    outputs = layer_outputs(tiny_vl_model, tiny_tokenizer, PROMPTS)
    assert len(outputs) == len(PROMPTS)
    for prompt, per_prompt in zip(PROMPTS, outputs, strict=True):
        tokens = len(chat_prompt_ids(tiny_tokenizer, prompt, thinking=False))
        assert len(per_prompt) == 4  # tiny-vl's layers
        assert all(output.shape == (tokens, 64) for output in per_prompt)


def test_the_median_pools_every_prompt_and_averages_the_middle_two() -> None:
    # Vectors along the axes, so every length is exact. The first rows (9
    # and 7) are left out: layer 0's lengths are 1, 4, 2 and 3.
    first = torch.tensor([[9.0, 0.0], [0.0, 1.0], [4.0, 0.0]])
    second = torch.tensor([[7.0, 0.0], [0.0, 2.0], [3.0, 0.0]])
    assert median_norms([[first, 2 * first], [second, 2 * second]]) == [2.5, 5.0]
    assert median_norms([[first[:2]], [second]]) == [2.0]  # 1, 2 and 3


def test_lengths_are_taken_in_float32_from_bfloat16_outputs(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    outputs = layer_outputs(tiny_model.to(torch.bfloat16), tiny_tokenizer, PROMPTS)
    assert {output.dtype for per_prompt in outputs for output in per_prompt} == {
        torch.bfloat16  # kept in the model's own type until the lengths are taken
    }
    as_float32 = [[output.float() for output in per_prompt] for per_prompt in outputs]
    assert median_norms(outputs) == median_norms(as_float32)


@pytest.mark.parametrize(
    ("add_push", "where"),
    [
        # A neuron push adds to an MLP value through a hook on down_proj.
        (
            lambda model: model.model.layers[1].mlp.down_proj.register_forward_pre_hook(
                lambda module, args: None
            ),
            r"model\.layers\.1\.mlp\.down_proj",
        ),
        # A vector push adds to the stream leaving a layer.
        (
            lambda model: model.model.layers[1].register_forward_hook(
                lambda module, args, output: None
            ),
            r"model\.layers\.1$",
        ),
    ],
    ids=["neuron push", "residual push"],
)
def test_a_model_with_a_push_active_is_refused(
    tiny_model: PreTrainedModel,
    tiny_tokenizer: PreTrainedTokenizerBase,
    add_push: Callable[[nn.Module], object],
    where: str,
) -> None:
    add_push(tiny_model)
    with pytest.raises(ValueError, match=rf"no pushes.*: {where}"):
        calibrate_norms(tiny_model, tiny_tokenizer, PROMPTS)


@pytest.mark.parametrize(
    "register",
    [
        torch.nn.modules.module.register_module_forward_hook,
        torch.nn.modules.module.register_module_forward_pre_hook,
    ],
    ids=["forward hook", "forward pre-hook"],
)
def test_a_global_hook_is_refused_too(
    tiny_model: PreTrainedModel,
    tiny_tokenizer: PreTrainedTokenizerBase,
    register: Callable[[Callable[..., None]], RemovableHandle],
) -> None:
    # A global hook runs on every module of every model, so it could push too.
    handle = register(lambda module, *args: None)
    try:
        with pytest.raises(ValueError, match="global forward hooks"):
            calibrate_norms(tiny_model, tiny_tokenizer, PROMPTS)
    finally:
        handle.remove()


def test_a_layer_that_does_not_run_is_noticed(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    # The model now stops after layer 0, so nothing may stand in for layer 1.
    tiny_model.config.num_hidden_layers = 1
    with pytest.raises(RuntimeError, match="did not run"):
        calibrate_norms(tiny_model, tiny_tokenizer, PROMPTS)


def test_a_single_string_is_not_taken_for_a_list_of_prompts(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    # Text is a sequence of characters: each would run as a prompt of its own.
    with pytest.raises(ValueError, match="list of prompts"):
        calibrate_norms(tiny_model, tiny_tokenizer, "Who are you?")


def test_transformers_own_recording_hooks_are_not_pushes(
    tiny_vl_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    norms = calibrate_norms(tiny_vl_model, tiny_tokenizer, PROMPTS)
    # A forward pass that asks for hidden states leaves transformers' own
    # hooks on the decoder layers. They only record, so they are allowed.
    ids = torch.tensor([chat_prompt_ids(tiny_tokenizer, "Hi", thinking=False)])
    with torch.no_grad():
        tiny_vl_model(input_ids=ids, output_hidden_states=True, use_cache=False)
    assert any(layer._forward_hooks for layer in find_decoder_layers(tiny_vl_model))
    assert calibrate_norms(tiny_vl_model, tiny_tokenizer, PROMPTS) == norms


def test_calibration_leaves_no_hook_behind_even_after_an_error(
    tiny_model: PreTrainedModel,
    tiny_tokenizer: PreTrainedTokenizerBase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    before = _hook_counts(tiny_model)
    calibrate_norms(tiny_model, tiny_tokenizer, PROMPTS)
    assert _hook_counts(tiny_model) == before

    def fail_on_the_second_prompt(tokenizer: object, prompt: str, **kwargs: object):
        if prompt == PROMPTS[1]:
            raise RuntimeError("the second prompt fails")
        return chat_prompt_ids(tiny_tokenizer, prompt, thinking=False)

    monkeypatch.setattr(calibrate, "chat_prompt_ids", fail_on_the_second_prompt)
    with pytest.raises(RuntimeError, match="second prompt"):
        calibrate_norms(tiny_model, tiny_tokenizer, PROMPTS)
    assert _hook_counts(tiny_model) == before


def test_calibration_needs_prompts_of_two_tokens_or_more(
    tiny_model: PreTrainedModel,
    tiny_tokenizer: PreTrainedTokenizerBase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="at least one prompt"):
        calibrate_norms(tiny_model, tiny_tokenizer, [])
    monkeypatch.setattr(calibrate, "chat_prompt_ids", lambda *args, **kwargs: [5])
    with pytest.raises(ValueError, match="only 1 token"):
        calibrate_norms(tiny_model, tiny_tokenizer, PROMPTS)


def test_the_prompt_set_holds_32_short_prompts_of_one_version() -> None:
    prompt_set = read_prompt_set(PROMPTS_FILE)
    assert PROMPTS_FILE == PROJECT_ROOT / "prompts" / "calibration.jsonl"
    assert prompt_set.version == 1
    assert len(prompt_set.prompts) == len(set(prompt_set.prompts)) == 32
    assert len(set(prompt_set.prompt_ids)) == 32
    assert all(len(prompt) <= 80 for prompt in prompt_set.prompts)
    assert prompt_set.sha256 == hashlib.sha256(PROMPTS_FILE.read_bytes()).hexdigest()


def test_docstring_examples_run() -> None:
    result = doctest.testmod(calibrate)
    assert result.attempted > 0
    assert result.failed == 0
