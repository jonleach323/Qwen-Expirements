"""Pushes: chosen neurons pushed during forward passes, while each push is active.

The tests drive the passes by hand, as a decoding loop will: the prompt pass,
then one pass per reply token with the model's cache. The reply tokens are
fixed in advance, so pushed and unpushed runs read exactly the same tokens.
"""

import doctest

import pytest
import torch
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from pushlab.engine import push as push_module
from pushlab.engine.push import Push, StepState, pushes_active
from pushlab.models.discovery import find_mlp_modules
from pushlab.units.calibrate import calibrate_norms
from pushlab.units.dose import amount
from pushlab.units.neuron import NeuronUnit

PROMPT = [5, 17, 42, 99, 3]
REPLY = [7, 8, 9, 10, 11, 12, 13, 14]  # reply tokens 0 to 7, one per pass
NORMS = [7.0, 10.0]  # tiny's two layers
VL_NORMS = [3.0, 5.0, 8.0, 13.0]  # tiny-vl's four layers
UNIT = NeuronUnit("tiny", layer=1, index=5)
MODELS = ["tiny_model", "tiny_vl_model"]


def _case(
    name: str, request: pytest.FixtureRequest
) -> tuple[PreTrainedModel, NeuronUnit, list[float]]:
    """A tiny model, a neuron of it in a middle or last layer, and its norms."""
    if name == "tiny_vl_model":
        request.getfixturevalue("tiny_downloads")  # registers tiny-vl
        return request.getfixturevalue(name), NeuronUnit("tiny-vl", 2, 100), VL_NORMS
    return request.getfixturevalue(name), UNIT, NORMS


def _push(
    unit: NeuronUnit = UNIT,
    alpha: float = 1.0,
    start: int = 0,
    end: int | None = None,
    include_prompt: bool = True,
) -> Push:
    return Push(
        unit=unit, alpha=alpha, start=start, end=end, include_prompt=include_prompt
    )


def _run(
    model: PreTrainedModel, state: StepState, batch: int = 1
) -> list[torch.Tensor]:
    """The logits of the prompt pass, then of each reply token's pass."""
    logits = []
    with torch.no_grad():
        state.at_prompt()
        output = model(input_ids=torch.tensor([PROMPT] * batch), use_cache=True)
        logits.append(output.logits)
        for k, token in enumerate(REPLY):
            state.at_reply_token(k)
            output = model(
                input_ids=torch.tensor([[token]] * batch),
                past_key_values=output.past_key_values,
                use_cache=True,
            )
            logits.append(output.logits)
    return logits


def _down_proj(model: PreTrainedModel, layer: int) -> nn.Module:
    return find_mlp_modules(model)[layer][1].down_proj


def _probed(
    model: PreTrainedModel,
    layer: int,
    pushes: list[Push],
    norms: list[float],
) -> list[torch.Tensor]:
    """down_proj's input at layer on every pass, as it is after any push.

    The probe is registered after the pushes' hook, so it sees what the
    push hands on.
    """
    inputs: list[torch.Tensor] = []
    state = StepState()
    with pushes_active(model, pushes, norms, state):
        handle = _down_proj(model, layer).register_forward_pre_hook(
            lambda module, args: inputs.append(args[0].detach().clone())
        )
        try:
            _run(model, state)
        finally:
            handle.remove()
    return inputs


def _hook_counts(model: nn.Module) -> list[tuple[int, int]]:
    return [(len(m._forward_hooks), len(m._forward_pre_hooks)) for m in model.modules()]


@pytest.mark.parametrize("name", MODELS)
def test_alpha_0_leaves_every_output_bit_identical(
    name: str, request: pytest.FixtureRequest
) -> None:
    model, unit, norms = _case(name, request)
    unsteered = _run(model, StepState())
    state = StepState()
    with pushes_active(model, [_push(unit, alpha=0.0)], norms, state):
        pushed = _run(model, state)
    for before, after in zip(unsteered, pushed, strict=True):
        assert torch.equal(before, after)


@pytest.mark.parametrize("name", MODELS)
def test_a_probe_sees_exactly_the_amount_at_the_neuron_on_active_passes_only(
    name: str, request: pytest.FixtureRequest
) -> None:
    model, unit, norms = _case(name, request)
    push = _push(unit, alpha=0.75, start=2, end=5)
    c = amount(unit, 0.75, model, norms)
    reference = _probed(model, unit.layer, [], norms)
    pushed = _probed(model, unit.layer, [push], norms)
    assert len(pushed) == 1 + len(REPLY)
    for step, (before, after) in enumerate(zip(reference, pushed, strict=True)):
        expected = before.clone()
        if step == 0 or 2 <= step - 1 < 5:  # the prompt pass, then tokens 2 to 4
            expected[..., unit.index] += c
        assert torch.equal(after, expected)
    # The layer before the pushed one reads exactly what it read unpushed.
    earlier = unit.layer - 1
    for before, after in zip(
        _probed(model, earlier, [], norms),
        _probed(model, earlier, [push], norms),
        strict=True,
    ):
        assert torch.equal(before, after)


@pytest.mark.parametrize("name", MODELS)
def test_pushing_the_neuron_equals_adding_its_write_direction_to_the_mlp_output(
    name: str, request: pytest.FixtureRequest
) -> None:
    model, unit, norms = _case(name, request)
    c = amount(unit, 1.0, model, norms)
    direction = unit.write_direction(model)
    mlp = find_mlp_modules(model)[unit.layer][1]
    outputs: list[torch.Tensor] = []
    record = mlp.register_forward_hook(lambda m, a, out: outputs.append(out.clone()))
    try:
        state = StepState()
        with pushes_active(model, [_push(unit)], norms, state):
            pushed = _run(model, state)
        pushed_outputs = outputs[:]
        outputs.clear()
        # The same push made by hand: c times the write direction added to
        # the MLP's output, at every position of every pass.
        add = mlp.register_forward_hook(lambda m, a, out: out + c * direction)
        try:
            by_hand = _run(model, StepState())
        finally:
            add.remove()
    finally:
        record.remove()
    for ours, theirs in zip(pushed_outputs, outputs, strict=True):
        torch.testing.assert_close(ours, theirs + c * direction, rtol=1e-5, atol=1e-5)
    for ours, theirs in zip(pushed, by_hand, strict=True):
        torch.testing.assert_close(ours, theirs, rtol=1e-4, atol=1e-4)


def test_a_window_is_active_on_its_reply_tokens_only(
    tiny_model: PreTrainedModel,
) -> None:
    push = _push(alpha=0.5, start=3, end=6, include_prompt=False)
    c = amount(UNIT, 0.5, tiny_model, NORMS)
    reference = _probed(tiny_model, UNIT.layer, [], NORMS)
    pushed = _probed(tiny_model, UNIT.layer, [push], NORMS)
    assert torch.equal(pushed[0], reference[0])  # the prompt pass is unchanged
    for k in range(len(REPLY)):
        expected = reference[1 + k].clone()
        if k in (3, 4, 5):
            expected[..., UNIT.index] += c
        assert torch.equal(pushed[1 + k], expected)


def test_no_hook_remains_after_the_context_even_after_an_error(
    tiny_model: PreTrainedModel,
) -> None:
    before = _hook_counts(tiny_model)
    unsteered = _run(tiny_model, StepState())
    state = StepState()
    with pushes_active(tiny_model, [_push()], NORMS, state):
        assert _hook_counts(tiny_model) != before
        _run(tiny_model, state)
    assert _hook_counts(tiny_model) == before
    with pytest.raises(RuntimeError, match="inside the context"):
        with pushes_active(tiny_model, [_push(alpha=2.0)], NORMS, state):
            _run(tiny_model, state)
            raise RuntimeError("an error inside the context")
    assert _hook_counts(tiny_model) == before
    for unpushed, after in zip(unsteered, _run(tiny_model, StepState()), strict=True):
        assert torch.equal(unpushed, after)


def test_pushes_add_up_with_one_hook_per_layer(tiny_model: PreTrainedModel) -> None:
    same = [_push(alpha=0.5), _push(alpha=0.25)]  # the same neuron twice
    other = _push(NeuronUnit("tiny", 1, 9), alpha=1.0)  # another neuron there
    lower = _push(NeuronUnit("tiny", 0, 3), alpha=1.0)  # and one in layer 0
    pushes = [*same, other, lower]
    state = StepState()
    with pushes_active(tiny_model, pushes, NORMS, state) as amounts:
        assert amounts == tuple(
            amount(p.unit, p.alpha, tiny_model, NORMS) for p in pushes
        )
        hooks = [
            len(_down_proj(tiny_model, layer)._forward_pre_hooks) for layer in (0, 1)
        ]
        assert hooks == [1, 1]
    # The layer-0 push changes what layer 1 reads, so layer 1 is compared
    # with a run that has that push alone.
    reference = _probed(tiny_model, 1, [lower], NORMS)
    pushed = _probed(tiny_model, 1, pushes, NORMS)
    for before, after in zip(reference, pushed, strict=True):
        expected = before.clone()
        expected[..., 5] += amounts[0] + amounts[1]
        expected[..., 9] += amounts[2]
        assert torch.equal(after, expected)
    reference = _probed(tiny_model, 0, [], NORMS)
    pushed = _probed(tiny_model, 0, pushes, NORMS)
    for before, after in zip(reference, pushed, strict=True):
        expected = before.clone()
        expected[..., 3] += amounts[3]
        assert torch.equal(after, expected)


def test_padding_is_never_pushed_and_every_row_gets_the_same_push(
    tiny_model: PreTrainedModel,
) -> None:
    c = amount(UNIT, 1.0, tiny_model, NORMS)
    down_proj = _down_proj(tiny_model, UNIT.layer)
    padding = torch.zeros(2, len(PROMPT), dtype=torch.bool)
    padding[1, :2] = True  # row 1's first two positions are padding
    finished = torch.tensor([[False], [True]])  # row 1 has stopped
    seen: dict[str, list[torch.Tensor]] = {"pushed": [], "unpushed": []}
    for key, pushes in [("unpushed", []), ("pushed", [_push()])]:
        state = StepState()
        with pushes_active(tiny_model, pushes, NORMS, state):
            handle = down_proj.register_forward_pre_hook(
                lambda module, args, key=key: seen[key].append(args[0].clone())
            )
            with torch.no_grad():
                state.at_prompt(padding=padding)
                output = tiny_model(
                    input_ids=torch.tensor([PROMPT] * 2), use_cache=True
                )
                state.at_reply_token(0, padding=finished)
                tiny_model(
                    input_ids=torch.tensor([[7], [7]]),
                    past_key_values=output.past_key_values,
                    use_cache=True,
                )
            handle.remove()
    for mask, before, after in zip(
        [padding, finished], seen["unpushed"], seen["pushed"], strict=True
    ):
        expected = before.clone()
        expected[..., UNIT.index] += c * (~mask).float()
        assert torch.equal(after, expected)


def test_in_bfloat16_the_amount_and_the_value_are_added_then_rounded_once(
    tiny_model: PreTrainedModel,
) -> None:
    model = tiny_model.to(torch.bfloat16)
    c = amount(UNIT, 1.0, model, NORMS)
    reference = _probed(model, UNIT.layer, [], NORMS)
    pushed = _probed(model, UNIT.layer, [_push()], NORMS)
    for before, after in zip(reference, pushed, strict=True):
        assert after.dtype == torch.bfloat16
        expected = before.clone()
        expected[..., UNIT.index] = (before[..., UNIT.index].float() + c).bfloat16()
        assert torch.equal(after, expected)


def test_the_amount_is_never_rounded_before_the_add() -> None:
    # 22.21 is no bfloat16 number: rounded first, it would be 22.25 before the
    # add. Over many values the two orders give different results somewhere.
    hook = push_module._down_proj_hook([(_push(), 22.21)], StepState())
    values = torch.linspace(-1, 1, 2001).bfloat16().reshape(1, -1, 1).repeat(1, 1, 8)
    (pushed,) = hook(nn.Identity(), (values,))
    column = values[..., UNIT.index]
    once = (column.float() + 22.21).bfloat16()
    assert torch.equal(pushed[..., UNIT.index], once)
    assert not torch.equal(once, column + torch.tensor(22.21).bfloat16())
    others = [i for i in range(8) if i != UNIT.index]
    assert torch.equal(pushed[..., others], values[..., others])


def test_the_hook_changes_a_copy_never_the_tensor_it_was_given(
    tiny_model: PreTrainedModel,
) -> None:
    # A hook that runs before the push's may keep the very tensor the push
    # hook is given; it must not change under that hook.
    kept: list[torch.Tensor] = []
    down_proj = _down_proj(tiny_model, UNIT.layer)
    early = down_proj.register_forward_pre_hook(lambda m, args: kept.append(args[0]))
    try:
        state = StepState()
        with pushes_active(tiny_model, [_push()], NORMS, state), torch.no_grad():
            state.at_prompt()
            tiny_model(input_ids=torch.tensor([PROMPT]))
    finally:
        early.remove()
    assert torch.equal(kept[0], _probed(tiny_model, UNIT.layer, [], NORMS)[0])


def test_a_padding_mask_of_the_wrong_shape_is_refused(
    tiny_model: PreTrainedModel,
) -> None:
    state = StepState()
    with pushes_active(tiny_model, [_push()], NORMS, state):
        state.at_prompt(padding=torch.zeros(1, len(PROMPT) + 1, dtype=torch.bool))
        with pytest.raises(ValueError, match="padding"), torch.no_grad():
            tiny_model(input_ids=torch.tensor([PROMPT]))


def test_a_push_for_another_model_is_refused_and_leaves_no_hook(
    tiny_vl_model: PreTrainedModel,
) -> None:
    before = _hook_counts(tiny_vl_model)
    with pytest.raises(ValueError, match="another model"):
        with pushes_active(tiny_vl_model, [_push()], VL_NORMS, StepState()):
            pass
    assert _hook_counts(tiny_vl_model) == before


def test_pushes_for_two_models_are_refused_even_of_the_same_shape(
    tiny_downloads: dict[str, object], tiny_model: PreTrainedModel
) -> None:
    # tiny and tiny-lm have the same shape, so only their names tell them apart.
    other = _push(NeuronUnit("tiny-lm", 1, 5))
    before = _hook_counts(tiny_model)
    with pytest.raises(ValueError, match="same model"):
        with pushes_active(tiny_model, [_push(), other], NORMS, StepState()):
            pass
    assert _hook_counts(tiny_model) == before


def test_pushes_active_needs_pushes_and_a_step_state(
    tiny_model: PreTrainedModel,
) -> None:
    with pytest.raises(ValueError, match="StepState"):
        with pushes_active(tiny_model, [_push()], NORMS, None):  # type: ignore[arg-type]
            pass
    with pytest.raises(ValueError, match="Push"):
        with pushes_active(tiny_model, [UNIT], NORMS, StepState()):  # type: ignore[list-item]
            pass


def test_calibration_refuses_to_run_while_pushes_are_active(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    with pushes_active(tiny_model, [_push()], NORMS, StepState()):
        with pytest.raises(ValueError, match="no pushes active"):
            calibrate_norms(tiny_model, tiny_tokenizer, ["Hi"])


def test_docstring_examples_run() -> None:
    result = doctest.testmod(push_module)
    assert result.attempted > 0
    assert result.failed == 0
