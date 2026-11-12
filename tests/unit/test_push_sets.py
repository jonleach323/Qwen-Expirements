"""Pushes of NeuronSets: each of a set's neurons gets the push's amount times
its coefficient, in its own layer, from that layer's hook (decision 0019).

As in test_push.py, the passes are driven by hand: the prompt pass, then one
pass per fixed reply token, with the model's cache.
"""

from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import torch
from torch import nn
from transformers import PreTrainedModel

from pushlab.engine.push import Push, StepState, pushes_active, schedule_of
from pushlab.models.discovery import find_mlp_modules
from pushlab.units.dose import amount
from pushlab.units.neuron import NeuronUnit
from pushlab.units.neuron_set import NeuronSet, unit_neurons

PROMPT = [5, 17, 42, 99, 3]
REPLY = [7, 8, 9, 10, 11, 12]  # reply tokens 0 to 5, one per pass
NORMS = [7.0, 10.0]  # tiny's two layers
A = NeuronUnit("tiny", 1, 5)
PAIR = NeuronSet("tiny", "pair", [(1, 5, 1.0), (1, 9, -0.5)], "test")
WIDE = NeuronSet(  # across both layers, sharing A's neuron
    "tiny", "wide", [(1, 9, 2.0), (0, 3, -1.5), (1, 5, 0.75)], "test", 12.0
)


def _run(model: PreTrainedModel, state: StepState) -> list[torch.Tensor]:
    """The logits of the prompt pass, then of each reply token's pass."""
    logits = []
    with torch.no_grad():
        state.at_prompt()
        output = model(input_ids=torch.tensor([PROMPT]), use_cache=True)
        logits.append(output.logits)
        for k, token in enumerate(REPLY):
            state.at_reply_token(k)
            output = model(
                input_ids=torch.tensor([[token]]),
                past_key_values=output.past_key_values,
                use_cache=True,
            )
            logits.append(output.logits)
    return logits


def _down_proj(model: PreTrainedModel, layer: int) -> nn.Module:
    return dict(find_mlp_modules(model))[layer].down_proj


@contextmanager
def _probe(model: PreTrainedModel, layers: list[int]) -> Iterator[dict[int, list]]:
    """down_proj's input at each layer on every pass: [before the push, after it]."""
    seen: dict[int, list[list[torch.Tensor]]] = {layer: [] for layer in layers}
    handles = []
    for layer in layers:
        down_proj = _down_proj(model, layer)
        # Registered before the pushes' hooks, so it sees the value unpushed.
        handles.append(
            down_proj.register_forward_pre_hook(
                lambda m, args, layer=layer: seen[layer].append([args[0].clone()])
            )
        )
        handles.append(  # a forward hook sees what the pushes handed on
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


def test_a_push_takes_a_set_and_refuses_what_is_no_unit() -> None:
    push = Push(unit=PAIR, alpha=-0.5, start=2, end=6, include_prompt=True)
    assert schedule_of([push]).units() == (PAIR,)
    assert [segment.unit for segment in push.segments()] == [PAIR, PAIR]
    with pytest.raises(ValueError, match="unit must be a NeuronUnit or a NeuronSet"):
        Push(unit="tiny:set:pair", alpha=1.0, include_prompt=True)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", ["tiny_model", "tiny_vl_model"])
@pytest.mark.parametrize(
    "window",
    [
        {"start": 0, "end": None, "include_prompt": True},
        {"start": 2, "end": 5, "include_prompt": False},
    ],
)
def test_a_one_neuron_set_pushes_exactly_as_its_neuron(
    name: str, window: dict, request: pytest.FixtureRequest
) -> None:
    if name == "tiny_vl_model":
        request.getfixturevalue("tiny_downloads")  # registers tiny-vl
        unit, other, norms = (
            NeuronUnit("tiny-vl", 2, 100),
            NeuronUnit("tiny-vl", 1, 7),
            [3.0, 5.0, 8.0, 13.0],
        )
    else:
        unit, other, norms = A, NeuronUnit("tiny", 0, 3), NORMS
    model = request.getfixturevalue(name)
    one = NeuronSet(unit.model_name, "one", [(unit.layer, unit.index, 1.0)], "test")
    runs = []
    for pushed in (unit, one):
        pushes = [
            Push(unit=pushed, alpha=0.75, **window),
            Push(unit=other, alpha=-0.5, start=1, include_prompt=False),
            Push(unit=pushed, alpha=0.25, start=3, include_prompt=False),
        ]
        state = StepState()
        with pushes_active(model, pushes, norms, state) as amounts:
            runs.append((amounts, _run(model, state)))
    (by_neuron, neuron_logits), (by_set, set_logits) = runs
    assert [x.hex() for x in by_neuron] == [x.hex() for x in by_set]
    for x, y in zip(neuron_logits, set_logits, strict=True):
        assert torch.equal(x, y)


def test_a_set_push_adds_its_share_to_each_neuron_in_every_layer(
    tiny_model: PreTrainedModel,
) -> None:
    pushes = [
        Push(unit=WIDE, alpha=0.5, start=1, end=4, include_prompt=True),
        Push(unit=A, alpha=-1.0, start=2, include_prompt=False),  # WIDE's N5 too
    ]
    wide, a = (amount(p.unit, p.alpha, tiny_model, NORMS) for p in pushes)
    with _probe(tiny_model, [0, 1]) as seen:
        state = StepState()
        with pushes_active(tiny_model, pushes, NORMS, state) as amounts:
            _run(tiny_model, state)
    assert amounts == (wide, a)
    for layer, passes in seen.items():
        assert len(passes) == 1 + len(REPLY)
        for p, (before, after) in enumerate(passes):
            reply_token = None if p == 0 else p - 1
            totals: dict[int, float] = {}  # as the hook adds them, in push order
            for push, value in zip(pushes, (wide, a), strict=True):
                if push.active_at(StepState(reply_token=reply_token)):
                    for neuron, c in unit_neurons(push.unit):
                        if neuron.layer == layer:
                            totals[neuron.index] = (
                                totals.get(neuron.index, 0.0) + value * c
                            )
            expected = before.clone()
            for index, total in totals.items():
                added = torch.tensor(total, dtype=torch.float32)  # rounded once
                expected[..., index] = before[..., index] + added
            assert torch.equal(after, expected), (layer, p)
    # WIDE pushes its layer-0 neuron on the prompt pass: layer 0 was pushed.
    assert not torch.equal(*seen[0][0])


def test_a_set_gets_one_hook_in_each_layer_it_touches(
    tiny_model: PreTrainedModel,
) -> None:
    def counts() -> list[int]:
        return [
            len(_down_proj(tiny_model, layer)._forward_pre_hooks) for layer in (0, 1)
        ]

    before = counts()
    wide = Push(unit=WIDE, alpha=1.0, include_prompt=True)
    with pushes_active(
        tiny_model,
        [wide, Push(unit=A, alpha=1.0, include_prompt=True)],
        NORMS,
        StepState(),
    ):
        assert counts() == [before[0] + 1, before[1] + 1]
    pair = Push(unit=PAIR, alpha=1.0, include_prompt=True)
    with pushes_active(tiny_model, [pair], NORMS, StepState()):
        assert counts() == [before[0], before[1] + 1]  # PAIR is all in layer 1
    assert counts() == before


def test_pushing_a_set_adds_its_amount_times_its_write_direction(
    tiny_model: PreTrainedModel,
) -> None:
    c = amount(PAIR, 1.0, tiny_model, NORMS)
    direction = PAIR.write_direction(tiny_model)  # w, the weighted columns
    mlp = dict(find_mlp_modules(tiny_model))[1]
    outputs: list[torch.Tensor] = []
    record = mlp.register_forward_hook(lambda m, a, out: outputs.append(out.clone()))
    try:
        state = StepState()
        with pushes_active(
            tiny_model, [Push(unit=PAIR, alpha=1.0, include_prompt=True)], NORMS, state
        ):
            pushed = _run(tiny_model, state)
        pushed_outputs = outputs[:]
        outputs.clear()
        # The same push by hand: c * w added to the MLP's output everywhere.
        add = mlp.register_forward_hook(lambda m, a, out: out + c * direction)
        try:
            by_hand = _run(tiny_model, StepState())
        finally:
            add.remove()
    finally:
        record.remove()
    assert abs(float(torch.linalg.vector_norm(c * direction)) - 10.0) < 1e-4  # r_1
    for ours, theirs in zip(pushed_outputs, outputs, strict=True):
        torch.testing.assert_close(ours, theirs + c * direction, rtol=1e-5, atol=1e-5)
    for ours, theirs in zip(pushed, by_hand, strict=True):
        torch.testing.assert_close(ours, theirs, rtol=1e-4, atol=1e-4)
