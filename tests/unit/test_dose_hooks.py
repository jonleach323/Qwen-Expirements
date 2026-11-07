"""The hooks that follow a dose table: each row's own amounts on its own passes.

A probe on down_proj sees its input before the pushes' hook and after it, on
every pass of a padded batch whose rows follow different schedules.
"""

import doctest
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from pushlab.engine import hooks
from pushlab.engine.decoding import Decoding
from pushlab.engine.generate import generate
from pushlab.engine.hooks import doses_active
from pushlab.engine.push import StepState, add_amounts
from pushlab.models.chat import chat_prompt_ids
from pushlab.models.discovery import find_mlp_modules
from pushlab.schedules.compile import compile
from pushlab.schedules.model import Schedule
from pushlab.schedules.shapes import alternate, constant, handoff, pulse, ramp
from pushlab.units.dose import amount
from pushlab.units.neuron import NeuronUnit

# Of different lengths in the chat template, so their batch is padded.
PROMPTS = ["Hi", "What is two plus two?", "Name a colour.", "Count to three."]
GREEDY = Decoding(strategy="greedy", max_new_tokens=10)


def _case(
    name: str, request: pytest.FixtureRequest
) -> tuple[PreTrainedModel, list[NeuronUnit], list[float]]:
    """A tiny model; two neurons of one layer and one of another; its norms."""
    if name == "tiny_vl_model":
        request.getfixturevalue("tiny_downloads")  # registers tiny-vl
        units = ["tiny-vl:L2:N100", "tiny-vl:L2:N7", "tiny-vl:L0:N3"]
        norms = [3.0, 5.0, 8.0, 13.0]
    else:
        units, norms = ["tiny:L1:N5", "tiny:L1:N9", "tiny:L0:N3"], [7.0, 10.0]
    return request.getfixturevalue(name), [NeuronUnit.parse(u) for u in units], norms


def _schedules(a: NeuronUnit, b: NeuronUnit, c: NeuronUnit) -> list[Schedule]:
    """No push, a ramp, a handoff across two layers, and an alternation."""
    return [
        Schedule("none"),
        ramp(a, 0.0, 1.0, 1, 6),
        handoff(a, c, 1.0, -1.0, 3),
        alternate(a, b, 1.0, -0.5, 2, include_prompt=True),
    ]


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


def _s1_amounts(
    schedule: Schedule, reply_token: int | None, model: PreTrainedModel, norms: list
) -> dict[NeuronUnit, float]:
    """Each unit's amount on the pass by S1's dose rule, segment by segment."""
    totals: dict[NeuronUnit, float] = {}
    for segment in schedule.segments:
        if segment.active_at(reply_token):
            value = amount(segment.unit, segment.alpha_at(reply_token), model, norms)
            totals[segment.unit] = totals.get(segment.unit, 0.0) + value
    return totals


@pytest.mark.parametrize("model_name", ["tiny_model", "tiny_vl_model"])
def test_each_row_gets_exactly_its_own_amounts_on_every_pass(
    model_name: str,
    request: pytest.FixtureRequest,
    tiny_tokenizer: PreTrainedTokenizerBase,
) -> None:
    model, (a, b, c), norms = _case(model_name, request)
    schedules = _schedules(a, b, c)
    with _probe(model, [a.layer, c.layer]) as seen:
        replies = generate(
            model, tiny_tokenizer, PROMPTS, [], GREEDY, norms, schedules=schedules
        )
    lengths = [len(chat_prompt_ids(tiny_tokenizer, p, thinking=False)) for p in PROMPTS]
    pushed_rows: set[int] = set()
    for layer, passes in seen.items():
        assert len(passes) == max(len(reply.token_ids) for reply in replies)
        for p, (before, after) in enumerate(passes):
            reply_token = None if p == 0 else p - 1
            expected = before.clone()
            for row, schedule in enumerate(schedules):
                if p >= len(replies[row].token_ids):
                    continue  # the row has finished: it is padding now
                first = max(lengths) - lengths[row] if p == 0 else 0  # left padding
                totals = _s1_amounts(schedule, reply_token, model, norms)
                for unit, value in totals.items():
                    if unit.layer == layer:
                        column = before[row, first:, unit.index].float()
                        added = column + torch.tensor(value, dtype=torch.float32)
                        expected[row, first:, unit.index] = added.to(before.dtype)
                        pushed_rows.add(row)
            assert torch.equal(after, expected), (layer, p)
    assert pushed_rows == {1, 2, 3}  # every row with a push was pushed somewhere


def test_a_pass_with_nothing_on_in_a_layer_leaves_its_input_alone(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    a = NeuronUnit("tiny", 1, 5)
    handed: list[bool] = []  # whether down_proj got the very tensor its probe saw
    down_proj = dict(find_mlp_modules(tiny_model))[1].down_proj
    kept: list[torch.Tensor] = []
    before = down_proj.register_forward_pre_hook(lambda m, args: kept.append(args[0]))
    after = down_proj.register_forward_hook(
        lambda m, args, out: handed.append(args[0] is kept[-1])
    )
    try:
        generate(
            tiny_model,
            tiny_tokenizer,
            ["Hi"],
            [],
            GREEDY,
            [7.0, 10.0],
            schedules=[pulse(a, 1.0, 2, 2)],
        )
    finally:
        before.remove()
        after.remove()
    # The prompt pass and tokens 0 and 1, then 2 and 3 pushed, then none.
    assert handed[:5] == [True, True, True, False, False]
    assert all(handed[5:])


def test_the_hooks_refuse_a_batch_of_another_size(tiny_model: PreTrainedModel) -> None:
    table = compile(
        [constant(NeuronUnit("tiny", 1, 5), 1.0)] * 2, tiny_model, [7.0, 10.0], 4
    )
    state = StepState()
    with doses_active(tiny_model, table, state), torch.no_grad():
        state.at_prompt()
        with pytest.raises(ValueError, match="the dose table has 2 rows"):
            tiny_model(input_ids=torch.tensor([[5, 17, 42]]))


def test_the_hooks_refuse_a_pass_past_the_table(tiny_model: PreTrainedModel) -> None:
    table = compile(
        [constant(NeuronUnit("tiny", 1, 5), 1.0)], tiny_model, [7.0, 10.0], 2
    )
    state = StepState()
    with doses_active(tiny_model, table, state), torch.no_grad():
        state.at_reply_token(1)  # 2 new tokens: only token 0 is ever read
        with pytest.raises(
            ValueError, match="covers the prompt pass and reply tokens 0 to 0"
        ):
            tiny_model(input_ids=torch.tensor([[5]]))


def _hook_counts(model: torch.nn.Module) -> list[int]:
    return [len(m._forward_pre_hooks) for m in model.modules()]


def test_the_hooks_go_when_the_block_ends_even_after_an_error(
    tiny_model: PreTrainedModel,
) -> None:
    unit = NeuronUnit("tiny", 1, 5)
    table = compile([constant(unit, 1.0)], tiny_model, [7.0, 10.0], 4)
    before = _hook_counts(tiny_model)
    with pytest.raises(RuntimeError, match="stop"):
        with doses_active(tiny_model, table, StepState()):
            assert _hook_counts(tiny_model) != before
            raise RuntimeError("stop")
    assert _hook_counts(tiny_model) == before
    empty = compile([Schedule("none")], tiny_model, [7.0, 10.0], 4)
    with doses_active(tiny_model, empty, StepState()):
        assert _hook_counts(tiny_model) == before  # nothing to push, no hook


def test_doses_active_needs_a_dose_table_and_a_step_state(
    tiny_model: PreTrainedModel,
) -> None:
    table = compile([Schedule("none")], tiny_model, [7.0, 10.0], 4)
    with pytest.raises(ValueError, match="DoseTable"):
        with doses_active(tiny_model, [Schedule("none")], StepState()):  # type: ignore[arg-type]
            pass
    with pytest.raises(ValueError, match="StepState"):
        with doses_active(tiny_model, table, None):  # type: ignore[arg-type]
            pass


VALUES = torch.linspace(-1, 1, 2 * 5 * 8).bfloat16().reshape(2, 5, 8)
NEURONS = torch.tensor([1, 6])


def test_each_row_adds_its_own_amounts_never_rounded_first() -> None:
    # 22.21 is no bfloat16 number: rounded first, it would be 22.25.
    amounts = torch.tensor([[22.21, -3.3], [0.5, 7.77]], dtype=torch.float32)
    active = torch.tensor([[True, True], [True, False]])
    pushed = add_amounts(VALUES, NEURONS, amounts, active, None)
    expected = VALUES.clone()
    for row, column, value in [(0, 1, 22.21), (0, 6, -3.3), (1, 1, 0.5)]:
        exact = VALUES[row, :, column].float() + torch.tensor(
            value, dtype=torch.float32
        )
        expected[row, :, column] = exact.bfloat16()
    assert torch.equal(pushed, expected)
    assert torch.equal(VALUES, torch.linspace(-1, 1, 80).bfloat16().reshape(2, 5, 8))


def test_what_is_not_pushed_keeps_every_bit_negative_zeros_too() -> None:
    values = torch.full((2, 3, 4), -0.0)
    amounts = torch.zeros(2, 1)
    padding = torch.tensor([[True, False, False], [False, False, False]])
    active = torch.tensor([[True], [False]])
    pushed = add_amounts(values, torch.tensor([2]), amounts, active, padding)
    signs = torch.signbit(pushed)
    # Row 0's real positions went through the add, at 0: -0.0 + 0.0 is +0.0.
    assert not signs[0, 1:, 2].any()
    signs[0, 1:, 2] = True
    assert signs.all()  # padding, row 1 and every other neuron kept -0.0


@pytest.mark.parametrize(
    "values, amounts, problem",
    [
        (VALUES[0], torch.zeros(2, 2), "expected \\(rows, positions, neurons\\)"),
        (VALUES, torch.zeros(3, 2), "amounts for 3 rows"),
    ],
)
def test_add_amounts_checks_its_shapes(
    values: torch.Tensor, amounts: torch.Tensor, problem: str
) -> None:
    active = torch.ones_like(amounts, dtype=torch.bool)
    with pytest.raises(ValueError, match=problem):
        add_amounts(values, NEURONS, amounts, active, None)


def test_doctests() -> None:
    result = doctest.testmod(hooks)
    assert result.attempted > 0
    assert result.failed == 0
