"""The hooks that follow a dose table: each row's own amounts on its own passes.

A probe on down_proj sees its input before the pushes' hook and after it, on
every pass of a padded batch whose rows follow different schedules.
"""

import doctest
import itertools
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from pushlab.engine import hooks
from pushlab.engine.decoding import Decoding
from pushlab.engine.generate import generate
from pushlab.engine.hooks import doses_active
from pushlab.engine.push import StepState
from pushlab.models.chat import chat_prompt_ids
from pushlab.models.discovery import find_mlp_modules
from pushlab.schedules.compile import compile
from pushlab.schedules.model import Schedule
from pushlab.schedules.shapes import alternate, constant, handoff, onset, pulse, ramp
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


def _check_probed(
    seen: dict[int, list],
    replies: list,
    schedules: list[Schedule],
    lengths: list[int],
    model: PreTrainedModel,
    norms: list[float],
) -> set[int]:
    """Check that each probed pass got exactly each running row's amounts at
    its real positions, and nothing else; return the rows that were pushed."""
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
    return pushed_rows


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
    pushed = _check_probed(seen, replies, schedules, lengths, model, norms)
    assert pushed == {1, 2, 3}  # every row with a push was pushed somewhere


@contextmanager
def _puppet(model: PreTrainedModel, scripts: list[list[int]]) -> Iterator[None]:
    """Make pass p choose scripts[row][p] for every row, as test_generate_stops does."""
    passes = itertools.count()

    def choose(module: object, inputs: object, scores: torch.Tensor) -> torch.Tensor:
        p = next(passes)
        chosen = torch.zeros_like(scores)
        for row, script in enumerate(scripts):
            chosen[row, -1, script[min(p, len(script) - 1)]] = 1.0
        return chosen

    handle = model.lm_head.register_forward_hook(choose)
    try:
        yield
    finally:
        handle.remove()


def test_rows_that_finish_early_are_never_pushed_again(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    (a, b, c), norms = (
        [NeuronUnit.parse(u) for u in ("tiny:L1:N5", "tiny:L1:N9", "tiny:L0:N3")],
        [7.0, 10.0],
    )
    schedules = _schedules(a, b, c)
    end = tiny_tokenizer.eos_token_id
    # Rows 0 and 2 end their turns after 2 and 4 tokens; rows 1 and 3 run on.
    scripts = [[70, end], [71] * 10, [72, 73, 74, end], [75] * 10]
    with _puppet(tiny_model, scripts), _probe(tiny_model, [1, 0]) as seen:
        replies = generate(
            tiny_model, tiny_tokenizer, PROMPTS, [], GREEDY, norms, schedules=schedules
        )
    assert [len(reply.token_ids) for reply in replies] == [2, 10, 4, 10]
    lengths = [len(chat_prompt_ids(tiny_tokenizer, p, thinking=False)) for p in PROMPTS]
    assert _check_probed(seen, replies, schedules, lengths, tiny_model, norms) == {
        1,
        2,
        3,
    }
    # One record per token, and touched counts only the passes a row ran:
    # row 2 (the handoff at 3) read tokens 0 to 2, all A's.
    assert [len(reply.amounts) for reply in replies] == [2, 10, 4, 10]
    assert replies[2].touched == (0, 3, 0)
    assert replies[2].amounts[3] == ((a, amount(a, 1.0, tiny_model, norms)),)


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


def test_the_hooks_check_the_batch_size_on_every_pass(
    tiny_model: PreTrainedModel,
) -> None:
    late = pulse(NeuronUnit("tiny", 1, 5), 1.0, 5, 2)  # nothing on the prompt pass
    table = compile([late, late], tiny_model, [7.0, 10.0], 8)
    state = StepState()
    with doses_active(tiny_model, table, state), torch.no_grad():
        state.at_prompt()
        with pytest.raises(ValueError, match="the dose table has 2 rows"):
            tiny_model(input_ids=torch.tensor([[5, 17, 42]]))


def test_a_table_of_one_pass_covers_the_prompt_pass_alone(
    tiny_model: PreTrainedModel,
) -> None:
    table = compile(
        [constant(NeuronUnit("tiny", 1, 5), 1.0)], tiny_model, [7.0, 10.0], 1
    )
    state = StepState()
    with doses_active(tiny_model, table, state), torch.no_grad():
        state.at_reply_token(0)
        with pytest.raises(ValueError, match="covers the prompt pass alone, not reply"):
            tiny_model(input_ids=torch.tensor([[5]]))


def test_the_hooks_refuse_a_model_without_the_tables_layer(
    tiny_downloads: dict[str, object],
    tiny_vl_model: PreTrainedModel,
    tiny_model: PreTrainedModel,
) -> None:
    deep = onset(NeuronUnit("tiny-vl", 3, 5), 1.0, 0)  # tiny-vl has 4 layers, tiny 2
    table = compile([deep], tiny_vl_model, [3.0, 5.0, 8.0, 13.0], 4)
    with pytest.raises(ValueError, match="pushes layer 3, which this model lacks"):
        with doses_active(tiny_model, table, StepState()):
            pass


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


def test_doctests() -> None:
    result = doctest.testmod(hooks)
    assert result.attempted > 0
    assert result.failed == 0
