"""generate with NeuronSets: each row gets exactly its own pushes, on every
neuron its sets have, in one padded batch (decision 0019).

The tiny models run in float32 on the CPU. A row of a left-padded batch and
its prompt alone give the same tokens, and scores within 1e-5 (decision
0009); in one batch layout a row's numbers depend on its own prompt and
schedule alone, bit for bit.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import pytest
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from pushlab.engine.decoding import Decoding
from pushlab.engine.generate import generate
from pushlab.engine.push import Push
from pushlab.models.chat import chat_prompt_ids
from pushlab.models.discovery import find_mlp_modules
from pushlab.schedules.compile import UnitAmount
from pushlab.schedules.model import Schedule
from pushlab.schedules.shapes import alternate, constant, handoff, pulse, ramp
from pushlab.units.dose import PerPromptNorms, amount
from pushlab.units.neuron import NeuronUnit
from pushlab.units.neuron_set import NeuronSet, Unit, unit_neurons

# Of different lengths in the chat template, so their batch is padded.
PROMPTS = ["Hi", "What is two plus two?", "Name a colour.", "Count to three.", "Yo"]
GREEDY = Decoding(strategy="greedy", max_new_tokens=12)
CLOSE = {"rtol": 0.0, "atol": 1e-5}  # as tests/unit/test_generate.py


def _case(
    name: str, request: pytest.FixtureRequest
) -> tuple[PreTrainedModel, NeuronUnit, NeuronUnit, NeuronSet, list[float]]:
    """A tiny model; a neuron; another of another layer; a set across layers
    that shares a neuron with each; strong norms."""
    if name == "tiny_vl_model":
        request.getfixturevalue("tiny_downloads")  # registers tiny-vl
        a, b = NeuronUnit("tiny-vl", 2, 100), NeuronUnit("tiny-vl", 1, 7)
        entries = [(2, 100, 1.0), (2, 7, -0.5), (1, 7, 0.75), (0, 3, 0.5)]
        wide = NeuronSet("tiny-vl", "wide", entries, "test", 12.0)
        return request.getfixturevalue(name), a, b, wide, [3.0, 5.0, 8.0, 13.0]
    a, b = NeuronUnit("tiny", 1, 5), NeuronUnit("tiny", 0, 3)
    entries = [(1, 5, 1.0), (1, 9, -0.5), (0, 3, 0.75), (0, 70, 0.5)]
    wide = NeuronSet("tiny", "wide", entries, "test", 15.0)
    return request.getfixturevalue(name), a, b, wide, [7.0, 10.0]


def _five(unit: Unit, other: NeuronUnit) -> list[Schedule]:
    """No push, a pulse, a handoff to the other, an alternation, and both at
    once: a push of the other while the unit's lasts."""
    return [
        Schedule("none"),
        pulse(unit, 2.0, 1, 4),
        handoff(unit, other, 2.0, -2.0, 3),
        alternate(unit, other, 2.0, -2.0, 2, include_prompt=True),
        Schedule(
            "both", constant(unit, 1.5).segments + pulse(other, -1.0, 2, 5).segments
        ),
    ]


def _scored(model: PreTrainedModel, run: Callable[[], Any]) -> tuple[Any, list]:
    """run()'s result, and each pass's scores for the next token: (rows, vocabulary)."""
    scores: list[torch.Tensor] = []
    hook = model.lm_head.register_forward_hook(
        lambda module, inputs, output: scores.append(output[:, -1].float().clone())
    )
    try:
        result = run()
    finally:
        hook.remove()
    return result, scores


def _by_neuron(
    schedule: Schedule, reply_token: int | None, model: PreTrainedModel, norms: list
) -> dict[NeuronUnit, float]:
    """Each neuron's amount on the pass: each unit's amount by S1's rule,
    segment by segment, times the neuron's coefficient, the units' parts
    added in the order the schedule names them, the first set as it is."""
    totals: dict[Unit, float] = {}
    for segment in schedule.segments:
        if segment.active_at(reply_token):
            value = amount(segment.unit, segment.alpha_at(reply_token), model, norms)
            totals[segment.unit] = totals.get(segment.unit, 0.0) + value
    neurons: dict[NeuronUnit, float] = {}
    for unit in schedule.units():
        if unit not in totals:
            continue
        for neuron, coefficient in unit_neurons(unit):
            part = totals[unit] * coefficient
            neurons[neuron] = neurons[neuron] + part if neuron in neurons else part
    return neurons


@pytest.mark.parametrize("model_name", ["tiny_model", "tiny_vl_model"])
def test_a_one_neuron_set_generates_exactly_as_its_neuron(
    model_name: str,
    request: pytest.FixtureRequest,
    tiny_tokenizer: PreTrainedTokenizerBase,
) -> None:
    model, a, b, _, norms = _case(model_name, request)
    one = NeuronSet(a.model_name, "one", [(a.layer, a.index, 1.0)], "test")
    runs = [
        _scored(
            model,
            lambda unit=unit: generate(
                model,
                tiny_tokenizer,
                PROMPTS,
                [],
                GREEDY,
                norms,
                schedules=[*_five(unit, b)[1:], ramp(unit, 0.0, 2.0, 0, 6)],
            ),
        )
        for unit in (a, one)
    ]
    (as_neuron, neuron_scores), (as_set, set_scores) = runs
    for x, y in zip(as_neuron, as_set, strict=True):
        assert (x.token_ids, x.text, x.touched) == (y.token_ids, y.text, y.touched)
        assert [lp.hex() for lp in x.logprobs] == [lp.hex() for lp in y.logprobs]
        for neuron_pass, set_pass in zip(x.amounts, y.amounts, strict=True):
            assert [(one if u == a else u, v.hex()) for u, v in neuron_pass] == [
                (u, v.hex()) for u, v in set_pass
            ]
    assert all(
        torch.equal(s, t) for s, t in zip(neuron_scores, set_scores, strict=True)
    )


@pytest.mark.parametrize("model_name", ["tiny_model", "tiny_vl_model"])
def test_each_row_with_a_set_gets_what_it_gets_alone(
    model_name: str,
    request: pytest.FixtureRequest,
    tiny_tokenizer: PreTrainedTokenizerBase,
) -> None:
    model, _, b, wide, norms = _case(model_name, request)
    schedules = _five(wide, b)
    batched, batched_scores = _scored(
        model,
        lambda: generate(
            model, tiny_tokenizer, PROMPTS, [], GREEDY, norms, schedules=schedules
        ),
    )
    for row, (prompt, schedule) in enumerate(zip(PROMPTS, schedules, strict=True)):
        (alone,), scores = _scored(
            model,
            lambda prompt=prompt, schedule=schedule: generate(
                model, tiny_tokenizer, [prompt], [], GREEDY, norms, schedules=[schedule]
            ),
        )
        assert batched[row] == alone  # tokens, text, stop, touched and amounts
        for k in range(len(alone.token_ids)):
            torch.testing.assert_close(batched_scores[k][row], scores[k][0], **CLOSE)
    # Each push changed its row, so its effect was there to compare.
    unpushed = generate(model, tiny_tokenizer, PROMPTS, [], GREEDY)
    assert batched[0] == unpushed[0]
    for row in (1, 2, 3, 4):
        assert batched[row].token_ids != unpushed[row].token_ids


def test_changing_one_rows_schedule_leaves_the_others_bit_for_bit(
    request: pytest.FixtureRequest, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    tiny_model, a, b, wide, norms = _case("tiny_model", request)
    first = _five(wide, b)
    second = [first[0], pulse(a, -2.0, 0, 9), first[2], Schedule("none"), first[4]]
    runs = [
        _scored(
            tiny_model,
            lambda schedules=schedules: generate(
                tiny_model,
                tiny_tokenizer,
                PROMPTS,
                [],
                GREEDY,
                norms,
                schedules=schedules,
            ),
        )
        for schedules in (first, second)
    ]
    (one, one_scores), (two, two_scores) = runs
    for row in (0, 2, 4):  # the rows whose schedules stayed
        assert one[row] == two[row]
        for k in range(len(one[row].token_ids)):
            assert torch.equal(one_scores[k][row], two_scores[k][row]), (row, k)
    assert one[1].token_ids != two[1].token_ids


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


@pytest.mark.parametrize("model_name", ["tiny_model", "tiny_vl_model"])
def test_a_probe_sees_each_rows_exact_amount_on_every_neuron(
    model_name: str,
    request: pytest.FixtureRequest,
    tiny_tokenizer: PreTrainedTokenizerBase,
) -> None:
    model, _, b, wide, norms = _case(model_name, request)
    schedules = _five(wide, b)
    with _probe(model, list(wide.layers)) as seen:
        replies = generate(
            model, tiny_tokenizer, PROMPTS, [], GREEDY, norms, schedules=schedules
        )
    lengths = [len(chat_prompt_ids(tiny_tokenizer, p, thinking=False)) for p in PROMPTS]
    assert len(set(lengths)) > 1  # so the batch was padded
    pushed: set[tuple[int, int]] = set()  # (layer, row) pairs seen pushed
    for layer, passes in seen.items():
        assert len(passes) == max(len(reply.token_ids) for reply in replies)
        for p, (before, after) in enumerate(passes):
            expected = before.clone()
            for row, schedule in enumerate(schedules):
                if p >= len(replies[row].token_ids):
                    continue  # the row has finished: it is padding now
                first = max(lengths) - lengths[row] if p == 0 else 0  # left padding
                reply_token = None if p == 0 else p - 1
                for neuron, total in _by_neuron(
                    schedule, reply_token, model, norms
                ).items():
                    if neuron.layer == layer:
                        column = before[row, first:, neuron.index]
                        added = torch.tensor(total, dtype=torch.float32)
                        expected[row, first:, neuron.index] = column + added
                        pushed.add((layer, row))
            assert torch.equal(after, expected), (layer, p)
    # Every row with a push was pushed in each of the set's layers.
    assert pushed == {(layer, row) for layer in wide.layers for row in (1, 2, 3, 4)}


@pytest.mark.parametrize("group_by_length", [False, True])
def test_each_prompt_can_take_its_own_sizes(
    request: pytest.FixtureRequest,
    tiny_tokenizer: PreTrainedTokenizerBase,
    group_by_length: bool,
) -> None:
    model, a, b, wide, fixed = _case("tiny_model", request)
    one = NeuronSet("tiny", "one", [(1, 5, 1.0)], "test")
    prompts = ["Hi", "Yo", "What is two plus two?", "Ok"]  # three of one length
    schedules = [
        pulse(one, 2.0, 0, 5),
        constant(a, 1.0),
        alternate(wide, b, 2.0, -2.0, 2),
        Schedule("none"),
    ]
    sizes = PerPromptNorms([fixed, [3.5, 20.0], [2.0, 4.0], [9.0, 9.0]])
    replies = generate(
        model,
        tiny_tokenizer,
        prompts,
        [],
        GREEDY,
        sizes,
        schedules=schedules,
        group_by_length=group_by_length,
    )
    for i, (prompt, schedule) in enumerate(zip(prompts, schedules, strict=True)):
        own = list(sizes.rows[i])  # the prompt alone, with its sizes as the fixed
        (alone,) = generate(
            model, tiny_tokenizer, [prompt], [], GREEDY, own, schedules=[schedule]
        )
        assert replies[i] == alone
    # Prompt 1's doses took its own sizes: layer 1's is 20, twice the fixed 10.
    assert replies[1].amounts[0] == (UnitAmount(a, 2 * amount(a, 1.0, model, fixed)),)


def test_s1s_pushes_take_each_prompts_own_sizes_too(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    pair = NeuronSet("tiny", "pair", [(1, 5, 1.0), (1, 9, -0.5)], "test")
    pushes = [Push(unit=pair, alpha=1.0, start=0, end=4, include_prompt=True)]
    sizes = PerPromptNorms([[7.0, 10.0], [3.5, 20.0]])
    replies = generate(tiny_model, tiny_tokenizer, ["Hi", "Yo"], pushes, GREEDY, sizes)
    for reply, own in zip(replies, sizes.rows, strict=True):
        assert reply.amounts[0] == ((pair, amount(pair, 1.0, tiny_model, list(own))),)
    with pytest.raises(
        ValueError, match="norms has sizes for 2 prompts, but there are 3"
    ):
        generate(tiny_model, tiny_tokenizer, ["Hi", "Yo", "Ok"], pushes, GREEDY, sizes)
