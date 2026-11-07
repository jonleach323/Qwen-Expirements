"""generate with one schedule per prompt: rows that never touch each other.

The tiny models run in float32 on the CPU. A row of a left-padded batch and
its prompt alone then give the same tokens, and scores within 1e-5: the
batch changes the shape of the arithmetic, so rounding differs a little
(decision 0009). In one batch layout, though, a row's numbers depend on its
own prompt and schedule alone, bit for bit.
"""

from collections.abc import Callable
from typing import Any

import pytest
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from pushlab.engine.decoding import Decoding
from pushlab.engine.generate import Generation, generate
from pushlab.engine.push import Push, schedule_of
from pushlab.schedules.compile import UnitAmount
from pushlab.schedules.model import Schedule, Segment
from pushlab.schedules.shapes import alternate, handoff, pulse
from pushlab.units.dose import amount
from pushlab.units.neuron import NeuronUnit

PROMPTS = ["Hi", "What is two plus two?", "Name a colour.", "Count to three."]
GREEDY = Decoding(strategy="greedy", max_new_tokens=12)
CLOSE = {"rtol": 0.0, "atol": 1e-5}  # as tests/unit/test_generate.py


def _case(
    name: str, request: pytest.FixtureRequest
) -> tuple[PreTrainedModel, NeuronUnit, NeuronUnit, list[float]]:
    """A tiny model, two neurons of different layers, and strong norms."""
    if name == "tiny_vl_model":
        request.getfixturevalue("tiny_downloads")  # registers tiny-vl
        a, b = NeuronUnit("tiny-vl", 2, 100), NeuronUnit("tiny-vl", 1, 7)
        return request.getfixturevalue(name), a, b, [3.0, 5.0, 8.0, 13.0]
    a, b = NeuronUnit("tiny", 1, 5), NeuronUnit("tiny", 0, 3)
    return request.getfixturevalue(name), a, b, [7.0, 10.0]


def _four(a: NeuronUnit, b: NeuronUnit) -> list[Schedule]:
    """No push, a pulse, a handoff and an alternation."""
    return [
        Schedule("none"),
        pulse(a, 2.0, 1, 4),
        handoff(a, b, 2.0, -2.0, 3),
        alternate(a, b, 2.0, -2.0, 2, include_prompt=True),
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


@pytest.mark.parametrize("model_name", ["tiny_model", "tiny_vl_model"])
def test_each_row_of_a_batch_gets_what_it_gets_alone(
    model_name: str,
    request: pytest.FixtureRequest,
    tiny_tokenizer: PreTrainedTokenizerBase,
) -> None:
    model, a, b, norms = _case(model_name, request)
    schedules = _four(a, b)
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
    for row in (1, 2, 3):
        assert batched[row].token_ids != unpushed[row].token_ids


@pytest.mark.parametrize("model_name", ["tiny_model", "tiny_vl_model"])
def test_changing_one_rows_schedule_leaves_the_others_bit_for_bit(
    model_name: str,
    request: pytest.FixtureRequest,
    tiny_tokenizer: PreTrainedTokenizerBase,
) -> None:
    model, a, b, norms = _case(model_name, request)
    first = _four(a, b)
    second = [first[0], pulse(b, -2.0, 0, 9), first[2], Schedule("none")]
    runs = [
        _scored(
            model,
            lambda schedules=schedules: generate(
                model, tiny_tokenizer, PROMPTS, [], GREEDY, norms, schedules=schedules
            ),
        )
        for schedules in (first, second)
    ]
    (one, one_scores), (two, two_scores) = runs
    for row in (0, 2):  # the rows whose schedules stayed
        assert one[row] == two[row]
        for k in range(len(one[row].token_ids)):
            assert torch.equal(one_scores[k][row], two_scores[k][row]), (row, k)
    assert one[1].token_ids != two[1].token_ids


def test_no_schedule_anywhere_is_s1s_unsteered_run(
    tiny_vl_model: PreTrainedModel,
    tiny_tokenizer: PreTrainedTokenizerBase,
    tiny_downloads: dict[str, object],
) -> None:
    empty = [Schedule("none")] * len(PROMPTS)
    scheduled, scheduled_scores = _scored(
        tiny_vl_model,
        lambda: generate(
            tiny_vl_model, tiny_tokenizer, PROMPTS, [], GREEDY, None, schedules=empty
        ),
    )
    unsteered, unsteered_scores = _scored(
        tiny_vl_model,
        lambda: generate(tiny_vl_model, tiny_tokenizer, PROMPTS, [], GREEDY),
    )
    assert scheduled == unsteered
    pairs = zip(scheduled_scores, unsteered_scores, strict=True)
    assert all(torch.equal(s, u) for s, u in pairs)
    assert all(reply.amounts == ((),) * len(reply.token_ids) for reply in scheduled)


def test_a_list_of_pushes_runs_as_its_schedule_on_every_row(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    unit, norms = NeuronUnit("tiny", 1, 5), [7.0, 10.0]
    pushes = [
        Push(unit=unit, alpha=1.0, start=1, end=6, include_prompt=True),
        Push(unit=NeuronUnit("tiny", 0, 3), alpha=-1.0, start=2, include_prompt=False),
    ]
    as_pushes, push_scores = _scored(
        tiny_model,
        lambda: generate(tiny_model, tiny_tokenizer, PROMPTS, pushes, GREEDY, norms),
    )
    schedule = schedule_of(pushes)
    as_schedules, schedule_scores = _scored(
        tiny_model,
        lambda: generate(
            tiny_model,
            tiny_tokenizer,
            PROMPTS,
            [],
            GREEDY,
            norms,
            schedules=[schedule] * len(PROMPTS),
        ),
    )
    for by_push, by_schedule in zip(as_pushes, as_schedules, strict=True):
        assert by_push.token_ids == by_schedule.token_ids
        assert by_push.amounts == by_schedule.amounts
        # touched counts each push, and each segment: the prompt pass is none.
        assert by_schedule.touched == (0, *by_push.touched)
    pairs = zip(push_scores, schedule_scores, strict=True)
    assert all(torch.equal(p, s) for p, s in pairs)


@pytest.mark.parametrize("model_name", ["tiny_model", "tiny_vl_model"])
def test_the_amounts_recorded_for_every_token_are_the_schedules(
    model_name: str,
    request: pytest.FixtureRequest,
    tiny_tokenizer: PreTrainedTokenizerBase,
) -> None:
    model, a, b, norms = _case(model_name, request)
    schedules = [*_four(a, b), Schedule("both", [Segment(a, 1.0, 0.0, 0, 5)] * 2)]
    prompts = [*PROMPTS, "Hello"]
    replies = generate(
        model, tiny_tokenizer, prompts, [], GREEDY, norms, schedules=schedules
    )
    for reply, schedule in zip(replies, schedules, strict=True):
        assert len(reply.amounts) == len(reply.token_ids)
        # Token 0 is chosen after the prompt pass, token j after the pass
        # that reads token j - 1.
        for j, recorded in enumerate(reply.amounts):
            reply_token = None if j == 0 else j - 1
            totals: dict[NeuronUnit, float] = {}
            for segment in schedule.segments:
                if segment.active_at(reply_token):
                    dose = segment.alpha_at(reply_token)
                    totals[segment.unit] = totals.get(segment.unit, 0.0) + amount(
                        segment.unit, dose, model, norms
                    )
            expected = tuple(
                UnitAmount(u, totals[u]) for u in schedule.units() if u in totals
            )
            assert recorded == expected, (schedule.name, j)


def test_touched_counts_each_segments_reply_tokens(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    a, b = NeuronUnit("tiny", 1, 5), NeuronUnit("tiny", 0, 3)
    (reply,) = generate(
        tiny_model,
        tiny_tokenizer,
        ["Hi"],
        [],
        GREEDY,
        [7.0, 10.0],
        schedules=[handoff(a, b, 1.0, -1.0, 4, overlap=1)],
    )
    assert len(reply.token_ids) == 12  # so the passes read reply tokens 0 to 10
    # The prompt pass; A on tokens 0 to 4; B from token 4 to 10.
    assert reply.touched == (0, 5, 7)


def test_grouped_by_length_each_batch_keeps_its_rows_schedules(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    a, b = NeuronUnit("tiny", 1, 5), NeuronUnit("tiny", 0, 3)
    prompts = ["Hi", "Yo", "What is two plus two?", "Ok"]
    schedules = [pulse(a, 2.0, 0, 5), Schedule("none"), _four(a, b)[3], _four(a, b)[2]]
    grouped = generate(
        tiny_model,
        tiny_tokenizer,
        prompts,
        [],
        GREEDY,
        [7.0, 10.0],
        schedules=schedules,
        group_by_length=True,
    )
    for prompt, schedule, reply in zip(prompts, schedules, grouped, strict=True):
        (alone,) = generate(
            tiny_model,
            tiny_tokenizer,
            [prompt],
            [],
            GREEDY,
            [7.0, 10.0],
            schedules=[schedule],
        )
        assert reply == alone


@pytest.mark.parametrize(
    "pushes, schedules, norms, problem",
    [
        (
            [Push(unit=NeuronUnit("tiny", 1, 5), alpha=1.0, include_prompt=True)],
            [Schedule("none")] * 2,
            [7.0, 10.0],
            "pushes or schedules, not both",
        ),
        ([], [Schedule("none")], [7.0, 10.0], "one schedule per prompt: 2 prompts, 1"),
        ([], Schedule("none"), [7.0, 10.0], "a list of schedules"),
        ([], [Schedule("none"), "none"], [7.0, 10.0], "a list of schedules"),
        (
            [],
            [Schedule("none"), pulse(NeuronUnit("tiny", 1, 5), 1.0, 0, 2)],
            None,
            "schedules need norms.*calibration",
        ),
        (
            [],
            [
                Schedule("none"),
                Schedule("x", [Segment(NeuronUnit("tiny", 1, 5), 0.0, 1.0, 3)]),
            ],
            [7.0, 10.0],
            "(?s)row 1: schedule 'x' is refused",
        ),
    ],
)
def test_generate_refuses_schedules_it_cannot_follow(
    tiny_model: PreTrainedModel,
    tiny_tokenizer: PreTrainedTokenizerBase,
    pushes: list,
    schedules: object,
    norms: list[float] | None,
    problem: str,
) -> None:
    with pytest.raises(ValueError, match=problem):
        generate(
            tiny_model,
            tiny_tokenizer,
            ["Hi", "Yo"],
            pushes,
            GREEDY,
            norms,
            schedules=schedules,  # type: ignore[arg-type]
        )


def test_schedules_may_come_with_pushes_none(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    schedules = [pulse(NeuronUnit("tiny", 1, 5), 1.0, 0, 2)]
    with_none = generate(
        tiny_model,
        tiny_tokenizer,
        ["Hi"],
        None,
        GREEDY,
        [7.0, 10.0],
        schedules=schedules,
    )
    assert with_none == generate(
        tiny_model, tiny_tokenizer, ["Hi"], [], GREEDY, [7.0, 10.0], schedules=schedules
    )
    with pytest.raises(ValueError, match="pushes must be a list of Push records"):
        generate(tiny_model, tiny_tokenizer, ["Hi"], "none", GREEDY)  # type: ignore[arg-type]


def test_a_generation_still_takes_s1s_fields_alone() -> None:
    reply = Generation("Hi", (13048, 248046), "end_of_turn", (1,))
    assert reply.amounts == ()
