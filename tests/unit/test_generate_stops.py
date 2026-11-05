"""When replies stop, and which reply tokens pushes touch: pushlab.engine.generate.

A tiny model never ends its turn by itself, so some tests make it a puppet:
a hook on its output layer makes each pass choose a scripted token. Every
forward pass still runs for real, pushes included.
"""

import itertools
from collections.abc import Iterator
from contextlib import contextmanager

import pytest
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from pushlab.engine.decoding import Decoding
from pushlab.engine.generate import generate
from pushlab.engine.push import Push
from pushlab.models.chat import chat_prompt_ids
from pushlab.models.discovery import find_mlp_modules
from pushlab.units.neuron import NeuronUnit

UNIT = NeuronUnit("tiny", layer=1, index=5)
NORMS = [7.0, 10.0]
FILLER = 65  # what a puppet row chooses once its script has run out


@contextmanager
def _puppet(model: PreTrainedModel, scripts: list[list[int]]) -> Iterator[None]:
    """Make pass p choose scripts[row][p] for every row, by its scores."""
    passes = itertools.count()

    def choose(module: object, inputs: object, scores: torch.Tensor) -> torch.Tensor:
        p = next(passes)
        chosen = torch.zeros_like(scores)
        for row, script in enumerate(scripts):
            chosen[row, -1, script[p] if p < len(script) else FILLER] = 1.0
        return chosen

    handle = model.lm_head.register_forward_hook(choose)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def _additions(model: PreTrainedModel, unit: NeuronUnit) -> Iterator[list]:
    """What the pushes add to the unit's value on each pass, one tensor per pass.

    Each tensor is (rows, positions): the value down_proj gets, minus the
    value it would have got without the pushes. A hook added before the
    pushes' sees the value unpushed, and a forward hook sees it pushed.
    """
    down_proj = dict(find_mlp_modules(model))[unit.layer].down_proj
    unpushed: list[torch.Tensor] = []
    added: list[torch.Tensor] = []
    before = down_proj.register_forward_pre_hook(
        lambda module, args: unpushed.append(args[0][..., unit.index].clone())
    )
    after = down_proj.register_forward_hook(
        lambda module, args, output: added.append(
            args[0][..., unit.index] - unpushed[-1]
        )
    )
    try:
        yield added
    finally:
        before.remove()
        after.remove()


def _greedy(count: int) -> Decoding:
    return Decoding(strategy="greedy", max_new_tokens=count)


@pytest.mark.parametrize("end", ["<|im_end|>", "<|endoftext|>"])
def test_stops_at_the_end_of_the_turn_and_keeps_its_token(
    tiny_vl_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase, end: str
) -> None:
    # <|im_end|> ends a chat turn; tiny-vl's config, like Qwen3.5-4B's, ends
    # text with <|endoftext|>. Either ends the reply.
    hi = tiny_tokenizer("Hi", add_special_tokens=False)["input_ids"]
    script = [*hi, tiny_tokenizer.convert_tokens_to_ids(end)]
    with _puppet(tiny_vl_model, [script]):
        (reply,) = generate(tiny_vl_model, tiny_tokenizer, ["Say hi."], [], _greedy(12))
    assert reply.token_ids == tuple(script)
    assert reply.stop_reason == "end_of_turn"
    assert reply.text == "Hi"  # without the end marker


def test_stops_at_max_new_tokens(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    (reply,) = generate(tiny_model, tiny_tokenizer, ["Hi"], [], _greedy(5))
    assert len(reply.token_ids) == 5
    assert reply.stop_reason == "max_new_tokens"
    assert tiny_tokenizer.eos_token_id not in reply.token_ids


def test_an_end_of_turn_on_the_last_allowed_token_still_ends_the_turn(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    script = [70, 71, tiny_tokenizer.eos_token_id]
    with _puppet(tiny_model, [script]):
        (reply,) = generate(tiny_model, tiny_tokenizer, ["Hi"], [], _greedy(3))
    assert reply.token_ids == tuple(script)
    assert reply.stop_reason == "end_of_turn"


def test_a_row_that_ends_early_leaves_the_others_as_they_are_alone(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    prompts = ["Hi", "What is two plus two?", "Name a colour."]
    free = generate(tiny_model, tiny_tokenizer, prompts, [], _greedy(12))
    # Make one of the first row's tokens an end token, one the others never
    # choose, so only the first row ends early.
    others = {token for reply in free[1:] for token in reply.token_ids}
    ends = [token for token in free[0].token_ids[:-1] if token not in others]
    assert ends, "no token ends only the first reply; choose other prompts"
    tiny_model.generation_config.eos_token_id = ends[0]
    batched = generate(tiny_model, tiny_tokenizer, prompts, [], _greedy(12))
    assert batched[0].stop_reason == "end_of_turn"
    assert batched[0].token_ids == free[0].token_ids[: len(batched[0].token_ids)]
    assert len(batched[0].token_ids) < 12
    assert batched[1:] == free[1:]
    alone = [
        generate(tiny_model, tiny_tokenizer, [p], [], _greedy(12))[0] for p in prompts
    ]
    assert batched == alone


@pytest.mark.parametrize("model_name", ["tiny_model", "tiny_vl_model"])
def test_a_window_from_2_to_4_touches_exactly_2_reply_tokens(
    model_name: str,
    request: pytest.FixtureRequest,
    tiny_tokenizer: PreTrainedTokenizerBase,
) -> None:
    model = request.getfixturevalue(model_name)
    unit, norms = UNIT, NORMS
    if model_name == "tiny_vl_model":
        request.getfixturevalue("tiny_downloads")  # registers tiny-vl
        unit, norms = NeuronUnit("tiny-vl", 2, 100), [3.0, 5.0, 8.0, 13.0]
    window = Push(unit=unit, alpha=1.0, start=2, end=4, include_prompt=False)
    with _additions(model, unit) as added:
        (reply,) = generate(model, tiny_tokenizer, ["Hi"], [window], _greedy(8), norms)
    assert len(reply.token_ids) == 8  # so passes read reply tokens 0 to 6
    assert reply.touched == (2,)
    # Pass 0 is the prompt pass, and pass k + 1 reads reply token k: the
    # push lands on the passes of tokens 2 and 3 only.
    assert [bool(a.any()) for a in added] == [False] * 3 + [True] * 2 + [False] * 3


def test_a_push_never_touches_the_last_token_since_no_pass_reads_it(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    window = Push(unit=UNIT, alpha=1.0, start=2, end=4, include_prompt=False)
    four = _greedy(4)  # reply tokens 0 to 3; token 3 ends the reply
    (reply,) = generate(tiny_model, tiny_tokenizer, ["Hi"], [window], four, NORMS)
    assert reply.touched == (1,)


def test_the_prompt_pass_is_pushed_but_counts_as_no_reply_token(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    constant = Push(unit=UNIT, alpha=1.0, include_prompt=True)
    with _additions(tiny_model, UNIT) as added:
        (reply,) = generate(
            tiny_model, tiny_tokenizer, ["Hi"], [constant], _greedy(5), NORMS
        )
    assert reply.touched == (4,)  # reply tokens 0 to 3
    assert len(added) == 5
    assert all(bool(a.ne(0).all()) for a in added)  # at every position of every pass


@pytest.mark.parametrize("model_name", ["tiny_model", "tiny_vl_model"])
def test_the_left_padding_of_a_batch_is_never_pushed(
    model_name: str,
    request: pytest.FixtureRequest,
    tiny_tokenizer: PreTrainedTokenizerBase,
) -> None:
    # The prompt pass of a batch whose prompts differ in length. A push there
    # would change no reply, since padding is masked, so only a probe sees it.
    model = request.getfixturevalue(model_name)
    unit, norms = UNIT, NORMS
    if model_name == "tiny_vl_model":
        request.getfixturevalue("tiny_downloads")  # registers tiny-vl
        unit, norms = NeuronUnit("tiny-vl", 2, 100), [3.0, 5.0, 8.0, 13.0]
    constant = Push(unit=unit, alpha=1.0, include_prompt=True)
    prompts = ["Hi", "What is two plus two?"]
    with _additions(model, unit) as added:
        generate(model, tiny_tokenizer, prompts, [constant], _greedy(2), norms)
    lengths = [len(chat_prompt_ids(tiny_tokenizer, p, thinking=False)) for p in prompts]
    pads = max(lengths) - lengths[0]
    prompt_pass = added[0]
    assert prompt_pass.shape == (2, max(lengths))
    assert not prompt_pass[0, :pads].any()  # row 0's padding
    assert prompt_pass[0, pads:].ne(0).all() and prompt_pass[1].ne(0).all()


def test_each_push_counts_its_own_reply_tokens(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    early = Push(unit=UNIT, alpha=1.0, start=0, end=2, include_prompt=False)
    late = Push(
        unit=NeuronUnit("tiny", 0, 3), alpha=-1.0, start=3, include_prompt=False
    )
    (reply,) = generate(
        tiny_model, tiny_tokenizer, ["Hi"], [early, late], _greedy(8), NORMS
    )
    assert reply.touched == (2, 4)  # tokens 0 and 1; tokens 3 to 6


def test_a_finished_row_is_never_pushed_again(
    tiny_model: PreTrainedModel, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    end = tiny_tokenizer.eos_token_id
    scripts = [[70, end], [71, 72, 73, 74, 75, 76]]  # row 0 ends after 2 tokens
    constant = Push(unit=UNIT, alpha=1.0, include_prompt=False)
    read: list[list[int]] = []  # the token each row reads on each reply pass
    reading = tiny_model.register_forward_pre_hook(
        lambda module, args, kwargs: read.append(kwargs["input_ids"][:, -1].tolist()),
        with_kwargs=True,
    )
    try:
        with _puppet(tiny_model, scripts), _additions(tiny_model, UNIT) as added:
            first, second = generate(
                tiny_model,
                tiny_tokenizer,
                ["Hi", "Hello"],
                [constant],
                _greedy(6),
                NORMS,
            )
    finally:
        reading.remove()
    assert first.token_ids == (70, end) and first.stop_reason == "end_of_turn"
    assert second.token_ids == (71, 72, 73, 74, 75, 76)
    assert second.stop_reason == "max_new_tokens"
    assert (first.touched, second.touched) == ((1,), (5,))
    reply_passes = added[1:]  # the passes that read reply tokens 0 to 4
    assert [bool(a[0].any()) for a in reply_passes] == [True] + [False] * 4
    assert [bool(a[1].all()) for a in reply_passes] == [True] * 5
    # Once finished, row 0 reads only the pad token.
    pad = tiny_tokenizer.pad_token_id
    assert [tokens[0] for tokens in read[1:]] == [70] + [pad] * 4
