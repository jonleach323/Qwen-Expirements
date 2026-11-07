"""The generation loop: replies to a batch of prompts, with pushes on chosen passes.

generate renders each prompt in the chat template with thinking off, runs
one prompt pass over the batch, then decodes one token at a time, reusing
the model's cache. Before every pass it sets the StepState the hooks read,
so each push lands exactly on the passes its schedule names. Each prompt can
follow its own schedule: they are compiled into one DoseTable, and each row
of a batch adds only its own amounts (pushlab.engine.hooks). S1's list of
pushes, the same for every prompt, runs as the schedule it makes.

Prompts of different lengths are padded on the left, with an attention
mask, and each row's positions count from its own first token, as in
Hugging Face's generate(). So a row of a batch gets the reply its prompt
gets alone. With group_by_length, only prompts of the same token length
share a batch, and nothing is padded.
"""

import itertools
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from pushlab.engine.decoding import Decoding, TokenChooser
from pushlab.engine.hooks import doses_active
from pushlab.engine.push import Push, StepState, schedule_of
from pushlab.models.chat import chat_prompt_ids, stop_token_ids
from pushlab.schedules.compile import DoseTable, UnitAmount
from pushlab.schedules.compile import compile as compile_schedules
from pushlab.schedules.model import Schedule, Segment

STOP_REASONS = ("end_of_turn", "max_new_tokens")


@dataclass(frozen=True)
class Generation:
    """One prompt's reply.

    token_ids are the reply's tokens; if the turn ended, the last is the
    token that ended it. text is the reply decoded without special tokens,
    so without that end marker. stop_reason is "end_of_turn" when the model
    ended its turn, with the chat's end-of-turn token or its own end-of-text
    token, or "max_new_tokens" when the reply reached its limit.

    touched[i] counts the reply tokens the i-th push touched, or with
    schedules the i-th segment of the row's schedule: those whose pass it was
    active on. That counts passes, not influence: a push on reply token k
    first changes the choice of token k + 1, and no pass reads a reply's
    last token, so a constant push on a reply of n tokens touches n - 1. The
    prompt pass is no reply token, though a push on it changes token 0.

    amounts[j] lists the units pushed on the pass that chose token j, each
    with the amount added to its neuron: the prompt pass for token 0, and
    the pass that read token j - 1 for token j (decision 0008). So there is
    one entry per token, and an empty one where nothing was pushed.

    Example:
        >>> reply = Generation("Hi", (13048, 248046), "end_of_turn", (1,))
        >>> reply.stop_reason, reply.touched, reply.amounts
        ('end_of_turn', (1,), ())
    """

    text: str
    token_ids: tuple[int, ...]
    stop_reason: str
    touched: tuple[int, ...]
    amounts: tuple[tuple[UnitAmount, ...], ...] = ()


@dataclass
class _Row:
    """A batch row's reply so far.

    counted holds the windows its touched counts follow: each push's reply
    segment, or each segment of the row's schedule.
    """

    counted: tuple[Segment, ...]
    tokens: list[int] = field(default_factory=list)
    stop_reason: str | None = None
    touched: list[int] = field(init=False)
    amounts: list[tuple[UnitAmount, ...]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.touched = [0] * len(self.counted)

    @property
    def running(self) -> bool:
        return self.stop_reason is None


def _checked_prompts(prompts: Sequence[str]) -> list[str]:
    if isinstance(prompts, str):  # else each character would be a prompt
        raise ValueError("prompts must be a list of prompts, not one text")
    prompts = list(prompts)
    if not prompts:
        raise ValueError("generate needs at least one prompt")
    for prompt in prompts:
        if not isinstance(prompt, str):
            raise ValueError(f"each prompt must be text, not {prompt!r}")
    return prompts


def _batches(lengths: Sequence[int], group_by_length: bool) -> list[list[int]]:
    """Which prompts run together: all of them, or those of each token length.

    Example:
        >>> _batches([9, 12, 9], group_by_length=True)
        [[0, 2], [1]]
    """
    if not group_by_length:
        return [list(range(len(lengths)))]
    batches: dict[int, list[int]] = {}
    for index, length in enumerate(lengths):
        batches.setdefault(length, []).append(index)
    return list(batches.values())


def _decode(
    model: PreTrainedModel,
    prompt_ids: list[list[int]],
    places: list[int],
    table: DoseTable | None,
    counted: list[tuple[Segment, ...]],
    state: StepState,
    decoding: Decoding,
    stops: frozenset[int],
    pad: int,
) -> list[_Row]:
    """Decode one batch: a prompt pass, then one pass per reply token.

    places are the batch's prompts' places in the call's list, which seed
    their sampling. table holds the batch's rows' amounts, or is None when
    nothing is pushed; counted each row's windows to count touched tokens by.
    """
    # Where the token IDs go in. Qwen3.5's first parameters belong to its
    # vision tower, which may sit elsewhere.
    device = model.get_input_embeddings().weight.device
    count = len(prompt_ids)
    width = max(len(ids) for ids in prompt_ids)
    input_ids = torch.tensor(
        [[pad] * (width - len(ids)) + ids for ids in prompt_ids], device=device
    )
    if any(len(ids) < width for ids in prompt_ids):
        attention_mask = torch.tensor(
            [[0] * (width - len(ids)) + [1] * len(ids) for ids in prompt_ids],
            device=device,
        )
        # As in Hugging Face's generate(): a row's positions count from its
        # first real token, and a padded position gets position 0.
        positions = attention_mask.cumsum(dim=-1) - 1
        positions = positions.masked_fill(attention_mask == 0, 0)
        state.at_prompt(padding=attention_mask == 0)
    else:
        attention_mask = None  # nothing to mask, as generate() decides too
        positions = torch.arange(width, device=device).repeat(count, 1)
        state.at_prompt()
    # Positions are always given: without them, Qwen3.5 would rebuild them
    # from rope offsets it keeps from its last generate() call.
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=positions,
        use_cache=True,
        logits_to_keep=1,
    )
    choose = TokenChooser(decoding, places, output.logits.device)
    rows = [_Row(windows) for windows in counted]
    for r, row in enumerate(rows):  # what the prompt pass pushed, behind token 0
        row.amounts.append(table.amounts(r, None) if table else ())
    for k in itertools.count():  # returns once every row has finished
        chosen = choose(output.logits[:, -1], [row.running for row in rows])
        for row, token in zip(rows, chosen, strict=True):
            if token is None:
                continue
            row.tokens.append(token)
            if token in stops:
                row.stop_reason = "end_of_turn"
            elif len(row.tokens) == decoding.max_new_tokens:
                row.stop_reason = "max_new_tokens"
        if not any(row.running for row in rows):
            return rows
        # The pass that reads reply token k. A row that has finished takes
        # part only as padding: its reply is complete, so it reads the pad
        # token and is never pushed.
        finished = [not row.running for row in rows]
        padding = torch.tensor(finished, device=device)[:, None]
        state.at_reply_token(k, padding=padding if any(finished) else None)
        for r, row in enumerate(rows):
            if row.running:  # this pass chooses its token k + 1
                for i, window in enumerate(row.counted):
                    row.touched[i] += int(window.active_at(k))
                row.amounts.append(table.amounts(r, k) if table else ())
        feed = [[row.tokens[-1] if row.running else pad] for row in rows]
        positions = positions[:, -1:] + 1
        if attention_mask is not None:
            ones = attention_mask.new_ones((count, 1))
            attention_mask = torch.cat([attention_mask, ones], dim=-1)
        output = model(
            input_ids=torch.tensor(feed, device=device),
            attention_mask=attention_mask,
            position_ids=positions,
            past_key_values=output.past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
    raise AssertionError("unreachable: itertools.count() never ends")


def _checked_schedules(schedules: object, count: int) -> list[Schedule]:
    if not isinstance(schedules, list | tuple) or not all(
        isinstance(schedule, Schedule) for schedule in schedules
    ):
        raise ValueError(
            f"schedules must be a list of schedules, one per prompt, not {schedules!r}"
        )
    if len(schedules) != count:
        raise ValueError(
            f"give one schedule per prompt: {count} prompts, {len(schedules)} schedules"
        )
    return list(schedules)


@torch.no_grad()
def generate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: Sequence[str],
    pushes: Sequence[Push],
    decoding: Decoding,
    norms: Sequence[float] | None = None,
    *,
    schedules: Sequence[Schedule] | None = None,
    group_by_length: bool = False,
) -> list[Generation]:
    """Reply to each prompt, with its pushes applied, one token at a time.

    Each prompt is one user turn, rendered in the chat template with
    thinking off. A reply ends after the end-of-turn token or the model's
    own end-of-text token, which it keeps, or after decoding.max_new_tokens
    tokens. norms are each layer's residual sizes from the model's
    calibration, which turn doses into amounts; they are needed only when
    something is pushed.

    pushes is S1's form: a list of Push that every prompt gets. schedules,
    the other form, gives each prompt its own Schedule, one per prompt and
    in the same order, with pushes left empty. Either way the pushes are
    compiled into one DoseTable (pushlab.schedules.compile), checked against
    the model, and each row adds only its own amounts. Each reply records
    the units and amounts behind every token (Generation.amounts).

    All prompts run as one batch, padded on the left, unless group_by_length
    is True: then prompts of the same token length run together, so none is
    padded, each batch with its rows' schedules. Either way the replies come
    back in the prompts' order. A sampled reply depends on its prompt, the
    seed and its place in the list, never on the other prompts
    (decoding.row_seed). The hooks are removed again when generate returns,
    even after an error.

    Example:
        >>> greedy = Decoding(strategy="greedy", max_new_tokens=200)
        >>> (reply,) = generate(model, tokenizer, ["Hi"], [], greedy)  # doctest: +SKIP
        >>> reply.stop_reason  # doctest: +SKIP
        'end_of_turn'
        >>> replies = generate(  # doctest: +SKIP
        ...     model,
        ...     tokenizer,
        ...     ["Hi", "Hello"],
        ...     [],
        ...     greedy,
        ...     norms,
        ...     schedules=[pulse(unit, 1.0, 0, 16), Schedule("none")],
        ... )
    """
    prompts = _checked_prompts(prompts)
    pushes = list(pushes)
    if not isinstance(decoding, Decoding):
        raise ValueError(f"decoding must be a Decoding, not {type(decoding).__name__}")
    if not isinstance(group_by_length, bool):
        raise ValueError(
            f"group_by_length must be True or False, not {group_by_length!r}"
        )
    if schedules is None:  # S1's pushes, the same for every prompt
        schedules = [schedule_of(pushes)] * len(prompts)
        # touched counts each push's reply window.
        counted = [tuple(push.segments()[-1] for push in pushes)] * len(prompts)
        needs = "pushes need norms"
    else:
        if pushes:
            raise ValueError("give pushes or schedules, not both")
        schedules = _checked_schedules(schedules, len(prompts))
        counted = [schedule.segments for schedule in schedules]
        needs = "schedules need norms"
    table = None
    if any(schedule.segments for schedule in schedules):
        if norms is None:
            raise ValueError(
                f"{needs}: each layer's residual size, from the model's "
                "calibration (pushlab calibrate)"
            )
        table = compile_schedules(schedules, model, norms, decoding.max_new_tokens)
    prompt_ids = [
        chat_prompt_ids(tokenizer, prompt, thinking=False) for prompt in prompts
    ]
    stops = frozenset(stop_token_ids(model, tokenizer))
    # Any token would do for padding: padded positions are masked, and what
    # a finished row reads is never used.
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = min(stops, default=0)
    state = StepState()
    rows: list[_Row | None] = [None] * len(prompts)
    for batch in _batches([len(ids) for ids in prompt_ids], group_by_length):
        own = table.select(batch) if table is not None else None
        with doses_active(model, own, state) if own is not None else nullcontext():
            replies = _decode(
                model,
                [prompt_ids[i] for i in batch],
                batch,
                own,
                [counted[i] for i in batch],
                state,
                decoding,
                stops,
                pad,
            )
        for index, row in zip(batch, replies, strict=True):
            rows[index] = row
    return [
        Generation(
            text=tokenizer.decode(row.tokens, skip_special_tokens=True),
            token_ids=tuple(row.tokens),
            stop_reason=row.stop_reason,
            touched=tuple(row.touched),
            amounts=tuple(row.amounts),
        )
        for row in rows
    ]
