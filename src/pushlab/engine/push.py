"""Pushes: adding to chosen neurons during forward passes, while each is active.

A decoding loop runs a prompt pass, then one pass per reply token. A Push
names a neuron, a dose and the passes it is active on: reply tokens start to
end (end not included), and the prompt pass if include_prompt. A StepState
says which pass is running, and pushes_active installs one hook per pushed
layer that adds each active push's amount to its neuron's MLP value (the
input of down_proj) at every real position of the pass.

A push on reply token k changes the computation at token k's position, so it
first shows in the choice of token k + 1, and it stays in the cache for the
tokens after. Only a push on the prompt pass reaches the choice of token 0.
"""

import math
import numbers
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn

from pushlab.models.discovery import find_mlp_modules
from pushlab.schedules.model import Schedule, Segment
from pushlab.units.dose import amount
from pushlab.units.neuron import NeuronUnit


def _is_whole(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True, kw_only=True)
class Push:
    """One neuron pushed by a dose alpha, on some passes of a reply.

    start is the first reply token the push applies to, counting from 0, and
    end the first it no longer applies to (None: to the end of the reply).
    include_prompt says whether it also applies on the prompt pass; it has
    no default, because a constant push like the evil-Qwen recipe's needs
    True and a window late in the reply needs False.

    A push on reply token k changes the pass that reads token k, so it first
    changes the choice of token k + 1. Only include_prompt reaches token 0.
    The research plan's pulse on tokens 0 to 15 is start=0, end=16.

    Example:
        >>> unit = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
        >>> recipe = Push(unit=unit, alpha=1.0, include_prompt=True)
        >>> recipe.start, recipe.end
        (0, None)
        >>> window = Push(unit=unit, alpha=1.0, start=3, end=6, include_prompt=False)
    """

    unit: NeuronUnit
    alpha: float
    start: int = 0
    end: int | None = None
    include_prompt: bool

    def __post_init__(self) -> None:
        """Refuse a push that is not a neuron, a finite dose and a real window."""
        if not isinstance(self.unit, NeuronUnit):
            raise ValueError(f"unit must be a NeuronUnit, not {self.unit!r}")
        alpha = self.alpha
        if not (
            isinstance(alpha, numbers.Real)
            and not isinstance(alpha, bool)
            and math.isfinite(alpha)
        ):
            raise ValueError(f"alpha must be a finite number, not {alpha!r}")
        if not _is_whole(self.start) or self.start < 0:
            raise ValueError(f"start must be a reply token from 0, not {self.start!r}")
        if self.end is not None and (not _is_whole(self.end) or self.end <= self.start):
            raise ValueError(
                f"end must be None or a reply token after start ({self.start}), "
                f"not {self.end!r}"
            )
        if not isinstance(self.include_prompt, bool):
            raise ValueError(
                f"include_prompt must be True or False, not {self.include_prompt!r}"
            )

    def active_at(self, state: "StepState") -> bool:
        """Whether this push applies on the pass that state describes.

        Example:
            >>> unit = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
            >>> push = Push(unit=unit, alpha=1.0, start=3, end=6, include_prompt=False)
            >>> state = StepState()
            >>> push.active_at(state)  # the prompt pass
            False
            >>> state.at_reply_token(5)
            >>> push.active_at(state)
            True
        """
        if state.reply_token is None:
            return self.include_prompt
        return self.start <= state.reply_token and (
            self.end is None or state.reply_token < self.end
        )

    def segments(self) -> tuple[Segment, ...]:
        """The push as schedule segments: the prompt pass if it is on, then its window.

        They push exactly the passes the push does, at its dose.

        Example:
            >>> unit = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
            >>> push = Push(unit=unit, alpha=1.0, start=3, end=6, include_prompt=True)
            >>> [
            ...     segment.describe() for segment in push.segments()
            ... ]  # doctest: +NORMALIZE_WHITESPACE
            ['qwen3.5-4b:L14:N8949 at +1 on the prompt pass',
             'qwen3.5-4b:L14:N8949 at +1 on reply tokens 3 to 5']
        """
        window = Segment(self.unit, self.alpha, self.alpha, self.start, self.end)
        if self.include_prompt:
            return (Segment(self.unit, self.alpha, self.alpha, region="prompt"), window)
        return (window,)


def schedule_of(pushes: Sequence[Push], name: str = "pushes") -> Schedule:
    """S1's pushes as one schedule, each push's segments in the pushes' order.

    Example:
        >>> unit = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
        >>> recipe = Push(unit=unit, alpha=1.0, include_prompt=True)
        >>> len(schedule_of([recipe]).segments)
        2
    """
    segments: list[Segment] = []
    for push in pushes:
        if not isinstance(push, Push):
            raise ValueError(f"pushes must be Push records, not {type(push).__name__}")
        segments.extend(push.segments())
    return Schedule(name, segments)


def _checked_padding(padding: torch.Tensor | None) -> torch.Tensor | None:
    if padding is not None and not (
        isinstance(padding, torch.Tensor)
        and padding.dtype == torch.bool
        and padding.dim() == 2
    ):
        raise ValueError(
            "padding must be None or a bool tensor of (rows, positions in this "
            f"pass), True where a position is padding, not {padding!r}"
        )
    return padding


@dataclass
class StepState:
    """Which pass runs now: the prompt pass, or the pass of reply token k.

    The decoding loop sets it before each forward pass, and the pushes' hooks
    read it. padding, if given, marks the positions of this pass that are
    padding (True), one row per batch row; they are never pushed.

    Example:
        >>> state = StepState()
        >>> state.in_prompt
        True
        >>> state.at_reply_token(0)
        >>> state.reply_token, state.in_prompt
        (0, False)
    """

    reply_token: int | None = None
    padding: torch.Tensor | None = None

    def __post_init__(self) -> None:
        """Check the fields as at_prompt and at_reply_token do."""
        if self.reply_token is not None and (
            not _is_whole(self.reply_token) or self.reply_token < 0
        ):
            raise ValueError(
                "the reply token must be None or a whole number from 0, "
                f"not {self.reply_token!r}"
            )
        _checked_padding(self.padding)

    @property
    def in_prompt(self) -> bool:
        """True during the prompt pass."""
        return self.reply_token is None

    def at_prompt(self, padding: torch.Tensor | None = None) -> None:
        """Set the state for the prompt pass.

        Example:
            >>> state = StepState(reply_token=7)
            >>> state.at_prompt(padding=torch.tensor([[True, False, False]]))
            >>> state.in_prompt, state.padding.tolist()
            (True, [[True, False, False]])
        """
        self.padding = _checked_padding(padding)
        self.reply_token = None

    def at_reply_token(self, k: int, padding: torch.Tensor | None = None) -> None:
        """Set the state for the pass that reads reply token k, counting from 0.

        Example:
            >>> state = StepState()
            >>> state.at_reply_token(2)
            >>> state.reply_token
            2
        """
        if not _is_whole(k) or k < 0:
            raise ValueError(
                f"the reply token must be a whole number from 0, not {k!r}"
            )
        self.padding = _checked_padding(padding)
        self.reply_token = k


def add_amounts(
    values: torch.Tensor,
    neurons: torch.Tensor,
    amounts: torch.Tensor,
    active: torch.Tensor,
    padding: torch.Tensor | None,
) -> torch.Tensor:
    """A copy of down_proj's input with each row's amounts added to its neurons.

    values is (rows, positions, neurons of the MLP). neurons holds the K
    pushed neurons' indices, each once; amounts (rows, K) what each row adds
    to each, and active (rows, K) which of them it pushes on this pass. Both
    may instead have one row, which every row shares. A pushed value and its amount are
    added in float32 (float64 for a float64 model) and rounded once to the
    activations' type (decision 0008); a neuron pushed at 0 still goes
    through the add. Everything else, padding positions included, keeps
    every bit.

    Example:
        >>> values = torch.zeros(2, 1, 4)
        >>> pushed = add_amounts(
        ...     values,
        ...     torch.tensor([1]),
        ...     torch.tensor([[2.5], [7.0]]),
        ...     torch.tensor([[True], [False]]),
        ...     None,
        ... )
        >>> pushed[:, 0].tolist()
        [[0.0, 2.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    """
    if values.dim() != 3:
        raise ValueError(
            f"down_proj's input has shape {tuple(values.shape)}; "
            "expected (rows, positions, neurons)"
        )
    rows, positions = values.shape[0], values.shape[1]
    # Shapes only: reading the indices themselves would make the GPU wait.
    if neurons.dim() != 1:
        raise ValueError(
            f"neurons must be one row of indices, not {tuple(neurons.shape)}"
        )
    if amounts.dim() != 2 or amounts.shape[1] != len(neurons):
        raise ValueError(
            f"amounts must be (rows, {len(neurons)}), one per neuron, not "
            f"{tuple(amounts.shape)}"
        )
    if amounts.shape[0] not in (1, rows):
        raise ValueError(
            f"amounts for {amounts.shape[0]} rows, but this pass has {rows} rows"
        )
    if active.shape != amounts.shape:
        raise ValueError(
            f"the active mask is {tuple(active.shape)}, but the amounts are "
            f"{tuple(amounts.shape)}"
        )
    real = torch.ones(rows, positions, dtype=torch.bool, device=values.device)
    padding = _checked_padding(padding)  # also when set directly
    if padding is not None:
        if tuple(padding.shape) != (rows, positions):
            raise ValueError(
                f"the padding mask is {tuple(padding.shape)}, but this "
                f"pass has {rows} rows of {positions} positions"
            )
        real = ~padding.to(values.device)
    # float32, or float64 for a float64 model: never less exact than the
    # activations, so a dose of 0 changes nothing in any type.
    compute = torch.promote_types(values.dtype, torch.float32)
    columns = values[..., neurons]  # a copy: (rows, positions, K)
    added = columns.to(compute) + amounts.to(compute)[:, None, :]
    pushed_here = active.to(values.device)[:, None, :] & real[:, :, None]
    pushed = values.clone()  # every other value stays exactly as it was
    pushed[..., neurons] = torch.where(pushed_here, added.to(values.dtype), columns)
    return pushed


def _down_proj_hook(
    pushes: Sequence[tuple[Push, float]], state: StepState
) -> Callable[[nn.Module, tuple], tuple | None]:
    """A forward pre-hook that adds the active pushes' amounts to down_proj's input."""

    def hook(module: nn.Module, args: tuple) -> tuple | None:
        totals: dict[int, float] = {}
        for push, value in pushes:
            if push.active_at(state):
                totals[push.unit.index] = totals.get(push.unit.index, 0.0) + value
        if not totals:
            return None  # nothing active: the input passes on untouched
        values = args[0]
        compute = torch.promote_types(values.dtype, torch.float32)
        # One row of amounts, which every row of the batch shares.
        amounts = torch.tensor([list(totals.values())], dtype=compute)
        neurons = torch.tensor(list(totals), device=values.device)
        active = torch.ones(amounts.shape, dtype=torch.bool)
        pushed = add_amounts(
            values, neurons, amounts.to(values.device), active, state.padding
        )
        return (pushed, *args[1:])

    return hook


@contextmanager
def pushes_active(
    model: nn.Module,
    pushes: Sequence[Push],
    norms: Sequence[float],
    state: StepState,
) -> Iterator[tuple[float, ...]]:
    """Install the pushes' hooks for the length of a with block.

    Each push's amount comes from its dose and norms (units.dose.amount),
    which also checks that the push is a neuron of this model. Every pushed
    layer gets one forward pre-hook on its down_proj, shared by its pushes;
    pushes add up, on one neuron or many. The hooks read state on every pass,
    and they are removed when the block ends, even after an error. The block
    gets each push's amount, in order.

    Example:
        >>> from transformers import Qwen3Config, Qwen3ForCausalLM
        >>> config = Qwen3Config(  # the registry's "tiny" shape
        ...     vocab_size=512,
        ...     hidden_size=64,
        ...     intermediate_size=128,
        ...     num_hidden_layers=2,
        ...     num_attention_heads=4,
        ...     num_key_value_heads=2,
        ...     head_dim=16,
        ... )
        >>> model = Qwen3ForCausalLM(config)
        >>> push = Push(unit=NeuronUnit("tiny", 1, 5), alpha=1.0, include_prompt=True)
        >>> state = StepState()
        >>> with pushes_active(model, [push], [7.0, 10.0], state) as amounts:
        ...     logits = model(input_ids=torch.tensor([[5, 17, 42]])).logits
        >>> len(amounts)
        1
    """
    if not isinstance(state, StepState):
        raise ValueError(f"state must be a StepState, not {type(state).__name__}")
    pushes = list(pushes)
    for push in pushes:
        if not isinstance(push, Push):
            raise ValueError(f"pushes must be Push records, not {type(push).__name__}")
    if len({push.unit.model_name for push in pushes}) > 1:
        names = sorted({push.unit.model_name for push in pushes})
        raise ValueError(f"all pushes must name the same model, not {names}")
    amounts = tuple(amount(push.unit, push.alpha, model, norms) for push in pushes)
    by_layer: dict[int, list[tuple[Push, float]]] = {}
    for push, value in zip(pushes, amounts, strict=True):
        by_layer.setdefault(push.unit.layer, []).append((push, value))
    mlps = dict(find_mlp_modules(model))
    handles = []
    try:
        for layer, here in by_layer.items():
            hook = _down_proj_hook(here, state)
            handles.append(mlps[layer].down_proj.register_forward_pre_hook(hook))
        yield amounts
    finally:
        for handle in handles:
            handle.remove()
