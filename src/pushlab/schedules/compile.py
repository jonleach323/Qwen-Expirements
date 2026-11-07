"""compile: schedules turned into the exact amount every row adds on every pass.

A DoseTable is built once per batch, from one schedule per row. For any row
and pass it says which units are pushed and the amount added to each one's
neuron, by S1's dose rule (pushlab.units.dose.amount), so alpha 1 pushes as
hard as before. Passes are counted as decision 0008 counts them: the prompt
pass, then the pass that reads reply token k. A reply of max_new_tokens
tokens has max_new_tokens passes: the prompt pass, and reply tokens 0 to
max_new_tokens - 2, since no pass reads the last. Segments of one unit that
push the same pass add their amounts, in the schedule's order, as S1's hook
adds pushes.

The engine reads the table one layer at a time. A LayerTable holds the
layer's pushed neurons and, for every pass and row, the amount added to each
and whether it is pushed at all: a segment at dose 0 is pushed, with 0. Its
size grows with the passes, the rows and the units pushed, never with the
model's layers times its neurons.
"""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn

from pushlab.models.discovery import model_info
from pushlab.schedules.model import Schedule
from pushlab.schedules.validation import ScheduleWarning, validate
from pushlab.units.dose import amount
from pushlab.units.neuron import NeuronUnit


class UnitAmount(NamedTuple):
    """A unit pushed on a pass, and the amount added to its neuron there."""

    unit: NeuronUnit
    amount: float


def pass_index(reply_token: int | None) -> int:
    """A pass's place in a DoseTable: 0 for the prompt pass (None), k + 1 for the
    pass that reads reply token k.

    Example:
        >>> pass_index(None), pass_index(0), pass_index(39)
        (0, 1, 40)
    """
    if reply_token is None:
        return 0
    if (
        not isinstance(reply_token, int)
        or isinstance(reply_token, bool)
        or reply_token < 0
    ):
        raise ValueError(
            "a pass is a reply token from 0, or None for the prompt pass, "
            f"not {reply_token!r}"
        )
    return reply_token + 1


@dataclass(frozen=True, eq=False)
class LayerTable:
    """One layer's pushed neurons, and what each row adds to them on every pass.

    units are the layer's pushed units, by neuron index. amounts[p, r, i] is
    what row r adds to units[i]'s neuron on pass p, in float64, and
    active[p, r, i] whether row r pushes it on that pass at all. busy[p]
    says whether any row pushes any of them on pass p.
    """

    layer: int
    units: tuple[NeuronUnit, ...]
    amounts: torch.Tensor
    active: torch.Tensor
    busy: tuple[bool, ...]

    def select(self, rows: list[int]) -> "LayerTable":
        """The table of the given rows, in that order."""
        active = self.active[:, rows]
        busy = tuple(bool(pushed) for pushed in active.flatten(1).any(dim=1))
        return LayerTable(self.layer, self.units, self.amounts[:, rows], active, busy)


def _amounts_on(
    schedule: Schedule,
    units: tuple[NeuronUnit, ...],
    scales: dict[NeuronUnit, float],
    reply_token: int | None,
) -> tuple[UnitAmount, ...]:
    """Each unit the schedule pushes on the pass, and its summed amount there."""
    totals: dict[NeuronUnit, float] = {}
    for segment in schedule.segments:
        if segment.active_at(reply_token):
            # S1's rule: alpha * (norm / length), the scale being dose 1's amount.
            value = segment.alpha_at(reply_token) * scales[segment.unit]
            totals[segment.unit] = totals.get(segment.unit, 0.0) + value
    return tuple(UnitAmount(unit, totals[unit]) for unit in units if unit in totals)


@dataclass(frozen=True, eq=False)
class DoseTable:
    """Which neurons each row pushes, and by how much, on every pass of a batch.

    Built by compile(). rows is the number of schedules, one per row of a
    batch; passes the number of passes a reply can have; layers one
    LayerTable for each pushed layer, by layer.

    Example:
        >>> from pushlab.schedules.shapes import pulse
        >>> table = compile(  # doctest: +SKIP
        ...     [pulse(unit, 1.0, 0, 16), Schedule("none")], model, norms, 200
        ... )
        >>> table.amounts(0, 15), table.amounts(0, 16)  # doctest: +SKIP
        ((UnitAmount(unit=..., amount=23.47...),), ())
    """

    schedules: tuple[Schedule, ...]
    max_new_tokens: int
    scales: dict[NeuronUnit, float]
    layers: tuple[LayerTable, ...]
    by_pass: tuple[tuple[tuple[UnitAmount, ...], ...], ...]  # [row][pass]
    order: tuple[tuple[NeuronUnit, ...], ...]  # each row's units, as they appear

    @property
    def rows(self) -> int:
        """How many rows, one per schedule."""
        return len(self.schedules)

    @property
    def passes(self) -> int:
        """The prompt pass, then one per reply token read: max_new_tokens."""
        return self.max_new_tokens

    def amounts(self, row: int, reply_token: int | None) -> tuple[UnitAmount, ...]:
        """The units row pushes on the pass of reply_token (None: the prompt
        pass), each with the amount added to its neuron, in the order the
        row's schedule names them. Any reply token, past the table's too."""
        if (
            not isinstance(row, int)
            or isinstance(row, bool)
            or not 0 <= row < self.rows
        ):
            raise ValueError(f"row must be 0 to {self.rows - 1}, not {row!r}")
        p = pass_index(reply_token)
        if p < self.passes:
            return self.by_pass[row][p]
        return _amounts_on(
            self.schedules[row], self.order[row], self.scales, reply_token
        )

    def select(self, rows: Sequence[int]) -> "DoseTable":
        """The table of the given rows, in that order, such as one batch's.

        Example:
            >>> table.select([2, 0]).rows  # doctest: +SKIP
            2
        """
        rows = list(rows)
        for row in rows:
            if (
                not isinstance(row, int)
                or isinstance(row, bool)
                or not 0 <= row < self.rows
            ):
                raise ValueError(f"each row must be 0 to {self.rows - 1}, not {row!r}")
        return DoseTable(
            tuple(self.schedules[row] for row in rows),
            self.max_new_tokens,
            self.scales,
            tuple(layer.select(rows) for layer in self.layers),
            tuple(self.by_pass[row] for row in rows),
            tuple(self.order[row] for row in rows),
        )


def _layer_tables(
    by_pass: list[list[tuple[UnitAmount, ...]]], units: list[NeuronUnit], passes: int
) -> tuple[LayerTable, ...]:
    """The amounts of every row and pass, gathered layer by layer."""
    tables = []
    for layer in sorted({unit.layer for unit in units}):
        here = sorted((u for u in units if u.layer == layer), key=lambda u: u.index)
        column = {unit: i for i, unit in enumerate(here)}
        amounts = [[[0.0] * len(here) for _ in by_pass] for _ in range(passes)]
        active = [[[False] * len(here) for _ in by_pass] for _ in range(passes)]
        for row, row_passes in enumerate(by_pass):
            for p, pushed in enumerate(row_passes):
                for unit, value in pushed:
                    if unit.layer == layer:
                        amounts[p][row][column[unit]] = value
                        active[p][row][column[unit]] = True
        busy = tuple(any(any(row) for row in rows) for rows in active)
        tables.append(
            LayerTable(
                layer,
                tuple(here),
                torch.tensor(amounts, dtype=torch.float64),
                torch.tensor(active, dtype=torch.bool),
                busy,
            )
        )
    return tuple(tables)


def compile(
    schedules: Sequence[Schedule],
    model: nn.Module,
    norms: Sequence[float],
    max_new_tokens: int,
) -> DoseTable:
    """Turn one schedule per row into the amount each row adds on every pass.

    model is the units' own model and norms its residual sizes, one per
    layer, from its calibration, as for pushlab.units.dose.amount. Every
    schedule is checked against the model as validate() checks it, with no
    ceiling: the engine pushes what it is given, and schedule files are
    held to the ceiling when they are read. Every unit must name one model.

    Example:
        >>> from transformers import Qwen3Config, Qwen3ForCausalLM
        >>> from pushlab.schedules.shapes import pulse
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
        >>> with torch.no_grad():  # 64 entries of 0.25 make a length of 2
        ...     model.model.layers[1].mlp.down_proj.weight[:, 5] = 0.25
        >>> unit = NeuronUnit("tiny", layer=1, index=5)
        >>> table = compile(
        ...     [pulse(unit, 1.0, 0, 2), Schedule("none")], model, [7.0, 10.0], 4
        ... )
        >>> [(str(u), a) for u, a in table.amounts(0, 1)]  # 1 * 10 / 2
        [('tiny:L1:N5', 5.0)]
        >>> table.amounts(0, 2), table.amounts(1, 1)
        ((), ())
    """
    if not isinstance(schedules, list | tuple) or not all(
        isinstance(schedule, Schedule) for schedule in schedules
    ):
        raise ValueError(
            f"schedules must be a list of schedules, one per row, not {schedules!r}"
        )
    if not schedules:
        raise ValueError("compile needs at least one schedule")
    if not isinstance(model, nn.Module):
        raise ValueError(
            "model must be the model itself, such as load_model(...).model, "
            f"not a {type(model).__name__}"
        )
    if (
        not isinstance(max_new_tokens, int)
        or isinstance(max_new_tokens, bool)
        or max_new_tokens < 1
    ):
        raise ValueError(
            f"max_new_tokens must be a whole number from 1, not {max_new_tokens!r}"
        )
    units = list(dict.fromkeys(u for schedule in schedules for u in schedule.units()))
    models = sorted({unit.model_name for unit in units})
    if len(models) > 1:
        raise ValueError(
            f"the schedules' units name more than one model: {', '.join(models)}"
        )
    if units:
        shape = model_info(model)
        for row, schedule in enumerate(schedules):
            with warnings.catch_warnings():  # warnings are for validate's callers
                warnings.simplefilter("ignore", ScheduleWarning)
                try:
                    validate(schedule, shape, max_new_tokens, ceiling=None)
                except ValueError as error:
                    raise ValueError(f"row {row}: {error}") from None
    # Dose 1's amount, which also checks each unit against the model and norms.
    scales = {unit: amount(unit, 1.0, model, norms) for unit in units}  # type: ignore[arg-type]
    order = [schedule.units() for schedule in schedules]
    by_pass = [
        [
            _amounts_on(schedule, order[row], scales, None if p == 0 else p - 1)
            for p in range(max_new_tokens)
        ]
        for row, schedule in enumerate(schedules)
    ]
    return DoseTable(
        tuple(schedules),
        max_new_tokens,
        scales,
        _layer_tables(by_pass, units, max_new_tokens),
        tuple(tuple(passes) for passes in by_pass),
        tuple(order),
    )
