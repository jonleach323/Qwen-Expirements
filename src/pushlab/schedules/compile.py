"""compile: schedules turned into the exact amount every row adds on every pass.

A DoseTable is built once per batch, from one schedule per row. For any row
and pass it says which units are pushed and the amount of each, by S1's dose
rule (pushlab.units.dose.amount), so alpha 1 pushes as hard as before.
Passes are counted as decision 0008 counts them: the prompt pass, then the
pass that reads reply token k. A reply of max_new_tokens tokens has
max_new_tokens passes: the prompt pass, and reply tokens 0 to
max_new_tokens - 2, since no pass reads the last. Segments of one unit that
push the same pass add their amounts, in the schedule's order, as S1's hook
adds pushes.

A unit is a neuron or a NeuronSet, whose amount is per unit of coefficient:
entry i's neuron gets it times c_i, in that neuron's own layer, so the set
pushes along its write direction (decision 0019). Where units of a row push
one neuron on a pass, their parts add in the order the row's own schedule
names the units, so a row's numbers never depend on another row's.

The engine reads the table one layer at a time. A LayerTable
(pushlab.schedules.layer_tables) holds the layer's pushed neurons and, for
every pass and row, the amount added to each and whether it is pushed at
all: a segment at dose 0 is pushed, with 0. Its size grows with the passes,
the rows and the neurons pushed, never with the model's layers times its
neurons.
"""

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import NamedTuple

import torch  # noqa: F401 (the examples use it)
from torch import nn

from pushlab.models.discovery import model_info
from pushlab.schedules.layer_tables import LayerTable, layer_tables
from pushlab.schedules.model import Schedule, Segment
from pushlab.schedules.validation import ScheduleWarning, validate
from pushlab.units.dose import PerPromptNorms, amount
from pushlab.units.neuron import NeuronUnit  # noqa: F401 (the examples use it)
from pushlab.units.neuron_set import NeuronSet, Unit  # noqa: F401 (NeuronSet: the examples use it)


class UnitAmount(NamedTuple):
    """A unit pushed on a pass, and its amount there: what a neuron gets, or a
    NeuronSet's amount per unit of coefficient (decision 0019).

    Example:
        >>> pushed = UnitAmount(NeuronUnit("tiny", 1, 5), 5.0)
        >>> str(pushed.unit), pushed.amount
        ('tiny:L1:N5', 5.0)
    """

    unit: Unit
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


# A schedule's segments, each with its unit's place in the row's units and
# that unit's dose-1 amount, so that no pass looks a unit up.
_Plan = list[tuple[Segment, int, float]]


def _plan(
    schedule: Schedule, units: tuple[Unit, ...], scales: Mapping[Unit, float]
) -> _Plan:
    """The schedule's segments, each with its unit's place in units and scale."""
    place = {unit: i for i, unit in enumerate(units)}
    return [(s, place[s.unit], scales[s.unit]) for s in schedule.segments]


def _amounts_on(
    plan: _Plan, units: tuple[Unit, ...], reply_token: int | None
) -> tuple[UnitAmount, ...]:
    """Each unit pushed on the pass, in units' order, with its amount there:
    its segments' amounts added in the schedule's order, as S1 adds pushes."""
    totals: dict[int, float] = {}
    for segment, i, scale in plan:
        if segment.active_at(reply_token):
            # S1's rule: alpha * (norm / length), the scale being dose 1's amount.
            totals[i] = totals.get(i, 0.0) + segment.alpha_at(reply_token) * scale
    return tuple(UnitAmount(units[i], totals[i]) for i in sorted(totals))


@dataclass(frozen=True, eq=False)
class DoseTable:
    """Which units each row pushes, and by how much, on every pass of a batch.

    Built by compile(). rows is the number of schedules, one per row of a
    batch; passes the number of passes a reply can have; layers one
    LayerTable for each pushed layer, by layer. scales[r] holds dose 1's
    amount of each of row r's units, from row r's residual sizes.

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
        >>> unit = NeuronUnit("tiny", layer=1, index=5)
        >>> table = compile(
        ...     [pulse(unit, 1.0, 0, 2), Schedule("none")],
        ...     Qwen3ForCausalLM(config),
        ...     [7.0, 10.0],
        ...     max_new_tokens=4,
        ... )
        >>> table.rows, table.passes, [layer.layer for layer in table.layers]
        (2, 4, [1])
        >>> [str(pushed.unit) for pushed in table.amounts(0, 1)], table.amounts(0, 2)
        (['tiny:L1:N5'], ())
    """

    schedules: tuple[Schedule, ...]
    max_new_tokens: int
    scales: tuple[dict[Unit, float], ...]  # [row]: dose 1's amount of its units
    layers: tuple[LayerTable, ...]
    by_pass: tuple[tuple[tuple[UnitAmount, ...], ...], ...]  # [row][pass]
    order: tuple[tuple[Unit, ...], ...]  # each row's units, as they appear

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
        pass), each with its amount (UnitAmount), in the order the row's
        schedule names them. Any reply token, past the table's too.

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
        >>> unit = NeuronUnit("tiny", layer=1, index=5)
        >>> table = compile(
        ...     [pulse(unit, 1.0, 0, 2), Schedule("none")],
        ...     Qwen3ForCausalLM(config),
        ...     [7.0, 10.0],
        ...     max_new_tokens=4,
        ... )
        >>> table.amounts(0, None), table.amounts(1, 0)  # no prompt pass; no push
        ((), ())
        >>> (
        ...     len(table.amounts(0, 0)),
        ...     len(table.amounts(0, 1)),
        ...     len(table.amounts(0, 99)),
        ... )
        (1, 1, 0)
        """
        if (
            not isinstance(row, int)
            or isinstance(row, bool)
            or not 0 <= row < self.rows
        ):
            raise ValueError(f"row must be 0 to {self.rows - 1}, not {row!r}")
        p = pass_index(reply_token)
        if p < self.passes:
            return self.by_pass[row][p]
        units = self.order[row]
        plan = _plan(self.schedules[row], units, self.scales[row])
        return _amounts_on(plan, units, reply_token)

    def select(self, rows: Sequence[int]) -> "DoseTable":
        """The table of the given rows, in that order, such as one batch's.

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
        >>> unit = NeuronUnit("tiny", layer=1, index=5)
        >>> table = compile(
        ...     [pulse(unit, 1.0, 0, 2), Schedule("none")],
        ...     Qwen3ForCausalLM(config),
        ...     [7.0, 10.0],
        ...     max_new_tokens=4,
        ... )
        >>> picked = table.select([1, 0])
        >>> picked.rows, picked.amounts(0, 0), len(picked.amounts(1, 0))
        (2, (), 1)
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
            tuple(self.scales[row] for row in rows),
            tuple(layer.select(rows) for layer in self.layers),
            tuple(self.by_pass[row] for row in rows),
            tuple(self.order[row] for row in rows),
        )


def _scales(
    units: list[Unit],
    order: list[tuple[Unit, ...]],
    model: nn.Module,
    norms: Sequence[float] | PerPromptNorms,
) -> tuple[dict[Unit, float], ...]:
    """Dose 1's amount of each row's units: from norms, which every row shares,
    each unit worked once; or from a PerPromptNorms's rows[r] for row r, each
    unit worked once per distinct row of sizes. amount() also checks each
    unit against the model and the sizes."""
    if not isinstance(norms, PerPromptNorms):
        shared = {unit: amount(unit, 1.0, model, norms) for unit in units}  # type: ignore[arg-type]
        return tuple({unit: shared[unit] for unit in row} for row in order)
    worked: dict[tuple[Unit, tuple[float, ...]], float] = {}
    for row, sizes in zip(order, norms.rows, strict=True):
        for unit in row:
            if (unit, sizes) not in worked:
                worked[unit, sizes] = amount(unit, 1.0, model, list(sizes))
    return tuple(
        {unit: worked[unit, sizes] for unit in row}
        for row, sizes in zip(order, norms.rows, strict=True)
    )


def compile(
    schedules: Sequence[Schedule],
    model: nn.Module,
    norms: Sequence[float] | PerPromptNorms,
    max_new_tokens: int,
) -> DoseTable:
    """Turn one schedule per row into the amount each row adds on every pass.

    model is the units' own model and norms its residual sizes, one per
    layer, as for pushlab.units.dose.amount: one list from its calibration,
    which every row shares, or a PerPromptNorms with one row of sizes per
    schedule, whose row r is row r's (the per_prompt calibration mode).
    Every schedule is checked against the model as validate() checks it,
    but with no ceiling: the engine pushes what it is given, as S1's pushes
    always were. Nothing on the way from a schedule file to the engine
    checks the ceiling yet, so whatever runs schedule files must first call
    validate(schedule, REGISTRY[model], max_new_tokens), which does. Every
    unit must name one model. A unit may be a NeuronSet, whose neurons each
    get its amount times their coefficient (decision 0019).

    Example:
        >>> from transformers import Qwen3Config, Qwen3ForCausalLM
        >>> from pushlab.schedules.shapes import constant, pulse
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
        >>> with torch.no_grad():  # 64 entries of 0.25 - 0.125: a length of 1
        ...     model.model.layers[1].mlp.down_proj.weight[:, 9] = 0.25
        >>> pair = NeuronSet("tiny", "pair", [(1, 5, 1.0), (1, 9, -0.5)], "made up")
        >>> table = compile([constant(pair, 1.0)], model, [7.0, 10.0], 4)
        >>> [(str(u), a) for u, a in table.amounts(0, 0)]  # 1 * 10 / 1
        [('tiny:set:pair', 10.0)]
        >>> layer = table.layers[0]  # each neuron gets 10 times its coefficient
        >>> [str(neuron) for neuron in layer.units], layer.amounts[1, 0].tolist()
        (['tiny:L1:N5', 'tiny:L1:N9'], [10.0, -5.0])
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
    if isinstance(norms, PerPromptNorms) and len(norms) != len(schedules):
        raise ValueError(
            f"norms has sizes for {len(norms)} prompts, but there are "
            f"{len(schedules)} schedules; give one row of sizes per schedule"
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
    order = [schedule.units() for schedule in schedules]
    # Dose 1's amounts, which also checks each unit against the model and norms.
    scales = _scales(units, order, model, norms)
    by_pass = []
    for row, schedule in enumerate(schedules):
        plan = _plan(schedule, order[row], scales[row])
        by_pass.append(
            [
                _amounts_on(plan, order[row], None if p == 0 else p - 1)
                for p in range(max_new_tokens)
            ]
        )
    return DoseTable(
        tuple(schedules),
        max_new_tokens,
        scales,
        layer_tables(by_pass, order),
        tuple(tuple(passes) for passes in by_pass),
        tuple(order),
    )
