"""Layer tables: what every row of a batch adds to each pushed neuron of a layer.

compile (pushlab.schedules.compile) works out each unit's amount on every
pass of every row; layer_tables spreads those amounts over the neurons the
units push, one LayerTable per pushed layer, which the engine's hooks read
(pushlab.engine.hooks). A NeuronUnit gives its own neuron its amount. A
NeuronSet's amount is per unit of coefficient: entry i's neuron gets it
times c_i, in that neuron's own layer (decision 0019). Where units of a row
push one neuron on a pass, their parts add in the order that row's own
schedule names the units, so a row's numbers never depend on another row's.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from operator import attrgetter

import torch

from pushlab.units.neuron import NeuronUnit
from pushlab.units.neuron_set import NeuronSet, Unit, unit_neurons  # noqa: F401 (NeuronSet: the examples use it)


@dataclass(frozen=True, eq=False)
class LayerTable:
    """One layer's pushed neurons, and what each row adds to them on every pass.

    units are the layer's pushed neurons, one NeuronUnit each, by neuron
    index, a NeuronSet's among them. amounts[p, r, i] is what row r adds to
    units[i]'s neuron on pass p, in float64, and active[p, r, i] whether row
    r pushes it on that pass at all. busy[p] says whether any row pushes any
    of them on pass p.
    """

    layer: int
    units: tuple[NeuronUnit, ...]
    amounts: torch.Tensor
    active: torch.Tensor
    busy: tuple[bool, ...]

    def select(self, rows: list[int]) -> "LayerTable":
        """The table of the given rows, in that order.

        Example:
        >>> from transformers import Qwen3Config, Qwen3ForCausalLM
        >>> from pushlab.schedules.compile import compile
        >>> from pushlab.schedules.model import Schedule
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
        >>> layer = table.layers[0].select([1])  # the row of no push
        >>> layer.amounts.shape, layer.busy
        (torch.Size([4, 1, 1]), (False, False, False, False))
        """
        active = self.active[:, rows]
        busy = tuple(bool(pushed) for pushed in active.flatten(1).any(dim=1))
        return LayerTable(self.layer, self.units, self.amounts[:, rows], active, busy)


def layer_tables(
    by_pass: Sequence[Sequence[Sequence[tuple[Unit, float]]]],
    order: Sequence[tuple[Unit, ...]],
) -> tuple[LayerTable, ...]:
    """Each pushed layer's table, from every row's unit amounts on every pass.

    by_pass[r][p] lists the units row r pushes on pass p, each with its
    amount, as DoseTable.by_pass does; order[r] holds every unit of row r's
    schedule, in the order it names them, and every row has as many passes.
    Every neuron of every unit gets a column, pushed or not. A unit gives
    its neurons in a layer their part of all its passes at once: its amount
    times each one's coefficient (1.0 for a NeuronUnit). The first of a
    row's units to push a neuron on a pass sets the neuron's amount, and
    each later one adds its part, so a neuron that one unit pushes keeps
    exactly that unit's part.

    Example:
        >>> a = NeuronUnit("tiny", 1, 5)
        >>> pair = NeuronSet("tiny", "pair", [(1, 5, 1.0), (0, 3, -0.5)], "x", 2.0)
        >>> one_row = [[(), ((pair, 4.0), (a, 1.0))]]  # two passes
        >>> zero, one = layer_tables(one_row, [(pair, a)])
        >>> [str(neuron) for neuron in zero.units], zero.amounts[:, 0, 0].tolist()
        (['tiny:L0:N3'], [0.0, -2.0])
        >>> one.amounts[:, 0, 0].tolist(), one.active[:, 0, 0].tolist(), one.busy
        ([0.0, 5.0], [False, True], (False, True))
    """
    units = list(dict.fromkeys(unit for row in order for unit in row))
    columns: dict[int, dict[NeuronUnit, int]] = {}  # each layer's, by index
    found = {neuron for unit in units for neuron, _ in unit_neurons(unit)}
    for neuron in sorted(found, key=attrgetter("layer", "index")):
        here = columns.setdefault(neuron.layer, {})
        here[neuron] = len(here)
    parts: dict[Unit, dict[int, tuple[list[int], list[float]]]] = {}
    for unit in units:  # each unit's columns and coefficients, layer by layer
        for neuron, coefficient in unit_neurons(unit):
            mine = parts.setdefault(unit, {}).setdefault(neuron.layer, ([], []))
            mine[0].append(columns[neuron.layer][neuron])
            mine[1].append(coefficient)
    shape = (len(by_pass[0]) if by_pass else 0, len(by_pass))  # (passes, rows)
    amounts = {
        layer: torch.zeros(*shape, len(here), dtype=torch.float64)
        for layer, here in columns.items()
    }
    active = {
        layer: torch.zeros(*shape, len(here), dtype=torch.bool)
        for layer, here in columns.items()
    }
    for row, (row_units, row_passes) in enumerate(zip(order, by_pass, strict=True)):
        place = {unit: i for i, unit in enumerate(row_units)}
        on: list[tuple[list[int], list[float]]] = [([], []) for _ in row_units]
        for p, listed in enumerate(row_passes):  # each unit's passes and amounts
            for unit, value in listed:
                on[place[unit]][0].append(p)
                on[place[unit]][1].append(value)
        for unit, (at, values) in zip(row_units, on, strict=True):
            if not at:
                continue  # it pushes nothing on the table's passes
            passes = torch.tensor(at)[:, None]
            unit_amounts = torch.tensor(values, dtype=torch.float64)[:, None]
            for layer, (where, coefficients) in parts[unit].items():
                neurons = torch.tensor(where)
                part = unit_amounts * torch.tensor(coefficients, dtype=torch.float64)
                row_amounts, row_active = amounts[layer][:, row], active[layer][:, row]
                had = row_active[passes, neurons]
                before = row_amounts[passes, neurons]
                row_amounts[passes, neurons] = torch.where(had, before + part, part)
                row_active[passes, neurons] = True
    return tuple(
        LayerTable(
            layer,
            tuple(here),
            amounts[layer],
            active[layer],
            tuple(bool(pushed) for pushed in active[layer].flatten(1).any(dim=1)),
        )
        for layer, here in columns.items()
    )
