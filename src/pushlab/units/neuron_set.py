"""A set of MLP neurons pushed together as one unit, such as a persona.

A NeuronSet names a registered model, a name, its entries and where they
came from. Each entry is a layer, a neuron and a coefficient, with layers
and neurons counted from 0. The set's write direction is the
coefficient-weighted sum of its neurons' down_proj columns,

    w = c_1 * W_down_{l_1}[:, n_1] + c_2 * W_down_{l_2}[:, n_2] + ...

and a dose alpha pushes the residual stream by alpha * size along w / |w|:
entry i's neuron gets alpha * c_i * size / |w| added to its MLP value, at
its own layer (decision 0019). size is the length of the set's push at dose
1. A set made from Mayner's persona data has his strength as its size, 4
times the length of the role's persona vector (pushlab.units.mayner), so
dose 1 steers as his code does. A set with no size takes the residual size
of its layer, as a NeuronUnit's dose does (pushlab.units.dose): a set of one
neuron with coefficient 1 then pushes exactly as that NeuronUnit. A residual
size belongs to one layer, so a set across layers needs a size.

A set's ID is model:set:name, such as qwen3-8b:set:pirate-top50, which can
never be read as a neuron's ID.
"""

import math
import numbers
import re
from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import nn

from pushlab.models import discovery
from pushlab.models.discovery import ModelInfo
from pushlab.models.registry import REGISTRY
from pushlab.units.neuron import UNQUANTIZED, NeuronUnit

# The same form as schedule and experiment names: lowercase letters and
# digits joined by single hyphens, so a name can go into IDs and file names.
SET_NAME = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")


def _is_whole(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite(value: object) -> bool:
    if not isinstance(value, numbers.Real) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # an int too large for a float
        return False


class SetEntry(NamedTuple):
    """One neuron of a set: its layer and index, counted from 0, and its
    coefficient, its weight in the set's write direction."""

    layer: int
    index: int
    coefficient: float


def _entry(where: str, item: object) -> SetEntry:
    """An entry checked and made a SetEntry, the coefficient a float."""
    if not isinstance(item, tuple | list) or len(item) != 3:
        raise ValueError(
            f"{where} must be (layer, neuron, coefficient), such as (14, 8949, "
            f"1.0), not {item!r}"
        )
    layer, index, coefficient = item
    for name, value in (("layer", layer), ("neuron", index)):
        if not _is_whole(value) or value < 0:
            raise ValueError(
                f"{where}: {name} must be a Python int from 0, not {value!r}"
            )
    if not _is_finite(coefficient) or coefficient == 0:
        raise ValueError(
            f"{where}: the coefficient must be a finite number other than 0, "
            f"not {coefficient!r}"
        )
    return SetEntry(layer, index, float(coefficient))


@dataclass(frozen=True)
class NeuronSet:
    """Some MLP neurons of one registered model, pushed together by one dose.

    entries are (layer, neuron, coefficient) triples, given as a list or a
    tuple and kept in order as SetEntry records; each neuron appears once
    and no coefficient is 0. source says where the set comes from, such as
    a data file and the rows taken. size, the length of the set's push at
    dose 1, is a finite number above 0, or None for the residual size of
    the set's one layer. A set is a plain value: equal fields make equal
    sets, which can be dictionary keys.

    Example:
        >>> pair = NeuronSet(
        ...     "tiny", "pair", [(1, 5, 1.0), (1, 9, -0.5)], "a made-up pair"
        ... )
        >>> str(pair), pair.layers
        ('tiny:set:pair', (1,))
        >>> [str(unit) for unit in pair.neurons()]
        ['tiny:L1:N5', 'tiny:L1:N9']
    """

    model_name: str
    name: str
    entries: tuple[SetEntry, ...]
    source: str
    size: float | None = None

    def __post_init__(self) -> None:
        """Check every field, as described in the class's docstring."""
        model_name, name, entries = self.model_name, self.name, self.entries
        if not isinstance(model_name, str) or model_name not in REGISTRY:
            known = ", ".join(sorted(REGISTRY))
            raise ValueError(f"{model_name!r} is not in the model registry ({known})")
        if not isinstance(name, str) or not SET_NAME.fullmatch(name):
            raise ValueError(
                "a set's name must be lowercase letters and digits joined by single "
                f"hyphens, such as pirate-top50, not {name!r}"
            )
        if not isinstance(entries, list | tuple) or not entries:
            raise ValueError(
                "entries must be a non-empty list of (layer, neuron, coefficient), "
                f"not {entries!r}"
            )
        checked = tuple(_entry(f"entries[{i}]", item) for i, item in enumerate(entries))
        seen: dict[tuple[int, int], int] = {}
        for i, entry in enumerate(checked):
            key = (entry.layer, entry.index)
            if key in seen:
                raise ValueError(
                    f"entries[{i}] repeats layer {entry.layer} neuron {entry.index} "
                    f"of entries[{seen[key]}]; give each neuron once"
                )
            seen[key] = i
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError(
                "source must say where the set comes from, such as a file and "
                f"the rows taken, not {self.source!r}"
            )
        size = self.size
        if size is not None and not (_is_finite(size) and size > 0):
            raise ValueError(
                f"size must be None or a finite number above 0, not {size!r}"
            )
        layers = sorted({entry.layer for entry in checked})
        if size is None and len(layers) > 1:
            raise ValueError(
                f"{model_name}:set:{name} has neurons in layers "
                f"{', '.join(map(str, layers))}, so it needs a size, the length of "
                "its push at dose 1: a residual size belongs to one layer"
            )
        # A frozen dataclass sets its fields through object.__setattr__.
        object.__setattr__(self, "entries", checked)
        object.__setattr__(self, "size", None if size is None else float(size))

    def __str__(self) -> str:
        """The set's ID, such as qwen3-8b:set:pirate-top50."""
        return f"{self.model_name}:set:{self.name}"

    @property
    def layers(self) -> tuple[int, ...]:
        """The layers the set has neurons in, from the lowest."""
        return tuple(sorted({entry.layer for entry in self.entries}))

    def neurons(self) -> tuple[NeuronUnit, ...]:
        """Each entry's neuron as a NeuronUnit, in the entries' order."""
        return tuple(
            NeuronUnit(self.model_name, e.layer, e.index) for e in self.entries
        )

    def validate(self, model_info: ModelInfo) -> None:
        """Raise ValueError unless a model of this shape has every neuron.

        Only the numbers are checked, as NeuronUnit.validate checks them.

        Example:
            >>> shape = ModelInfo(layers=2, hidden_size=64, mlp_size=128)
            >>> NeuronSet("tiny", "x", [(1, 128, 1.0)], "made up").validate(
            ...     shape
            ... )  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            ValueError: tiny:set:x: tiny:L1:N128: neuron 128 is out of range; ...
        """
        for unit in self.neurons():
            try:
                unit.validate(model_info)
            except ValueError as error:
                raise ValueError(f"{self}: {error}") from None

    def write_direction(self, model: nn.Module) -> torch.Tensor:
        """The coefficient-weighted sum of the neurons' down_proj columns, in float32.

        model is the model itself, as for NeuronUnit.write_direction, and must
        have the registry's shape for the set's model. The sum is taken in
        float64, layer by layer and each layer's entries in order, and
        rounded to float32 once, so a set of one neuron with coefficient 1
        gives exactly that neuron's write direction, every bit of it. The
        vector is a new one, on the device of the first layer's weight.

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
            >>> one = NeuronSet("tiny", "one", [(1, 5, 1.0)], "made up")
            >>> unit = NeuronUnit("tiny", 1, 5)
            >>> bool(
            ...     torch.equal(one.write_direction(model), unit.write_direction(model))
            ... )
            True
        """
        if not isinstance(model, nn.Module):
            raise ValueError(
                "model must be the model itself, such as load_model(...).model, "
                f"not a {type(model).__name__}"
            )
        shape = discovery.model_info(model)
        spec = REGISTRY[self.model_name]
        expected = ModelInfo(spec.layers, spec.hidden_size, spec.mlp_size)
        if shape != expected:
            raise ValueError(
                f"{self} is a set of {self.model_name}'s neurons, shaped {expected}, "
                f"but this model is shaped {shape}; is it another model?"
            )
        self.validate(shape)
        mlps = dict(discovery.find_mlp_modules(model))
        total: torch.Tensor | None = None
        for layer in self.layers:
            weight = mlps[layer].down_proj.weight
            if weight.dtype not in UNQUANTIZED:
                raise ValueError(
                    f"{self}: layer {layer}'s down_proj holds {weight.dtype} weights; "
                    "quantized models are not supported"
                )
            here = [entry for entry in self.entries if entry.layer == layer]
            columns = weight.detach()[:, [entry.index for entry in here]]
            coefficients = torch.tensor(
                [entry.coefficient for entry in here],
                dtype=torch.float64,
                device=weight.device,
            )
            # Products and sums in float64, the entries added in order from the
            # first product, so a lone column times 1.0 stays exact, -0.0 too.
            products = columns.to(torch.float64) * coefficients
            part = products[:, 0]
            for k in range(1, products.shape[1]):
                part = part + products[:, k]
            total = part if total is None else total + part.to(total.device)
        assert total is not None  # a set has at least one entry
        return total.to(torch.float32)


# Anything a schedule can push: one neuron, or a set of them.
Unit = NeuronUnit | NeuronSet
UNIT_TYPES = (NeuronUnit, NeuronSet)


def is_unit(value: object) -> bool:
    """True for a NeuronUnit or a NeuronSet.

    Example:
        >>> is_unit(NeuronUnit("tiny", 1, 5)), is_unit("tiny:L1:N5")
        (True, False)
    """
    return isinstance(value, UNIT_TYPES)


def unit_layers(unit: Unit) -> tuple[int, ...]:
    """The layers a unit pushes: a neuron's own, or each of a set's.

    Example:
        >>> unit_layers(NeuronUnit("tiny", 1, 5))
        (1,)
        >>> unit_layers(NeuronSet("tiny", "x", [(1, 5, 1.0), (0, 3, 2.0)], "x", 9.0))
        (0, 1)
    """
    if isinstance(unit, NeuronSet):
        return unit.layers
    if isinstance(unit, NeuronUnit):
        return (unit.layer,)
    raise ValueError(f"a unit is a NeuronUnit or a NeuronSet, not {unit!r}")


def unit_neurons(unit: Unit) -> tuple[tuple[NeuronUnit, float], ...]:
    """Each neuron a unit pushes, with what it gets per unit of the unit's amount.

    A NeuronUnit gets the whole amount (1.0); a set's neuron gets its
    coefficient times the set's amount (decision 0019).

    Example:
        >>> [
        ...     (str(n), c)
        ...     for n, c in unit_neurons(
        ...         NeuronSet("tiny", "x", [(1, 5, 1.0), (1, 9, -0.5)], "made up")
        ...     )
        ... ]
        [('tiny:L1:N5', 1.0), ('tiny:L1:N9', -0.5)]
    """
    if isinstance(unit, NeuronSet):
        return tuple(
            (neuron, entry.coefficient)
            for neuron, entry in zip(unit.neurons(), unit.entries, strict=True)
        )
    if isinstance(unit, NeuronUnit):
        return ((unit, 1.0),)
    raise ValueError(f"a unit is a NeuronUnit or a NeuronSet, not {unit!r}")
