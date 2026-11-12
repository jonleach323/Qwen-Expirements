"""The dose rule: how much a dose alpha adds to one neuron, or to a set's.

The research plan's rule, used by the evil-Qwen recipe and the Qwen3-8B
persona library, is c = alpha * r_L / ||W_down[:, n]||. Here r_L is the
median residual size at the neuron's layer L, and ||W_down[:, n]|| the
length of its write direction. Adding c to the neuron adds c times its
write direction to the residual stream, a push of length |alpha| * r_L: at
dose 1, as large as the layer's typical signal. A negative dose pushes the
other way.

A NeuronSet follows the same rule with its own write direction w, the
coefficient-weighted sum of its neurons' columns, and its own size: its
amount is alpha * size / ||w||, and entry i's neuron gets that amount times
its coefficient c_i. The set's push is then alpha * size * w / ||w||, of
length |alpha| * size. size is the set's own when it has one, as a set from
Mayner's persona data does, and else r_L of its one layer, so a set of one
neuron with coefficient 1 adds exactly that neuron's amount (decision 0019).
"""

import math
import numbers
from dataclasses import dataclass

import torch
from torch import nn

from pushlab.models.discovery import model_info
from pushlab.units.neuron import NeuronUnit
from pushlab.units.neuron_set import NeuronSet, Unit


def _is_finite_number(value: object) -> bool:
    """True for a finite int or float, such as 2 or 10.5; False for True or nan."""
    return (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


@dataclass(frozen=True)
class PerPromptNorms:
    """Each prompt's own residual sizes: the per_prompt calibration mode.

    rows[i] holds prompt i's median residual size at every layer, from layer
    0, measured on that prompt's own tokens
    (pushlab.units.calibration_mode.per_prompt_norms). Given as the norms of
    compile or generate, the row of prompt i takes rows[i], where a plain
    list of sizes, the fixed calibration, is shared by every row. rows may
    be given as lists; they are kept as tuples of floats, each finite and
    above 0, every row as long.

    Example:
        >>> norms = PerPromptNorms([[7.0, 10.0], [6.5, 11.0]])
        >>> len(norms), norms.rows[1]
        (2, (6.5, 11.0))
    """

    rows: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        """Refuse rows that are not one list of sizes per prompt."""
        if not isinstance(self.rows, list | tuple) or not self.rows:
            raise ValueError(
                "rows must be a non-empty list, one list of sizes per prompt, "
                f"not {self.rows!r}"
            )
        rows = []
        for i, row in enumerate(self.rows):
            if not isinstance(row, list | tuple) or not row:
                raise ValueError(
                    f"rows[{i}] must be a list of sizes, one per layer, not {row!r}"
                )
            for layer, norm in enumerate(row):
                if not (_is_finite_number(norm) and norm > 0):
                    raise ValueError(
                        f"rows[{i}][{layer}] must be a finite number above 0, "
                        f"not {norm!r}"
                    )
            rows.append(tuple(float(norm) for norm in row))
        if len({len(row) for row in rows}) != 1:
            raise ValueError("every prompt's row needs the same number of layers")
        # A frozen dataclass sets its fields through object.__setattr__.
        object.__setattr__(self, "rows", tuple(rows))

    def __len__(self) -> int:
        """How many prompts, one row each."""
        return len(self.rows)


def amount(
    unit: Unit,
    alpha: float,
    model: nn.Module,
    norms: list[float] | tuple[float, ...],
) -> float:
    """The number to add to unit's neuron for dose alpha: alpha * r_L / length.

    model is the unit's own model (see NeuronUnit.write_direction). norms is
    a list (or tuple) of the model's median residual sizes, one per layer
    from layer 0, each a finite number above 0; calibration will measure
    them. The write direction's length is taken in float32, as the published
    replications did. The result is a Python float.

    For a NeuronSet it is the amount per unit of coefficient, alpha * size /
    length, with the set's size, or r_L of its one layer if it has none:
    entry i's neuron gets it times its coefficient
    (pushlab.units.neuron_set.unit_neurons). norms are checked all the same.

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
        >>> with torch.no_grad():  # 64 entries of 0.25 make a length of 2
        ...     model.model.layers[1].mlp.down_proj.weight[:, 5] = 0.25
        >>> unit = NeuronUnit("tiny", layer=1, index=5)
        >>> amount(unit, 1.0, model, norms=[7.0, 10.0])  # 1 * 10 / 2
        5.0
        >>> amount(unit, -0.5, model, norms=[7.0, 10.0])
        -2.5
        >>> doubled = NeuronSet("tiny", "doubled", [(1, 5, 2.0)], "made up")
        >>> amount(doubled, 1.0, model, norms=[7.0, 10.0])  # 10 / 4, times 2 is 5
        2.5
    """
    if not isinstance(unit, NeuronUnit | NeuronSet):
        raise ValueError(f"unit must be a NeuronUnit or a NeuronSet, not {unit!r}")
    if not _is_finite_number(alpha):
        raise ValueError(f"alpha must be a finite number, not {alpha!r}")
    direction = unit.write_direction(model)  # checks the model and the neuron
    layers = model_info(model).layers
    if not isinstance(norms, list | tuple):
        raise ValueError(
            "norms must be a list of numbers, one per layer, "
            f"not a {type(norms).__name__}"
        )
    if len(norms) != layers:
        raise ValueError(
            f"norms has {len(norms)} values, but the model has {layers} layers; "
            "give one per layer, from layer 0"
        )
    for layer, norm in enumerate(norms):
        if not (_is_finite_number(norm) and norm > 0):
            raise ValueError(
                f"norms[{layer}] must be a finite number above 0, not {norm!r}"
            )
    length = float(torch.linalg.vector_norm(direction))
    if not (math.isfinite(length) and length > 0):
        raise ValueError(
            f"{unit}'s write direction has length {length}, so no dose can scale it"
        )
    if isinstance(unit, NeuronSet):
        size = unit.size if unit.size is not None else norms[unit.layers[0]]
    else:
        size = norms[unit.layer]
    # size / length first, so the amount is exactly alpha times dose 1's amount.
    return float(alpha) * (float(size) / length)
