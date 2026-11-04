"""The dose rule: how much a dose alpha adds to one neuron.

The research plan's rule, used by the evil-Qwen recipe and the Qwen3-8B
persona library, is c = alpha * r_L / ||W_down[:, n]||. Here r_L is the
median residual size at the neuron's layer L, and ||W_down[:, n]|| the
length of its write direction. Adding c to the neuron adds c times its
write direction to the residual stream, a push of length |alpha| * r_L: at
dose 1, as large as the layer's typical signal. A negative dose pushes the
other way.
"""

import math
import numbers
from collections.abc import Sequence

import torch
from torch import nn

from pushlab.models.discovery import model_info
from pushlab.units.neuron import NeuronUnit


def _is_finite_number(value: object) -> bool:
    """True for a finite int or float, such as 2 or 10.5; False for True or nan."""
    return (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def amount(
    unit: NeuronUnit, alpha: float, model: nn.Module, norms: Sequence[float]
) -> float:
    """The number to add to unit's neuron for dose alpha: alpha * r_L / length.

    model is the unit's own model (see NeuronUnit.write_direction). norms is
    a list (or tuple) of the model's median residual sizes, one per layer
    from layer 0, each a finite number above 0; calibration will measure
    them. The write direction's length is taken in float32, as the published
    replications did. The result is a Python float.

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
    """
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
    # r_L / length first, so the amount is exactly alpha times dose 1's amount.
    return float(alpha) * (float(norms[unit.layer]) / length)
