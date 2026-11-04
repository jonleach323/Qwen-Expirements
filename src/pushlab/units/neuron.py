"""One MLP neuron as a unit that can be pushed.

Each layer's MLP computes intermediate values, its neurons. Neuron n writes
into the residual stream through column n of the layer's down_proj, its
"write direction" (research plan, "MLP neurons"). A neuron's ID is
model:L<layer>:N<index>, such as qwen3.5-4b:L14:N8949, with layers and
neurons counted from 0, as in the published evil-neuron recipe.
"""

import re
from dataclasses import dataclass

import torch
from torch import nn

from pushlab.models import discovery
from pushlab.models.discovery import ModelInfo
from pushlab.models.registry import REGISTRY

# The one spelling of an ID: the model's registry name, then L and the layer,
# then N and the neuron, in ASCII digits with no sign, spaces or leading
# zeros. One spelling per neuron lets IDs be compared as text. [0-9] rather
# than \d, because \d (and int()) also accept other scripts' digits, such as
# Arabic-Indic ones.
NEURON_ID = re.compile(
    r"(?P<model>[^:\s]+):L(?P<layer>0|[1-9][0-9]*):N(?P<index>0|[1-9][0-9]*)"
)

# Weight types that hold each write direction as it is. Quantized weights
# (int8, 4-bit, float8) also need their scales, so they are refused.
UNQUANTIZED = (torch.float32, torch.bfloat16, torch.float16, torch.float64)


@dataclass(frozen=True)
class NeuronUnit:
    """One MLP neuron of a registered model: its layer, and its index there.

    Layers and indexes count from 0. str() gives the unit's ID and parse()
    reads it back. Creating a unit checks that its model is in the registry
    and that layer and index are whole numbers from 0; validate() checks
    them against a model's shape.

    Example:
        >>> unit = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
        >>> unit
        NeuronUnit(model_name='qwen3.5-4b', layer=14, index=8949)
        >>> str(unit)
        'qwen3.5-4b:L14:N8949'
    """

    model_name: str
    layer: int
    index: int

    def __post_init__(self) -> None:
        """Refuse a model outside the registry, or a layer or index below 0."""
        if not isinstance(self.model_name, str) or self.model_name not in REGISTRY:
            known = ", ".join(sorted(REGISTRY))
            raise ValueError(
                f"{self.model_name!r} is not in the model registry ({known})"
            )
        for name in ("layer", "index"):
            value = getattr(self, name)
            # bool is a kind of int in Python, but True would be written LTrue.
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a whole number from 0, not {value!r}")

    def __str__(self) -> str:
        """The unit's ID, such as qwen3.5-4b:L14:N8949, which parse() reads."""
        return f"{self.model_name}:L{self.layer}:N{self.index}"

    @classmethod
    def parse(cls, text: str) -> "NeuronUnit":
        """Read an ID, written exactly as str() writes it, into a unit.

        Anything else is refused rather than corrected: spaces, a sign,
        leading zeros, a lowercase l or n. The model must be in the registry.

        Example:
            >>> NeuronUnit.parse("qwen3.5-4b:L13:N7956")
            NeuronUnit(model_name='qwen3.5-4b', layer=13, index=7956)
            >>> NeuronUnit.parse("qwen3.5-4b:L013:N7956")  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            ValueError: 'qwen3.5-4b:L013:N7956' is not a neuron ID such as ...
        """
        match = NEURON_ID.fullmatch(text) if isinstance(text, str) else None
        if match is None:
            raise ValueError(
                f"{text!r} is not a neuron ID such as qwen3.5-4b:L14:N8949: the "
                "model's registry name, then L and the layer, then N and the "
                "neuron, both counted from 0 and written in digits with no "
                "sign, spaces or leading zeros"
            )
        return cls(match["model"], int(match["layer"]), int(match["index"]))

    def validate(self, model_info: ModelInfo) -> None:
        """Raise ValueError unless a model of this shape has the neuron.

        model_info is a model's shape, as discovery.model_info() or
        LoadedModel.info gives it. Only the numbers are checked here;
        write_direction() also checks that a model is the unit's own.

        Example:
            >>> qwen3_5_4b = ModelInfo(layers=32, hidden_size=2560, mlp_size=9216)
            >>> NeuronUnit("qwen3.5-4b", layer=31, index=9215).validate(qwen3_5_4b)
            >>> too_deep = NeuronUnit("qwen3.5-4b", layer=32, index=0)
            >>> too_deep.validate(qwen3_5_4b)  # doctest: +ELLIPSIS
            Traceback (most recent call last):
            ...
            ValueError: qwen3.5-4b:L32:N0: layer 32 is out of range; ...
        """
        if self.layer >= model_info.layers:
            raise ValueError(
                f"{self}: layer {self.layer} is out of range; "
                f"the model's layers are 0 to {model_info.layers - 1}"
            )
        if self.index >= model_info.mlp_size:
            raise ValueError(
                f"{self}: neuron {self.index} is out of range; "
                f"each layer's neurons are 0 to {model_info.mlp_size - 1}"
            )

    def write_direction(self, model: nn.Module) -> torch.Tensor:
        """The neuron's down_proj column, as a new float32 vector.

        model is the model itself, such as load_model(...).model. Its shape
        must be the registry's for the unit's model, which catches most
        mix-ups (L14 N8949 is in range for Qwen3-8B too), and it must have
        the neuron. The vector is hidden-size long, on the weight's device,
        and a copy: changing it never changes the model. bfloat16 weights
        become float32 exactly.

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
            >>> direction = NeuronUnit("tiny", layer=1, index=5).write_direction(model)
            >>> direction.shape, direction.dtype
            (torch.Size([64]), torch.float32)
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
                f"{self} is a neuron of {self.model_name}, shaped {expected}, "
                f"but this model is shaped {shape}; is it another model?"
            )
        self.validate(shape)
        _, mlp = discovery.find_mlp_modules(model)[self.layer]
        weight = mlp.down_proj.weight
        if weight.dtype not in UNQUANTIZED:
            raise ValueError(
                f"{self}: down_proj holds {weight.dtype} weights; quantized "
                "models are not supported"
            )
        # Without copy=True, a float32 model would hand back a view of its own
        # weight, and changing the vector would change the model.
        return weight.detach()[:, self.index].to(torch.float32, copy=True)
