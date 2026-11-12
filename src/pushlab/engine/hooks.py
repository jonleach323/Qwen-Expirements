"""Hooks that follow a DoseTable: every row adds its own amounts on its own passes.

doses_active installs one forward pre-hook on the down_proj of each layer the
table pushes. Before each pass the decoding loop sets the StepState, and the
hook reads that pass's amounts for every row from the table and adds them in
one vectorized step (pushlab.engine.push.add_amounts): row r's amounts go to
row r's values only, and never at a position the state marks as padding,
such as a left-padded prompt position or a row that has finished. A pass on
which no row pushes anything in the layer leaves down_proj's input as it
was, as S1's hook does.

The table holds one amount per pushed neuron, a NeuronSet's neurons among
them, each already its share of the set's push (decision 0019), so a set
across layers is pushed by the hook of each of its layers.

Rows never touch each other: a row's values change only by its own amounts,
and a value that row does not push keeps every bit, so in one batch layout a
row's numbers depend on its own prompt and schedule alone.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch
from torch import nn

from pushlab.engine.push import StepState, add_amounts
from pushlab.models.discovery import find_mlp_modules
from pushlab.schedules.compile import DoseTable, LayerTable, pass_index


def _reach(passes: int) -> str:
    """The passes a table of this many covers, in words."""
    if passes == 1:
        return "the prompt pass alone"
    return f"the prompt pass and reply tokens 0 to {passes - 2}"


def _table_hook(
    table: LayerTable, state: StepState
) -> Callable[[nn.Module, tuple], tuple | None]:
    """A forward pre-hook that adds the current pass's amounts, row by row, to
    the table's pushed neurons."""
    neurons = torch.tensor([neuron.index for neuron in table.units])
    # The table moved to the activations' device and add type, once.
    moved: dict[tuple[torch.device, torch.dtype], tuple] = {}

    def hook(module: nn.Module, args: tuple) -> tuple | None:
        p = pass_index(state.reply_token)
        if p >= len(table.busy):
            raise ValueError(
                f"the dose table covers {_reach(len(table.busy))}, not reply token "
                f"{state.reply_token}; compile it with a larger max_new_tokens"
            )
        values = args[0]
        rows = table.amounts.shape[1]
        if values.dim() == 3 and values.shape[0] != rows:
            raise ValueError(
                f"the dose table has {rows} rows, but this pass has {values.shape[0]}"
            )
        if not table.busy[p]:
            return None  # nothing pushed in this layer: the input passes on untouched
        compute = torch.promote_types(values.dtype, torch.float32)
        key = (values.device, compute)
        if key not in moved:
            # float64 to the add's type on the CPU, then over to the device.
            moved[key] = (
                neurons.to(values.device),
                table.amounts.to(dtype=compute).to(values.device),
                table.active.to(values.device),
            )
        index, amounts, active = moved[key]
        pushed = add_amounts(values, index, amounts[p], active[p], state.padding)
        return (pushed, *args[1:])

    return hook


@contextmanager
def doses_active(
    model: nn.Module, table: DoseTable, state: StepState
) -> Iterator[DoseTable]:
    """Install the table's hooks for the length of a with block.

    Each pushed layer gets one forward pre-hook on its down_proj, which reads
    state on every pass; a table that pushes nothing installs none. The hooks
    are removed when the block ends, even after an error. The table must
    have been compiled for this model (compile checks its units, neurons
    and NeuronSets, against it), and a pass must have as many rows as the
    table.

    Example:
        >>> from transformers import Qwen3Config, Qwen3ForCausalLM
        >>> from pushlab.schedules.compile import compile
        >>> from pushlab.schedules.model import Schedule
        >>> from pushlab.schedules.shapes import constant
        >>> from pushlab.units.neuron import NeuronUnit
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
        >>> recipe = constant(NeuronUnit("tiny", 1, 5), 1.0)
        >>> table = compile([recipe, Schedule("none")], model, [7.0, 10.0], 4)
        >>> state = StepState()
        >>> with doses_active(model, table, state), torch.no_grad():
        ...     state.at_prompt()
        ...     logits = model(
        ...         input_ids=torch.tensor([[5, 17, 42], [5, 17, 42]])
        ...     ).logits
        >>> bool(torch.equal(logits[0], logits[1]))  # only row 0 was pushed
        False
    """
    if not isinstance(table, DoseTable):
        raise ValueError(f"table must be a DoseTable, not {type(table).__name__}")
    if not isinstance(state, StepState):
        raise ValueError(f"state must be a StepState, not {type(state).__name__}")
    mlps = dict(find_mlp_modules(model))
    for layer in table.layers:
        if layer.layer not in mlps:
            raise ValueError(
                f"the dose table pushes layer {layer.layer}, which this model lacks"
            )
    handles = []
    try:
        for layer in table.layers:
            hook = _table_hook(layer, state)
            handles.append(mlps[layer.layer].down_proj.register_forward_pre_hook(hook))
        yield table
    finally:
        for handle in handles:
            handle.remove()
