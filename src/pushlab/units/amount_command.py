"""`pushlab amount`: what a dose adds to one neuron, from the saved calibration.

The amount is alpha * r_L / ||W_down[:, n]|| (pushlab.units.dose): the
layer's residual size comes from the model's calibration file, and the length
of the neuron's write direction from its weights, so the command loads the
model. It refuses before loading when the unit, the dose, the download or
the calibration is missing or wrong.
"""

import math
import sys

from pushlab.models.discovery import ModelInfo
from pushlab.models.registry import REGISTRY
from pushlab.store.folders import shown_path
from pushlab.units.neuron import NeuronUnit


def _refuse(message: str) -> int:
    print(f"pushlab amount: {message}", file=sys.stderr)
    return 1


def explain_amount(
    unit: NeuronUnit, alpha: float, size: float, length: float, value: float
) -> list[str]:
    """The lines `pushlab amount` prints for a dose's amount.

    size is the layer's residual size and length the length of the neuron's
    write direction; value, alpha * size / length, is the amount.

    Example:
        >>> unit = NeuronUnit.parse("qwen3.5-4b:L14:N8949")
        >>> print("\\n".join(explain_amount(unit, 1.0, 10.917, 0.46472, 23.4917)))
        qwen3.5-4b:L14:N8949 at dose +1 adds +23.4917 to the neuron's value.
          Layer 14's residual size is 10.9170 and the neuron's write direction is
          0.4647 long, so the push moves the residual stream 10.9170 along it.
    """
    return [
        f"{unit} at dose {alpha:+g} adds {value:+.4f} to the neuron's value.",
        f"  Layer {unit.layer}'s residual size is {size:.4f} and the neuron's "
        "write direction is",
        f"  {length:.4f} long, so the push moves the residual stream "
        f"{abs(alpha) * size:.4f} along it.",
    ]


def run_amount(unit_id: str, alpha: float) -> int:
    """Print what dose alpha adds to the neuron unit_id; return the exit status.

    Returns 0 once printed, and 1 with the reason on stderr when it refuses.

    Example:
        >>> run_amount("qwen3.5-4b:L14:N8949", 1.0)  # doctest: +SKIP
        0
    """
    try:
        unit = NeuronUnit.parse(unit_id)
        spec = REGISTRY[unit.model_name]
        unit.validate(ModelInfo(spec.layers, spec.hidden_size, spec.mlp_size))
    except ValueError as error:
        return _refuse(f"--unit {unit_id!r}: {error}")
    if not math.isfinite(alpha):
        return _refuse(f"--alpha must be a finite number, not {alpha}")

    # Imported here, so that --help answers without first loading torch.
    import torch

    from pushlab.models import load
    from pushlab.units import calibrate
    from pushlab.units.dose import amount

    try:
        revision = load.downloaded_revision(unit.model_name)
        calibration = calibrate.load_calibration(unit.model_name, revision)
    except (FileNotFoundError, ValueError) as error:
        return _refuse(str(error))  # names the command that makes a calibration
    path = calibrate.calibration_path(unit.model_name, revision)
    device = load.default_device()
    print(f"Loading {unit.model_name} to read the neuron's weights...")
    loaded = load.load_model(unit.model_name, load.default_dtype(device), device=device)
    value = amount(unit, alpha, loaded.model, list(calibration.norms))
    length = float(torch.linalg.vector_norm(unit.write_direction(loaded.model)))
    size = calibration.norms[unit.layer]
    print("\n".join(explain_amount(unit, alpha, size, length, value)))
    print(f"  Calibration: {shown_path(path)}")
    return 0
