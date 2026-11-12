"""|v_r| exactly as Mayner's code takes it, from his persona-vector files.

A set from Mayner's data has size STRENGTH * |v_r| (pushlab.units.mayner),
where |v_r| is the length of the role's persona vector at layer 18 with
massive-activation tokens left out. His step 6 takes it from the role
vectors in vectors/without/ of his data bundle
(huggingface.co/datasets/wmayner/persona-neurons, CC BY 4.0): each file holds
a "vector" of one row per layer, stored in bfloat16, and step 6 converts the
role's and the default's to float64 before subtracting them at layer 18
(steps/6_role_steering.py, load_vector and role_inputs). vector_norm_from_vectors
does the same, so a set it sizes pushes exactly his strength.

The split's |v_r|, which mayner.vector_norm_from_shares reads back from
split/top_neurons.parquet, measures the same difference from his step 3's
float64 sums of float32 means instead: on his data it is 0.1 to 0.2% off
step 6's (decision 0019), so it only comes close to his strength.

The files are pickled torch files, which could hold code. They are loaded
with torch.load(..., weights_only=True), which reads tensors and plain
containers and refuses anything else; never load them with
weights_only=False.
"""

import math
from pathlib import Path

import numpy as np
import torch

# Mayner's config.TARGET_LAYER: the layer of his persona vectors, the middle
# of Qwen3-8B's 36 layers, counted from 0.
TARGET_LAYER = 18


def load_vector(path: str | Path) -> torch.Tensor:
    """The "vector" of one of Mayner's vector files: (layers, hidden size).

    The file is loaded on the CPU with weights_only=True, so a file that
    holds anything but tensors and plain containers is refused. The vector
    keeps its stored type, bfloat16 in his bundle.

    Example:
        >>> vector = load_vector("data/vectors/without/pirate.pt")  # doctest: +SKIP
        >>> vector.dtype, vector.shape  # doctest: +SKIP
        (torch.bfloat16, torch.Size([36, 4096]))
        >>> load_vector("no/such/pirate.pt")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        FileNotFoundError: there is no vector file at no/such/pirate.pt; ...
    """
    try:
        data = torch.load(Path(path), map_location="cpu", weights_only=True)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"there is no vector file at {path}; vectors/without/<role>.pt and "
            "default.pt come with Mayner's data bundle "
            "(huggingface.co/datasets/wmayner/persona-neurons)"
        ) from None
    except Exception as error:  # an unsafe or broken pickle
        raise ValueError(
            f"{path} is not a plain tensor file that weights_only=True can read: "
            f"{error}"
        ) from None
    vector = data.get("vector") if isinstance(data, dict) else None
    if not isinstance(vector, torch.Tensor) or vector.dim() != 2:
        raise ValueError(
            f"{path} must hold a 'vector', a tensor of one row per layer, as "
            "Mayner's step 4 writes it"
        )
    return vector


def vector_norm_from_vectors(
    role_file: str | Path, default_file: str | Path, layer: int = TARGET_LAYER
) -> float:
    """|v_r| as Mayner's step 6 takes it: the role's vector minus the default's.

    Both vectors are converted to float64 before they are subtracted at
    layer (18, his target layer, unless given), and the length of the
    difference is taken in float64 with numpy, as his load_vector,
    run_prepare and role_inputs do.
    Subtracting the bfloat16 vectors first would move it by up to a few in
    10,000.

    Example:
        >>> norm = vector_norm_from_vectors(  # doctest: +SKIP
        ...     "data/vectors/without/pirate.pt", "data/vectors/without/default.pt"
        ... )
        >>> pirate = top_neurons_set(top, "pirate", 50, norm)  # doctest: +SKIP
    """
    role, default = load_vector(role_file), load_vector(default_file)
    if role.shape != default.shape:
        raise ValueError(
            f"{role_file} holds a vector of shape {tuple(role.shape)} and "
            f"{default_file} one of {tuple(default.shape)}; they must match"
        )
    if not (isinstance(layer, int) and not isinstance(layer, bool)) or not (
        0 <= layer < role.shape[0]
    ):
        raise ValueError(
            f"layer must be a layer of the vectors, 0 to {role.shape[0] - 1}, "
            f"not {layer!r}"
        )
    # His order of operations: float64, subtract, then numpy's norm.
    difference = role.double()[layer] - default.double()[layer]
    norm = float(np.linalg.norm(difference.numpy()))
    if not (math.isfinite(norm) and norm > 0):
        raise ValueError(
            f"{role_file} and {default_file} give |v_r| = {norm} at layer {layer}, "
            "not a finite length above 0; are they the same file?"
        )
    return norm
