"""The models pushlab supports, with their published shapes."""

from dataclasses import dataclass
from typing import Literal

# How a model is packaged, which decides how it gets loaded:
# "causal_lm" is a plain text model, "vision_language" has its language model
# nested inside a vision-language wrapper (Qwen3.5), and "test" is built in
# code by the test suite and never downloaded.
Kind = Literal["causal_lm", "vision_language", "test"]


@dataclass(frozen=True)
class ModelSpec:
    """One supported model: where it comes from and the shape of its MLPs.

    Example:
        >>> spec = REGISTRY["qwen3-8b"]
        >>> spec.hf_id, spec.layers, spec.mlp_size
        ('Qwen/Qwen3-8B', 36, 12288)
    """

    name: str
    hf_id: str
    revision: str
    layers: int
    hidden_size: int
    mlp_size: int
    kind: Kind


# Each Hugging Face model gets pinned to a commit hash rather than "main", so
# a new upload cannot change results unnoticed. Qwen3.5-4B is pinned to the
# model commit the S0 baseline used (docs/decisions/0005-s0-baseline.md).
# TODO: pin qwen3-8b, still on "main", once it is first downloaded and run.
# The tiny test model has nothing to pin.
_SPECS = [
    ModelSpec(
        name="qwen3.5-4b",
        hf_id="Qwen/Qwen3.5-4B",
        revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
        layers=32,
        hidden_size=2560,
        mlp_size=9216,
        kind="vision_language",
    ),
    ModelSpec(
        name="qwen3-8b",
        hf_id="Qwen/Qwen3-8B",
        revision="main",
        layers=36,
        hidden_size=4096,
        mlp_size=12288,
        kind="causal_lm",
    ),
    # The tiny test model from tests/conftest.py. It is built in code, so its
    # ID is a local name, not a Hugging Face repository, and nothing is pinned.
    ModelSpec(
        name="tiny",
        hf_id="local:tiny-qwen3",
        revision="main",
        layers=2,
        hidden_size=64,
        mlp_size=128,
        kind="test",
    ),
]

REGISTRY: dict[str, ModelSpec] = {spec.name: spec for spec in _SPECS}
