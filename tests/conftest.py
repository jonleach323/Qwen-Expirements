"""Shared test setup: seeds, GPU-test skipping, and the tiny test model."""

import os
import random
import re
from collections.abc import Callable

import numpy as np
import pytest
import torch

# Tests never download from Hugging Face; an accidental download fails fast.
# Set before transformers is imported: huggingface_hub reads it once, then.
os.environ["HF_HUB_OFFLINE"] = "1"

from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

pytest_plugins = ["pytester"]

TEST_SEED = 0


def _gpu_requested(markexpr: str) -> bool:
    """True if gpu appears in a -m expression under an even number of `not`s."""
    negated = [False]  # one entry per open parenthesis
    pending_not = False
    for token in re.findall(r"[()]|[^\s()]+", markexpr):
        if token == "not":
            pending_not = not pending_not
        elif token == "(":
            negated.append(negated[-1] != pending_not)
            pending_not = False
        elif token == ")":
            negated.pop()
        else:
            if token == "gpu" and negated[-1] == pending_not:
                return True
            pending_not = False
    return False


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip gpu-marked tests unless the run was started with -m gpu."""
    if _gpu_requested(config.getoption("markexpr")):
        return
    skip_gpu = pytest.mark.skip(reason="needs a GPU; run with -m gpu")
    for item in items:
        if item.get_closest_marker("gpu") is not None:
            item.add_marker(skip_gpu)


def _seed_everything() -> None:
    random.seed(TEST_SEED)
    np.random.seed(TEST_SEED)
    torch.manual_seed(TEST_SEED)  # also seeds every GPU


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Seed before any fixture is set up, so module-scoped ones are fixed too."""
    _seed_everything()


@pytest.fixture(autouse=True)
def fixed_seeds() -> int:
    """Seed Python, NumPy and torch again right before each test runs."""
    _seed_everything()
    return TEST_SEED


def _build_tiny_model() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        # Qwen3 sets head_dim apart from the hidden size. Here the attention
        # width (4 x 32 = 128) differs from the hidden size (64), as it does
        # in Qwen3.5-4B (16 x 256 vs 2560). The class default would be 128.
        head_dim=32,
        vocab_size=512,
        dtype="float32",
    )
    # Seed only inside a forked random state, so the weights are always the
    # same and the test's own random numbers are left untouched.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(TEST_SEED)
        model = Qwen3ForCausalLM(config)
    return model.to(torch.float32).eval()


@pytest.fixture
def make_tiny_model() -> Callable[[], Qwen3ForCausalLM]:
    """A function that builds a fresh tiny model, identical on every call."""
    return _build_tiny_model


@pytest.fixture
def tiny_model(make_tiny_model: Callable[[], Qwen3ForCausalLM]) -> Qwen3ForCausalLM:
    """A 2-layer Qwen3-style model with seeded random weights, float32, eval mode."""
    return make_tiny_model()
