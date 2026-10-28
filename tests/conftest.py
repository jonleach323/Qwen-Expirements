"""Shared test setup: fixed random seeds, and GPU tests skipped unless asked for."""

import random

import numpy as np
import pytest
import torch

pytest_plugins = ["pytester"]

TEST_SEED = 0


def _gpu_requested(markexpr: str) -> bool:
    """True if a -m expression names the gpu marker without negating it."""
    tokens = markexpr.replace("(", " ").replace(")", " ").split()
    return any(
        token == "gpu" and (i == 0 or tokens[i - 1] != "not")
        for i, token in enumerate(tokens)
    )


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


@pytest.fixture(autouse=True)
def fixed_seeds() -> int:
    """Seed Python, NumPy and torch (CPU and every GPU) before each test."""
    random.seed(TEST_SEED)
    np.random.seed(TEST_SEED)
    torch.manual_seed(TEST_SEED)
    return TEST_SEED
