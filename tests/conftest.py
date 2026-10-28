"""Shared test setup: fixed random seeds, and GPU tests skipped unless asked for."""

import random
import re

import numpy as np
import pytest
import torch

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
