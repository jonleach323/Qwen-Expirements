"""The shared setup fixes every random seed before each test and its fixtures."""

import random

import numpy as np
import pytest
import torch


# Two identical runs: the second only passes if seeds are reset for every test.
@pytest.mark.parametrize("run", [1, 2])
def test_seeds_are_fixed_before_each_test(
    request: pytest.FixtureRequest, run: int
) -> None:
    # Not requested by name, so this fails if the fixture stops being autouse.
    assert "fixed_seeds" in request.fixturenames
    seed = request.getfixturevalue("fixed_seeds")
    assert random.random() == random.Random(seed).random()
    assert np.random.random() == np.random.RandomState(seed).random_sample()
    expected = torch.rand(3, generator=torch.Generator().manual_seed(seed))
    assert torch.equal(torch.rand(3), expected)


@pytest.fixture(scope="module")
def module_scoped_draw() -> torch.Tensor:
    """Stands in for a module-scoped fixture such as a tiny random model."""
    return torch.rand(3)


def test_seeds_are_fixed_before_module_scoped_fixtures(
    request: pytest.FixtureRequest, module_scoped_draw: torch.Tensor
) -> None:
    seed = request.getfixturevalue("fixed_seeds")
    expected = torch.rand(3, generator=torch.Generator().manual_seed(seed))
    assert torch.equal(module_scoped_draw, expected)
