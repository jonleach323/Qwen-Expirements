"""The shared fixture fixes every random seed before each test."""

import random

import numpy as np
import pytest
import torch


# Two identical runs: the second only passes if seeds are reset for every test.
@pytest.mark.parametrize("run", [1, 2])
def test_seeds_are_fixed_before_each_test(fixed_seeds: int, run: int) -> None:
    assert random.random() == random.Random(fixed_seeds).random()
    assert np.random.random() == np.random.RandomState(fixed_seeds).random_sample()
    expected = torch.rand(3, generator=torch.Generator().manual_seed(fixed_seeds))
    assert torch.equal(torch.rand(3), expected)
