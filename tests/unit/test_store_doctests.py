"""The docstring examples of the store's steps, manifest and query modules run."""

import doctest
import importlib

import pytest


@pytest.mark.parametrize("name", ["steps", "manifest", "query"])
def test_doctests(name: str) -> None:
    result = doctest.testmod(importlib.import_module(f"pushlab.store.{name}"))
    assert result.attempted > 0
    assert result.failed == 0
