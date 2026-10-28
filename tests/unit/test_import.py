"""Smoke tests: the package installs, imports and reports its version."""

import importlib

import pytest

import pushlab

SUBPACKAGES = [
    "models",
    "units",
    "schedules",
    "engine",
    "runs",
    "store",
    "scoring",
    "readouts",
    "discovery",
    "style",
    "conversations",
    "backlog",
    "repair",
    "training",
    "analysis",
]


def test_version_is_a_string() -> None:
    assert isinstance(pushlab.__version__, str)
    assert pushlab.__version__


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    importlib.import_module(f"pushlab.{name}")
