"""The judge's backends by name (pushlab.scoring.backends): their default
models, their refusals, and the kind of backend each name makes. A stand-in
anthropic package takes the place of the real one, which pushlab does not
install yet; nothing here reaches the network."""

import doctest
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from pushlab.scoring import backends
from pushlab.scoring.api_backend import ApiBackend
from pushlab.scoring.backends import (
    BACKENDS,
    DEFAULT_MODELS,
    judge_model,
    make_backend,
)
from pushlab.scoring.local_backend import LOCAL_URL, LocalBackend

KEY = "test-key-not-real"
# What importing the factory must not load: the package pushlab does not
# install, and the heavy ones, which would slow every --help.
NOT_LOADED = ("anthropic", "torch", "pandas", "pyarrow")
IMPORT = f"""
import sys
import pushlab.scoring.backends
print([name for name in {NOT_LOADED!r} if name in sys.modules])
"""


@pytest.fixture
def anthropic(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """A stand-in anthropic package and a key; the options each client is
    made with."""
    made: list[dict[str, Any]] = []

    class Anthropic:
        def __init__(self, **options: Any) -> None:
            made.append(options)

    module = types.ModuleType("anthropic")
    module.Anthropic = Anthropic  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", module)
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    return made


def test_the_backends_and_their_default_models() -> None:
    assert BACKENDS == ("api", "local")
    assert DEFAULT_MODELS == {"api": "claude-haiku-5-5"}  # the exact ID, no date


@pytest.mark.parametrize(
    ("backend", "model", "expected"),
    [
        ("api", None, "claude-haiku-5-5"),
        ("api", "", "claude-haiku-5-5"),
        ("api", "claude-sonnet-5-5", "claude-sonnet-5-5"),
        ("local", "granite-4.1-8b", "granite-4.1-8b"),
        ("local", " granite-4.1-8b\n", "granite-4.1-8b"),
    ],
)
def test_a_judges_model_is_the_one_given_else_its_backends_default(
    backend: str, model: str | None, expected: str
) -> None:
    assert judge_model(backend, model) == expected


@pytest.mark.parametrize("model", [None, "", "  "])
def test_a_local_judge_needs_its_model_named(model: str | None) -> None:
    with pytest.raises(ValueError) as raised:
        judge_model("local", model)
    assert str(raised.value) == (
        "the local backend needs --model: the name the vLLM server serves"
    )
    with pytest.raises(ValueError, match="needs --model"):
        make_backend("local", model)


@pytest.mark.parametrize("backend", ["zoo", "API", ""])
def test_an_unknown_backend_is_refused(backend: str) -> None:
    for refused in (judge_model, make_backend):
        with pytest.raises(ValueError) as raised:
            refused(backend, "granite-4.1-8b")
        assert str(raised.value) == (
            f"unknown judge backend {backend!r}; there are api, local"
        )


def test_api_makes_an_api_backend_on_claude_haiku(
    anthropic: list[dict[str, Any]],
) -> None:
    backend = make_backend("api")
    assert isinstance(backend, ApiBackend)
    assert (backend.name, backend.model, backend.effort) == (
        "api",
        "claude-haiku-5-5",
        "low",
    )
    assert anthropic == [{"api_key": KEY, "max_retries": 0, "timeout": 60.0}]
    assert make_backend("api", "claude-sonnet-5-5").model == "claude-sonnet-5-5"


def test_api_takes_no_server_url(anthropic: list[dict[str, Any]]) -> None:
    with pytest.raises(ValueError) as raised:
        make_backend("api", base_url="http://127.0.0.1:8000/v1")
    assert str(raised.value) == (
        "the api backend calls the Anthropic API, so it takes no server URL: "
        "a URL is for the local backend"
    )
    assert anthropic == []  # refused before any client was made


def test_api_without_the_package_or_the_key_is_refused(
    monkeypatch: pytest.MonkeyPatch, anthropic: list[dict[str, Any]]
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(ValueError, match="reads ANTHROPIC_API_KEY, which is not set"):
        make_backend("api")
    monkeypatch.setitem(sys.modules, "anthropic", None)  # import anthropic fails
    with pytest.raises(ValueError, match="needs the anthropic package"):
        make_backend("api")


def test_local_makes_a_local_backend_at_the_default_url_or_the_one_given() -> None:
    backend = make_backend("local", "granite-4.1-8b")
    assert isinstance(backend, LocalBackend)
    assert (backend.name, backend.model, backend.base_url) == (
        "local",
        "granite-4.1-8b",
        LOCAL_URL,
    )
    other = make_backend("local", "granite-4.1-8b", "http://127.0.0.1:8001/v1/")
    assert other.url == "http://127.0.0.1:8001/v1/chat/completions"  # type: ignore[attr-defined]


def test_a_bad_local_url_is_refused() -> None:
    with pytest.raises(ValueError, match="must be http:// or https:// and a host"):
        make_backend("local", "granite-4.1-8b", "127.0.0.1:8000")


def test_importing_the_factory_loads_no_anthropic_and_nothing_heavy(
    tmp_path: Path,
) -> None:
    result = subprocess.run(
        [sys.executable, "-c", IMPORT],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "[]"


def test_docstring_examples_run() -> None:
    result = doctest.testmod(backends)
    assert result.attempted > 0
    assert result.failed == 0
