"""The judge's stand-in backend for tests (pushlab.scoring.fake_backend)."""

import doctest

import pytest

from pushlab.scoring import fake_backend
from pushlab.scoring.fake_backend import FakeBackend, busy
from pushlab.scoring.judge import CallError, Reply
from pushlab.scoring.rubrics import load_rubric

REQUEST = load_rubric("coherence").request("Hi.")


def test_a_script_answers_in_order_and_keeps_each_request() -> None:
    reply = Reply("{}", "length", 5, 6)
    fake = FakeBackend(["first", reply], name="api", model="m", settings="effort=low")
    assert fake.complete(REQUEST) == Reply("first", "end", 100, 20)
    assert fake.complete(REQUEST) is reply
    assert fake.requests == [REQUEST, REQUEST] and fake.calls == 2
    assert (fake.name, fake.model, fake.settings) == ("api", "m", "effort=low")


def test_a_scripted_error_is_raised() -> None:
    fake = FakeBackend([busy("slow down", 3.0)])
    with pytest.raises(CallError, match="slow down") as raised:
        fake.complete(REQUEST)
    assert raised.value.retryable and raised.value.retry_after == 3.0
    assert not raised.value.fatal


def test_a_function_answers_each_request() -> None:
    fake = FakeBackend(lambda request: request.text.upper())
    assert fake.complete(REQUEST).text == "HI."


def test_a_call_past_the_script_fails_the_test() -> None:
    with pytest.raises(AssertionError, match="an unexpected call"):
        FakeBackend([]).complete(REQUEST)


def test_docstring_examples_run() -> None:
    result = doctest.testmod(fake_backend)
    assert result.attempted > 0
    assert result.failed == 0
