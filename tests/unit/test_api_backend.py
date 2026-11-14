"""The judge's api backend (pushlab.scoring.api_backend): Claude through the
Anthropic API. A fake client stands in for the SDK's, and classes with the
SDK's names for its errors, so nothing here reaches the network; one test
checks the backend against the anthropic package that pushlab installs."""

import doctest
import inspect
import sys
import types
import typing
from types import SimpleNamespace as NS
from typing import Any

import pytest

from pushlab.scoring import api_backend
from pushlab.scoring.api_backend import ApiBackend, reply_of, retry_after
from pushlab.scoring.judge import CallError, Judge, JudgeUnavailable
from pushlab.scoring.personas import Persona
from pushlab.scoring.rubrics import load_rubric

OWL = Persona("Brass Owl", "A pompous mechanical owl who lectures in riddles.")
GOOD = '{"reason": "Clear.", "score": 9}'
KEY = "test-key-not-real"
REQUEST = load_rubric("coherence").request("Hello there.")


# The SDK's error classes, standing in, with its names and family tree: an
# APIStatusError has a status and a response with headers; a connection
# error has neither. SDK holds them all, as the anthropic module would.
class APIError(Exception):
    """The SDK's base error: a message, and the answer's body if any."""

    def __init__(self, message: str, *, body: object = None) -> None:
        super().__init__(message)
        self.message = message
        self.body = body


class APIStatusError(APIError):
    """An HTTP error: its status, and the response with its headers."""

    def __init__(
        self,
        message: str,
        status_code: int,
        headers: dict[str, str] | None = None,
        body: object = None,
    ) -> None:
        super().__init__(message, body=body)
        self.status_code = status_code
        self.response = NS(status_code=status_code, headers=dict(headers or {}))


SDK = NS(APIError=APIError, APIStatusError=APIStatusError)


def _subclass(name: str, base: type[APIError]) -> Any:
    """The SDK's error class of this name, under base, standing in; kept in SDK."""
    kind = type(name, (base,), {})
    setattr(SDK, name, kind)
    return kind


BadRequestError = _subclass("BadRequestError", APIStatusError)  # 400
AuthenticationError = _subclass("AuthenticationError", APIStatusError)  # 401
PermissionDeniedError = _subclass("PermissionDeniedError", APIStatusError)  # 403
NotFoundError = _subclass("NotFoundError", APIStatusError)  # 404
RateLimitError = _subclass("RateLimitError", APIStatusError)  # 429
InternalServerError = _subclass("InternalServerError", APIStatusError)  # 500 and up
# A newer SDK's own class for 529: beside InternalServerError, not under it.
OverloadedError = _subclass("OverloadedError", APIStatusError)
APIConnectionError = _subclass("APIConnectionError", APIError)
APITimeoutError = _subclass("APITimeoutError", APIConnectionError)
THINKING = NS(type="thinking", thinking="", signature="made-up")
USAGE = NS(input_tokens=310, output_tokens=42)


class Messages:
    """client.messages, standing in: each call takes the next step of a
    script, an answer to return or an error to raise, and keeps what it was
    sent."""

    def __init__(self, steps: tuple[object, ...]) -> None:
        self.steps = list(steps)
        self.sent: list[dict[str, Any]] = []

    def create(self, **sent: Any) -> object:
        self.sent.append(sent)
        step = self.steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        return step


def _text(text: str) -> NS:
    return NS(type="text", text=text)


def _response(*blocks: object, stop: object = "end_turn", usage: object = USAGE) -> NS:
    """A Messages API response, standing in: its blocks (GOOD's text if
    none), why it stopped and its usage."""
    return NS(content=list(blocks) or [_text(GOOD)], stop_reason=stop, usage=usage)


def _backend(*steps: object, **options: Any) -> tuple[ApiBackend, Messages]:
    """An ApiBackend on a fake client that follows steps, and its messages."""
    messages = Messages(steps)
    return ApiBackend(client=NS(messages=messages), sdk=SDK, **options), messages


def _failure(*steps: object) -> CallError:
    """The CallError a call raises when the client meets steps."""
    backend, _ = _backend(*steps)
    with pytest.raises(CallError) as raised:
        backend.complete(REQUEST)
    return raised.value


def _judge(backend: ApiBackend) -> tuple[Judge, list[float]]:
    """A judge on backend whose waits are recorded, not slept."""
    waits: list[float] = []
    return Judge(backend, sleep=waits.append, jitter=lambda: 1.0), waits


def test_a_request_sends_the_rubric_the_text_and_its_schema_and_nothing_else() -> None:
    answer = '{"reason": "Hoo.", "label": "speaks_as"}'
    backend, messages = _backend(_response(_text(answer)))
    request = load_rubric("speaks_as").request("Hoo! Riddle me this.", OWL)
    assert backend.complete(request).text == answer
    (sent,) = messages.sent
    assert sent == {
        "model": "claude-haiku-5-5",
        "max_tokens": request.max_tokens,
        "system": request.system,
        "messages": [{"role": "user", "content": request.user}],
        "output_config": {
            "effort": "low",
            "format": {"type": "json_schema", "schema": request.schema},
        },
    }
    # Claude Haiku 5.5 answers sampling settings with HTTP 400, and thinking
    # stays at its default, adaptive, kept short by the low effort.
    assert not {"temperature", "top_p", "top_k", "thinking"} & set(sent)
    assert request.max_tokens == 1024  # room for thinking and the answer
    assert "Brass Owl" in request.user and "Hoo! Riddle me this." in request.user


def test_the_answer_is_read_by_block_type_so_thinking_before_it_is_skipped() -> None:
    first, rest = GOOD[:20], GOOD[20:]
    backend, _ = _backend(_response(THINKING, _text(first), _text(rest)))
    assert backend.complete(REQUEST).text == GOOD
    # Only text blocks count, and only their text.
    tool = NS(type="tool_use", name="x", input={})
    reply = reply_of(_response(tool, _text("{}"), NS(type="text", text=None)))
    assert (reply.text, reply.stop) == ("{}", "end")
    assert (reply_of(NS()).text, reply_of(NS()).stop) == ("", "")


@pytest.mark.parametrize(
    ("stop_reason", "stop"),
    [("end_turn", "end"), ("max_tokens", "length"), ("refusal", "refusal")]
    + [("stop_sequence", "stop_sequence"), (None, "")],
)
def test_each_stop_reason_has_its_word(stop_reason: str | None, stop: str) -> None:
    backend, _ = _backend(_response(stop=stop_reason))
    assert backend.complete(REQUEST).stop == stop


def test_the_tokens_are_counted_and_are_zero_when_not_given() -> None:
    usages = (USAGE, None, NS(input_tokens=5))
    backend, _ = _backend(*(_response(usage=usage) for usage in usages))
    replies = [backend.complete(REQUEST) for _ in usages]
    counts = [(reply.input_tokens, reply.output_tokens) for reply in replies]
    assert counts == [(310, 42), (0, 0), (5, 0)]


@pytest.mark.parametrize(
    ("error", "hint"),
    [
        (
            AuthenticationError("bad key", 401),
            "the key in ANTHROPIC_API_KEY was rejected",
        ),
        (PermissionDeniedError("not allowed", 403), "this key may not make this call"),
        (
            NotFoundError("model: claude-haiku-5-5", 404),
            "the model claude-haiku-5-5 is not there, or this key cannot use it",
        ),
        (
            APIStatusError("credit too low", 402),
            "the account's billing stops every call",
        ),
    ],
)
def test_a_key_or_model_that_no_call_can_pass_stops_the_job(
    error: APIStatusError, hint: str
) -> None:
    message = f"HTTP {error.status_code}: {error.message} ({hint})"
    failure = _failure(error)
    assert (failure.fatal, failure.retryable, str(failure)) == (True, False, message)
    assert failure.__cause__ is error
    judge, _ = _judge(_backend(error)[0])
    with pytest.raises(JudgeUnavailable) as stopped:
        judge.verdict(load_rubric("coherence"), "Hello there.")
    assert str(stopped.value) == f"api:claude-haiku-5-5: {message}"


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (RateLimitError("rate limited", 429), "HTTP 429: rate limited"),
        (InternalServerError("internal error", 500), "HTTP 500: internal error"),
        (InternalServerError("overloaded", 529), "HTTP 529: overloaded"),
        # A newer SDK's own class for 529, not under InternalServerError.
        (OverloadedError("overloaded", 529), "HTTP 529: overloaded"),
        (APIStatusError("unavailable", 503), "HTTP 503: unavailable"),
        # The SDK's own retries would try these again too.
        (APIStatusError("request timeout", 408), "HTTP 408: request timeout"),
        (APIStatusError("conflict", 409), "HTTP 409: conflict"),
        (APIConnectionError("lost"), "no answer from the Anthropic API: lost"),
        (APITimeoutError("timed out"), "no answer from the Anthropic API: timed out"),
        (
            APIError("unreadable"),
            "the Anthropic API's answer could not be read: unreadable",
        ),
    ],
)
def test_a_failure_that_may_pass_is_retryable(error: Exception, message: str) -> None:
    failure = _failure(error)
    assert (failure.retryable, failure.fatal, str(failure)) == (True, False, message)


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (BadRequestError("prompt is too long", 400), "HTTP 400: prompt is too long"),
        (APIStatusError("request too large", 413), "HTTP 413: request too large"),
        (APIStatusError("unprocessable", 422), "HTTP 422: unprocessable"),
    ],
)
def test_any_other_http_error_fails_the_text_at_once(
    error: Exception, message: str
) -> None:
    failure = _failure(error)
    assert (failure.retryable, failure.fatal, str(failure)) == (False, False, message)


@pytest.mark.parametrize(
    ("headers", "wait"),
    [({"retry-after": "12"}, 12.0), ({"retry-after": " 0.5 "}, 0.5), ({}, None)]
    + [({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}, None)]
    + [({"retry-after": "soon"}, None), ({"retry-after": "-3"}, None)],
)
def test_a_rate_limit_waits_as_the_server_asks_when_it_says_seconds(
    headers: dict[str, str], wait: float | None
) -> None:
    failure = _failure(RateLimitError("rate limited", 429, headers))
    assert failure.retryable and failure.retry_after == wait


def test_an_overloaded_servers_retry_after_is_read_too() -> None:
    failure = _failure(InternalServerError("overloaded", 529, {"retry-after": "20"}))
    assert failure.retry_after == 20.0
    assert _failure(APIConnectionError("Connection error.")).retry_after is None
    # Only a plain number of seconds is read.
    assert retry_after({"retry-after": "1e3"}) is None
    assert retry_after(None) is None and retry_after("30") is None


def test_the_apis_own_message_is_kept_on_one_line_and_cut_short() -> None:
    inner = {"type": "rate_limit_error", "message": "Too many\n  requests."}
    body = {"type": "error", "error": inner, "request_id": "req_made_up"}
    error = RateLimitError(f"Error code: 429 - {body}", 429, body=body)
    assert str(_failure(error)) == "HTTP 429: Too many requests."
    long = BadRequestError("x" * 1000, 400)
    assert str(_failure(long)) == "HTTP 400: " + "x" * 300


def test_an_error_that_is_not_the_sdks_is_a_bug_raised_as_it_is() -> None:
    backend, _ = _backend(TypeError("create() got an unexpected keyword argument"))
    with pytest.raises(TypeError, match="unexpected keyword"):
        backend.complete(REQUEST)


def _anthropic(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Put a stand-in anthropic package in sys.modules; return the options
    each client is made with."""
    made: list[dict[str, Any]] = []

    class Anthropic:
        def __init__(self, **options: Any) -> None:
            made.append(options)
            self.messages = Messages((_response(),))

    module = types.ModuleType("anthropic")
    module.Anthropic = Anthropic  # type: ignore[attr-defined]
    for name, kind in vars(SDK).items():
        setattr(module, name, kind)
    monkeypatch.setitem(sys.modules, "anthropic", module)
    return made


def test_with_no_client_one_is_made_with_the_key_and_no_retries_of_its_own(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made = _anthropic(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", f"  {KEY}\n")
    backend = ApiBackend()
    # The Judge retries, with backoff, and counts every attempt.
    assert made == [{"api_key": KEY, "max_retries": 0, "timeout": 60.0}]
    assert backend.sdk is sys.modules["anthropic"]
    assert backend.complete(REQUEST).text == GOOD
    ApiBackend(timeout=5.0)
    assert made[-1]["timeout"] == 5.0


@pytest.mark.parametrize("value", [None, "", "  \n"])
def test_a_missing_or_blank_key_is_refused(
    value: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    made = _anthropic(monkeypatch)
    if value is None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    else:
        monkeypatch.setenv("ANTHROPIC_API_KEY", value)
    with pytest.raises(ValueError) as raised:
        ApiBackend()
    assert str(raised.value) == (
        "the api backend reads ANTHROPIC_API_KEY, which is not set"
    )
    assert made == []


def test_without_the_anthropic_package_it_says_how_to_install_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)  # import anthropic fails
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    with pytest.raises(ValueError) as raised:
        ApiBackend()
    assert str(raised.value) == (
        "the api backend needs the anthropic package, which this environment "
        "lacks: run uv sync, or on the GPU pod bash scripts/setup_pod.sh"
    )
    # A client given still needs the package, for its error classes.
    with pytest.raises(ValueError, match="needs the anthropic package"):
        ApiBackend(client=NS(messages=Messages(())))


def test_the_installed_sdk_has_what_the_backend_uses() -> None:
    # The real anthropic package, which pushlab installs. Making a client calls
    # nothing, and tests/conftest.py leaves no key and a closed URL besides.
    import anthropic
    from anthropic.types import JSONOutputFormatParam, OutputConfigParam, StopReason

    for name in api_backend.CAUGHT:
        kind = getattr(anthropic, name, None)
        assert isinstance(kind, type) and issubclass(kind, BaseException), name
    client = anthropic.Anthropic(
        api_key=KEY, max_retries=0, timeout=api_backend.TIMEOUT
    )
    sent = inspect.signature(client.messages.create).parameters
    assert {"model", "max_tokens", "system", "messages", "output_config"} <= set(sent)
    config = typing.get_type_hints(OutputConfigParam)
    efforts = {
        v for arg in typing.get_args(config["effort"]) for v in typing.get_args(arg)
    }
    assert set(api_backend.EFFORTS) <= efforts
    schema = typing.get_type_hints(JSONOutputFormatParam)
    assert set(schema) == {"type", "schema"}
    assert typing.get_args(schema["type"]) == ("json_schema",)
    assert set(api_backend.STOPS) <= set(typing.get_args(StopReason))


def test_the_model_and_effort_can_be_chosen_and_bad_ones_are_refused() -> None:
    backend, messages = _backend(_response(), model="claude-sonnet-5-5", effort="max")
    backend.complete(REQUEST)
    assert (backend.name, messages.sent[0]["model"]) == ("api", "claude-sonnet-5-5")
    assert messages.sent[0]["output_config"]["effort"] == "max"
    with pytest.raises(ValueError, match="needs a model, such as claude-haiku-5-5"):
        ApiBackend(" ", client=NS(), sdk=SDK)
    with pytest.raises(ValueError) as raised:
        ApiBackend(client=NS(), sdk=SDK, effort="lowest")
    assert str(raised.value) == (
        "effort must be one of low, medium, high, xhigh, max, not 'lowest'"
    )


def test_a_judge_on_the_api_backend_gives_a_verdict() -> None:
    backend, messages = _backend(_response(THINKING, _text(GOOD)))
    judge, waits = _judge(backend)
    verdict = judge.verdict(load_rubric("coherence"), "Hello there.")
    assert (verdict.status, verdict.value, verdict.reason) == ("ok", 9.0, "Clear.")
    tokens = (verdict.input_tokens, verdict.output_tokens)
    assert (verdict.attempts, tokens) == (1, (310, 42))
    assert judge.name == "api:claude-haiku-5-5" and waits == []
    user = messages.sent[0]["messages"][0]["content"]
    assert user.endswith("<text>\nHello there.\n</text>")


def test_a_rate_limit_then_an_answer_is_ok_at_the_second_attempt() -> None:
    busy = RateLimitError("rate limited", 429, {"retry-after": "12"})
    backend, messages = _backend(busy, _response())
    judge, waits = _judge(backend)
    verdict = judge.verdict(load_rubric("coherence"), "Hello there.")
    assert (verdict.status, verdict.value, verdict.attempts) == ("ok", 9.0, 2)
    assert waits == [12.0]  # as long as the server asked
    assert len(messages.sent) == 2 and messages.sent[0] == messages.sent[1]


def test_a_refusal_or_a_bad_request_fails_the_text_at_once_with_no_label() -> None:
    refusal = _response(_text(""), stop="refusal")
    backend, _ = _backend(refusal, BadRequestError("prompt is too long", 400))
    judge, waits = _judge(backend)
    refused = judge.verdict(load_rubric("coherence"), "Hello there.")
    assert (refused.status, refused.value, refused.attempts) == ("failed", None, 1)
    assert "refusal" in refused.error
    bad = judge.verdict(load_rubric("coherence"), "Hello again.")
    assert (bad.status, bad.attempts, waits) == ("failed", 1, [])
    assert bad.error == "HTTP 400: prompt is too long"


def test_docstring_examples_run() -> None:
    result = doctest.testmod(api_backend)
    assert result.attempted > 0
    assert result.failed == 0
