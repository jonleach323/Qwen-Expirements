"""The judge's local backend (pushlab.scoring.local_backend): an open model
that a vLLM server serves, over HTTP with urllib. A fake post, or fakes of
urllib's openers, stand in for the server, so nothing here opens a socket."""

import doctest
import http.client
import io
import json
import os
import socket
import urllib.error
import urllib.request
from collections.abc import Callable
from types import SimpleNamespace as NS
from typing import Any

import pytest

from pushlab.scoring import local_backend
from pushlab.scoring.judge import CallError, Judge, JudgeUnavailable
from pushlab.scoring.local_backend import (
    LOCAL_URL,
    LocalBackend,
    is_loopback,
    read_completion,
)
from pushlab.scoring.personas import Persona
from pushlab.scoring.rubrics import load_rubric

MODEL = "granite-4.1-8b"
OWL = Persona("Brass Owl", "A pompous mechanical owl who lectures in riddles.")
GOOD = '{"reason": "Clear.", "score": 9}'
URL = "http://127.0.0.1:8000/v1/chat/completions"
REQUEST = load_rubric("coherence").request("Hello there.")
# What an HTTP 401 or 403 adds, and what a refused connection says.
NO_KEY = (
    "(the server refuses the call; pushlab sends no key, so serve the judge "
    "without --api-key)"
)
NOTHING = (
    f"nothing answers at {URL}: start the vLLM server first (docs/runbooks/judge.md)"
)


def _completion(content: object = GOOD, finish: object = "stop") -> dict[str, Any]:
    """A chat completion as a vLLM server answers one, decoded."""
    message = {"role": "assistant", "content": content}
    choices = [{"index": 0, "message": message, "finish_reason": finish}]
    usage = {"prompt_tokens": 80, "completion_tokens": 12, "total_tokens": 92}
    return {"object": "chat.completion", "choices": choices, "usage": usage}


def _body(content: object = GOOD) -> bytes:
    """A chat completion as the server sends it: JSON bytes."""
    return json.dumps(_completion(content)).encode()


class FakePost:
    """A post that answers from a script and keeps each call's url, payload
    and timeout."""

    def __init__(self, *answers: object) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[str, dict[str, Any], float]] = []

    def __call__(self, url: str, payload: dict[str, Any], timeout: float) -> object:
        self.calls.append((url, payload, timeout))
        return self.answers.pop(0)


class Server:
    """urllib's openers, standing in for the network: each call is kept, as
    (which opener, the request, its timeout), and answers with the next
    body of a script, or raises the next error."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *steps: object) -> None:
        self.steps = list(steps)
        self.opened: list[tuple[str, urllib.request.Request, float]] = []
        self.handlers: list[object] = []
        monkeypatch.setattr(urllib.request, "build_opener", self.build_opener)
        monkeypatch.setattr(urllib.request, "urlopen", self.urlopen)

    def answer(self, how: str, request: Any, timeout: float) -> io.BytesIO:
        self.opened.append((how, request, timeout))
        step = self.steps.pop(0)
        if isinstance(step, BaseException):
            raise step
        return io.BytesIO(step)  # type: ignore[arg-type]

    def build_opener(self, *handlers: object) -> NS:
        self.handlers.extend(handlers)
        return NS(
            open=lambda request, timeout: self.answer("no proxy", request, timeout)
        )

    def urlopen(self, request: Any, timeout: float) -> io.BytesIO:
        return self.answer("default", request, timeout)


def _http_error(
    code: int, body: bytes = b"", headers: dict[str, str] | None = None
) -> urllib.error.HTTPError:
    """The HTTPError urllib raises for a status, with its body and headers."""
    message = http.client.HTTPMessage()
    for name, value in (headers or {}).items():
        message[name] = value
    return urllib.error.HTTPError(URL, code, "made-up", message, io.BytesIO(body))


def _failure(monkeypatch: pytest.MonkeyPatch, step: object) -> CallError:
    """The CallError a call through the default post raises on step."""
    Server(monkeypatch, step)
    with pytest.raises(CallError) as raised:
        LocalBackend(MODEL).complete(REQUEST)
    return raised.value


def test_a_request_is_one_chat_completion_at_temperature_0_with_the_schema() -> None:
    post = FakePost(_completion('{"reason": "Hoo.", "label": "speaks_as"}'))
    judge = Judge(LocalBackend(MODEL, post=post), sleep=lambda seconds: None)
    verdict = judge.verdict(load_rubric("speaks_as"), "Hoo! Riddle me this.", OWL)
    assert (verdict.status, verdict.label, verdict.attempts) == ("ok", "speaks_as", 1)
    assert (verdict.input_tokens, verdict.output_tokens) == (80, 12)
    assert judge.name == "local:granite-4.1-8b"
    request = load_rubric("speaks_as").request("Hoo! Riddle me this.", OWL)
    ((url, payload, timeout),) = post.calls
    assert (url, timeout) == (URL, 120.0)
    assert payload == {
        "model": "granite-4.1-8b",
        "messages": [
            {"role": "system", "content": request.system},
            {"role": "user", "content": request.user},
        ],
        "max_tokens": request.max_tokens,
        "temperature": 0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "speaks_as",
                "schema": request.schema,
                "strict": True,
            },
        },
    }
    assert "Brass Owl" in request.user and "Hoo! Riddle me this." in request.user


@pytest.mark.parametrize(
    ("finish", "stop"),
    [("stop", "end"), ("length", "length"), ("abort", "abort"), (None, "")],
)
def test_each_finish_reason_has_its_word(finish: str | None, stop: str) -> None:
    post = FakePost(_completion(finish=finish))
    reply = LocalBackend(MODEL, post=post).complete(REQUEST)
    assert (reply.text, reply.stop) == (GOOD, stop)


def test_the_tokens_are_counted_and_no_content_reads_as_empty() -> None:
    bare, odd = _completion(None), _completion()
    del bare["usage"]
    odd["usage"] = {"prompt_tokens": "80", "completion_tokens": None}
    replies = [read_completion(body) for body in (_completion(), bare, odd)]
    counts = [(r.input_tokens, r.output_tokens) for r in replies]
    assert counts == [(80, 12), (0, 0), (0, 0)]
    assert replies[1].text == ""


@pytest.mark.parametrize(
    "body",
    [[], "ok", {}, {"choices": []}, {"choices": [{}]}, {"choices": [{"message": 1}]}]
    + [{"choices": [{"message": {"content": 7}}]}, {"object": "error", "message": "x"}],
)
def test_an_answer_of_another_shape_is_retryable(body: object) -> None:
    with pytest.raises(CallError) as raised:
        LocalBackend(MODEL, post=FakePost(body)).complete(REQUEST)
    assert raised.value.retryable and not raised.value.fatal
    prefix = "the server's answer is not a chat completion: "
    assert str(raised.value).startswith(prefix)


def test_the_default_post_sends_the_payload_as_json_and_reads_the_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = Server(monkeypatch, _body())
    backend = LocalBackend(MODEL)
    assert backend.base_url == LOCAL_URL == "http://127.0.0.1:8000/v1"
    reply = backend.complete(REQUEST)
    assert (reply.text, reply.stop, reply.input_tokens) == (GOOD, "end", 80)
    ((how, request, timeout),) = server.opened
    assert (how, request.full_url, request.get_method()) == ("no proxy", URL, "POST")
    assert timeout == 120.0
    assert request.get_header("Content-type") == "application/json"
    assert request.get_header("User-agent") == "pushlab"
    assert json.loads(request.data) == backend.payload(REQUEST)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("base_url", "how"),
    [
        ("http://127.0.0.1:8000/v1/", "no proxy"),
        ("http://localhost:8000/v1", "no proxy"),
        ("http://[::1]:8000/v1", "no proxy"),
        ("http://judge.example.invalid:8000/v1", "default"),
        ("https://10.0.0.7/v1", "default"),
    ],
)
def test_only_a_server_on_this_machine_is_called_past_the_environments_proxy(
    base_url: str, how: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("http_proxy", "http://proxy.example.invalid:3128")
    server = Server(monkeypatch, _body())
    LocalBackend(MODEL, base_url, timeout=30.0).complete(REQUEST)
    ((opened, request, timeout),) = server.opened
    assert (opened, timeout) == (how, 30.0)
    assert request.full_url == base_url.rstrip("/") + "/chat/completions"
    if how == "no proxy":
        (handler,) = server.handlers
        assert isinstance(handler, urllib.request.ProxyHandler)
        assert handler.proxies == {}
    else:
        assert server.handlers == []
    # The environment's proxy is left as it was, for everything else.
    assert os.environ["http_proxy"] == "http://proxy.example.invalid:3128"


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (
            lambda: _http_error(401, b'{"error": "Unauthorized"}'),
            f'HTTP 401 from {URL}: {{"error": "Unauthorized"}} {NO_KEY}',
        ),
        (lambda: _http_error(403), f"HTTP 403 from {URL} {NO_KEY}"),
        (
            lambda: _http_error(404, b"The model does not exist."),
            f"HTTP 404 from {URL}: The model does not exist. (no model "
            "granite-4.1-8b is served there, or the path is wrong: --model must be "
            "the name the vLLM server serves)",
        ),
        (lambda: urllib.error.URLError(ConnectionRefusedError(111, "x")), NOTHING),
        (lambda: ConnectionRefusedError(111, "Connection refused"), NOTHING),
    ],
)
def test_a_server_no_call_can_pass_stops_the_job(
    error: Callable[[], Exception], message: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    failure = _failure(monkeypatch, error())
    assert (failure.fatal, failure.retryable, str(failure)) == (True, False, message)


# Failures that may pass, and what the CallError says after "<URL>".
RETRYABLE = [
    (lambda: _http_error(429, b"Too many"), "HTTP 429 from", ": Too many"),
    (lambda: _http_error(500, b"Server\n  Error"), "HTTP 500 from", ": Server Error"),
    (lambda: _http_error(503), "HTTP 503 from", ""),
    (lambda: _http_error(408), "HTTP 408 from", ""),
    (
        lambda: urllib.error.URLError(socket.gaierror(-2, "Name or service not known")),
        "no answer from",
        ": [Errno -2] Name or service not known",
    ),
    (lambda: urllib.error.URLError(TimeoutError("x")), "no answer from", ": x"),
    (lambda: TimeoutError("timed out"), "no answer from", ": timed out"),
    (
        lambda: ConnectionResetError(104, "reset"),
        "no answer from",
        ": [Errno 104] reset",
    ),
    (
        lambda: http.client.RemoteDisconnected("Remote end closed connection"),
        "no answer from",
        ": Remote end closed connection",
    ),
    (
        lambda: http.client.IncompleteRead(b"{", 100),
        "no answer from",
        ": IncompleteRead(1 bytes read, 100 more expected)",
    ),
]


@pytest.mark.parametrize(("error", "what", "said"), RETRYABLE)
def test_a_busy_server_or_a_lost_connection_is_retryable(
    error: Callable[[], Exception],
    what: str,
    said: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = _failure(monkeypatch, error())
    assert (failure.retryable, failure.fatal) == (True, False)
    assert str(failure) == f"{what} {URL}{said}"


@pytest.mark.parametrize(
    ("body", "problem"),
    [
        (b"<html>Bad gateway</html>", "Expecting value: line 1 column 1 (char 0)"),
        (
            b"\xff",
            "'utf-8' codec can't decode byte 0xff in position 0: invalid start byte",
        ),
    ],
)
def test_an_answer_that_is_not_json_is_retryable(
    body: bytes, problem: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    failure = _failure(monkeypatch, body)
    assert (failure.retryable, failure.fatal) == (True, False)
    assert str(failure) == f"the server's answer is not JSON: {problem}"


@pytest.mark.parametrize(
    ("status", "body", "said"),
    [(400, b"x" * 1000, ": " + "x" * 300), (422, b"bad\n  schema", ": bad schema")]
    + [(413, b"", "")],
)
def test_any_other_http_error_fails_the_text_at_once(
    status: int, body: bytes, said: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    failure = _failure(monkeypatch, _http_error(status, body))
    assert (failure.retryable, failure.fatal) == (False, False)
    assert str(failure) == f"HTTP {status} from {URL}{said}"


@pytest.mark.parametrize(
    ("status", "value", "wait"),
    [(429, "7", 7.0), (429, "1.5", 1.5), (429, None, None), (429, "soon", None)]
    + [(429, "Wed, 21 Oct 2026 07:28:00 GMT", None), (503, "20", 20.0)],
)
def test_a_busy_server_is_waited_for_as_it_asks_when_it_says_seconds(
    status: int, value: str | None, wait: float | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    headers = {} if value is None else {"Retry-After": value}
    failure = _failure(monkeypatch, _http_error(status, b"", headers))
    assert failure.retryable and failure.retry_after == wait


@pytest.mark.parametrize("model", ["", "  "])
def test_a_local_backend_needs_the_served_models_name(model: str) -> None:
    with pytest.raises(ValueError) as raised:
        LocalBackend(model)
    assert str(raised.value) == (
        "the local backend needs --model: the name the vLLM server serves"
    )


@pytest.mark.parametrize(
    "base_url",
    ["127.0.0.1:8000/v1", "localhost:8000/v1", "ftp://127.0.0.1/v1", "http:///v1"]
    + ["http://[::1/v1"]
    # A port that is not a number, or past 65535.
    + ["http://127.0.0.1:800O/v1", "http://127.0.0.1:99999/v1"],
)
def test_a_url_that_is_not_http_to_a_host_is_refused(base_url: str) -> None:
    with pytest.raises(ValueError) as raised:
        LocalBackend(MODEL, base_url)
    assert str(raised.value) == (
        "the local backend's URL must be http:// or https:// and a host, such as "
        f"http://127.0.0.1:8000/v1, not {base_url!r}"
    )


def test_this_machine_is_localhost_or_a_loopback_address() -> None:
    here = ["127.0.0.1:8000", "localhost:8000", "LOCALHOST", "[::1]:8000", "127.0.1.1"]
    elsewhere = ["10.0.0.7:8000", "judge.example.invalid", "localhost.example.invalid"]
    assert all(is_loopback(f"http://{host}/v1") for host in here)
    assert not any(is_loopback(f"http://{host}/v1") for host in elsewhere)
    assert not is_loopback("http://[::1/v1")  # not a URL at all


def test_a_busy_server_then_an_answer_is_ok_at_the_second_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    Server(monkeypatch, _http_error(503, b"busy", {"Retry-After": "3"}), _body())
    waits: list[float] = []
    judge = Judge(LocalBackend(MODEL), sleep=waits.append, jitter=lambda: 1.0)
    verdict = judge.verdict(load_rubric("coherence"), "Hello there.")
    assert (verdict.status, verdict.value, verdict.attempts) == ("ok", 9.0, 2)
    assert waits == [3.0]  # as long as the server asked


def test_no_server_stops_the_job(monkeypatch: pytest.MonkeyPatch) -> None:
    Server(monkeypatch, urllib.error.URLError(ConnectionRefusedError(111, "x")))
    judge = Judge(LocalBackend(MODEL), sleep=lambda seconds: None)
    with pytest.raises(JudgeUnavailable, match="^local:granite-4.1-8b: nothing"):
        judge.verdict(load_rubric("coherence"), "Hello there.")


def test_docstring_examples_run() -> None:
    result = doctest.testmod(local_backend)
    assert result.attempted > 0
    assert result.failed == 0
