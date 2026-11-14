"""The judge's local backend: an open model that a vLLM server serves.

LocalBackend asks a model through vLLM's OpenAI-compatible API, at
http://127.0.0.1:8000/v1 unless told another URL (docs/runbooks/judge.md
starts one on the pod). The model must be of another family than the model
under test (pushlab.scoring.family.check_family). It speaks HTTP with urllib,
from the standard library: no openai package, and no new dependency
(decision 0021).

A request is one chat completion: the rubric's instructions as the system
message, the question and the text as the user's message, at temperature
0, with the rubric's JSON schema as the answer's required format (vLLM's
structured outputs). A server on this machine (localhost, 127.0.0.1, ::1)
is called directly, never through a proxy that the environment names,
which could not reach it; any other host is called through those proxies,
as usual. TLS is always verified.

Failures become CallErrors (pushlab.scoring.judge). A busy or failing
server (HTTP 408, 429, 500 and up), a lost or timed-out connection, and an
answer that is not JSON or not a chat completion may pass if tried again
(retryable). A server that refuses the call (HTTP 401, 403), serves no
such model or path (404), or does not answer at all stops the job (fatal).
Any other HTTP error fails the text at once, with the start of the
server's answer.
"""

import http.client
import ipaddress
import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

from pushlab.scoring.judge import CallError, Reply
from pushlab.scoring.rubrics import Request

LOCAL_URL = "http://127.0.0.1:8000/v1"
# Its answers' settings, part of the cache key: greedy, as payload asks.
SETTINGS = "temperature=0"
TIMEOUT = 120.0  # seconds a call may wait for the server
USER_AGENT = "pushlab"  # a plain name, nothing about who runs it
RUNBOOK = "docs/runbooks/judge.md"
# Why the model stopped, in a Reply's words; any other reason is passed on.
STOPS = {"stop": "end", "length": "length"}
# HTTP statuses that may pass if tried again, besides 500 and up.
RETRY_STATUSES = (408, 429)
SHOWN = 300  # characters of the server's answer kept in an error
NEEDS_MODEL = "the local backend needs --model: the name the vLLM server serves"
NOT_A_COMPLETION = "the server's answer is not a chat completion"
_SECONDS = re.compile(r"\d+(\.\d+)?")

# Sends a payload as JSON to a URL, waiting at most timeout seconds, and
# returns the answer's JSON, decoded.
Post = Callable[[str, dict[str, Any], float], Any]


class LocalBackend:
    """An open model that vLLM serves through its OpenAI-compatible API, as a
    judge's backend.

    model is the name the server serves (its --served-model-name); base_url
    the server's API root, before /chat/completions. post sends a request:
    http_post unless a test gives another. ValueError for a blank model, or
    a URL that is not http:// or https:// to a host.

    Example:
        >>> from pushlab.scoring.rubrics import load_rubric
        >>> message = {"content": '{"reason": "Clear.", "score": 9}'}
        >>> answer = {"choices": [{"message": message, "finish_reason": "stop"}]}
        >>> backend = LocalBackend("granite-4.1-8b", post=lambda *sent: answer)
        >>> backend.complete(load_rubric("coherence").request("Hi.")).stop
        'end'
        >>> backend.url
        'http://127.0.0.1:8000/v1/chat/completions'
    """

    name = "local"

    def __init__(
        self,
        model: str,
        base_url: str = LOCAL_URL,
        *,
        post: Post | None = None,
        timeout: float = TIMEOUT,
    ) -> None:
        if not model.strip():
            raise ValueError(NEEDS_MODEL)
        try:
            parts = urlsplit(base_url)
        except ValueError:  # such as an unclosed [ of an IPv6 address
            parts = None
        try:
            port = None if parts is None else parts.port  # a bad one raises
        except ValueError:
            parts = None
        if parts is None or parts.scheme not in ("http", "https") or not parts.hostname:
            raise ValueError(
                "the local backend's URL must be http:// or https:// and a host, "
                f"such as {LOCAL_URL}, not {base_url!r}"
            )
        self.model = model
        self.base_url = base_url
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.post: Post = http_post if post is None else post
        self.timeout = timeout
        self.port = port
        # What changes its answers besides the model: part of the cache key.
        self.settings = SETTINGS

    def payload(self, request: Request) -> dict[str, Any]:
        """The chat completion that asks the model about the request's text:
        the instructions as the system message, the question and the text as
        the user's, greedy, and the rubric's schema as the answer's format."""
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": request.system},
                {"role": "user", "content": request.user},
            ],
            "max_tokens": request.max_tokens,
            "temperature": 0,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": request.rubric,
                    "schema": request.schema,
                    "strict": True,
                },
            },
        }

    def complete(self, request: Request) -> Reply:
        """Ask the model about the request's text; raise CallError when the
        call fails."""
        try:
            body = self.post(self.url, self.payload(request), self.timeout)
        except urllib.error.HTTPError as error:
            raise self._refused(error) from error
        except (OSError, http.client.HTTPException) as error:  # URLError is one
            raise self._lost(error) from error
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            message = f"the server's answer is not JSON: {error}"
            raise CallError(message, retryable=True) from error
        return read_completion(body)

    def _refused(self, error: urllib.error.HTTPError) -> CallError:
        """The CallError an HTTP error is: fatal, retryable, or failing the
        text at once."""
        status = error.code
        said = _excerpt(error)
        where = f"HTTP {status} from {self.url}" + (f": {said}" if said else "")
        if status == 404:
            hint = (
                f"no model {self.model} is served there, or the path is wrong: "
                "--model must be the name the vLLM server serves"
            )
            return CallError(f"{where} ({hint})", fatal=True)
        if status in (401, 403):
            hint = (
                "the server refuses the call; pushlab sends no key, so serve the "
                "judge without --api-key"
            )
            return CallError(f"{where} ({hint})", fatal=True)
        if status in RETRY_STATUSES or status >= 500:
            wait = _retry_after(error.headers)
            return CallError(where, retryable=True, retry_after=wait)
        return CallError(where)

    def _lost(self, error: BaseException) -> CallError:
        """The CallError a call that got no answer is: fatal when nothing
        listens at the URL, retryable otherwise."""
        reason = error.reason if isinstance(error, urllib.error.URLError) else error
        if isinstance(reason, ConnectionRefusedError):
            return CallError(
                f"nothing answers at {self.url}: start the vLLM server first "
                f"({RUNBOOK})",
                fatal=True,
            )
        said = " ".join(str(reason).split()) or type(reason).__name__
        return CallError(f"no answer from {self.url}: {said}", retryable=True)


def is_loopback(url: str) -> bool:
    """Whether url's host is this machine: localhost or a loopback address.

    Example:
        >>> [is_loopback(f"http://{host}:8000/v1") for host in ("[::1]", "10.0.0.7")]
        [True, False]
    """
    try:
        host = (urlsplit(url).hostname or "").rstrip(".")
    except ValueError:  # not a URL at all, such as an unclosed [
        return False
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:  # a name, not an address
        return False


def http_post(url: str, payload: dict[str, Any], timeout: float) -> Any:
    """POST payload to url as JSON and return the JSON answer, decoded.

    A server on this machine is called with no proxy, as one the environment
    names could not reach it; any other through the environment's proxies.
    urllib's errors are raised as they are, and json's for an answer that
    is not JSON.

    Example:
        >>> url = f"{LOCAL_URL}/chat/completions"
        >>> http_post(url, {"model": "granite-4.1-8b"}, 120.0)  # doctest: +SKIP
        {'id': 'chatcmpl-...', 'object': 'chat.completion', 'choices': [...], ...}
    """
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "User-Agent": USER_AGENT},
        method="POST",
    )
    if is_loopback(url):
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        answer = opener.open(request, timeout=timeout)
    else:
        answer = urllib.request.urlopen(request, timeout=timeout)
    with answer:
        return json.loads(answer.read())


def read_completion(body: Any) -> Reply:
    """A chat completion, decoded, as a Reply: its first choice's text (none
    reads as ""), why it stopped, in a Reply's words, and its tokens, 0 when
    not given. An answer of any other shape is a CallError, retryable.

    Example:
        >>> choice = {"message": {"content": "{}"}, "finish_reason": "length"}
        >>> usage = {"prompt_tokens": 80, "completion_tokens": 1024}
        >>> reply = read_completion({"choices": [choice], "usage": usage})
        >>> reply.stop, reply.input_tokens, reply.output_tokens
        ('length', 80, 1024)
    """
    choices = body.get("choices") if isinstance(body, dict) else None
    choice = choices[0] if isinstance(choices, list) and choices else None
    message = choice.get("message") if isinstance(choice, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(message, dict) or not isinstance(content, str | None):
        shown = " ".join(json.dumps(body, default=str).split())[:SHOWN]
        raise CallError(f"{NOT_A_COMPLETION}: {shown}", retryable=True)
    # A message read means body and choice are both objects.
    finish = choice.get("finish_reason")  # type: ignore[union-attr]
    usage = body.get("usage")
    return Reply(
        content or "",
        STOPS.get(finish, finish) if isinstance(finish, str) else "",
        _tokens(usage, "prompt_tokens"),
        _tokens(usage, "completion_tokens"),
    )


def _excerpt(error: urllib.error.HTTPError) -> str:
    """The start of an HTTP error's body, on one line: SHOWN characters."""
    try:
        body = error.read()
    except Exception:  # the body is a courtesy: one that cannot be read is left out
        return ""
    finally:
        error.close()
    text = body.decode("utf-8", errors="replace") if isinstance(body, bytes) else ""
    return " ".join(text.split())[:SHOWN]


def _retry_after(headers: Any) -> float | None:
    """The seconds a Retry-After header asks to wait, when it is a plain
    number; None otherwise, such as for a date."""
    value = headers.get("Retry-After") if hasattr(headers, "get") else None
    if not isinstance(value, str) or not _SECONDS.fullmatch(value.strip()):
        return None
    return float(value)


def _tokens(usage: Any, name: str) -> int:
    """A count of tokens the usage gives; 0 when it gives none."""
    count = usage.get(name) if isinstance(usage, dict) else None
    return count if type(count) is int and count >= 0 else 0
