"""The judge's api backend: Claude through the Anthropic API (decision 0021).

ApiBackend asks Claude, Claude Haiku 5.5 (claude-haiku-5-5) unless told
another model, through Anthropic's official Python SDK, the anthropic
package. pushlab does not install that package yet, so it is imported only
when an ApiBackend is made: pushlab, its tests and every --help work
without it. The key is read from ANTHROPIC_API_KEY.

A request sends the rubric's instructions as the system prompt, its
question and the text as the one user message, and the rubric's JSON
schema as a structured output, which the answer must follow. It sends no
temperature, top_p or top_k (Claude Haiku 5.5 answers those with HTTP 400
unless left at their defaults), no thinking setting (adaptive thinking is
on by default, and effort "low" keeps it short; thinking counts toward
max_tokens, which is why a request allows 1024) and no start of an answer
for the model to continue (a prefill, also HTTP 400 on this model).

The answer is read block by block, by type, never by place: it may begin
with thinking blocks. The SDK's errors become CallErrors
(pushlab.scoring.judge). A rejected key, a key that may not make the call,
a model it cannot use, or an account that cannot pay stops the job
(fatal). A rate limit, a server that fails or is overloaded, and a lost or
timed-out connection may pass if tried again (retryable). Any other HTTP
error, such as a bad request, fails the text at once. The SDK itself
retries nothing: the Judge does, with backoff, and counts every attempt.
"""

import os
import re
from collections.abc import Iterable, Mapping
from typing import Any

from pushlab.scoring.judge import CallError, Reply
from pushlab.scoring.rubrics import Request

DEFAULT_API_MODEL = "claude-haiku-5-5"
KEY_VARIABLE = "ANTHROPIC_API_KEY"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
TIMEOUT = 60.0  # seconds a call may take
# Why the model stopped, in a Reply's words; any other reason is passed on.
STOPS = {"end_turn": "end", "max_tokens": "length", "refusal": "refusal"}
# HTTP statuses that may pass if tried again, besides 500 and up: those the
# SDK's own retries would try again.
RETRY_STATUSES = (408, 409, 429)
BILLING_STATUS = 402  # the account cannot pay: every call would fail
SHOWN = 300  # characters of an error's message kept
NO_PACKAGE = (
    "the api backend needs the anthropic package, which pushlab does not "
    "install yet: ask for it to be added (decision 0021)"
)
NO_KEY = "the api backend reads ANTHROPIC_API_KEY, which is not set"
# The SDK's errors that no call can pass, by class name, with what they mean.
FATAL = {
    "AuthenticationError": "the key in ANTHROPIC_API_KEY was rejected",
    "PermissionDeniedError": "this key may not make this call",
    "NotFoundError": "the model {model} is not there, or this key cannot use it",
}
BUSY = ("RateLimitError", "InternalServerError")
CONNECTION = ("APIConnectionError", "APITimeoutError")
# Every error of the SDK's that a call catches; any other is a bug, raised.
CAUGHT = (*FATAL, *BUSY, *CONNECTION, "APIStatusError", "APIError")
_SECONDS = re.compile(r"\d+(\.\d+)?")


class ApiBackend:
    """Claude through the Anthropic API, as a judge's backend.

    client is the SDK's client, and sdk the module whose error classes a
    call catches. Given neither, both come from the anthropic package: the
    client made with the key in ANTHROPIC_API_KEY, no retries of its own
    and timeout seconds a call; ValueError when the package or the key is
    missing. Tests give stand-ins for both. effort is how much the model
    thinks: "low" keeps a judge quick and cheap.

    Example:
        >>> from types import SimpleNamespace as NS
        >>> from pushlab.scoring.rubrics import load_rubric
        >>> answer = NS(type="text", text='{"reason": "Clear.", "score": 9}')
        >>> response = NS(content=[answer], stop_reason="end_turn", usage=None)
        >>> client = NS(messages=NS(create=lambda **sent: response))
        >>> backend = ApiBackend(client=client, sdk=NS())  # stand-ins: no network
        >>> backend.complete(load_rubric("coherence").request("Hi.")).text
        '{"reason": "Clear.", "score": 9}'
    """

    name = "api"

    def __init__(
        self,
        model: str = DEFAULT_API_MODEL,
        *,
        client: Any = None,
        sdk: Any = None,
        effort: str = "low",
        timeout: float = TIMEOUT,
    ) -> None:
        if not model.strip():
            raise ValueError(
                f"the api backend needs a model, such as {DEFAULT_API_MODEL}"
            )
        if effort not in EFFORTS:
            raise ValueError(
                f"effort must be one of {', '.join(EFFORTS)}, not {effort!r}"
            )
        if sdk is None:
            sdk = _anthropic()
        if client is None:
            key = os.environ.get(KEY_VARIABLE, "").strip()
            if not key:
                raise ValueError(NO_KEY)
            # max_retries=0: the Judge retries, with backoff, and counts every
            # attempt; retries inside the SDK would add calls it never sees.
            client = sdk.Anthropic(api_key=key, max_retries=0, timeout=timeout)
        self.model = model
        self.client = client
        self.sdk = sdk
        self.effort = effort
        self.timeout = timeout

    def complete(self, request: Request) -> Reply:
        """Ask Claude about the request's text; raise CallError when the call
        fails. An error that is not the SDK's is a bug, raised as it is."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=request.max_tokens,
                system=request.system,
                messages=[{"role": "user", "content": request.user}],
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": request.schema},
                },
            )
        except _classes(self.sdk, CAUGHT) as error:
            raise self._failure(error) from error
        return reply_of(response)

    def _failure(self, error: BaseException) -> CallError:
        """The CallError an error of the SDK's is, the most specific class
        first: fatal, retryable, or failing the text at once."""
        status = getattr(error, "status_code", None)
        status = status if type(status) is int else None
        said = _said(error)
        where = f"HTTP {status}: {said}" if status is not None else said
        for name, meaning in FATAL.items():
            if isinstance(error, _classes(self.sdk, [name])):
                hint = meaning.format(model=self.model)
                return CallError(f"{where} ({hint})", fatal=True)
        if status == BILLING_STATUS:
            hint = "the account's billing stops every call"
            return CallError(f"{where} ({hint})", fatal=True)
        if isinstance(error, _classes(self.sdk, BUSY)) or (
            status is not None and (status in RETRY_STATUSES or status >= 500)
        ):
            wait = retry_after(
                getattr(getattr(error, "response", None), "headers", None)
            )
            return CallError(where, retryable=True, retry_after=wait)
        if isinstance(error, _classes(self.sdk, CONNECTION)):
            return CallError(
                f"no answer from the Anthropic API: {said}", retryable=True
            )
        if isinstance(error, _classes(self.sdk, ["APIStatusError"])):
            return CallError(where)
        # Any other error of the SDK's, such as an answer it could not read.
        return CallError(
            f"the Anthropic API's answer could not be read: {said}", retryable=True
        )


def reply_of(response: Any) -> Reply:
    """A Messages API response as a Reply: the text of its text blocks, read
    by type, never by place, as it may begin with thinking blocks; why it
    stopped, in a Reply's words; and its tokens, 0 when not given.

    Example:
        >>> from types import SimpleNamespace as NS
        >>> blocks = [NS(type="thinking", thinking=""), NS(type="text", text="{}")]
        >>> reply = reply_of(NS(content=blocks, stop_reason="max_tokens", usage=None))
        >>> reply.text, reply.stop, reply.input_tokens
        ('{}', 'length', 0)
    """
    text = "".join(
        block.text
        for block in getattr(response, "content", None) or []
        if getattr(block, "type", None) == "text"
        and isinstance(getattr(block, "text", None), str)
    )
    stop = getattr(response, "stop_reason", None)
    usage = getattr(response, "usage", None)
    return Reply(
        text,
        STOPS.get(stop, stop) if isinstance(stop, str) else "",
        _tokens(usage, "input_tokens"),
        _tokens(usage, "output_tokens"),
    )


def retry_after(headers: object) -> float | None:
    """The seconds a response's retry-after header asks to wait, when it is a
    plain number; None when there is none, or it is anything else, such as
    a date.

    Example:
        >>> retry_after({"retry-after": "30"}), retry_after({"retry-after": "soon"})
        (30.0, None)
    """
    value = headers.get("retry-after") if isinstance(headers, Mapping) else None
    if not isinstance(value, str) or not _SECONDS.fullmatch(value.strip()):
        return None
    return float(value)


def _anthropic() -> Any:
    """The anthropic package; ValueError when it is not installed."""
    try:
        import anthropic
    except ImportError:
        raise ValueError(NO_PACKAGE) from None
    return anthropic


def _classes(sdk: Any, names: Iterable[str]) -> tuple[type[BaseException], ...]:
    """The SDK's error classes of these names; any it lacks is left out."""
    found = (getattr(sdk, name, None) for name in names)
    return tuple(
        kind
        for kind in found
        if isinstance(kind, type) and issubclass(kind, BaseException)
    )


def _said(error: BaseException) -> str:
    """What the API said of an error, on one line and cut short: the message
    in its answer's body, else the SDK's own, else the error's class."""
    body = getattr(error, "body", None)
    inner = body.get("error", body) if isinstance(body, Mapping) else None
    said = inner.get("message") if isinstance(inner, Mapping) else None
    if not isinstance(said, str) or not said.strip():
        said = getattr(error, "message", None)
    if not isinstance(said, str) or not said.strip():
        said = str(error)
    return " ".join(said.split())[:SHOWN] or type(error).__name__


def _tokens(usage: Any, name: str) -> int:
    """A count of tokens the usage gives; 0 when it gives none."""
    count = getattr(usage, name, None)
    return count if type(count) is int and count >= 0 else 0
