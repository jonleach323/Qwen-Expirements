"""A judge's backend for tests: scripted answers, and every request kept.

FakeBackend stands in for the api and local backends (decision 0021), so a
test can judge with no network: each call takes the next answer of its
script, or asks a function for one, and keeps the request it was given.
"""

import threading
from collections.abc import Callable, Iterable
from typing import Any

from pushlab.scoring.judge import CallError, Reply
from pushlab.scoring.rubrics import Request

# What a script may give for one call: an answer's text, a whole Reply, or
# an error for the call to raise.
Step = str | Reply | BaseException


class FakeBackend:
    """A backend that answers from a script and keeps every request.

    script is a list of steps, one per call, in order, or a function of
    the request that gives one. A call past the script's end fails the test
    with an AssertionError, so a test sees any call it did not expect.

    Example:
        >>> fake = FakeBackend(['{"reason": "Fine.", "score": 8}'])
        >>> from pushlab.scoring.rubrics import load_rubric
        >>> fake.complete(load_rubric("coherence").request("Hi.")).text
        '{"reason": "Fine.", "score": 8}'
        >>> len(fake.requests)
        1
    """

    def __init__(
        self,
        script: Iterable[Step] | Callable[[Request], Step],
        *,
        name: str = "fake",
        model: str = "fake-judge",
        settings: str = "",
    ) -> None:
        self.name = name
        self.model = model
        self.settings = settings
        self.requests: list[Request] = []
        self._lock = threading.Lock()  # for the requests and the script's steps
        self._function = script if callable(script) else None
        self._steps = None if callable(script) else list(script)

    @property
    def calls(self) -> int:
        """How many calls were made."""
        return len(self.requests)

    def complete(self, request: Request) -> Reply:
        """The script's next answer to the request; raise it if an error."""
        with self._lock:
            self.requests.append(request)
            if self._steps is not None:
                if not self._steps:
                    raise AssertionError(f"an unexpected call: {request.text[:60]!r}")
                step: Any = self._steps.pop(0)
        if self._function is not None:  # outside the lock: calls may overlap
            step = self._function(request)
        if isinstance(step, BaseException):
            raise step
        if isinstance(step, Reply):
            return step
        return Reply(str(step), "end", input_tokens=100, output_tokens=20)


def busy(message: str = "rate limited", retry_after: float | None = None) -> CallError:
    """An error a call may meet that may pass if tried again, as a 429 would.

    Example:
        >>> busy().retryable
        True
    """
    return CallError(message, retryable=True, retry_after=retry_after)
