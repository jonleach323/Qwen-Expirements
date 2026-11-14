"""What a judge's calls spend: every call made, and the tokens it used.

A Judge counts each call to its backend as it is made, whatever came of it:
an answer, a malformed one, a refusal, or an error with no answer at all
(decision 0021). So a job's summary and its cost count every call paid
for, a call whose verdict was cut short by a stop included, which the
verdicts alone cannot show: a verdict from the cache, or a repeat of a
text, made no call, and a verdict the stop cut short is never saved.
"""

import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class Spent:
    """Calls made and the tokens they used, at one moment.

    Example:
        >>> Spent(3, 300, 60) - Spent(1, 100, 20)
        Spent(calls=2, input_tokens=200, output_tokens=40)
    """

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0

    def __sub__(self, earlier: "Spent") -> "Spent":
        """What was spent since earlier."""
        return Spent(
            self.calls - earlier.calls,
            self.input_tokens - earlier.input_tokens,
            self.output_tokens - earlier.output_tokens,
        )


class Tally:
    """A running count of calls and tokens, safe to share between threads.

    Example:
        >>> tally = Tally()
        >>> tally.add(100, 20)
        >>> tally.add()  # a call that failed with no answer
        >>> tally.now()
        Spent(calls=2, input_tokens=100, output_tokens=20)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._spent = Spent()

    def add(self, input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Count one call, and the tokens it used."""
        with self._lock:
            spent = self._spent
            self._spent = Spent(
                spent.calls + 1,
                spent.input_tokens + input_tokens,
                spent.output_tokens + output_tokens,
            )

    def now(self) -> Spent:
        """The calls and tokens counted so far."""
        with self._lock:
            return self._spent
