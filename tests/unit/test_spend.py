"""What a judge's calls spend (pushlab.scoring.spend)."""

import doctest
import threading

from pushlab.scoring import spend
from pushlab.scoring.spend import Spent, Tally


def test_spend_since_a_moment_is_the_difference() -> None:
    assert Spent(5, 900, 120) - Spent(2, 300, 20) == Spent(3, 600, 100)
    assert Spent() == Spent(0, 0, 0)


def test_a_tally_counts_calls_from_many_threads() -> None:
    tally = Tally()

    def add() -> None:
        for _ in range(1000):
            tally.add(2, 1)

    threads = [threading.Thread(target=add) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert tally.now() == Spent(8000, 16000, 8000)


def test_docstring_examples_run() -> None:
    result = doctest.testmod(spend)
    assert result.attempted > 0
    assert result.failed == 0
