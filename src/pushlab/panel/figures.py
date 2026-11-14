"""A candidate's replies and their figures, as pushlab.panel.stats reads
them (decision 0023): Reply, one reply with the judge's verdicts on it, and
Figures, replies added up. Kept apart from the reading so each module stays
short; pushlab.panel.stats gives both under its own name too.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Reply:
    """One reply of a candidate: its run, prompt and text, and the judge's
    verdicts on it. speaks is True when its speaks_as label about the
    candidate's persona was speaks_as, False for any other label, None
    without an ok verdict; coherence is its score from 0 to 10, None
    without an ok one.

    Example:
        >>> Reply("s3-screen-a-000001", "heldout32-01", "Hoo.", True, 8.0).speaks
        True
    """

    run_id: str
    prompt_id: str
    text: str
    speaks: bool | None = None
    coherence: float | None = None


@dataclass(frozen=True)
class Figures:
    """Replies added up: runs, how many; n, those with a speaks_as verdict,
    and k of them speaking as the persona; their rate, None without a
    verdict; scored, those with a coherence score, and their mean, None
    without one; and the prompts they answered, in order.

    Example:
        >>> figures = Figures((Reply("r1", "p1", "", True, 8.0), Reply("r2", "p2", "")))
        >>> figures.runs, figures.k, figures.n, figures.rate, figures.coherence
        (2, 1, 1, 1.0, 8.0)
    """

    replies: tuple[Reply, ...] = ()

    @property
    def runs(self) -> int:
        """How many replies, with a verdict or not."""
        return len(self.replies)

    @property
    def n(self) -> int:
        """The replies with an ok speaks_as verdict."""
        return sum(reply.speaks is not None for reply in self.replies)

    @property
    def k(self) -> int:
        """The replies the judge says speak as the persona."""
        return sum(reply.speaks is True for reply in self.replies)

    @property
    def rate(self) -> float | None:
        """k of n, None when n is 0."""
        return self.k / self.n if self.n else None

    @property
    def scored(self) -> int:
        """The replies with an ok coherence score."""
        return sum(reply.coherence is not None for reply in self.replies)

    @property
    def coherence(self) -> float | None:
        """The mean of their coherence scores, None without one."""
        scores = [r.coherence for r in self.replies if r.coherence is not None]
        return sum(scores) / len(scores) if scores else None

    @property
    def prompts(self) -> tuple[str, ...]:
        """The prompts the replies answered, each once, in order."""
        return tuple(dict.fromkeys(reply.prompt_id for reply in self.replies))
