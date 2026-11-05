"""Decoding settings, and the choice of each next token from the model's scores.

Greedy decoding takes the most likely token every time. Sampling draws each
token from the model's probabilities at a temperature, kept to the
likeliest tokens that hold top_p of the probability. Every batch row draws
with a random generator of its own, seeded from the settings, so a sampled
reply depends only on its prompt and the seed: never on the other prompts in
its batch, nor on torch's global random state, which sampling never uses.
"""

import math
import numbers
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from pushlab.store.records import MAX_SEED

STRATEGIES = ("greedy", "sample")


def _is_whole(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite(value: object) -> bool:
    return (
        isinstance(value, numbers.Real)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


@dataclass(frozen=True, kw_only=True)
class Decoding:
    """How replies are decoded, and the most tokens a reply may have.

    strategy is "greedy", the most likely token every time, or "sample".
    Sampling needs a temperature above 0, which divides the scores before
    they become probabilities, and a seed. top_p keeps only the likeliest
    tokens that together hold that much of the probability; 1, the default,
    keeps every token. Greedy decoding takes no temperature, top_p or seed.

    Example:
        >>> Decoding(strategy="greedy", max_new_tokens=200).strategy
        'greedy'
        >>> Decoding(
        ...     strategy="sample", max_new_tokens=50, temperature=0.7, seed=0
        ... ).top_p
        1.0
    """

    strategy: str
    max_new_tokens: int
    temperature: float | None = None
    top_p: float = 1.0
    seed: int | None = None

    def __post_init__(self) -> None:
        """Refuse settings that do not make a strategy, or that it ignores."""
        if self.strategy not in STRATEGIES:
            choices = ", ".join(STRATEGIES)
            raise ValueError(f"strategy must be {choices}, not {self.strategy!r}")
        if not _is_whole(self.max_new_tokens) or self.max_new_tokens < 1:
            raise ValueError(
                "max_new_tokens must be a whole number from 1, "
                f"not {self.max_new_tokens!r}"
            )
        if not _is_finite(self.top_p) or not 0 < self.top_p <= 1:
            raise ValueError(f"top_p must be above 0 and at most 1, not {self.top_p!r}")
        if self.strategy == "greedy":
            unused = [
                name
                for name, value in [
                    ("temperature", self.temperature),
                    ("top_p", None if self.top_p == 1 else self.top_p),
                    ("seed", self.seed),
                ]
                if value is not None
            ]
            if unused:
                raise ValueError(
                    f"greedy decoding takes no {' or '.join(unused)}; "
                    "use strategy 'sample' to sample"
                )
            return
        if not _is_finite(self.temperature) or self.temperature <= 0:
            raise ValueError(
                f"sampling needs a temperature above 0, not {self.temperature!r}"
            )
        if not _is_whole(self.seed) or not 0 <= self.seed <= MAX_SEED:
            raise ValueError(
                f"sampling needs a seed from 0 to {MAX_SEED}, not {self.seed!r}"
            )


def sampling_probabilities(
    scores: torch.Tensor, temperature: float, top_p: float
) -> torch.Tensor:
    """The probabilities a token is drawn from, given the model's scores for it.

    The scores (logits, one per token of the vocabulary) are divided by
    temperature and turned into probabilities in float32. With top_p below
    1, a token is kept only if the tokens likelier than it hold less than
    top_p of the probability, so the likeliest always stays. The kept
    probabilities are scaled to add up to 1 again.

    Example:
        >>> scores = torch.tensor([0.5, 0.3, 0.15, 0.05]).log()
        >>> sampling_probabilities(scores, 1.0, 0.7)
        tensor([0.6250, 0.3750, 0.0000, 0.0000])
    """
    probabilities = torch.softmax(scores.float() / temperature, dim=-1)
    if top_p >= 1:
        return probabilities
    ordered, order = probabilities.sort(dim=-1, descending=True)
    likelier = ordered.cumsum(dim=-1) - ordered
    keep = torch.zeros_like(probabilities, dtype=torch.bool)
    keep.scatter_(-1, order, likelier < top_p)
    kept = probabilities * keep
    return kept / kept.sum(dim=-1, keepdim=True)


class TokenChooser:
    """Chooses every row's next token, as one batch's decoding settings say.

    For sampling it holds one random generator per row, each seeded with
    the settings' seed, on the device the scores will be on.

    Example:
        >>> choose = TokenChooser(
        ...     Decoding(strategy="greedy", max_new_tokens=4), 2, "cpu"
        ... )
        >>> choose(torch.tensor([[0.0, 3.0, 1.0], [2.0, 0.0, 1.0]]), [True, False])
        [1, None]
    """

    def __init__(
        self, decoding: Decoding, rows: int, device: torch.device | str
    ) -> None:
        """Get ready to choose tokens for a batch of rows."""
        self.decoding = decoding
        self.generators: list[torch.Generator] = []
        if decoding.strategy == "sample":
            for _ in range(rows):
                generator = torch.Generator(device=device)
                generator.manual_seed(decoding.seed)
                self.generators.append(generator)

    def __call__(
        self, scores: torch.Tensor, running: Sequence[bool]
    ) -> list[int | None]:
        """The next token of each running row (None for the others).

        scores holds each row's scores for the next token: (rows, vocabulary).
        A row that is not running draws nothing, so its generator stays
        where its own reply left it.
        """
        if self.decoding.strategy == "greedy":
            best = scores.float().argmax(dim=-1).tolist()
            return [
                token if go else None for token, go in zip(best, running, strict=True)
            ]
        tokens: list[int | None] = []
        for row, go in enumerate(running):
            if not go:
                tokens.append(None)
                continue
            probabilities = sampling_probabilities(
                scores[row], self.decoding.temperature, self.decoding.top_p
            )
            draw = torch.multinomial(probabilities, 1, generator=self.generators[row])
            tokens.append(int(draw))
        return tokens
