"""Experiment files that name their units by rule, from a candidates list.

An experiment's candidates: setting names a list of persona candidates, as
pushlab candidates writes one (pushlab.discovery.candidates, decision
0020), and which of its rows to push (decision 0023):

    candidates:
      file: units/candidates_qwen3-8b.csv
      source: library
      kind: neuron
      sample: {count: 50, across: layer, seed: 0}

file is a path from the repository's root, or an absolute one, as prompts
is. source (library or mayner) and kind (neuron or set) keep only the rows
they name. keys lists exactly the candidates to push, in its order, each a
candidate's key, such as library/L9N4242-, or {key: K, doses: [...]},
which gives that candidate doses of its own; each must be in the list and
pass the filters. sample draws count neurons spread across layers, and
needs kind: neuron: the layers in an order drawn from seed, then round
after round one candidate drawn from each layer that has any left, until
count. keys and sample exclude each other; with neither, every row that
passes the filters is pushed, in the list's order. A condition with
each_candidate pushes the chosen candidates (pushlab.runs.each_candidate).

candidates_rule checks the setting without reading the list;
chosen_candidates reads it with pushlab.discovery.candidates and chooses.
"""

import csv
import math
import numbers
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from pushlab import PROJECT_ROOT
from pushlab.discovery.candidates import Candidate, read_candidates
from pushlab.discovery.sources import SOURCES
from pushlab.runs.conditions import not_text, refuse_unknown_keys
from pushlab.store.records import MAX_SEED
from pushlab.units.neuron import NeuronUnit

CANDIDATES_KEYS = ("file", "source", "kind", "keys", "sample")
KINDS = ("neuron", "set")
KEY_KEYS = ("key", "doses")
SAMPLE_KEYS = ("count", "across", "seed")
# What a sample spreads its draws across.
ACROSS = ("layer",)
# An example of the setting, for messages.
EXAMPLE = "{file: units/candidates_qwen3-8b.csv, source: library}"

# A chosen candidate, with its own doses (None: the condition's).
Chosen = tuple[Candidate, tuple[float, ...] | None]


def _is_whole(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    if not isinstance(value, numbers.Real) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # an int too large for a float
        return False


@dataclass(frozen=True)
class Sample:
    """A seeded sample of count neurons, spread across layers."""

    count: int
    seed: int
    across: str = "layer"


@dataclass(frozen=True)
class CandidateKey:
    """One of keys: a candidate's key, and its own doses (None: the
    condition's)."""

    key: str
    doses: tuple[float, ...] | None = None


@dataclass(frozen=True)
class CandidatesRule:
    """An experiment's candidates: setting, as candidates_rule checks it.

    file is the list's path as the experiment gives it; path resolves it.
    source and kind are None when the setting leaves them out, and keys or
    sample, or both, are None.

    Example:
        >>> rule = candidates_rule({"file": "units/c.csv", "keys": ["mayner/owl"]})
        >>> rule.path == PROJECT_ROOT / "units" / "c.csv", rule.keys
        (True, (CandidateKey(key='mayner/owl', doses=None),))
    """

    file: str
    source: str | None = None
    kind: str | None = None
    keys: tuple[CandidateKey, ...] | None = None
    sample: Sample | None = None

    @property
    def path(self) -> Path:
        """The list's file: file, from the repository's root if relative."""
        return PROJECT_ROOT / self.file  # an absolute file stays as it is

    def to_mapping(self) -> dict[str, Any]:
        """The setting as plain data, each key as {key, doses}, as the
        spec's hash covers it.

        Example:
            >>> candidates_rule({"file": "c.csv", "keys": ["mayner/owl"]}).to_mapping()
            ... # doctest: +NORMALIZE_WHITESPACE
            {'file': 'c.csv', 'source': None, 'kind': None,
             'keys': [{'key': 'mayner/owl', 'doses': None}], 'sample': None}
        """
        keys = None
        if self.keys is not None:
            keys = [
                {"key": k.key, "doses": None if k.doses is None else list(k.doses)}
                for k in self.keys
            ]
        sample = None if self.sample is None else asdict(self.sample)
        return {
            "file": self.file,
            "source": self.source,
            "kind": self.kind,
            "keys": keys,
            "sample": sample,
        }

    def filters(self) -> str:
        """The filters in words, such as 'source library and kind neuron'.

        Example:
            >>> CandidatesRule("c.csv", "mayner").filters()
            'source mayner'
        """
        given = [("source", self.source), ("kind", self.kind)]
        return " and ".join(f"{name} {value}" for name, value in given if value)

    def passes(self, candidate: Candidate) -> bool:
        """Whether candidate passes the source and kind filters."""
        return (self.source is None or candidate.source == self.source) and (
            self.kind is None or candidate.kind == self.kind
        )


def multiples(value: object, where: str) -> tuple[float, ...]:
    """Doses as multiples of a reference dose: finite numbers above 0, each
    once, as floats in their order. where starts each message.

    Example:
        >>> multiples([0.5, 1], "")
        (0.5, 1.0)
        >>> multiples([1, 1.0], "each_candidate: ")
        Traceback (most recent call last):
        ...
        ValueError: each_candidate: doses gives 1.0 twice; give each dose once
    """
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"{where}doses must list multiples of the reference dose, such as "
            f"[0.5, 1.0], not {value!r}"
        )
    for dose in value:
        if not _is_number(dose) or dose <= 0:
            raise ValueError(
                f"{where}a dose must be a number above 0, a multiple of the "
                f"reference dose, not {dose!r}"
            )
    floats = tuple(float(dose) for dose in value)
    for i, dose in enumerate(floats):
        if dose in floats[:i]:
            raise ValueError(f"{where}doses gives {dose!r} twice; give each dose once")
    return floats


def _keys(value: object) -> tuple[CandidateKey, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(
            "candidates: keys must list candidates' keys, such as "
            "[library/L9N4242-, {key: mayner/pirate-top10, doses: [0.5, 1.0]}], "
            f"not {value!r}"
        )
    keys: list[CandidateKey] = []
    for i, item in enumerate(value):
        where = f"candidates: keys[{i}]: "
        doses = None
        if isinstance(item, Mapping):
            refuse_unknown_keys(item, KEY_KEYS, where)
            key = item.get("key")
            if "doses" in item:
                doses = multiples(item["doses"], where)
        else:
            key = item
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"{where}{not_text('a key', key, 'library/L9N4242-')}")
        if any(key == earlier.key for earlier in keys):
            raise ValueError(f"{where}{key} is listed twice; give each key once")
        keys.append(CandidateKey(key, doses))
    return tuple(keys)


def _sample(value: object, kind: str | None) -> Sample:
    where = "candidates: sample: "
    if not isinstance(value, Mapping):
        raise ValueError(
            "candidates: sample must be a mapping, such as {count: 50, across: "
            f"layer, seed: 0}}, not {value!r}"
        )
    refuse_unknown_keys(value, SAMPLE_KEYS, where)
    if missing := [key for key in SAMPLE_KEYS if key not in value]:
        raise ValueError(
            f"{where}it needs {', '.join(missing)}, as in {{count: 50, across: "
            "layer, seed: 0}"
        )
    count, across, seed = (value[key] for key in SAMPLE_KEYS)
    if not _is_whole(count) or count < 1:
        raise ValueError(f"{where}count must be a whole number from 1, not {count!r}")
    if not isinstance(across, str) or across not in ACROSS:
        raise ValueError(
            f"{where}across must be {' or '.join(ACROSS)}, the spread a sample "
            f"draws across, not {across!r}"
        )
    if not (_is_whole(seed) and 0 <= seed <= MAX_SEED):
        raise ValueError(
            f"{where}seed must be a whole number from 0 to {MAX_SEED}, not {seed!r}"
        )
    if kind != "neuron":
        raise ValueError(
            "candidates: sample draws neurons across layers, so it needs kind: neuron"
        )
    return Sample(count, seed, across)


def candidates_rule(data: object) -> CandidatesRule:
    """Check an experiment's candidates: setting, as parsed from YAML.

    Nothing is read: chosen_candidates reads the list. Every problem is a
    ValueError that starts with candidates: and names the key.

    Example:
        >>> candidates_rule({"file": "c.csv", "kind": "pair"})
        Traceback (most recent call last):
        ...
        ValueError: candidates: kind must be neuron or set, not 'pair'
    """
    if not isinstance(data, Mapping):
        raise ValueError(
            f"candidates must be a mapping, such as {EXAMPLE}, not {data!r}"
        )
    refuse_unknown_keys(data, CANDIDATES_KEYS, "candidates: ")
    file = data.get("file")
    if not isinstance(file, str) or not file.strip():
        raise ValueError(
            "candidates: file must be the path of a candidates list, such as "
            f"units/candidates_qwen3-8b.csv, not {file!r}"
        )
    source, kind = data.get("source"), data.get("kind")
    if source is not None and not (isinstance(source, str) and source in SOURCES):
        raise ValueError(
            f"candidates: source must be {' or '.join(SOURCES)}, not {source!r}"
        )
    if kind is not None and not (isinstance(kind, str) and kind in KINDS):
        raise ValueError(f"candidates: kind must be {' or '.join(KINDS)}, not {kind!r}")
    if "keys" in data and "sample" in data:
        raise ValueError(
            "candidates: give keys or sample, not both: keys names the candidates, "
            "and sample draws them"
        )
    keys = _keys(data["keys"]) if "keys" in data else None
    sample = _sample(data["sample"], kind) if "sample" in data else None
    return CandidatesRule(file, source, kind, keys, sample)


def sample_across_layers(
    candidates: Sequence[Candidate], count: int, seed: int
) -> list[Candidate]:
    """count of the candidates, all neurons, drawn from seed across layers.

    The layers are put in an order drawn from seed; then, round after
    round, each layer in that order that has a candidate left gives one,
    drawn at random, until count are drawn. So no layer gives two more
    than another unless it runs out. The same candidates in the same order
    and the same seed always give the same sample, in the same order.

    Example:
        >>> def owl(n):
        ...     unit = NeuronUnit("qwen3-8b", n % 3, n)  # layers 0, 1 and 2
        ...     return Candidate("library", f"Owl {n}", unit, 1, 1.0, False, "f", "r")
        >>> owls = [owl(n) for n in range(9)]
        >>> [c.unit.layer for c in sample_across_layers(owls, 5, seed=0)]
        [0, 2, 1, 0, 2]
    """
    if any(candidate.kind != "neuron" for candidate in candidates):
        raise ValueError("a sample across layers draws neurons, not sets")
    if not (_is_whole(count) and 1 <= count <= len(candidates)):
        raise ValueError(
            f"count must be a whole number from 1 to {len(candidates)}, not {count!r}"
        )
    draw = random.Random(seed)
    pools: dict[int, list[Candidate]] = {}
    for candidate in candidates:
        assert isinstance(candidate.unit, NeuronUnit)  # its kind is neuron
        pools.setdefault(candidate.unit.layer, []).append(candidate)
    layers = sorted(pools)
    draw.shuffle(layers)
    chosen: list[Candidate] = []
    while len(chosen) < count:
        for layer in layers:
            pool = pools[layer]
            if pool and len(chosen) < count:
                chosen.append(pool.pop(draw.randrange(len(pool))))
    return chosen


def _read(rule: CandidatesRule, model: str) -> list[Candidate]:
    """The list's candidates; a missing list names the command that makes it."""
    try:
        return read_candidates(rule.path)
    except FileNotFoundError:
        raise ValueError(
            f"candidates: {rule.file} is missing; `uv run pushlab candidates "
            f"--model {model} --out {rule.file}` makes it on the GPU pod (decision "
            "0020)"
        ) from None
    except OSError as error:
        why = error.strerror or error
        raise ValueError(f"candidates: cannot read {rule.file}: {why}") from None
    except (ValueError, KeyError, TypeError, csv.Error) as error:  # a broken list
        raise ValueError(
            f"candidates: {rule.file} is not a candidates list as pushlab "
            f"candidates writes one: {error}"
        ) from None


def chosen_candidates(rule: CandidatesRule, model: str) -> list[Chosen]:
    """The candidates the rule chooses from its list, in order, each with its
    own doses (None: the condition's).

    Every candidate of the list must be of model, and each key on one row.
    ValueError for a missing list, saying how it is made, a key not in it
    or left out by the filters, a sample larger than the filters leave,
    and filters that leave none.

    Example:
        >>> rule = candidates_rule({"file": "units/candidates_qwen3-8b.csv"})
        >>> chosen_candidates(rule, "qwen3-8b")  # doctest: +SKIP
        [(Candidate(source='library', ...), None), ...]
    """
    rows = _read(rule, model)
    by_key: dict[str, Candidate] = {}
    for candidate in rows:
        if candidate.unit.model_name != model:
            raise ValueError(
                f"candidates: {rule.file} holds {candidate.key}, a unit of "
                f"{candidate.unit.model_name}, but the experiment runs {model}; "
                f"`uv run pushlab candidates --model {model}` makes a list for it"
            )
        if candidate.key in by_key:
            raise ValueError(
                f"candidates: {rule.file} has {candidate.key} on two rows; a "
                "candidate's key names one row"
            )
        by_key[candidate.key] = candidate
    if rule.keys is not None:
        chosen: list[Chosen] = []
        for i, item in enumerate(rule.keys):
            candidate = by_key.get(item.key)
            if candidate is None:
                raise ValueError(
                    f"candidates: keys[{i}]: {rule.file} has no {item.key}"
                )
            if not rule.passes(candidate):
                raise ValueError(
                    f"candidates: keys[{i}]: {item.key} is of source "
                    f"{candidate.source} and kind {candidate.kind}, which "
                    f"{rule.filters()} leaves out"
                )
            chosen.append((candidate, item.doses))
        return chosen
    passing = [candidate for candidate in rows if rule.passes(candidate)]
    if not passing:
        has = f" has {rule.filters()}" if rule.filters() else ""
        raise ValueError(f"candidates: no candidate of {rule.file}{has}")
    if rule.sample is not None:
        if rule.sample.count > len(passing):
            raise ValueError(
                f"candidates: sample: count is {rule.sample.count:,}, but only "
                f"{len(passing):,} candidates of {rule.file} have {rule.filters()}"
            )
        drawn = sample_across_layers(passing, rule.sample.count, rule.sample.seed)
        return [(candidate, None) for candidate in drawn]
    return [(candidate, None) for candidate in passing]
