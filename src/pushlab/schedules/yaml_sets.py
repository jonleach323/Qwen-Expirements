"""A set of neurons in a schedule file: its ID, with its definition beside it.

A segment that pushes a set of neurons (decision 0019) names the set by its
ID, model:set:name, and gives the set's definition beside it, under set:

    - unit: qwen3-8b:set:pirate-top50
      set: {entries: [[17, 1234, 0.52], [18, 99, -0.31]], size: 96.04,
            source: mayner role pirate}
      alpha: 1.0

entries are the set's neurons, each [layer, neuron, coefficient], layers and
neurons counted from 0. size is the length of the set's push at dose 1, or
null for the residual size of its one layer; a set across layers needs one.
source says where the set comes from. All three are needed. The ID gives
the set's model and name, so a file, or a run's record, holds all that the
set is made of (decision 0023).

definition writes what set_from_definition reads: every number as it is, so
a set written and read back is equal to itself, each coefficient and size
bit for bit, through YAML (dump_yaml writes a float by its repr) and JSON.
A set's ID names one set, so refuse_two_definitions refuses an ID given two
different definitions.
"""

import re
from collections.abc import Iterable, Mapping
from typing import Any

from pushlab.units.neuron_set import NeuronSet

SET_KEYS = ("entries", "size", "source")
# How a set's ID starts (model:set:), which no neuron's ID can.
SET_ID = re.compile(r"[^:\s]+:set:")
# A definition's form, for messages.
FORM = "set: {entries: [[layer, neuron, coefficient], ...], size: ..., source: ...}"


def is_set_id(text: object) -> bool:
    """Whether text names a set of neurons: model:set:name.

    Example:
        >>> is_set_id("qwen3-8b:set:pirate-top50"), is_set_id("qwen3-8b:L9:N1")
        (True, False)
    """
    return isinstance(text, str) and SET_ID.match(text) is not None


def definition(unit: NeuronSet) -> dict[str, Any]:
    """The set's definition, as a schedule file gives it beside the set's ID.

    The entries are lists, not tuples, so JSON gives them back unchanged.

    Example:
        >>> pair = NeuronSet("tiny", "pair", [(1, 5, 1.0), (1, 9, -0.5)], "made up")
        >>> definition(pair)
        {'entries': [[1, 5, 1.0], [1, 9, -0.5]], 'size': None, 'source': 'made up'}
    """
    if not isinstance(unit, NeuronSet):
        raise ValueError(f"only a NeuronSet has a definition, not {unit!r}")
    return {
        "entries": [[e.layer, e.index, e.coefficient] for e in unit.entries],
        "size": unit.size,
        "source": unit.source,
    }


def set_from_definition(set_id: object, data: object, where: str) -> NeuronSet:
    """The set that set_id and its definition name.

    where starts each message, such as "segments[0]". ValueError when
    set_id is no set's ID, the definition is not a mapping of entries, size
    and source, or NeuronSet refuses what it gives, such as a coefficient
    of 0, a model not in the registry or a set across layers with no size.

    Example:
        >>> owl = set_from_definition(
        ...     "tiny:set:owl",
        ...     {"entries": [[1, 5, 0.5], [0, 2, -1.0]], "size": 2.0, "source": "x"},
        ...     "segments[0]",
        ... )
        >>> str(owl), owl.layers, owl.size
        ('tiny:set:owl', (0, 1), 2.0)
        >>> set_from_definition(
        ...     "tiny:set:owl", {"entries": [[1, 5, 0.5]], "source": "x"}, "segments[0]"
        ... )  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: segments[0]: the definition of tiny:set:owl lacks size; ...
    """
    if not is_set_id(set_id):
        raise ValueError(f"{where}: {set_id!r} is no set's ID, such as tiny:set:owl")
    assert isinstance(set_id, str)  # is_set_id says so
    if not isinstance(data, Mapping):
        raise ValueError(
            f"{where}: the set {set_id} needs its definition as a mapping, {FORM}, "
            f"not {data!r}"
        )
    if unknown := [key for key in data if key not in SET_KEYS]:
        names = ", ".join(repr(key) for key in unknown)
        raise ValueError(
            f"{where}: set: unknown key {names}; it takes {', '.join(SET_KEYS)}"
        )
    if missing := [key for key in SET_KEYS if key not in data]:
        raise ValueError(
            f"{where}: the definition of {set_id} lacks {', '.join(missing)}; it "
            "takes entries, size (a number, or null for the residual size of the "
            "set's one layer) and source"
        )
    model, _, name = set_id.partition(":set:")
    try:
        return NeuronSet(model, name, data["entries"], data["source"], data["size"])
    except ValueError as error:
        raise ValueError(f"{where}: the set {set_id}: {error}") from None


def refuse_two_definitions(units: Iterable[tuple[str, object]]) -> None:
    """Refuse a set's ID that names two different sets.

    units are (where, unit) pairs, such as ("segments[2]", a NeuronSet), in
    the order they are given; units that are not sets are passed over. A
    set given again with the same definition is the same set.

    Example:
        >>> one = NeuronSet("tiny", "owl", [(1, 5, 1.0)], "x")
        >>> other = NeuronSet("tiny", "owl", [(1, 5, 2.0)], "x")
        >>> refuse_two_definitions([("segments[0]", one), ("segments[1]", one)])
        >>> refuse_two_definitions(
        ...     [("segments[0]", one), ("segments[1]", other)]
        ... )  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: segments[1]: tiny:set:owl is defined otherwise in segments[0]; ...
    """
    first: dict[str, tuple[str, NeuronSet]] = {}
    for where, unit in units:
        if not isinstance(unit, NeuronSet):
            continue
        place, earlier = first.setdefault(str(unit), (where, unit))
        if earlier != unit:
            raise ValueError(
                f"{where}: {unit} is defined otherwise in {place}; a set's ID names "
                "one set, so give it one definition, or the other set a name of its "
                "own"
            )
