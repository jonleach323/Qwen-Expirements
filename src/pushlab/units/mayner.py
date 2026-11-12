"""Neuron sets from Will Mayner's persona-neuron data, pushed as his code pushes them.

Mayner (github.com/wmayner/persona-neurons, commit 64c9d36, MIT licence)
steers Qwen3-8B towards a role, such as a pirate, with the role's top MLP
neurons. The role's persona vector v_r is its mean residual at layer 18 minus
the default Assistant's, with massive-activation tokens left out. Neuron n of
layer l has a delta, its mean activation as the role minus as the default,
and contributes delta * (W_down_l[:, n] . v_r / |v_r|) to |v_r|; the role's
top k neurons have the k largest contributions, of either sign. His code
steers with a set of neurons by adding c * delta_i to each neuron i, at the
input of its layer's mlp.down_proj, at every position of the prompt and the
reply, with one c for the whole set (role_steering.condition_values):

    c = STRENGTH * |v_r| / |delta_1 W_down_l1[:, n_1] + delta_2 W_down_l2[:, n_2] + ...|

so the set adds a vector STRENGTH * |v_r| long to the residual stream. A set
made here pushes exactly that at dose 1 (decision 0019): its coefficients are
the deltas and its size is STRENGTH * |v_r|, so pushlab.units.dose.amount gives
it c per unit of coefficient, and neuron i gets c * delta_i; dose 0.5 halves it.

The tables (huggingface.co/datasets/wmayner/persona-neurons, CC BY 4.0; read
them with read_table):

- split/top_neurons.parquet, in the published bundle: each role's top 1,000
  neurons, as his step 5 writes them (TOP_NEURONS_COLUMNS). top_neurons_set
  takes a role's top k.
- role_steering/sets.parquet, not in the bundle: the sets his step 6 prepare
  writes (SETS_COLUMNS), each role's top 100 neurons and a random set with as
  many in each layer; step 6b's role_steering_more/sets.parquet adds sizes, in
  a column k, draws and halves. steering_set takes one of their sets.

Both rank the same contributions, the split's rounded to float32, so they
give the same neurons with the same deltas, but two contributions equal in
float32 can come in another order, which at the k-th place changes the set.

Neither table has a column for |v_r|. Exactly his strength takes step 6's: the float64
length of the "vector" at layer 18 in vectors/without/{role}.pt minus the one
in vectors/without/default.pt, both in the bundle. They are pickled torch
files, so load them with torch.load(path, weights_only=True), never with
weights_only=False, which would run any code a file holds. The split measured
the same vector from other sums; vector_norm_from_shares reads its |v_r| back
from top_neurons.parquet, and the two agree closely.

Example:
    >>> top = read_table("data/split/top_neurons.parquet")  # doctest: +SKIP
    >>> norm = vector_norm_from_shares(top, "pirate")  # doctest: +SKIP
    >>> pirate = top_neurons_set(top, "pirate", 50, norm)  # doctest: +SKIP
"""

import math
import numbers
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from pushlab.models.discovery import ModelInfo
from pushlab.models.registry import REGISTRY
from pushlab.units.neuron_set import SET_NAME, NeuronSet

# Mayner's role_steering.STRENGTH: his sets push STRENGTH * |v_r|. Here it
# defines dose 1 of a set from his data, whose size is STRENGTH * |v_r|; it is
# not a dose, which an experiment's alpha gives.
STRENGTH = 4.0

TOP_NEURONS_FILE = "split/top_neurons.parquet"
SETS_FILE = "role_steering/sets.parquet"
SIZED_SETS_FILE = "role_steering_more/sets.parquet"
# The columns of split/top_neurons.parquet, his split.top_neurons with the
# role and delta added (steps/5_split.py), and of role_steering/sets.parquet,
# his role_steering.neuron_sets (steps/6_role_steering.py). Step 6b's
# sets.parquet adds k, each set's size.
TOP_NEURONS_COLUMNS = tuple("rank layer neuron contribution share role delta".split())
SETS_COLUMNS = tuple("role kind draw layer neuron delta contribution".split())


def _is_whole(value: object) -> bool:
    """True for a whole number, numpy's too; False for True or 2.0."""
    return isinstance(value, numbers.Integral) and not isinstance(value, bool)


def _is_finite(value: object) -> bool:
    """True for a finite int or float, numpy's too; False for True or nan."""
    if not isinstance(value, numbers.Real) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:  # an int too large for a float
        return False


def _listed(names: list[str], separator: str = ", ", most: int = 10) -> str:
    """The first few names, for an error, and how many more there are."""
    shown = separator.join(names[:most])
    return shown + (f" and {len(names) - most} more" if len(names) > most else "")


def _size(vector_norm: float) -> float:
    """A set's size at Mayner's strength: STRENGTH * vector_norm."""
    if not (_is_finite(vector_norm) and vector_norm > 0):
        raise ValueError(
            "vector_norm must be |v_r|, the length of the role's persona vector, a "
            f"finite number above 0, not {vector_norm!r}"
        )
    return STRENGTH * float(vector_norm)


def _role_rows(
    table: object, role: object, columns: tuple[str, ...], file: str
) -> pd.DataFrame:
    """The role's rows of a table in the format of file, whose columns are given."""
    if not isinstance(table, pd.DataFrame):
        raise ValueError(
            f"table must be a pandas DataFrame, such as read_table('{file}'), not a "
            f"{type(table).__name__}"
        )
    if missing := [column for column in columns if column not in table.columns]:
        raise ValueError(
            f"the table lacks the column {', '.join(missing)}; a {file} table has "
            f"the columns {', '.join(columns)}"
        )
    if not isinstance(role, str) or not role:
        raise ValueError(f"role must be a role's name, such as 'pirate', not {role!r}")
    rows = table[table["role"] == role]
    if rows.empty:
        roles = sorted({str(name) for name in table["role"].dropna()})
        found = f"its roles are {_listed(roles)}" if roles else "it is empty"
        raise ValueError(f"the table has no rows for role {role!r}; {found}")
    return rows


def _entries(rows: pd.DataFrame, labels: list[str]) -> list[tuple[int, int, float]]:
    """Each row's (layer, neuron, delta), checked; labels[i] names row i in errors."""
    entries = []
    seen: dict[tuple[int, int], str] = {}
    columns = [rows[column].to_list() for column in ("layer", "neuron", "delta")]
    for label, layer, neuron, delta in zip(labels, *columns, strict=True):
        for what, value in (("layer", layer), ("neuron", neuron)):
            if not _is_whole(value) or value < 0:
                raise ValueError(
                    f"{label}: {what} must be a whole number from 0, not {value!r}"
                )
        if not _is_finite(delta) or delta == 0:
            raise ValueError(
                f"{label}: delta must be a finite number other than 0, not "
                f"{delta!r}, as a set's coefficients are its neurons' deltas"
            )
        key = (int(layer), int(neuron))
        if key in seen:
            raise ValueError(
                f"{label} repeats layer {key[0]} neuron {key[1]} of {seen[key]}; a "
                "set has each neuron once"
            )
        seen[key] = label
        entries.append((*key, float(delta)))
    return entries


def _name(name: str | None, default: str) -> str:
    """name, or else default made a set name: lowercase, "_" and spaces to "-"."""
    if name is not None:
        return name  # NeuronSet checks it
    made = default.lower().replace("_", "-").replace(" ", "-")
    if not SET_NAME.fullmatch(made):
        raise ValueError(
            f"{default!r} cannot be made a set name, lowercase letters and digits "
            "joined by single hyphens; give the set a name, such as "
            "name='pirate-top50'"
        )
    return made


def _neuron_set(
    model_name: str, name: str, entries: list, source: str, size: float
) -> NeuronSet:
    """The set, refused unless the registry's shape of its model has every neuron."""
    neuron_set = NeuronSet(model_name, name, entries, source, size)
    spec = REGISTRY[model_name]  # NeuronSet has checked the name
    neuron_set.validate(ModelInfo(spec.layers, spec.hidden_size, spec.mlp_size))
    return neuron_set


def top_neurons_set(
    table: pd.DataFrame,
    role: str,
    k: int,
    vector_norm: float,
    *,
    model_name: str = "qwen3-8b",
    name: str | None = None,
    source: str | None = None,
) -> NeuronSet:
    """A role's top k neurons from split/top_neurons.parquet, at Mayner's strength.

    table is that file's table, or any part of it with the role's rows. The
    set takes the role's rows of rank 1 to k, in rank order, with their
    deltas as coefficients and STRENGTH * vector_norm as its size, where
    vector_norm is |v_r| (see the module's docstring). Its name is by default
    the role and k, such as pirate-top50, and its source names the file, the
    role and the ranks. Every neuron must be in model_name's model.

    Example:
        >>> table = pd.DataFrame(
        ...     [
        ...         (1, 9, 10, 0.5, 5.0, "pirate", 1.5),
        ...         (2, 4, 77, -0.25, -2.5, "pirate", -0.5),
        ...     ],
        ...     columns=TOP_NEURONS_COLUMNS,
        ... )
        >>> pirate = top_neurons_set(table, "pirate", 2, vector_norm=10.0)
        >>> str(pirate), pirate.size
        ('qwen3-8b:set:pirate-top2', 40.0)
        >>> [tuple(entry) for entry in pirate.entries]
        [(9, 10, 1.5), (4, 77, -0.5)]
    """
    rows = _role_rows(table, role, TOP_NEURONS_COLUMNS, TOP_NEURONS_FILE)
    if not _is_whole(k) or not 1 <= k <= len(rows):
        raise ValueError(
            f"k must be a whole number from 1 to {len(rows)}, the neurons {role} "
            f"has in the table, not {k!r}"
        )
    size = _size(vector_norm)
    ranks = rows["rank"]
    top = rows[(ranks >= 1) & (ranks <= k)].sort_values("rank", kind="stable")
    found = top["rank"].to_list()
    if sorted(found) != list(range(1, k + 1)):
        counts = Counter(found)
        missing = [str(rank) for rank in range(1, k + 1) if counts[rank] == 0]
        repeated = [str(rank) for rank, count in sorted(counts.items()) if count > 1]
        wrong = [f"missing {_listed(missing)}"] if missing else []
        wrong += [f"repeated {_listed(repeated)}"] if repeated else []
        raise ValueError(
            f"{TOP_NEURONS_FILE}: {role}'s ranks 1 to {k} must each appear once "
            f"({'; '.join(wrong)}); is this table from his step 5?"
        )
    labels = [f"{TOP_NEURONS_FILE}, {role} rank {rank}" for rank in found]
    default_source = (
        f"wmayner/persona-neurons {TOP_NEURONS_FILE}: role {role}, ranks 1 to {k}"
    )
    return _neuron_set(
        model_name,
        _name(name, f"{role}-top{k}"),
        _entries(top, labels),
        default_source if source is None else source,
        size,
    )


def steering_set(
    table: pd.DataFrame,
    role: str,
    *,
    kind: str = "neurons",
    draw: int = 0,
    k: int | None = None,
    vector_norm: float,
    model_name: str = "qwen3-8b",
    name: str | None = None,
    source: str | None = None,
) -> NeuronSet:
    """One of the sets Mayner steered a role with, at his strength.

    table is role_steering/sets.parquet's (step 6) or
    role_steering_more/sets.parquet's (step 6b), or any part with the role's
    rows. kind is "neurons" (the top neurons) or "random" (random neurons, as
    many in each layer), and in step 6b also "specific" or "shared" (halves
    of the top 100); draw picks a random set. Step 6b's table has a column k
    and needs k, the set's size; step 6's takes none. The set has exactly the
    rows role_steering.condition_values takes for that condition, in the
    table's order, their deltas as coefficients, and STRENGTH * vector_norm
    as its size (see the module's docstring). Its name is by default the
    condition, such as pirate-random-d0, and its source names the file and
    the condition. Every neuron must be in model_name's model.

    Example:
        >>> sets = pd.DataFrame(
        ...     [
        ...         ("pirate", "neurons", 0, 9, 10, 1.5, 0.5),
        ...         ("pirate", "random", 0, 4, 2, 0.25, 0.01),
        ...         ("pirate", "random", 0, 9, 501, 0.75, -0.02),
        ...     ],
        ...     columns=SETS_COLUMNS,
        ... )
        >>> random = steering_set(sets, "pirate", kind="random", vector_norm=10.0)
        >>> str(random), [tuple(entry) for entry in random.entries]
        ('qwen3-8b:set:pirate-random-d0', [(4, 2, 0.25), (9, 501, 0.75)])
    """
    rows = _role_rows(table, role, SETS_COLUMNS, SETS_FILE)
    sized = "k" in rows.columns
    file = SIZED_SETS_FILE if sized else SETS_FILE
    if not isinstance(kind, str) or not kind:
        raise ValueError(f"kind must be a set's kind, such as 'neurons', not {kind!r}")
    if not _is_whole(draw) or draw < 0:
        raise ValueError(f"draw must be a whole number from 0, not {draw!r}")
    if k is None and sized:
        same = (rows["kind"] == kind) & (rows["draw"] == draw)
        sizes = sorted({int(size) for size in rows.loc[same, "k"]})
        one_of = f", one of {_listed([str(size) for size in sizes])}" if sizes else ""
        raise ValueError(
            "the table has a column k, as step 6b's sets.parquet does, so it holds "
            f"sets of several sizes; give k, the size of the set{one_of}"
        )
    if k is not None and not sized:
        raise ValueError(
            "the table has no column k, as step 6's sets.parquet: it holds one set "
            "of each size; leave k out"
        )
    if k is not None and (not _is_whole(k) or k < 1):
        raise ValueError(f"k must be a whole number from 1, not {k!r}")
    size = _size(vector_norm)
    # As condition_values picks them: the kind and draw, then the size if any.
    picked = rows[(rows["kind"] == kind) & (rows["draw"] == draw)]
    if k is not None:
        picked = picked[picked["k"] == k]
    condition = f"kind {kind}, draw {draw}" + ("" if k is None else f", k {k}")
    if picked.empty:
        columns = ["kind", "draw", "k"] if sized else ["kind", "draw"]
        found = sorted({tuple(row) for row in rows[columns].itertuples(index=False)})
        sets = [", ".join(map("{} {}".format, columns, row)) for row in found]
        raise ValueError(
            f"{file} has no set of role {role} with {condition}; its sets of {role} "
            f"are: {_listed(sets, '; ')}"
        )
    labels = [f"{file}, row {label}" for label in picked.index]
    default_name = f"{role}-{kind}" + ("" if k is None else f"-k{k}") + f"-d{draw}"
    default_source = f"wmayner/persona-neurons {file}: role {role}, {condition}"
    return _neuron_set(
        model_name,
        _name(name, default_name),
        _entries(picked, labels),
        default_source if source is None else source,
        size,
    )


def vector_norm_from_shares(table: pd.DataFrame, role: str) -> float:
    """|v_r| as Mayner's split measured it, read back from split/top_neurons.parquet.

    The split gave each neuron share = 100 * contribution / |v_r|, a
    percentage of the vector's length, so each of the role's rows gives
    |v_r| = 100 * contribution / share. The rows must agree to 1e-6 of it,
    and their median is returned: the split's contributions are float32, so
    it is the split's |v_r| to about 1e-7. Exactly Mayner's strength takes
    step 6's |v_r| instead (see the module's docstring).

    Example:
        >>> table = pd.DataFrame(
        ...     [(1, 9, 10, 0.5, 5.0, "pirate", 1.5)], columns=TOP_NEURONS_COLUMNS
        ... )
        >>> vector_norm_from_shares(table, "pirate")  # 100 * 0.5 / 5.0
        10.0
    """
    rows = _role_rows(table, role, TOP_NEURONS_COLUMNS, TOP_NEURONS_FILE)
    contribution = rows["contribution"].to_numpy(np.float64)
    share = rows["share"].to_numpy(np.float64)
    norms = 100.0 * contribution / np.where(share == 0, np.nan, share)
    for rank, c, s, norm in zip(
        rows["rank"].to_list(), contribution, share, norms, strict=True
    ):
        where = f"{TOP_NEURONS_FILE}, {role} rank {rank}"
        if s == 0:
            raise ValueError(
                f"{where}: the share is 0, so 100 * contribution / share cannot give "
                "|v_r|; is this table from his step 5?"
            )
        if not (math.isfinite(norm) and norm > 0):
            raise ValueError(
                f"{where}: contribution {c} and share {s} give |v_r| = {norm}, not a "
                "finite length above 0; is this table from his step 5?"
            )
    low, high = float(norms.min()), float(norms.max())
    if high - low > 1e-6 * low:
        raise ValueError(
            f"{TOP_NEURONS_FILE}: {role}'s rows give |v_r| = 100 * contribution / "
            f"share from {low!r} to {high!r}, more than 1e-6 of it apart; is this "
            "table from his step 5?"
        )
    return float(np.median(norms))


def read_table(path: str | Path) -> pd.DataFrame:
    """A parquet table of Mayner's, such as split/top_neurons.parquet, as a DataFrame.

    Example:
        >>> read_table("no/such/top_neurons.parquet")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        FileNotFoundError: there is no table at no/such/top_neurons.parquet; ...
    """
    try:
        return pd.read_parquet(path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"there is no table at {path}; {TOP_NEURONS_FILE} comes with Mayner's "
            "data bundle (huggingface.co/datasets/wmayner/persona-neurons), and "
            f"{SETS_FILE} is written by his `uv run steps/6_role_steering.py "
            "prepare` (github.com/wmayner/persona-neurons)"
        ) from None
