"""pushlab.units.mayner on small made-up tables: the names and sources it
gives, everything it refuses, read_table, and its docstring examples.

tests/unit/test_mayner_sets.py checks its sets against Mayner's own code.
"""

import doctest
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pushlab.units import mayner

NORM = 10.0  # the made-up pirate's |v_r|: 100 * contribution / share


def top_table() -> pd.DataFrame:
    """A made-up split/top_neurons.parquet: the pirate's top 3, the sommelier's 2."""
    return pd.DataFrame(
        {
            "rank": [1, 2, 3, 1, 2],
            "layer": [1, 0, 1, 0, 1],
            "neuron": [5, 9, 77, 5, 3],
            "contribution": [0.5, -0.25, 0.125, 0.2, -0.1],
            "share": [5.0, -2.5, 1.25, 5.0, -2.5],
            "role": ["pirate"] * 3 + ["sommelier"] * 2,
            "delta": [1.5, -0.5, 0.75, 2.0, 0.25],
        }
    )


def sets_table() -> pd.DataFrame:
    """A made-up role_steering/sets.parquet: the pirate's top 2 and a random
    pair, and the sommelier's top 1."""
    return pd.DataFrame(
        {
            "role": ["pirate"] * 4 + ["sommelier"],
            "kind": ["neurons", "neurons", "random", "random", "neurons"],
            "draw": 0,
            "layer": [1, 0, 0, 1, 0],
            "neuron": [5, 9, 30, 100, 5],
            "delta": [1.5, -0.5, 0.25, -0.75, 2.0],
            "contribution": [0.5, -0.25, 0.01, 0.02, 0.2],
        }
    )


def sized_sets_table() -> pd.DataFrame:
    """sets_table as step 6b's, with each set's size k, and the pirate's top 1."""
    table = sets_table().assign(k=[2, 2, 2, 2, 1])
    return pd.concat([table, table.iloc[[0]].assign(k=1)], ignore_index=True)


def test_the_default_names_and_sources_say_where_a_set_comes_from() -> None:
    top = mayner.top_neurons_set(top_table(), "pirate", 3, NORM, model_name="tiny")
    assert (str(top), top.size) == ("tiny:set:pirate-top3", 40.0)
    assert top.source == (
        "wmayner/persona-neurons split/top_neurons.parquet: role pirate, ranks 1 to 3"
    )
    random = mayner.steering_set(
        sets_table(), "pirate", kind="random", vector_norm=NORM, model_name="tiny"
    )
    assert str(random) == "tiny:set:pirate-random-d0"
    assert random.entries == ((0, 30, 0.25), (1, 100, -0.75))
    assert random.source == (
        "wmayner/persona-neurons role_steering/sets.parquet: role pirate, kind "
        "random, draw 0"
    )
    top_1 = mayner.steering_set(sized_sets_table(), "pirate", k=1, vector_norm=NORM)
    assert str(top_1) == "qwen3-8b:set:pirate-neurons-k1-d0"
    assert top_1.entries == ((1, 5, 1.5),)
    assert top_1.source == (
        "wmayner/persona-neurons role_steering_more/sets.parquet: role pirate, "
        "kind neurons, draw 0, k 1"
    )


def test_a_given_name_and_source_are_kept_and_numpy_numbers_taken() -> None:
    pirate = mayner.top_neurons_set(
        top_table(), "pirate", np.int64(2), np.float32(2.5), name="arr", source="mine"
    )
    assert (pirate.name, pirate.source, pirate.size) == ("arr", "mine", 10.0)
    assert len(pirate.entries) == 2


def test_a_roles_name_is_made_a_set_name() -> None:
    table = top_table().replace({"role": {"pirate": "Sea_Dog Captain"}})
    sea_dog = mayner.top_neurons_set(table, "Sea_Dog Captain", 1, NORM)
    assert sea_dog.name == "sea-dog-captain-top1"


@pytest.mark.parametrize("role", ["café", "sea__dog", "pirate!"])
def test_a_role_that_cannot_give_a_set_name_needs_one(role: str) -> None:
    top = top_table().replace({"role": {"pirate": role}})
    with pytest.raises(ValueError, match="give the set a name, such as"):
        mayner.top_neurons_set(top, role, 1, NORM)
    sets = sets_table().replace({"role": {"pirate": role}})
    with pytest.raises(ValueError, match="give the set a name, such as"):
        mayner.steering_set(sets, role, vector_norm=NORM)
    named = mayner.top_neurons_set(top, role, 1, NORM, name="pirate-top1")
    assert named.name == "pirate-top1"


# Each reader of a role's rows: a call on a table and a role, its kind of
# table, and that format's columns.
READERS = {
    "top_neurons_set": (
        lambda table, role: mayner.top_neurons_set(table, role, 1, NORM),
        top_table,
        mayner.TOP_NEURONS_COLUMNS,
    ),
    "steering_set": (
        lambda table, role: mayner.steering_set(table, role, vector_norm=NORM),
        sets_table,
        mayner.SETS_COLUMNS,
    ),
    "vector_norm_from_shares": (
        mayner.vector_norm_from_shares,
        top_table,
        mayner.TOP_NEURONS_COLUMNS,
    ),
}


@pytest.mark.parametrize("reader", READERS)
@pytest.mark.parametrize("table", ["split/top_neurons.parquet", None, [1, 2]])
def test_a_table_must_be_a_data_frame(reader: str, table: object) -> None:
    call, _, _ = READERS[reader]
    with pytest.raises(ValueError, match="must be a pandas DataFrame, such as"):
        call(table, "pirate")


@pytest.mark.parametrize("reader", READERS)
def test_every_column_of_the_format_is_needed(reader: str) -> None:
    call, table, columns = READERS[reader]
    assert tuple(table().columns) == columns
    for column in columns:
        with pytest.raises(ValueError, match=f"lacks the column {column}; a "):
            call(table().drop(columns=column), "pirate")


@pytest.mark.parametrize("reader", READERS)
def test_an_unknown_role_is_refused(reader: str) -> None:
    call, table, _ = READERS[reader]
    with pytest.raises(ValueError, match="no rows for role 'alien'; its roles are "):
        call(table(), "alien")
    with pytest.raises(ValueError, match="no rows for role 'pirate'; it is empty"):
        call(table().iloc[:0], "pirate")
    many = pd.concat([table().assign(role=f"r{i:02}") for i in range(12)])
    with pytest.raises(ValueError, match="roles are r00, r01, .*, r09 and 2 more$"):
        call(many, "pirate")
    for role in ("", 3, None):
        with pytest.raises(ValueError, match="role must be a role's name"):
            call(table(), role)


@pytest.mark.parametrize("k", [0, -1, 4, True, 2.0, None, "2"])
def test_k_must_be_from_1_to_the_roles_neurons(k: object) -> None:
    with pytest.raises(ValueError, match="k must be a whole number from 1 to 3, the"):
        mayner.top_neurons_set(top_table(), "pirate", k, NORM)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("ranks", "wrong"),
    [([1, 1, 3], "missing 2; repeated 1"), ([2, 3, 4], "missing 1")],
)
def test_ranks_1_to_k_must_each_appear_once(ranks: list[int], wrong: str) -> None:
    table = top_table()
    table.loc[table.role == "pirate", "rank"] = ranks
    with pytest.raises(ValueError, match=re.escape(f"appear once ({wrong})")):
        mayner.top_neurons_set(table, "pirate", 3, NORM)


@pytest.mark.parametrize("delta", [0.0, -0.0, float("nan"), float("inf")])
def test_a_delta_must_be_finite_and_not_0(delta: float) -> None:
    top, sets = top_table(), sets_table()
    top.loc[1, "delta"] = sets.loc[1, "delta"] = delta  # the pirate's second
    with pytest.raises(ValueError, match="pirate rank 2: delta must be a finite"):
        mayner.top_neurons_set(top, "pirate", 2, NORM)
    with pytest.raises(ValueError, match="sets.parquet, row 1: delta must be a"):
        mayner.steering_set(sets, "pirate", vector_norm=NORM)
    # Only the rows taken are checked.
    assert len(mayner.top_neurons_set(top, "pirate", 1, NORM).entries) == 1


def test_layers_and_neurons_must_be_whole_numbers_from_0() -> None:
    table = top_table()
    table.loc[0, "layer"] = -1
    with pytest.raises(ValueError, match="rank 1: layer must be a whole number"):
        mayner.top_neurons_set(table, "pirate", 1, NORM)
    floats = top_table().assign(neuron=[5.0, 9.0, 77.0, 5.0, 3.0])
    with pytest.raises(ValueError, match="rank 1: neuron must be a whole number"):
        mayner.top_neurons_set(floats, "pirate", 1, NORM)


def test_a_neuron_must_be_in_the_model_and_in_the_set_once() -> None:
    table = top_table()
    table.loc[0, "neuron"] = 128  # the tiny model's neurons are 0 to 127
    with pytest.raises(ValueError, match="tiny:L1:N128: neuron 128 is out of range"):
        mayner.top_neurons_set(table, "pirate", 1, NORM, model_name="tiny")
    assert mayner.top_neurons_set(table, "pirate", 1, NORM).entries[0].index == 128
    table.loc[1, ["layer", "neuron"]] = [1, 128]  # rank 2 is rank 1's neuron
    with pytest.raises(ValueError, match="rank 2 repeats layer 1 neuron 128 of"):
        mayner.top_neurons_set(table, "pirate", 2, NORM)


@pytest.mark.parametrize(
    "norm", [0, -1.0, float("nan"), float("inf"), True, "3", 10**400]
)
def test_the_vector_norm_must_be_finite_and_above_0(norm: object) -> None:
    refusal = re.escape("vector_norm must be |v_r|, the length of the role's")
    with pytest.raises(ValueError, match=refusal):
        mayner.top_neurons_set(top_table(), "pirate", 1, norm)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=refusal):
        mayner.steering_set(sets_table(), "pirate", vector_norm=norm)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("table", "choice", "message"),
    [
        (
            sets_table,
            {"kind": "vector"},
            "has no set of role pirate with kind vector, draw 0; its sets of "
            "pirate are: kind neurons, draw 0; kind random, draw 0",
        ),
        (sets_table, {"draw": 1}, "no set of role pirate with kind neurons, draw 1;"),
        (sets_table, {"kind": ""}, "kind must be a set's kind"),
        (sets_table, {"draw": -1}, "draw must be a whole number from 0"),
        (sets_table, {"draw": True}, "draw must be a whole number from 0"),
        (sets_table, {"k": 2}, "has no column k, as step 6's sets.parquet"),
        (sized_sets_table, {}, "give k, the size of the set, one of 1, 2"),
        (
            sized_sets_table,
            {"k": 3},
            "with kind neurons, draw 0, k 3; its sets of pirate are: kind neurons, "
            "draw 0, k 1; kind neurons, draw 0, k 2; kind random, draw 0, k 2",
        ),
        (sized_sets_table, {"k": 0}, "k must be a whole number from 1"),
    ],
)
def test_steering_set_takes_only_a_set_the_table_has(
    table: Callable[[], pd.DataFrame], choice: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=re.escape(message)):
        mayner.steering_set(table(), "pirate", vector_norm=NORM, **choice)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("share", 0.0, "pirate rank 2: the share is 0, so"),
        ("share", np.nan, "contribution -0.25 and share nan give |v_r| = nan, not"),
        ("contribution", np.inf, "contribution inf and share -2.5 give |v_r| = -inf"),
        ("share", 2.5, "contribution -0.25 and share 2.5 give |v_r| = -10.0, not"),
        ("share", -2.5 * (1 + 2e-6), "from 9.99998"),
    ],
)
def test_shares_that_do_not_give_one_length_are_refused(
    column: str, value: float, message: str
) -> None:
    table = top_table()
    table.loc[1, column] = value  # the pirate's rank 2
    with pytest.raises(ValueError, match=re.escape(message)):
        mayner.vector_norm_from_shares(table, "pirate")


def test_the_norm_is_the_rows_median_once_they_agree() -> None:
    table = top_table()
    assert mayner.vector_norm_from_shares(table, "sommelier") == 4.0
    table.loc[1, "share"] = -2.5 * (1 + 5e-7)  # within 1e-6 of the others
    assert mayner.vector_norm_from_shares(table, "pirate") == NORM


def test_read_table_reads_a_table_and_says_where_a_missing_one_comes_from(
    tmp_path: Path,
) -> None:
    path = tmp_path / "top_neurons.parquet"
    top_table().to_parquet(path, index=False)
    pd.testing.assert_frame_equal(mayner.read_table(str(path)), top_table())
    missing = tmp_path / "split" / "top_neurons.parquet"
    with pytest.raises(FileNotFoundError, match="there is no table at .*wmayner"):
        mayner.read_table(missing)


def test_docstring_examples_run() -> None:
    result = doctest.testmod(mayner)
    assert result.attempted > 0
    assert result.failed == 0
