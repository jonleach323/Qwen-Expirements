"""Neuron sets from Mayner's tables push exactly what his code pushes.

A fixture in his data format, seeded deltas and p of two roles on the tiny
model's 2 x 128 neurons, is written as his steps write sets.parquet and
top_neurons.parquet, and read back. Our sets from those tables are checked
against what his code (an unchanged copy, tests/unit/mayner_reference.py)
builds from the same rows.
"""

import importlib.util
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from transformers import Qwen3ForCausalLM

from pushlab.units import mayner
from pushlab.units.dose import amount
from pushlab.units.neuron_set import NeuronSet, unit_neurons

# Mayner's code, copied unchanged with his MIT licence notice, is in
# tests/unit/mayner_reference.py; it is loaded by its path, as test modules
# import nothing from one another.
_SPEC = importlib.util.spec_from_file_location(
    "mayner_reference", Path(__file__).with_name("mayner_reference.py")
)
reference = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(reference)
# === The fixture in his data format =========================================

# Two roles, sorted as his role_indices sorts them, so each role's index
# (which seeds its random sets) is its position. Their |v_r| are made up.
ROLES = ("pirate", "sommelier")
ROLE = "pirate"
ROLE_INDEX = {role: i for i, role in enumerate(ROLES)}
VECTOR_NORM = {"pirate": 3.0, "sommelier": 5.0}
# delta_a and p as split/arrays.npz holds them, (roles, layers, neurons) in
# float32, here on the tiny model's 2 layers of 128 neurons.
_DRAWS = np.random.default_rng(20261112)
DELTA = _DRAWS.standard_normal((len(ROLES), 2, 128)).astype(np.float32)
P = _DRAWS.standard_normal((len(ROLES), 2, 128)).astype(np.float32)
NORMS = [7.0, 10.0]  # residual sizes for amount(); his sets have their own size
SIZES = (10, 50)  # the set sizes of the fixture's step 6b table


def _written(table: pd.DataFrame, path: Path) -> pd.DataFrame:
    """table written to path as his steps write it, then read back."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return mayner.read_table(path)


@pytest.fixture
def sets_table(tmp_path: Path) -> pd.DataFrame:
    """role_steering/sets.parquet of both roles, as his step 6 prepare writes it."""
    sets = [
        reference.neuron_sets(r, ROLE_INDEX[r], DELTA[i], P[i])
        for i, r in enumerate(ROLES)
    ]
    path = tmp_path / "role_steering" / "sets.parquet"
    return _written(pd.concat(sets, ignore_index=True), path)


@pytest.fixture
def sized_sets_table(tmp_path: Path) -> pd.DataFrame:
    """role_steering_more/sets.parquet of both roles, as his step 6b prepare
    writes it (more_neuron_sets), for SIZES and random draw 1 of K neurons."""
    parts = []
    for i, role in enumerate(ROLES):
        index = ROLE_INDEX[role]
        for k in SIZES:
            parts.append(
                reference.neuron_sets(role, index, DELTA[i], P[i], k).assign(k=k)
            )
        drawn = reference.neuron_sets(role, index, DELTA[i], P[i], reference.K, 1)
        parts.append(drawn[drawn.kind == "random"].assign(k=reference.K))
    path = tmp_path / "role_steering_more" / "sets.parquet"
    return _written(pd.concat(parts, ignore_index=True), path)


@pytest.fixture
def top_table(tmp_path: Path) -> pd.DataFrame:
    """split/top_neurons.parquet of both roles, as his step 5 split writes it."""
    c = DELTA * P  # split.contributions
    norm = np.array([VECTOR_NORM[role] for role in ROLES])  # float64, as there
    tops = []
    for i, role in enumerate(ROLES):
        top = reference.split_top_neurons(
            c[i], norm[i], min(1000, c[i].size)
        )  # its K_MAX
        tops.append(top.assign(role=role, delta=DELTA[i, top.layer, top.neuron]))
    path = tmp_path / "split" / "top_neurons.parquet"
    return _written(pd.concat(tops, ignore_index=True), path)


def w_down(model: Qwen3ForCausalLM) -> Callable[[int], torch.Tensor]:
    """His w_down for model (steering_groups): a layer's down_proj weight."""
    return lambda layer: model.model.layers[layer].mlp.down_proj.weight


def their_payload(
    sets: pd.DataFrame, role: str, model: Qwen3ForCausalLM, **condition: object
) -> pd.DataFrame:
    """The rows (layer, neuron, value) condition_values hands RowIntervention for
    a set of role: kind "neurons", draw DRAW and no k, unless condition says."""
    cond = {"role": role, "kind": "neurons", "draw": reference.DRAW} | condition
    mine = sets[sets.role == role]  # his step 6 reads only the role's rows
    target_norm = reference.STRENGTH * VECTOR_NORM[role]
    kind, payload = reference.condition_values(
        cond, mine, {}, w_down(model), target_norm
    )
    assert kind == "rows"
    return payload


def our_set(table: pd.DataFrame, role: str, **choice: object) -> NeuronSet:
    """steering_set of role, at |v_r| VECTOR_NORM[role], in the tiny model."""
    norm = VECTOR_NORM[role]
    return mayner.steering_set(
        table, role, vector_norm=norm, model_name="tiny", **choice
    )


def assert_values_match(
    neuron_set: NeuronSet, payload: pd.DataFrame, model: Qwen3ForCausalLM
) -> None:
    """At dose 1, each neuron of neuron_set gets payload's value, in its order."""
    per_coefficient = amount(neuron_set, 1.0, model, NORMS)
    neurons = unit_neurons(neuron_set)
    theirs = zip(payload.layer.to_list(), payload.neuron.to_list(), strict=True)
    assert [(neuron.layer, neuron.index) for neuron, _ in neurons] == list(theirs)
    ours = [per_coefficient * coefficient for _, coefficient in neurons]
    np.testing.assert_allclose(ours, payload.value.to_numpy(), rtol=1e-6)


# === Our sets against his code ==============================================


@pytest.mark.parametrize("kind", ["neurons", "random"])
def test_a_set_adds_his_values_and_pushes_the_vector_his_code_builds(
    sets_table: pd.DataFrame, tiny_model: Qwen3ForCausalLM, kind: str
) -> None:
    assert mayner.STRENGTH == reference.STRENGTH  # dose 1 is his strength
    neuron_set = our_set(sets_table, ROLE, kind=kind)
    payload = their_payload(sets_table, ROLE, tiny_model, kind=kind)
    assert_values_match(neuron_set, payload, tiny_model)
    dose_1 = amount(neuron_set, 1.0, tiny_model, NORMS)
    ours = (dose_1 * neuron_set.write_direction(tiny_model)).double()
    theirs = reference.write_vector(payload, w_down(tiny_model))
    target_norm = reference.STRENGTH * VECTOR_NORM[ROLE]
    # Ours is float32 and his a float64 sum of float32 products; an entry near
    # 0 is compared to 1e-6 of the vector's length.
    torch.testing.assert_close(ours, theirs, rtol=1e-6, atol=1e-6 * target_norm)
    assert math.isclose(float(ours.norm()), target_norm, rel_tol=1e-6)


@pytest.mark.parametrize("role", ROLES)
def test_both_tables_give_the_same_set_and_the_split_gives_the_norm(
    sets_table: pd.DataFrame, top_table: pd.DataFrame, role: str
) -> None:
    norm = VECTOR_NORM[role]
    from_top = mayner.top_neurons_set(
        top_table, role, reference.K, norm, model_name="tiny"
    )
    assert from_top.entries == our_set(sets_table, role).entries
    assert len(from_top.entries) == reference.K
    assert from_top.size == reference.STRENGTH * norm
    i = ROLE_INDEX[role]  # the role's own deltas, not the other role's
    assert all(e.coefficient == DELTA[i, e.layer, e.index] for e in from_top.entries)
    # The contributions are float32, so each row gives |v_r| to about 6e-8.
    found = mayner.vector_norm_from_shares(top_table, role)
    assert found == pytest.approx(norm, rel=1e-7)


@pytest.mark.parametrize(
    ("kind", "draw", "k"),
    [
        ("neurons", 0, 10),
        ("neurons", 0, 50),
        ("random", 0, 10),
        ("random", 1, reference.K),
    ],
)
def test_a_sized_tables_sets_get_his_values(
    sized_sets_table: pd.DataFrame,
    top_table: pd.DataFrame,
    tiny_model: Qwen3ForCausalLM,
    kind: str,
    draw: int,
    k: int,
) -> None:
    neuron_set = our_set(sized_sets_table, ROLE, kind=kind, draw=draw, k=k)
    assert len(neuron_set.entries) == k
    choice = {"kind": kind, "draw": draw, "k": k}
    payload = their_payload(sized_sets_table, ROLE, tiny_model, **choice)
    assert_values_match(neuron_set, payload, tiny_model)
    if kind == "neurons":
        norm = VECTOR_NORM[ROLE]
        top = mayner.top_neurons_set(top_table, ROLE, k, norm, model_name="tiny")
        assert neuron_set.entries == top.entries


# === Our pushes against his RowIntervention, on the tiny model ==============
#
# In test_mayner_steering.py, which loads the same copy of his code: there
# our engine and his RowIntervention push the same sets and give the same
# scores and greedy replies.
# ============================================================================
