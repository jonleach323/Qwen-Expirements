"""Mayner's reference code, which the Mayner tests compare pushlab against.

An unchanged copy of the few functions of github.com/wmayner/persona-neurons
that build a neuron set's steering rows and push them, under his MIT licence
(below). test_mayner_sets.py and test_mayner_steering.py load this file by
its path; it is not a test module, so pytest does not collect it.
"""

# The copy keeps his line lengths and names (l for a layer), and ruff format
# leaves it as it is.
# ruff: noqa: E501, E741

import numpy as np
import pandas as pd
import torch

# ===========================================================================
# Mayner's code, copied unchanged from github.com/wmayner/persona-neurons at
# commit 64c9d36 ("Initial release"), each piece headed by its file and
# lines there. split.py's top_neurons is renamed split_top_neurons, as
# role_steering.py has one too; component_add_values is only there because
# condition_values calls it for direction sets.
#
# MIT License
#
# Copyright (c) 2026 Will Mayner
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ===========================================================================
# fmt: off
# persona_neurons/role_steering.py:33-35, 38 and 40
K = 100
STRENGTH = 4.0
DRAW = 0   # the one random set and the one rotated basis
REPLY_SEED, RANDOM_SEED, ROTATION_SEED = 20261103, 20261104, 20261106
DIRECTION_SETS = ("components", "rotation_matched")


# persona_neurons/role_steering.py:51-83
def top_neurons(delta: np.ndarray, p: np.ndarray, k: int = K) -> pd.DataFrame:
    """The k neurons (delta, p: layers × neurons) with the largest |Δa · p|, largest first (ties in index order).
    Columns: layer, neuron, delta, contribution."""
    c = (np.asarray(delta, np.float64) * np.asarray(p, np.float64)).ravel()
    order = np.argsort(-np.abs(c), kind="stable")[:k]
    layer, neuron = np.divmod(order, delta.shape[1])
    return pd.DataFrame({"layer": layer, "neuron": neuron, "delta": np.asarray(delta).ravel()[order],
                         "contribution": c[order]})


def layer_matched_random(chosen: pd.DataFrame, n_neurons: int, seed) -> pd.DataFrame:
    """A random neuron set with as many neurons in each layer as `chosen` (columns layer, neuron), drawn uniformly
    without replacement within each layer. Columns: layer, neuron."""
    rng = np.random.default_rng(seed)
    rows = []
    for layer, k in chosen.layer.value_counts().sort_index().items():
        rows += [(int(layer), int(n)) for n in np.sort(rng.choice(n_neurons, size=int(k), replace=False))]
    return pd.DataFrame(rows, columns=["layer", "neuron"])


def neuron_sets(role: str, role_index: int, delta: np.ndarray, p: np.ndarray, k: int = K,
                draw: int = DRAW) -> pd.DataFrame:
    """A role's top k neurons and a random set with the same count per layer, drawn with
    np.random.default_rng([RANDOM_SEED, role_index, k, draw]). Columns: role, kind ("neurons" or "random"), draw, layer,
    neuron, delta, contribution."""
    top = top_neurons(delta, p, k)
    out = pd.concat([top[["layer", "neuron"]].assign(kind="neurons", draw=0),
                     layer_matched_random(top, delta.shape[1], [RANDOM_SEED, role_index, k, draw]).assign(
                         kind="random", draw=draw)], ignore_index=True)
    d64, p64 = np.asarray(delta, np.float64), np.asarray(p, np.float64)
    out["delta"] = d64[out.layer, out.neuron]
    out["contribution"] = out.delta * p64[out.layer, out.neuron]
    return out.assign(role=role)[["role", "kind", "draw", "layer", "neuron", "delta", "contribution"]]


# persona_neurons/role_steering.py:114-174
def component_add_values(by_layer: dict) -> pd.DataFrame:
    """What a set of components adds to the neurons before scaling. by_layer maps l to (V_l, coef_l): the chosen
    components as columns (n_neurons × j) and coef_l = V_lᵀ Δa_l; the addition is V_l coef_l. Columns: layer, neuron, value."""
    parts = []
    for layer, (v, coef) in sorted(by_layer.items()):
        x = np.asarray(v, np.float64) @ np.asarray(coef, np.float64)
        parts.append(pd.DataFrame({"layer": layer, "neuron": np.arange(len(x)), "value": x}))
    return pd.concat(parts, ignore_index=True)


def write_vector(values: pd.DataFrame, w_down) -> torch.Tensor:
    """What adding `values` (layer, neuron, value) to the neurons outputs to the residual stream, Σ W_down_l[:, n] ×
    value, in float64 on the CPU. w_down(layer) is that layer's down_proj weight (d_model × n_neurons)."""
    total = None
    for layer, g in values.groupby("layer", sort=True):
        w = w_down(int(layer)).detach()
        idx = torch.tensor(g.neuron.to_numpy(), dtype=torch.long, device=w.device)
        x = (w[:, idx].float() @ torch.tensor(g.value.to_numpy(), dtype=torch.float32, device=w.device)).double().cpu()
        total = x if total is None else total + x
    return total


def matched_scale(values: pd.DataFrame, w_down, target_norm: float) -> float:
    """The c for which c × values outputs a vector of norm target_norm."""
    return float(target_norm / write_vector(values, w_down).norm().item())


class RowIntervention:
    """Per-row additions to neurons while active: rows[i] is None (row i untouched) or a DataFrame with columns layer,
    neuron, value, added to the input of that layer's mlp.down_proj at every position of row i."""

    def __init__(self, model, rows: list):
        self.model, self.rows, self.handles = model, rows, []

    def _tensors(self):
        n = self.model.model.layers[0].mlp.down_proj.in_features
        dev = next(self.model.parameters()).device
        out = {}
        for i, r in enumerate(self.rows):
            if r is None or len(r) == 0:
                continue
            for layer, g in r.groupby("layer"):
                add = out.setdefault(int(layer), torch.zeros(len(self.rows), n))
                add[i, torch.tensor(g.neuron.to_numpy(), dtype=torch.long)] += torch.tensor(g.value.to_numpy(), dtype=torch.float32)
        return {l: x.to(dev) for l, x in out.items()}

    def __enter__(self):
        for layer, add in self._tensors().items():

            def hook(module, args, add=add):
                x = args[0]
                if x.shape[0] != add.shape[0]:
                    raise ValueError(f"batch of {x.shape[0]} rows under a RowIntervention of {add.shape[0]}")
                return (x + add[:, None, :].to(x.dtype),) + tuple(args[1:])
            self.handles.append(self.model.model.layers[layer].mlp.down_proj.register_forward_pre_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        self.handles = []


# persona_neurons/role_steering.py:184-209
def condition_values(cond: dict, sets: pd.DataFrame, comps: dict, w_down, target_norm: float):
    """What a condition does:
    - ("vector", None) for the vector (the caller adds it at layer 18), and ("none", None) for no steering;
    - ("rows", values): the neuron values a set adds, scaled so that its output has norm target_norm.
    sets: neuron_sets rows of this role; comps[(kind, draw)] for the direction sets (components and rotation_matched):
    {"rank": DataFrame with layer and coef (each direction's projection of Δa), "v": (n_neurons, n_directions) the
    directions as columns, in the same order}. A condition with a set size k takes the sets' rows of that size."""
    kind = cond["kind"]
    if kind in ("vector", "none"):
        return kind, None
    if kind in DIRECTION_SETS:
        d = comps[(kind, int(cond["draw"]))]
        k = int(cond.get("k", K))
        top = d["rank"].head(k)
        if kind == "rotation_matched" and len(d["rank"]) > k:   # per layer, as many as the top k neurons have there
            n = sets[(sets.kind == "neurons") & (sets.k == k)].layer.value_counts()
            r = d["rank"]
            top = r[r.groupby("layer").cumcount() < r.layer.map(n).fillna(0)]
        vals = component_add_values({int(l): (d["v"][:, g.index.to_numpy()], g.coef.to_numpy())
                                     for l, g in top.groupby("layer", sort=True)})
    else:
        s = sets[(sets.kind == kind) & (sets.draw == cond["draw"])]
        s = s[s.k == cond["k"]] if "k" in cond else s
        vals = s[["layer", "neuron"]].assign(value=s.delta.to_numpy().astype(np.float64))
    c = matched_scale(vals, w_down, target_norm)
    return "rows", vals.assign(value=c * vals.value.to_numpy()).reset_index(drop=True)


# persona_neurons/split.py:144-152, top_neurons renamed split_top_neurons
def split_top_neurons(c: np.ndarray, norm: float, k: int) -> pd.DataFrame:
    """The k neurons with the largest |contribution| in c (layers × neurons): rank, layer, neuron, contribution,
    share (% of norm)."""
    flat = c.ravel()
    k = min(k, flat.size)
    idx = np.argsort(-np.abs(flat), kind="stable")[:k]
    layer, neuron = np.unravel_index(idx, c.shape)
    return pd.DataFrame({"rank": np.arange(1, k + 1), "layer": layer, "neuron": neuron, "contribution": flat[idx],
                         "share": 100 * flat[idx] / norm})
