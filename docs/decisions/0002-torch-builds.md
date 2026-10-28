# 0002: CPU torch by default; CUDA torch and the fast kernels in the `gpu` extra

Date: 2026-10-28 · Sprint: S0 · Status: accepted

## Context

- Cloud sessions have no GPU, so they need only PyTorch's CPU-only build
  (about 190 MB to download). The GPU pod needs a CUDA build (about 3 GB
  including NVIDIA's CUDA libraries), plus `flash-linear-attention` and
  `causal-conv1d` for Qwen3.5's linear-attention layers.
- Both machines run Linux x86_64. uv's platform-marker recipe therefore can't
  tell them apart. PyTorch publishes each build on its own package index.
- The S0 brief requires torch to be a core dependency, and plain `uv sync` and
  `uv run pytest -q` to work in cloud sessions.

## Decision

| Machine | Install command | torch build installed |
| --- | --- | --- |
| Cloud session | `uv sync` | `2.14.1+cpu` |
| GPU pod | `uv sync --no-group cpu --extra gpu` | `2.14.1+cu130`, plus the two kernels |

In `pyproject.toml`:

- torch is a core dependency, `>=2.14,<2.15`.
- A dependency group `cpu`, on by default, takes torch from PyTorch's CPU index.
- The `gpu` extra takes torch from PyTorch's CUDA 13.0 index. It adds
  `flash-linear-attention` and a prebuilt `causal-conv1d` wheel from Astral's
  CUDA 13.0 index.
- `cpu` and `gpu` are declared as conflicting. uv locks both and never mixes
  them.

## Why this shape

uv's PyTorch guide switches builds with two conflicting extras and
extra-scoped sources ("Configuring accelerators with optional dependencies").
In the guide, torch appears only in those extras. Here torch is also a core
dependency, as the brief requires. That forces one change: the CPU side has to
be a default dependency group instead of an extra. Tested with uv 0.11.32 and
0.13.6 in October 2026:

- uv has no default extras. With pure extras and torch as a core dependency,
  plain `uv sync` installs PyPI's torch, which on Linux is the full CUDA 13
  build.
- `uv run` re-syncs to the exact build the lockfile assigns to the active
  selection. So after `uv sync --extra cpu`, a plain `uv run pytest` would
  swap the CPU build for PyPI's CUDA build. A default group makes the CPU
  build exactly what a plain `uv sync` or `uv run` expects.
- An unconditional CPU index next to the `gpu` extra's CUDA index is rejected
  by `uv lock` ("conflicting indexes for package `torch`").
- Asking for `--extra gpu` without `--no-group cpu` stops with a conflict
  error, not a mixed install.

**Why CUDA 13.0.** torch 2.14.1 comes in cu126, cu130 and cu132 builds (no
cu128). Astral publishes prebuilt `causal-conv1d` wheels for torch 2.14 on all
three. cu130 was picked for two reasons:

- It is the CUDA version of PyPI's default Linux torch, so it is the most
  widely used build.
- Unlike cu126, it also covers newer GPUs (Blackwell).

The costs are a newer driver (R580 or later, while cu126 runs on older CUDA
12 drivers) and no support for GPUs older than Turing, such as the V100 and
P100. The plan's RTX 4090 and A100 are fine on both counts. cu126 is the
fallback if the pod's driver or GPU rules cu130 out (see Consequences).

**Why a prebuilt `causal-conv1d`.** PyPI has only its source code. Its
metadata is dynamic, so uv would have to build it just to write the lockfile.
Building needs a CUDA compiler whose version matches torch's CUDA version, and
cloud sessions have no such compiler. uv's PyTorch guide points to Astral's GPU
indexes for exactly these extensions. Each Astral build requires the torch
minor version it was built for. The locked wheel,
`1.7.0+cu.13.0.torch.2.14`, requires `torch==2.14.*`, so a plain
`causal-conv1d>=1.7` picks the build for the locked torch. The CUDA version
comes from the index instead, so Astral's index and PyTorch's must name the
same CUDA version (both are cu130 here).

**Why cap torch below 2.15.** torch 2.15.0 is on the CPU index but not yet on
the CUDA 13.0 index. Without the cap, a fresh lock would put cloud sessions on
2.15 while the pod ran 2.14.1. The cap is what keeps the two machines on the
same torch.

## Consequences

- The pod needs an NVIDIA driver that supports CUDA 13.0 (R580 or newer).
  Pick the pod template with that in mind. The pod does not need a system CUDA
  toolkit, because the CUDA libraries arrive as pip packages. If only older
  drivers or GPUs are available, point both cu130 indexes (PyTorch's and
  Astral's) at cu126 and re-lock.
- On the pod, plain `uv sync` or `uv run` would switch torch back to the CPU
  build, because the `cpu` group is on by default. Pod setup must always sync
  with `uv sync --no-group cpu --extra gpu`, then set `UV_NO_SYNC=1` so that
  `uv run` uses the environment as installed. The pod setup script, a later
  S0 task, should do both, and CLAUDE.md should then gain the pod install
  command.
- In a cloud session, any selection that installs the project with the `cpu`
  group off and without `gpu` silently downloads PyPI's CUDA torch stack
  (about 3 GB).
  Examples: `--no-default-groups`, `UV_NO_DEFAULT_GROUPS=1`, or
  `--no-group cpu` on its own. To skip the dev tools, use `--no-dev`.
- Upgrading torch means raising the cap, checking that the chosen CUDA index
  and Astral's index both have the new version, re-locking, rerunning the
  golden tests on the pod, and writing a decision note.
- `uv.lock` holds four torch entries: CPU, CPU for macOS, CUDA 13.0, and PyPI's
  default for when neither `cpu` nor `gpu` is selected. Only one is ever
  installed.
- The `gpu` extra resolves in the lockfile but has not yet been installed on a
  GPU. Its first real test is the S0 pod task.
