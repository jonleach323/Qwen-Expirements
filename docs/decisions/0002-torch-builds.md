# 0002: CPU torch by default; CUDA torch and the fast kernels in the `gpu` extra

Date: 2026-10-28 · Sprint: S0 · Status: accepted

## Context

- Cloud sessions have no GPU, so they need only PyTorch's CPU-only build
  (about 190 MB). The GPU pod needs a CUDA build (several GB including its
  CUDA libraries), plus `flash-linear-attention` and `causal-conv1d` for
  Qwen3.5's linear-attention layers.
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
This setup follows the guide with one change: the CPU side is a default
dependency group, not an extra. Tested with uv 0.11.32 in October 2026:

- uv has no default extras. With pure extras and torch as a core dependency,
  plain `uv sync` installs PyPI's torch, which on Linux is the full CUDA 13
  build.
- `uv run` re-syncs to the exact build the lockfile assigns to the active
  selection. So after `uv sync --extra cpu`, a plain `uv run pytest` would
  swap the CPU build for PyPI's CUDA build. A default group makes the CPU
  build exactly what a plain `uv sync` or `uv run` expects.
- Asking for `--extra gpu` without `--no-group cpu` stops with a conflict
  error, not a mixed install.

**Why CUDA 13.0.** torch 2.14.1 has cu126, cu130 and cu132 builds, and no
cu128. Astral's prebuilt `causal-conv1d` exists for torch 2.14 with cu130 and
cu132, but its cu126 builds stop at torch 2.9. cu130 is also the CUDA version
of PyPI's default Linux torch.

**Why a prebuilt `causal-conv1d`.** PyPI has only its source code. Its
metadata is dynamic, so uv would have to build it just to write the lockfile.
Building needs a CUDA compiler whose version matches torch's CUDA version, and
cloud sessions have no such compiler. uv's PyTorch guide points to Astral's GPU
indexes for exactly these extensions. The pin names the exact build
(`1.7.0+cu.13.0.torch.2.14`) because a compiled extension only works with the
torch it was built for. Without the pin, a later re-lock could pick a build
for a newer torch.

**Why cap torch below 2.15.** torch 2.15.0 is on the CPU index but not yet on
the CUDA 13.0 index. Without the cap, cloud sessions would test against 2.15
while the pod ran 2.14.1.

## Consequences

- The pod needs an NVIDIA driver that supports CUDA 13.0 (R580 or newer).
  Pick the pod template with that in mind. The pod does not need a CUDA
  toolkit, because the wheels bundle their CUDA runtime.
- On the pod, plain `uv sync` or `uv run` would switch torch back to the CPU
  build, because the `cpu` group is on by default. Pod setup must always sync
  with `uv sync --no-group cpu --extra gpu`, then set `UV_NO_SYNC=1` so that
  `uv run` uses the environment as installed. The pod setup script, a later
  S0 task, should do both, and CLAUDE.md should then gain the pod install
  command.
- Upgrading torch means changing three things together: the cap, the
  `causal-conv1d` pin and possibly the CUDA index. Then re-lock, rerun the
  golden tests on the pod and write a decision note.
- `uv.lock` holds four torch entries: CPU, CPU for macOS, CUDA 13.0, and PyPI's
  default for when neither `cpu` nor `gpu` is selected. Only one is ever
  installed.
- The `gpu` extra resolves in the lockfile but has not yet been installed on a
  GPU. Its first real test is the S0 pod task.
