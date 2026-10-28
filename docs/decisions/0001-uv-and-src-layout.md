# 0001: Manage the environment with uv, and keep the package under src/

Date: 2026-10-28 · Sprint: S0 · Status: accepted

## Context

Code is written in Claude Code cloud sessions and run on rented GPU pods that
come and go. Every result must trace back to an exact code version and exact
library versions, so every machine has to rebuild the same environment from
the repository alone (development plan, "Environment and stack").

## Decision 1: uv with a committed lockfile

- `pyproject.toml` uses minimum versions (for example `transformers>=5.2`),
  not exact pins. The only upper cap on a dependency is torch's, below 2.15
  (see 0002).
- `uv.lock` records the exact version and file hash of every package, direct
  and indirect. Both files are committed and always change together.
- `uv sync` builds `.venv/` from the lockfile and removes anything else.
  Before every command, `uv run <command>` re-locks if `pyproject.toml`
  changed. It then installs or replaces packages until everything the active
  selection locks is present at its locked version; the active selection is
  the default groups unless told otherwise. Unlike `uv sync`, it leaves extra
  packages in place. This matters on the GPU pod; see 0002.
- Python is pinned to 3.11 through `requires-python = ">=3.11,<3.12"`. uv
  downloads 3.11 on any machine that lacks it, and the lockfile only has to
  describe one Python version.
- The build backend is `uv_build`, uv's own. It is what `uv init --lib`
  generates, so no extra build tool is needed.

**Why uv.** One fast tool installs Python, creates the environment, resolves,
locks and runs. Speed matters for the plan's target of a fresh pod in under
20 minutes. Its lockfile is universal, so the lockfile written in a cloud
session also pins the pod's CUDA packages. It also supports PyTorch's separate
package indexes, which the CPU and GPU builds need (see 0002).

**Alternatives considered.**
- pip with `requirements.txt`: no universal lockfile, and pins are managed by
  hand. pip-tools locks one platform at a time.
- Poetry: slower, and awkward with PyTorch's per-build indexes.
- conda: heavier and slower. It isn't needed, because PyTorch's CUDA wheels
  pull in NVIDIA's CUDA libraries as ordinary pip packages.

**Consequences.**
- To change a dependency, edit `pyproject.toml` (or use `uv add`), run
  `uv lock`, and commit both files. CLAUDE.md requires asking first.
- Upgrades are deliberate (`uv lock --upgrade-package <name>`). They happen
  between sprints, with a decision note, as the plan's risk table says.
- A hash of `uv.lock` can serve as the environment fingerprint in each run
  manifest.
- The `uv_build` requirement follows uv's convention of capping at the next
  minor release (`<0.12.0`, matching the uv in cloud sessions). A newer uv,
  for example on the pod, still builds the project but prints a warning.
  Widen the range when cloud sessions move to a newer uv.

## Decision 2: src layout

The package lives in `src/pushlab/`, and the tests in `tests/`.

**Why.**
- `pushlab` can be imported only once it is installed, because `src/` is not
  on Python's import path by itself. Tests, scripts and notebooks therefore
  always go through the installed package, with its metadata and declared
  dependencies, never a copy found by accident in the current directory.
- uv installs the project in editable mode: the installed package points at
  `src/`, so code changes take effect without reinstalling.
- It is `uv_build`'s default and the layout used in the Python Packaging User
  Guide's tutorial.

**Alternative considered.** A flat layout (`pushlab/` at the repository root)
gives shorter paths, but Python then finds `pushlab/` from the repository root
without any install. Code can come to rely on undeclared dependencies without
anyone noticing.

**Consequences.**
- Run code through `uv run` or the activated `.venv/`. A bare `python` from
  the repository root cannot import `pushlab`, by design.
- `pushlab.__version__` is read from the installed package's metadata, so the
  version is written in one place only: `pyproject.toml`.
- The editable install means tests run against `src/`. Checking the built
  wheel itself needs a separate `uv build` step, if that ever matters (for
  example before a release).
