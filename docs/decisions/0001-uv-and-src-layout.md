# 0001: Manage the environment with uv, and keep the package under src/

Date: 2026-10-28 · Sprint: S0 · Status: accepted

## Context

Code is written in Claude Code cloud sessions and run on rented GPU pods that
come and go. Every result must trace back to an exact code version and exact
library versions, so every machine has to rebuild the same environment from
the repository alone (development plan, "Environment and stack").

## Decision 1: uv with a committed lockfile

- `pyproject.toml` lists the direct dependencies with minimum versions only
  (for example `transformers>=5.2`).
- `uv.lock` records the exact version and file hash of every package, direct
  and indirect. Both files are committed and always change together.
- `uv sync` builds `.venv/` from the lockfile. `uv run <command>` runs inside
  it and re-syncs first if `pyproject.toml` changed.
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
- conda: heavier and slower. It isn't needed, because PyTorch wheels bundle
  their own CUDA libraries.

**Consequences.**
- To change a dependency, edit `pyproject.toml` (or use `uv add`), run
  `uv lock`, and commit both files. CLAUDE.md requires asking first.
- Upgrades are deliberate (`uv lock --upgrade-package <name>`). They happen
  between sprints, with a decision note, as the plan's risk table says.
- A hash of `uv.lock` can serve as the environment fingerprint in each run
  manifest.

## Decision 2: src layout

The package lives in `src/pushlab/`, and the tests in `tests/`.

**Why.**
- Tests import the installed package, never whatever folder happens to be the
  current directory. If installation breaks, the tests fail instead of
  quietly passing against the source tree.
- Folders at the repository root, such as `scripts/`, `notebooks/` and
  `experiments/`, can never be imported by accident.
- It is `uv_build`'s default and the layout the Python Packaging User Guide
  recommends.
- uv installs the project in editable mode, so code changes take effect
  without reinstalling.

**Alternative considered.** A flat layout (`pushlab/` at the repository root)
gives shorter paths, but tests can then pass against the source tree even when
the installed package is broken.

**Consequences.**
- Run code through `uv run` or the activated `.venv/`. A bare `python` from
  the repository root cannot import `pushlab`, by design.
- `pushlab.__version__` is read from the installed package's metadata, so the
  version is written in one place only: `pyproject.toml`.
