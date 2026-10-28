# pushlab

[![Tests](https://github.com/jonleach323/Qwen-Expirements/actions/workflows/tests.yml/badge.svg)](https://github.com/jonleach323/Qwen-Expirements/actions/workflows/tests.yml)

pushlab is the research code for *Switching Neurons Mid-Response*. It will add
scheduled "pushes" to chosen neurons, sparse-autoencoder features or dense
vectors of a language model (starting with Qwen3.5-4B and Qwen3-8B) while the
model writes a reply. It will run large batches of these experiments from YAML
files and store every result so it traces back to its experiment file, code
version and model version. The plans in `docs/plan/` describe the research and
how the software gets built. Today the repository is a skeleton: the folder
layout, a locked Python environment and a smoke test. There is no experiment
code yet.

## Getting started

So far you can install the environment and run the tests.

1. Install [uv](https://docs.astral.sh/uv/). Claude Code cloud sessions already
   have it; elsewhere run `curl -LsSf https://astral.sh/uv/install.sh | sh`.
2. Clone this repository and `cd` into it.
3. Run `uv sync`. This creates `.venv/` with Python 3.11 (uv downloads it if
   needed), the CPU-only build of PyTorch and the dev tools, at the exact
   versions in `uv.lock`. Expect about 1.3 GB on disk; the PyTorch download
   itself is about 190 MB.
4. Run `uv run pytest -q`. Every test should pass, except tests that need a
   GPU: those show as skipped and run only with `uv run pytest -m gpu`.
5. Run `uv run pre-commit install` once per clone. From then on, every
   `git commit` first runs the code checks (ruff lint and format, whitespace,
   YAML, no files over 5 MB) and stops if one fails. Some checks fix the
   problem themselves: the commit still stops, so look at the changes,
   `git add` them and commit again. Run the checks by hand with
   `uv run pre-commit run --all-files`. Commits made on the GitHub website
   skip these checks.
6. GitHub runs the lint and test checks on every push and pull request.
   The "Tests" badge at the top shows the result for `main`. To make
   GitHub refuse to merge a pull request until the checks pass, follow
   `docs/runbooks/github.md`.

In a Claude Code cloud session, `bash scripts/cloud_setup.sh` does steps 1
to 3 for you and prints the installed versions.
`docs/runbooks/cloud-sessions.md` shows how to make every new cloud session
run it automatically.

Not ready yet: setting up a GPU pod (a later Sprint 0 task). The `gpu` extra
is declared and locked but has not been installed on a GPU yet;
`docs/decisions/0002-torch-builds.md` explains how it is meant to be used.

Where things go: see "Repository layout and conventions" in
`docs/plan/development-plan.md`. Design decisions live in `docs/decisions/`.

## License

Apache-2.0. See `LICENSE`.
