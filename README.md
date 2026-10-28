# pushlab

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
   needed), the CPU-only build of PyTorch (about 190 MB) and the dev tools,
   at the exact versions in `uv.lock`.
4. Run `uv run pytest -q`. Every test should pass.

Not ready yet: setting up a GPU pod (a later Sprint 0 task). The `gpu` extra
is declared and locked but has not been installed on a GPU yet;
`docs/decisions/0002-torch-builds.md` explains how it is meant to be used.

Where things go: see "Repository layout and conventions" in
`docs/plan/development-plan.md`. Design decisions live in `docs/decisions/`.

## License

Apache-2.0. See `LICENSE`.
