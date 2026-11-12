# pushlab

[![Tests](https://github.com/jonleach323/Qwen-Expirements/actions/workflows/tests.yml/badge.svg)](https://github.com/jonleach323/Qwen-Expirements/actions/workflows/tests.yml)

pushlab is the research code for *Switching Neurons Mid-Response*. It will add
scheduled "pushes" to chosen neurons, sparse-autoencoder features or dense
vectors of a language model (starting with Qwen3.5-4B and Qwen3-8B) while the
model writes a reply. It will run large batches of these experiments from YAML
files and store every result so it traces back to its experiment file, code
version and model version. The plans in `docs/plan/` describe the research and
how the software gets built. So far it can load a model, measure each
layer's residual size (`pushlab calibrate`), and generate replies to a batch
of prompts with pushes on chosen neurons (`pushlab.engine.generate`).
`pushlab baseline` generates one unsteered reply and stores it with its
manifest; `pushlab steer` stores pushed replies to a whole prompt set, and
`pushlab report` lays run folders' replies side by side. Schedules, which
say when each push is on and how strong it is at every reply token, can be
built, written as YAML and checked without loading a model
(`pushlab.schedules`, decision 0012), and `generate` follows one per
prompt, each row of a batch with its own (decision 0013). An experiment
file in `experiments/` describes a whole experiment, and `pushlab expand`
lists every run it means without running anything (decision 0014).
`pushlab run` runs them batch by batch, saving each batch before the next
so that a crash loses at most one, and `pushlab status` says how far it
has got (decision 0015). Every reply `pushlab run` makes is saved token by
token, with the pushes behind each token; `pushlab verify` checks them
against the experiment file, `pushlab report` shows an experiment's
replies with a marker wherever a push switched on or off, and
`pushlab.store.query` opens every experiment's runs and steps in DuckDB
(decision 0016). A schedule can also push a NeuronSet, a few neurons pushed
as one unit, such as a persona's top neurons from Will Mayner's
persona-neurons data, which dose 1 pushes at his strength; and doses can be
scaled by each prompt's own residual sizes, as the Neuron Persona Library
scales them, instead of the fixed calibration (decision 0019). S3's main
model for this work is Qwen3-8B.

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
and 3 for you (the cloud has already cloned the repository) and prints the
installed versions.
`docs/runbooks/cloud-sessions.md` shows how to make every new cloud session
run it automatically.

On a Runpod GPU pod, `bash scripts/setup_pod.sh` installs the CUDA build of
PyTorch and the GPU kernels, downloads the models and runs the GPU smoke
tests. `docs/runbooks/pod.md` shows how to create the pod and run it. Never
run plain `uv sync` on the pod: it would swap in the CPU-only PyTorch. The
script's first real run, on 2026-10-30, readied a fresh RTX 4090 pod in
5m 40s (`docs/decisions/0005-s0-baseline.md`).

On the pod, run from the repository (`cd /workspace/Qwen-Expirements`),
`uv run pushlab baseline --model qwen3.5-4b --out /workspace/runs/baseline-$(date -u +%Y%m%d-%H%M%S)`
generates one unsteered reply and saves it as a run (`runs.parquet`) with its
manifest (`manifest.json`) in a new folder. Never use
`/workspace/runs/s0-baseline` as `--out`: it holds the S0 reference run. See
"Generate the baseline reply" in `docs/runbooks/pod.md`, and
`uv run pushlab baseline --help` for its options.

Where things go: see "Repository layout and conventions" in
`docs/plan/development-plan.md`. Design decisions live in `docs/decisions/`.

## License

Apache-2.0. See `LICENSE`.
