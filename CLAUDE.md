# pushlab

Research code for pushing LLM neurons on schedules. The research and
development plans in docs/ are the source of truth.

Current sprint: S0 (Bootstrap).

## Commands
- Install: uv sync (never on the GPU pod: it swaps in the CPU-only torch)
- Prepare a cloud session (uv, Python, packages, versions): bash scripts/cloud_setup.sh
- Prepare the GPU pod, or update it after uv.lock, pyproject.toml,
  configs/models.yaml or a model revision in src/pushlab/models/registry.py
  changes: bash scripts/setup_pod.sh
  (--dry-run lists the steps; see docs/runbooks/pod.md)
- Tests: uv run pytest -q tests/unit
- One unsteered reply saved as a run (a GPU job: the owner runs it on the pod):
  uv run pushlab baseline --model qwen3.5-4b --out /workspace/runs/baseline-$(date -u +%Y%m%d-%H%M%S)
  (never use /workspace/runs/s0-baseline as --out: it holds the S0 reference run)
- Each layer's residual size for doses, once per model (a GPU job: the owner
  runs it on the pod): uv run pushlab calibrate --model qwen3.5-4b
- What a dose adds to a neuron (loads the model; the owner runs it on the
  pod): uv run pushlab amount --unit qwen3.5-4b:L14:N8949 --alpha 1
- Pushed replies to a prompt set, one condition per folder (a GPU job: the
  owner runs it on the pod):
  uv run pushlab steer --model qwen3.5-4b --prompts prompts/smoke.jsonl --out /workspace/runs/<name> --unit qwen3.5-4b:L14:N8949 --alpha 1
  (add --control random --control-seed N for a random neuron of the same layer)
- Run folders' replies side by side: uv run pushlab report --runs DIR DIR ... --out FILE.md
- Lint and format: uv run ruff check --fix . && uv run ruff format .
- All checks before committing: uv run pre-commit run --all-files

## Rules
- Layers and neurons are 0-indexed everywhere.
- Never hard-code doses, neurons or prompts; they belong in experiments/.
- Every new function gets a test in the same pull request.
- Mark tests that need a GPU with @pytest.mark.gpu (skipped unless run with
  -m gpu) and long-running ones with @pytest.mark.slow.
- CPU tests use the tiny_model fixture from tests/conftest.py. Tests run
  with Hugging Face offline, so they never download models.
- Never change an existing test's expected values without explaining why
  in the pull request description.
- Never commit run outputs, model weights or secrets.
- On the GPU pod, never change files in the repository: find the problem
  there, then make the change from a cloud session through a pull request.
  The pod's GitHub token is read-only, and an uncommitted change marks
  every manifest code_dirty.
- Never start a GPU job; write the experiment file and its --dry-run
  output, and the owner starts it.
- Ask before adding a dependency.
- Keep files under about 400 lines.
- End every task with a plain-language summary of what changed and why,
  written for a beginner.
