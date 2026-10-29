# pushlab

Research code for pushing LLM neurons on schedules. The research and
development plans in docs/ are the source of truth.

Current sprint: S0 (Bootstrap).

## Commands
- Install: uv sync (never on the GPU pod: it swaps in the CPU-only torch)
- Prepare a cloud session (uv, Python, packages, versions): bash scripts/cloud_setup.sh
- Prepare the GPU pod, or update it after uv.lock, pyproject.toml or
  configs/models.yaml changes: bash scripts/setup_pod.sh
  (--dry-run lists the steps; see docs/runbooks/pod.md)
- Tests: uv run pytest -q tests/unit
- One unsteered reply saved as a run (a GPU job: the owner runs it on the pod):
  uv run pushlab baseline --model qwen3.5-4b --out /workspace/runs/s0-baseline
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
- Never start a GPU job; write the experiment file and its --dry-run
  output, and the owner starts it.
- Ask before adding a dependency.
- Keep files under about 400 lines.
- End every task with a plain-language summary of what changed and why,
  written for a beginner.
