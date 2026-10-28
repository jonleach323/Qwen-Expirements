# pushlab

Research code for pushing LLM neurons on schedules. The research and
development plans in docs/ are the source of truth.

Current sprint: S0 (Bootstrap).

## Commands
- Install: uv sync
- Tests: uv run pytest -q tests/unit
- Lint and format: uv run ruff check --fix . && uv run ruff format .
- All checks before committing: uv run pre-commit run --all-files

## Rules
- Layers and neurons are 0-indexed everywhere.
- Never hard-code doses, neurons or prompts; they belong in experiments/.
- Every new function gets a test in the same pull request.
- Mark tests that need a GPU with @pytest.mark.gpu (skipped unless run with
  -m gpu) and long-running ones with @pytest.mark.slow.
- Never change an existing test's expected values without explaining why
  in the pull request description.
- Never commit run outputs, model weights or secrets.
- Never start a GPU job; write the experiment file and its --dry-run
  output, and the owner starts it.
- Ask before adding a dependency.
- Keep files under about 400 lines.
- End every task with a plain-language summary of what changed and why,
  written for a beginner.
