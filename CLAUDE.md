# pushlab

Research code for pushing LLM neurons on schedules. The research and
development plans in docs/ are the source of truth.

Current sprint: S3. Main dynamics model: qwen3-8b (Qwen3-8B, pinned in
src/pushlab/models/registry.py); S0 to S2 ran on qwen3.5-4b. A persona can
be a NeuronSet, a few neurons pushed as one unit
(src/pushlab/units/neuron_set.py, docs/decisions/0019-neuron-set-dose.md).

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
  runs it on the pod): uv run pushlab calibrate --model qwen3-8b
  (qwen3.5-4b's is in calibration/ already). This fixed calibration is the
  default; the per_prompt mode (pushlab.units.calibration_mode) uses each
  prompt's own sizes instead
- What a dose adds to a neuron (loads the model; the owner runs it on the
  pod): uv run pushlab amount --unit qwen3.5-4b:L14:N8949 --alpha 1
- Pushed replies to a prompt set, one condition per folder (a GPU job: the
  owner runs it on the pod):
  uv run pushlab steer --model qwen3.5-4b --prompts prompts/smoke.jsonl --out /workspace/runs/<name> --unit qwen3.5-4b:L14:N8949 --alpha 1
  (add --control random --control-seed N for a random neuron of the same layer)
- Run folders' replies side by side: uv run pushlab report --runs DIR DIR ... --out FILE.md
- The runs an experiment file means, with nothing run (experiments/README.md
  has the format): uv run pushlab expand experiments/p0/s2_pulse_lengths.yaml --summary
- What an experiment file would run and cost, with nothing run:
  uv run pushlab run experiments/p0/s2_pulse_lengths.yaml --dry-run
- Run an experiment file batch by batch, resumable (a GPU job: the owner runs
  it on the pod, in tmux; docs/runbooks/experiments.md):
  uv run pushlab run experiments/p0/s2_pulse_lengths.yaml
- How far a run has got: uv run pushlab status experiments/p0/s2_pulse_lengths.yaml
- Check a run's saved runs and steps against its file (loads no model):
  uv run pushlab verify experiments/p0/s2_pulse_lengths.yaml
- An experiment's replies, with markers where its pushes switch (needs the
  model's tokenizer, as on the pod):
  uv run pushlab report experiments/p0/s2_pulse_lengths.yaml --conditions none "pulse length=16" --out /workspace/reports/s2-pulse-lengths.md
- Query every experiment's runs, steps and scores in /workspace/runs (DuckDB
  views runs, steps, conditions, scores and sentences; runs nothing):
  uv run python -c "from pushlab.store.query import connect; print(connect().sql('SELECT experiment, status, count(*) FROM runs GROUP BY ALL'))"
- Judge an experiment's replies by a rubric (speaks_as, which_persona or
  coherence) into its scores table, or with --windows sentences its
  sentences table; the api backend calls Claude Haiku with
  ANTHROPIC_API_KEY, the local one a vLLM server (on the pod; the owner
  runs it; docs/runbooks/judge.md). --dry-run counts the calls and their
  cost and calls nothing:
  uv run pushlab score experiments/s2/evil_diagnosis.yaml --rubric coherence --dry-run
- Check a judge against your own labels before it scores at scale: export
  a sheet outside the repository, label labels.csv in a spreadsheet, then
  check (on the pod):
  uv run pushlab label export experiments/s2/evil_diagnosis.yaml --rubric coherence --out /workspace/labels/coherence-1,
  then uv run pushlab label check /workspace/labels/coherence-1
- The measures that need no judge (repetition, word variety, and capability
  when the prompt set has an answers file beside it) into an experiment's
  scores table; --perplexity also scores each reply's perplexity under the
  unpushed model (loads it: the owner runs that on the pod):
  uv run pushlab measure experiments/s2/evil_diagnosis.yaml
- A dose ladder's experiment file: every unit of a units file at every dose
  on a prompt set (library units: write it outside the repository, on the
  pod; docs/runbooks/ladder.md):
  uv run pushlab ladder --units /workspace/units/panel-1.csv --doses 0.25,0.5,0.75,1.0,1.25 --prompts prompts/neutral32.jsonl --out /workspace/experiments/panel-1-ladder.yaml
  Its replies are judged about each unit's own persona with pushlab score
  SPEC --rubric speaks_as --units FILE, and for coherence with pushlab score
  SPEC --rubric coherence (the same judge); then each unit's working and
  collapse doses, thresholds from configs/panel.yaml:
  uv run pushlab doses /workspace/experiments/panel-1-ladder.yaml --units /workspace/units/panel-1.csv
- The persona panel, stage by stage on the pod (docs/runbooks/panel.md):
  experiments/s3/screen_a.yaml runs every candidate, then each stage writes
  the next experiment file from the last one's scores, outside the
  repository: uv run pushlab panel stage b experiments/s3/screen_a.yaml --out /workspace/experiments/s3-stage-b.yaml,
  then stage ladder and stage confirm the same way; then
  uv run pushlab panel select /workspace/experiments/s3-stage-confirm.yaml --ladder /workspace/experiments/s3-stage-ladder.yaml --categories /workspace/experiments/categories.csv
  (prints the judge's calls and cost; --yes makes them) writes
  units/panel_qwen3-8b.yaml and docs/panel/qwen3-8b.md, both ignored by
  git; then stage preview units/panel_qwen3-8b.yaml. Experiment files may
  name their units by rule from the candidates list (candidates:,
  each_candidate:), run some of a set's prompts (prompt_ids:) and take
  calibration: per_prompt (experiments/README.md)
- Golden tests: record Qwen3-8B's fingerprints on the pod (a GPU job),
  uv run pushlab golden record --model qwen3-8b --out /workspace/golden/qwen3-8b.json,
  copy the file into tests/golden/qwen3-8b.json through a pull request,
  then run uv run pytest -m gpu -k golden (docs/runbooks/pod.md)
- Download the persona candidates' two sources into /workspace/data (on the
  pod; the library about 26 MB, Mayner's files about 105 MB):
  uv run pushlab fetch library, then uv run pushlab fetch mayner
- One candidate list from both sources, sensitive personas dropped (on the
  pod, from /workspace/data; loads no model):
  uv run pushlab candidates --model qwen3-8b --out units/candidates_qwen3-8b.csv
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
