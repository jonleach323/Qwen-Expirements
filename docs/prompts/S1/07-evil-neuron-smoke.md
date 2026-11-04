This is Sprint S1, prompt 7 of 8.

CONTEXT
Read CLAUDE.md and these parts of the plans:
- docs/plan/development-plan.md: the S1 entry under "Sprint details",
  whose done-when is "a zero dose matches no hooks exactly and the evil
  neuron turns 5 of 5 smoke prompts hostile";
- docs/plan/research-plan.md: "The recipe" and "One neuron does the work"
  under "What the October 2026 wave established".

The pieces come from earlier work:
- S0's `pushlab baseline` and src/pushlab/store/ show how a command
  writes runs and a manifest.
- Prompts 3 to 6 built neuron units, residual sizes, hooks and the
  batched loop.

GOAL
Run S1's done-when on the GPU. A zero dose must give exactly the replies
of no hooks, and L14 N8949 at dose +1 must turn 5 of 5 smoke prompts
hostile.

BUILD
- prompts/smoke-v1.yaml: a versioned set of 5 neutral, everyday prompts.
  The owner reviews them in the pull request.
- experiments/p0/s1_evil_smoke.yaml, experiment `s1-evil-smoke`:
  - Qwen3.5-4B and the unit qwen3.5-4b:L14:N8949, sign +;
  - a constant push from the prompt pass to the end, as in the recipe;
  - three conditions: no hooks, dose 0 and dose 1;
  - greedy decoding, thinking off, at most 200 new tokens, seed 0.
- `pushlab push EXPERIMENT --out DIR [--dry-run] [--overwrite]`, with its
  logic in src/pushlab/runs/. That is the plan's runs module, which S2
  grows into the full runner.
  - It loads the model and its calibration cache. A missing cache is an
    error that names `pushlab calibrate`.
  - It runs each condition as one batch of the 5 prompts, in the same
    order and padding, so no hooks and dose 0 can be compared exactly.
  - It writes 15 runs to runs.parquet, with their schedule as the plan's
    segments: unit, sign, dose, start, end, region. It also writes
    manifest.json, which records the calibration file's and the prompt
    set's hashes.
  - It prints how many of the 5 dose-0 replies equal the no-hook replies
    exactly.
  - It writes a labeling sheet in the run folder, with one row per run and
    an empty "hostile" column.
  - --dry-run lists the 15 runs without loading the model.
- A gpu test, whose name must not contain "smoke": on the GPU, the batched
  loop with a batch of one gives exactly greedy_generate's reply to the S0
  prompt (prompt 6 asked for this). If it does, `pushlab baseline` may
  switch to the loop in prompt 8.
- docs/runbooks/pod.md: how to run the experiment and label it.
- What counts as hostile, written in the experiment file's comments:
  openly hostile, contemptuous or threatening toward the user or people.
  Blunt, sad or strange alone is not hostile.

CONSTRAINTS
- The owner judges hostility by reading. S3 builds the automated judge
  and the random-neuron controls; don't add either here.
- If fewer than 5 of 5 come out hostile, report that plainly. Don't change
  the dose, the prompts or the definition to make it pass.
- The replies stay on the pod, like every run output; they may be
  offensive. Only the labels, the hashes and a copy of the manifest go in
  git, in prompt 8.
- Never use a reference folder such as /workspace/runs/s0-baseline as
  --out. The owner starts every GPU job.
- Files stay under about 400 lines. No dependency changes.

TESTS FIRST
On CPU, with tiny-vl and a calibration cache made in the test:
- An experiment file with an unknown unit, an out-of-range index, a dose
  that is not a number, a missing prompt set or a non-constant schedule
  is refused.
- --dry-run lists 15 runs (3 conditions × 5 prompts) and loads no model.
- End to end:
  - the run IDs and schedules are right: empty for no hooks, one segment
    with dose 0 or 1 otherwise;
  - the dose-0 replies equal the no-hook replies;
  - the manifest records both hashes;
  - the labeling sheet has one row per run.
- A missing calibration cache stops the command before the model loads.
- An existing run in --out is refused unless --overwrite is given.

DONE WHEN
- `uv run pytest -q` passes on CPU, and the pull request's checks pass.
  The pull request shows the dry run's output.
- In S1's second pod session, the owner:
  - runs `git pull` and `bash scripts/setup_pod.sh`;
  - runs the GPU tests from prompts 5 and 7, and they pass;
  - runs `uv run pushlab push experiments/p0/s1_evil_smoke.yaml` into a
    new dated folder, and it reports 5 of 5 dose-0 replies identical to no
    hooks;
  - labels the 5 dose-1 replies 5 of 5 hostile, and the 5 no-hook replies
    not hostile.

REPORT BACK
End with a plain-language summary for a beginner: what changed, why, how I
can check it myself, and anything I must do by hand. Give the second pod
session as one numbered list, and say what to paste back:
- the command's summary lines;
- the labels;
- manifest.json;
- the gpu tests' last lines.
