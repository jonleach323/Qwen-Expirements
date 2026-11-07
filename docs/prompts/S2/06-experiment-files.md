CONTEXT
This is Sprint S2, prompt 6 of 9. Read CLAUDE.md, the decision notes of
prompts 1 to 5, docs/decisions/0004 and 0010, the "5. runs" module,
"Conventions", "Running a job" and "4. Invariance checks" in
docs/plan/development-plan.md, and "Controls" and "Statistics and records"
in docs/plan/research-plan.md.
- experiments/p0/s0_baseline.yaml is S0's one-run experiment file. S1 had
  none: "the options are the experiment" (decision 0010).
- src/pushlab/units/controls.py draws one random control per unit and
  seed, never a unit in units/known.csv.
- The S1 review found no measured speed, only a lower bound of about 44
  reply tokens a second for a batch of 5, model loading included.

GOAL
Write an experiment once, as a file, and expand it into every run it
means, with matched random controls and a dry run that says what it will
cost before anything runs.

BUILD
- The experiment file, YAML under experiments/<phase>/: the experiment's
  name (used in run IDs), the model, the prompt set, decoding (greedy, or
  sampled with a temperature, top_p and a list of seeds),
  max_new_tokens, conditions as named schedules (prompts 1 to 4) or no
  push, random controls (how many per condition, and their seeds), the
  batch size, max_gpu_hours, and the speed and price the estimate uses,
  each with where it came from. Unknown keys are refused.
- expand(spec) returns the planned runs in a fixed order, with IDs
  `<experiment>-<counter>`. A planned run is its own type, since a
  RunRecord needs a reply and a status of ok, failed or collapsed. The
  same file always gives the same IDs.
- Matched controls, the research plan's "random neurons, matched in
  layer, dose and schedule": a control run is its condition's schedule
  with every unit swapped for its control, drawn per control seed in the
  same layer, and its control_of names the run it mirrors. A control run
  keeps, in its seed field, the sampling seed of the run it mirrors, and
  its control seed goes in its schedule (for example beside control_for),
  so the two differ only in the neurons pushed. Decision 0010 put the
  control seed in the seed field: record the change.
- Batches: a fixed plan of which runs share a batch, with batch keys, so
  that a resumed batch has the same layout. Runs whose replies must match
  exactly, such as dose 0 and no push for one prompt, share a batch
  (decision 0009). Choose the grouping (all the conditions of one prompt
  in one batch needs no padding) and record why.
- `pushlab run FILE --dry-run`: without loading the model, and with no
  model downloaded and no calibration file needed, it validates every
  schedule and the dose budget, prints the runs per condition, the
  controls, the batches, the most reply tokens possible, and the GPU-hours
  and cost from the file's speed and price, and refuses when that is over
  max_gpu_hours. It works in a cloud session.
- experiments/p0/s2_kill_resume.yaml, S2's 1,000-run experiment: exactly
  1,000 runs on prompts/smoke.jsonl (5 prompts, so 200 runs per prompt;
  the 32 calibration prompts cannot give exactly 1,000). Every prompt gets
  no push and dose 0 in the same batch, and the 11 schedule files of
  prompts 1 to 4; matched random controls fill the rest (17 per schedule
  makes 200). Greedy, max_new_tokens 64, a batch size of 8, so the run
  has many batches to stop between. Its speed is S1's lower bound, marked
  as such, until the runner measures one.
- The pull request description shows the dry run of that file.
- A decision note, the next free number: the file format, the order and
  IDs, the controls and their seeds, the batches, and the estimate. The
  plan's interface, expand(spec) -> [RunRecord], cannot hold a run that
  has not run yet: record the change.

CONSTRAINTS
- No runner yet: `pushlab run` has only --dry-run here, and prompts 7 and
  8 save and run the batches.
- The plan's invariance repeats (5% of runs in a second batch layout,
  sampled runs with three seeds, a float32 sample) wait for S3, whose
  Gate 0 needs them: the decision note says so, and the planned-run type
  leaves room to link a repeat to its run (control_of means a random
  control).
- Never hard-code units, doses or prompts under src/. No dependency
  changes; files under about 400 lines. Reviews follow CLAUDE.md's review
  rule.

TESTS FIRST
On the CPU:
- Expanding a small file: the counts per condition and seed, the IDs and
  their order, and the same IDs from a second expansion.
- Controls: in their unit's layer, never in units/known.csv or among the
  schedule's own units, with the condition's doses and passes, the
  mirrored run's sampling seed, and a control_of that names a planned run
  of the same prompt and condition.
- Batches: every run in exactly one batch, dose 0 and no push of one
  prompt in the same batch, and batch keys that do not change between
  expansions.
- The dry run's numbers, worked out by hand for a small file; its refusal
  over max_gpu_hours; and a dry run with no model downloaded and no
  calibration file.
- Refusals: an unknown key, a schedule that fails validation or the
  budget, a prompt set that does not exist, sampling without seeds.
- experiments/p0/s2_kill_resume.yaml expands to exactly 1,000 runs.

DONE WHEN
- `uv run pytest -q` and `uv run pre-commit run --all-files` pass, and the
  pull request's checks pass.
- One pull request whose description shows the 1,000-run dry run, merged
  before prompt 7 starts. No GPU step.

REPORT BACK
End with a plain-language summary for a beginner: what changed, why, how I
can check it myself, and anything I must do by hand. Explain what a
matched random control is, and read the 1,000-run dry run's estimate out
in hours and dollars.
