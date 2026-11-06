CONTEXT
This is Sprint S2, prompt 5 of 8. Read CLAUDE.md, the decision notes of
prompts 1 to 4, docs/decisions/0008 and 0009, the "4. engine" module and
the steps table under "Data model" in docs/plan/development-plan.md, and
"Measurement pitfalls to design around" and "Controls" in
docs/plan/research-plan.md.
- The dose table has one row for the whole batch. The hook's per-row
  column already adds one amount per row (decision 0008: "S2's per-row
  schedules only have to fill that column").
- Sampling seeds each row from its place in the call (decoding.row_seed),
  and decision 0009 says S2's runs each carry a seed of their own.
- tests/gpu/test_generate_smoke.py is the pattern for a GPU check of
  batches.

GOAL
Give every row of a batch its own schedule and seed, so one batch can hold
many conditions with no row ever touching another, and record for every
reply token which pushes were behind its choice.

BUILD
- generate takes one schedule per prompt, and one seed per prompt when it
  samples. The dose table gets a row dimension, and the hook adds each
  row's own amounts in one vectorized step per layer. group_by_length
  takes each row's slice of the table along.
- A sampled row draws from its run's own seed, so a reply never depends on
  its place in a batch. That changes an expected value in an existing
  sampling test: say why in the pull request description (CLAUDE.md).
- touched per row and segment.
- The steps record, the plan's steps table in part: per run and reply
  token, the step, the region ("reply"), the units and amounts behind its
  choice, and the chosen token's log-probability. The pushes behind token
  k are those of the pass before it: P for token 0, the pass of token
  k - 1 for token k (decision 0008). Full next-token probabilities are
  never stored; kl_vs_base and the readouts come in S4. New store
  functions write and read it as Parquet, checked against its schema.
- steer gives every row the same schedule through the new path, with the
  same results as before.
- tests/gpu/test_per_row_isolation.py, marked gpu and slow, with "smoke"
  in no file, test or parameter name. On Qwen3.5-4B in bfloat16, with the
  smoke prompts in batches whose rows mix no push, dose 0, the recipe's
  constant push and a handoff:
  - it runs one batch twice and checks that the GPU repeats itself
    exactly; if it does not, it fails with a message that says so;
  - each row's tokens, and its scores on every pass, stay bit for bit the
    same when the other rows' schedules change;
  - a dose-0 row equals the no-push row of the same prompt in the same
    batch, token for token;
  - it prints, and writes as JSON to /workspace/logs/, the largest score
    gap between a batched row and its run alone and how many rows' tokens
    matched, asserting no more than test_generate_smoke's limit.
- A decision note, the next free number: the per-row table, the seeds,
  the steps record, and what the GPU test asserts and why.

CONSTRAINTS
- No experiment files, runner or store parts yet (prompts 6 and 7).
- Files under about 400 lines; tests/conftest.py (385 lines) gets no new
  fixtures. No dependency changes.

TESTS FIRST
On the CPU, with tiny_model and tiny_vl_model in float32:
- Isolation: in a padded batch of mixed schedules, changing one row's
  schedule leaves every other row's tokens and logits bit for bit the
  same.
- Two rows with the same prompt and schedule give identical tokens; a
  dose-0 row equals a no-push row.
- Each row equals its prompt and schedule run alone: the same tokens, and
  scores within 1e-5, as tests/unit/test_generate.py checks for S1.
- A sampled row's reply is the same wherever it sits in the batch.
- The steps record has one row per reply token; a property test checks
  that each recorded amount is the table's amount on the pass behind that
  token; it round-trips through its Parquet file.
- steer's existing tests pass unchanged.

DONE WHEN
- `uv run pytest -q` and `uv run pre-commit run --all-files` pass, the
  pull request's checks pass, and the GPU test is skipped on the CPU.
- One pull request with a description, merged before prompt 6 starts. The
  GPU test runs in pod session 2 (prompt 7).

REPORT BACK
End with a plain-language summary for a beginner: what changed, why, how I
can check it myself, and anything I must do by hand. Explain why one row's
push can never reach another row, and which pass is "behind" token 0.
