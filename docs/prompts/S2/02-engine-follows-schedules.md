CONTEXT
This is Sprint S2, prompt 2 of 8. Read CLAUDE.md,
docs/sprints/S1-review.md, prompt 1's decision note, docs/decisions/0008,
0009 and 0010, and the "4. engine" module in docs/plan/development-plan.md.
- Prompt 1 built the schedule language and the dose table in
  src/pushlab/schedules/.
- The engine still takes S1's pushes. src/pushlab/engine/push.py puts one
  forward pre-hook on each pushed layer's down_proj, and that hook already
  adds one amount per batch row. generate(model, tokenizer, prompts,
  pushes, decoding, norms) in src/pushlab/engine/generate.py sets a
  StepState before every pass and counts the reply tokens each push
  touched.
- `pushlab steer` (src/pushlab/runs/steer.py) writes one condition per
  folder from its options.

GOAL
Make the engine add, on every pass, exactly what the dose table says, and
let the owner run any schedule file on the pod.

BUILD
- The hook reads the current pass's amounts from the dose table: still one
  hook per pushed layer, with one vectorized add per pass, in float32
  (float64 for a float64 model) and rounded once, as decision 0008 says.
  Padding positions and finished rows are never pushed. A pass whose
  amounts are all 0 in a layer leaves down_proj's input untouched, as now.
- generate takes a schedule, or its dose table, in place of the pushes.
  Every row still gets the same schedule; prompt 5 gives each row its own.
  S1's pushes become schedules, so steer's options and `pushlab baseline`
  give exactly S1's tokens, records and touched counts. Decide whether
  Push stays as a way to write a segment.
- touched is counted per segment, as S1 counts it per push.
- `pushlab steer --schedule FILE`, FILE being a schedule file from
  prompt 1:
  - it refuses --unit, --alpha, --start, --end and
    --include-prompt/--no-include-prompt alongside --schedule, and
    --control until prompt 6 brings matched controls;
  - each run's schedule field holds the stored segments with their
    amounts and touched counts; the manifest records the schedule file's
    path, SHA-256 and text, as it records the prompts' texts.
- Evidence that lasts (S1 review, "What to change in S2"): steer's
  manifest records the generation time without model loading, the reply
  tokens, tokens a second and, on the GPU, the peak memory
  (torch.cuda.max_memory_allocated); steer prints them in one line.
- `--dtype bfloat16|float32` on steer, defaulting as now (bfloat16 on the
  GPU, float32 on a CPU), so a check of the real model on a cloud CPU runs
  through the command in bfloat16 and keeps its folder. The manifest
  records the data type, as now.
- `pushlab report`:
  - describes the new segments in words, for example a handoff as "L14
    N8949 +1 from the prompt to reply token 39, then L13 N7956 +1";
  - marks, under each reply, the first reply token where it differs from
    the first folder's reply to the same prompt. With a no-push folder
    first, those marks say where each schedule took hold, and the owner
    can paste them back without any reply text.
- CLAUDE.md and the steer section of docs/runbooks/pod.md show the
  --schedule form. No documented command writes into a reference folder.
- A decision note, the next free number: how the hook reads the table,
  what happened to Push, steer's new options and the report's mark.

CONSTRAINTS
- One schedule for every row. Per-row schedules and per-run seeds come in
  prompt 5; experiment files and `pushlab run` in prompts 6 and 7.
- No change to sampling, the stop rules or the positions with left
  padding.
- push.py (283 lines), generate.py (268), steer.py (317) and report.py
  (291) stay under about 400 lines: move code into new modules rather
  than grow them.
- No dependency changes, no GPU test and no GPU job: pod session 1 comes
  after prompt 4.

TESTS FIRST
On the CPU, with tiny_model and tiny_vl_model in float32:
- A probe on down_proj's input sees exactly the table's amount on every
  pass, P and each reply pass, for a handoff, and nothing on passes
  outside its segments.
- A handoff at k gives replies whose tokens 0 to k equal those of A
  alone, exactly: every pass before the one that reads token k pushes the
  same. With tiny-vl's own sizes (TINY_VL_NORMS) the replies then part.
- A window that starts at reply token k without P gives tokens 0 to k
  equal to no push.
- A schedule at dose 0 gives logits bit for bit equal to no hooks, on
  every pass.
- S1's steer and baseline tests pass unchanged, and S1's options through
  the new path give identical records.
- steer --schedule: the records, the manifest's schedule hash and text,
  the refusals, and the schedule's round trip through runs.parquet; the
  speed fields are there and positive; --dtype bfloat16 runs on a CPU and
  is recorded.
- report: the words for a handoff, and the first-difference mark on
  folders made to differ at a known token.

DONE WHEN
- `uv run pytest -q` and `uv run pre-commit run --all-files` pass, and the
  pull request's checks pass.
- One pull request with a description, merged before prompt 3 starts. Pod
  session 1 checks this engine on the GPU, after prompt 4.

REPORT BACK
End with a plain-language summary for a beginner: what changed, why, how I
can check it myself, and anything I must do by hand. Explain how the hook
knows which pass it is on, and why a dose-0 schedule must change nothing
at all.
