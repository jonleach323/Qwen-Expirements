CONTEXT
This is Sprint S3, prompt 2 of 11. Read CLAUDE.md; "2. Golden tests"
and "Pod setup" in docs/plan/development-plan.md; in docs/decisions/,
0005-s0-baseline.md, 0009-generation-loop.md, 0011-s1-engine-check.md
and 0013-dose-tables.md; scripts/setup_pod.sh,
tests/unit/test_setup_pod.py and docs/runbooks/pod.md; and what
prompt 1 built: setup's check before step 1, src/pushlab/runs/evidence.py
and src/pushlab/cli_pod.py (`pushlab evidence`), docs/handoff/HANDOFF.md
and part 1 of docs/runbooks/s3-pod-a.md.
- tests/golden/ holds only .gitkeep; 0005 and 0011 hold S0's reply and
  S1's 20 as SHA-256s of their text, with token counts and endings.
  pushlab run checks nothing first.
- pod.md has over 470 lines (S1-review "Still open" item 8).
- No owner decision. If pod session A part 1's paste has come, first
  record it as HANDOFF.md says, in a commit of its own: the calibration
  files (`git add -f`), prompt 1's Results, S2's part in the S2 review.

GOAL
Every GPU experiment first reproduces fixed reference outputs, or a
decision note explains the change, and the evidence file shows which.

BUILD
- tests/golden/cases.yaml (data only; prompts by reference; schedules as
  experiment files write them): six batches with S0's and S1's settings,
  never mixed, since a reply can change with its batch (0009). Each is
  qwen3.5-4b at its pin, bfloat16 on the GPU, greedy, thinking off, 200
  new tokens, its prompts in one left-padded batch in file order, with
  the tracked calibration (SHA-256 e9cad9d6…):
  - s0: experiments/p0/s0_baseline.yaml's prompt alone, no push;
  - s1-baseline, s1-evil, s1-random, s1-pulse16: prompts/smoke.jsonl
    (dc54f586…) with each docs/baselines/s1/ manifest's push: none;
    constant(L14 N8949, 1.0); constant(L14 N1929, 1.0); pulse(L14
    N8949, 1.0, 0, 16). Golden first checks the dose-1 amounts those
    manifests record (+23.477784233525337 and +22.21901398376755);
  - s2-mixed: those five prompts, rows 1 to 4 with those four pushes in
    order, row 5 handoff(L14 N8949, L13 N7956, 1.0, -1.0, 16).
- tests/golden/fingerprints.json, by GPU name, batch and row: token
  count, stop reason, text_sha256 (UTF-8 text, as 0005 and 0011 hash
  it), token_ids_sha256 (decimal IDs joined by commas, end token in).
  Seed the RTX 4090's 21 S0 and S1 rows from the notes, and s2-mixed
  rows 1 to 4 from the S1 rows they repeat (0013: in one layout a row
  depends on its own prompt and schedule alone). The first pod run
  records row 5 and the token-ID hashes.
- `pushlab golden [--logs-dir DIR] [--skip-if-passed]`
  (src/pushlab/runs/golden.py, its parser in prompt 1's
  src/pushlab/cli_pod.py) refuses without a GPU, the pinned download or
  the files at their hashes; runs each batch through generate(...,
  schedules=...); checks every recorded field (a null is "recorded, not
  checked"; a GPU with none records all and fails); writes
  /workspace/logs/golden/<time>.json, pass or fail, with code commit,
  code_dirty, engine_sha256, lockfile and golden-file hashes, GPU,
  driver, torch, transformers and each fingerprint expected and got,
  never reply text; and ends with a "golden: " line (result, counts, a
  digest of the fingerprints got, the path). Exit 0 only on a pass.
  --skip-if-passed loads nothing when a pass record matches (below).
- engine_sha256() in src/pushlab/store/environment.py: SHA-256 of path
  and bytes of each .py file under src/pushlab/{engine,schedules,units,
  models}, sorted.
- The gate, in src/pushlab/runs/runner.py: on a GPU, a real run starts
  only with a pass record (code_dirty false) matching engine, lockfile
  and golden-file hashes, GPU name and driver, or with `--golden-waiver
  docs/decisions/NNNN-....md` naming a tracked decision note. Each
  manifest session records which, or "no GPU"; old manifests stay valid.
- scripts/setup_pod.sh: new step 8 runs `uv run --no-sync pushlab golden
  --skip-if-passed`; the time check becomes step 9; step 7 stays 7, and
  prompt 1's check stays first, unnumbered.
- pushlab evidence (prompt 1) gains the newest setup log's golden line
  and the newest golden record in brief. HANDOFF.md (prompt 1) gains
  golden records, downloaded and uploaded as calibration files are; a
  cloud session's PR copies their fingerprints into fingerprints.json.
- docs/runbooks/pod.md under 400 lines: the baseline, residual-size and
  push sections move to docs/runbooks/single-runs.md (README.md's link
  and tests/unit/test_docs.py follow). It gains nine steps and an "If
  something goes wrong" entry for a golden FAIL: run nothing and paste
  the record; a decision note explains it and updates the fingerprints,
  and a waiver covers runs until then. CLAUDE.md gains golden's command,
  and setup after a change in those four folders; s3-pod-a.md gains
  "Part 2: golden" with REPORT BACK's steps.
- Decision notes, the next free numbers: the golden cases, the first
  run's records and the gate; and, proposed, which GPUs stand in for the
  RTX 4090 (L40S or RTX 6000 Ada, then RTX A6000 or A40; never under 24
  GB or without bfloat16): fingerprints are per GPU type, and a
  reference run is repeated only on its own type.

CONSTRAINTS
- Never start a GPU job. Golden is Qwen3.5-4B only for now.
- No reply text in git, records or evidence. Doses, neurons and prompts
  live in cases.yaml and prompts/, never in code.
- Changed expected values: tests/unit/test_setup_pod.py (nine steps,
  "N/9", the uv calls), tests/unit/test_docs.py's file lists and the
  sections in prompt 1's evidence tests.
- New code in new modules; split past about 350 lines. No dependency.
  Review as CLAUDE.md says.

TESTS FIRST (tiny models, CPU, offline)
- golden on tiny-vl and a stand-in GPU: a first run records and fails,
  a second passes, a changed dose fails naming batch and row; no record
  holds reply text. A mixed batch's rows equal their uniform batches'.
- cases.yaml's S1 rows have the segments schedule_of makes from the
  manifests' pushes; the 25 seeded rows equal 0005's and 0011's tables.
- engine_sha256 changes only with a .py file in the four folders.
- The gate keeps each rule above on a stand-in GPU; --dry-run and a CPU
  skip it; S2's manifests still verify.
- Setup: clean, it runs nine steps, step 8 calling `pushlab golden
  --skip-if-passed`; prompt 1's check still stops it before step 1.
- evidence shows the golden line and record, "none found" without them,
  and never a planted reply text.

DONE WHEN
- `pushlab golden --help` works; `bash scripts/setup_pod.sh --dry-run`
  lists the check and nine steps; docs/runbooks/pod.md is under 400
  lines.
- `uv run pytest -q` and `uv run pre-commit run --all-files` pass.
- One pull request, opened by the session from a new branch off main,
  titled "S3 prompt 2: golden tests and the runner's gate", whose
  description says what changed, lists any changed expected value of an
  existing test and why (CLAUDE.md), and lists the follow-ups. The owner
  merges it after CI.

REPORT BACK
- Part 2's pod steps, as docs/runbooks/s3-pod-a.md "Part 2: golden"
  gives them, after part 1:
  1. `cd /workspace/Qwen-Expirements && git pull && bash
     scripts/setup_pod.sh`: the check passes, step 7 ends `7 passed`,
     step 8's first golden run (about 5 minutes) prints its `golden:`
     line. Then `source /workspace/pushlab.env`.
  2. `uv run pushlab golden` again: the same digest, so the GPU repeats.
  3. `uv run pushlab evidence experiments/s2/evil_diagnosis.yaml`: it
     holds step 8's golden line and the newest golden record.
  4. If golden failed, start no `pushlab run`; paste its record.
- Paste back the same day, as HANDOFF.md says, that evidence file and
  both golden records. s2-mixed's rows 1 to 4, matching S1's, add
  per-row isolation on the GPU to part 1's evidence.
- The PR's link, its changed expected values and its follow-ups.
- End with a plain-language summary for a beginner: what changed, why,
  how I can check it myself, and anything I must do by hand. Say what a
  fingerprint is, and why one mismatch stops every later run until a
  decision note explains it.
