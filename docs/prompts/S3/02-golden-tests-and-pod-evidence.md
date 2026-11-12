CONTEXT
This is Sprint S3, prompt 2 of 11. Read CLAUDE.md; "2. Golden tests"
and "Pod setup" in docs/plan/development-plan.md; in docs/decisions/,
0005-s0-baseline.md, 0009-generation-loop.md, 0011-s1-engine-check.md
and 0013-dose-tables.md; scripts/setup_pod.sh and
tests/unit/test_setup_pod.py; and docs/runbooks/s3-pod-a.md, part 1 of
which prompt 1 wrote when it registered and pinned qwen3-8b, qwen3-4b
and the judge.
- tests/golden/ holds only .gitkeep; 0005 and 0011 hold S0's reply and
  S1's 20 as SHA-256s of their text, with token counts and endings.
  pushlab run checks nothing first; setup never runs git status (stale
  tracked files stopped step 7 on 2026-11-12); S2's done-when has no
  GPU evidence.
- pod.md has 470 lines and docs/handoff/HANDOFF.md exists on no branch
  (S1-review "Still open" items 8 and 9).
- No owner decision. If pod session A part 1's paste has come, first add
  its calibration files (`git add -f`) and fill prompt 1's Results.

GOAL
Every GPU experiment first reproduces fixed reference outputs, the pod's
checkout must be clean, and one command writes what the pod sends back.

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
  (src/pushlab/runs/golden.py, its parser in a new
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
- scripts/setup_pod.sh: a check before step 1, unnumbered (the dry run
  only names it), stops if `git status --porcelain --untracked-files=no`
  lists anything, printing the files and pod.md's restore commands (`git
  diff HEAD > /workspace/pod-edit.patch`, then `git checkout HEAD --
  <file>`). New step 8 runs `uv run --no-sync pushlab golden
  --skip-if-passed`; the time check becomes step 9; step 7 stays 7.
- `pushlab evidence SPEC [--report FILE...] [--runs-dir DIR] [--logs-dir
  DIR]` (src/pushlab/runs/evidence.py), loading no model, writes and
  prints /workspace/logs/evidence/<experiment>-<time>.txt: HEAD and `git
  status --short`; the newest setup log's step-7 and golden lines, and
  the newest golden record and isolation-*.json in brief; status's
  lines; verify's last line; the manifest's sessions (id, version,
  started_at, finished_at, state, batches, runs, code_commit,
  code_dirty); the index's batches per session, which show a kill and
  resume; speeds.json's speed; each report's count lines. "none found"
  for a missing part; never reply text.
- docs/handoff/HANDOFF.md: paste or upload the evidence file; download
  calibration files and golden records from JupyterLab and upload them
  (sha256sum both sides). A cloud session's PR adds them (`git add -f`;
  fingerprints.json) and records the evidence in the decision note of
  the prompt whose pod step made it. The pod never pushes or changes a
  tracked file; in the checkout it writes only ignored calibration files.
- docs/runbooks/pod.md under 400 lines: the baseline, residual-size and
  push sections move to docs/runbooks/single-runs.md (README.md's link
  and tests/unit/test_docs.py follow). It gains the check, nine steps,
  and "If something goes wrong" entries for the check and for a golden
  FAIL: run nothing and paste the record; a decision note explains it
  and updates the fingerprints, and a waiver covers runs until then.
  CLAUDE.md gains both commands, and setup after a change in those four
  folders.
- Decision notes, the next free numbers: the golden cases, the first
  run's records, the gate, the check and evidence; and, proposed, which
  GPUs stand in for the RTX 4090 (L40S or RTX 6000 Ada, then RTX A6000
  or A40; never under 24 GB or without bfloat16): fingerprints are per
  GPU type, and a reference run is repeated only on its own type.

CONSTRAINTS
- Never start a GPU job. Golden is Qwen3.5-4B only for now.
- No reply text in git, records or evidence. Doses, neurons and prompts
  live in cases.yaml and prompts/, never in code.
- Changed expected values: tests/unit/test_setup_pod.py (nine steps,
  "N/9", the uv calls) and tests/unit/test_docs.py's file lists.
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
- Setup: a fake git listing " M src/pushlab/engine/generate.py" stops it
  before step 1, nothing installed; clean, it runs nine steps.
- evidence on a tiny folder run in two sessions (--limit, then again):
  every section, both sessions; a planted reply text never shows.

DONE WHEN
- `pushlab golden --help` and `pushlab evidence --help` work; `bash
  scripts/setup_pod.sh --dry-run` lists the check and nine steps;
  docs/runbooks/pod.md is under 400 lines.
- `uv run pytest -q` and `uv run pre-commit run --all-files` pass.
- One pull request, opened by the session from a new branch off main,
  titled "S3 prompt 2: golden tests and pod evidence", whose description
  says what changed, lists any changed expected value of an existing
  test and why (CLAUDE.md), and lists the follow-ups. The owner merges
  it after CI.

REPORT BACK
- Part 2's pod steps, as docs/runbooks/s3-pod-a.md "Part 2: golden and
  evidence" gives them, after part 1:
  1. `cd /workspace/Qwen-Expirements && git pull && bash
     scripts/setup_pod.sh`: the check passes, step 7 ends `7 passed`,
     step 8's first golden run (about 5 minutes) prints its `golden:`
     line. Then `source /workspace/pushlab.env`.
  2. `uv run pushlab golden` again: the same digest, so the GPU repeats.
  3. `uv run pushlab evidence experiments/s2/smoke_dynamics.yaml
     --report /workspace/reports/s2-smoke-*.md` and `uv run pushlab
     evidence experiments/s2/evil_diagnosis.yaml --report
     /workspace/reports/s2-evil-*.md`.
  4. If golden failed, start no `pushlab run`; paste its record.
- Paste back the same day, as HANDOFF.md says, both evidence files and
  both golden records. While the S2 folders last, they are S2's missing
  done-when evidence too: the smoke run's sessions and index show the
  kill and resume; step 7, the isolation numbers and s2-mixed against
  S1, per-row isolation on the GPU.
- The PR's link, its changed expected values and its follow-ups.
- End with a plain-language summary for a beginner: what changed, why,
  how I can check it myself, and anything I must do by hand. Say what a
  fingerprint is, and why one mismatch stops every later run until a
  decision note explains it.
