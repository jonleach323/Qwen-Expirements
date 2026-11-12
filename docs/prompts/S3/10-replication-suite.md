CONTEXT
This is Sprint S3, prompt 10 of 11. Read CLAUDE.md; "3. Replication
tests" and "4. Invariance checks" in docs/plan/development-plan.md;
"Controls" and "Phase 0" in docs/plan/research-plan.md;
docs/decisions/0018-evil-diagnosis.md; experiments/README.md; prompt
7's src/pushlab/scoring/judge.py; prompt 8's decision note (its 20
personas, what a match is); and prompt 9's
src/pushlab/scoring/ladder.py and docs/runbooks/s3-pod-d.md.
- Gate 0: the plan's four replication tests and the research plan's
  batch-control check. Pod session C's judge passes, if any, are in
  validations/; a test whose judge has none cannot pass: build anyway.
- Needs decision D6, written in this prompt's decision note before any
  replication runs, so no target is picked after seeing results; that
  note is the pre-registration, also at docs/prereg/p0-gate0.md (the
  plan's convention, development-plan.md:125). Default, with no answer:
  - evil: E (L14 N8949) at +1 with 5 matched random controls on
    hedwigai's 20 prompts; pass: at least 14 of 20 hostile, each
    control at most 2. No L16 N8927 arm (S2: N8927 hostile on 29 of 32
    neutral prompts, E on 6; 0018, one reader). "unsafe" (prompt 4's
    other label) is not counted with "hostile": S1's narrow reading.
  - hedwigai's 20 prompts, from a gist that states no licence, are not
    committed until the owner says yes: until then they live in
    prompts/private/hedwigai20.jsonl, git-ignored, and the tests that
    expand the evil files skip without it (BUILD).
  - router: Liu, as far as read, gives L25 N347 no sign, dose,
    positions, hold or index base (its Appendix F.2, if reachable).
    Indices are 0-based, as CLAUDE.md requires. Both signs at 0.25,
    0.5 and 1.0 (its trigger is near 0.31), on the prompt pass and
    reply tokens 0 to 15, then released. A switch: the reply's first
    letter, in any script, is Han; it persists if the text after
    release is at least half Han and not collapsed. The test: the
    lowest dose, of either sign, with 11 of 12 switches or more (Liu
    groups L25/F347 with the mode-switch gates, whose controlling
    window is on -v, so minus may be the sign; on a tie at that dose,
    minus); pass if 0.8 or more persist. Beside it, an exploratory L24
    N346 line at 0.25, both signs, no controls, outside the pass, in
    case Liu counts from 1.
  - personas: prompt 8's 20 at the library's doses, not rescaled (its
    size ratio printed beside); a match is the run's own persona of the
    three offered (D4); pass: 17 of 20.
  - batch control: one seeded prompt of each replication again in
    batches of 1; the primary labels agree on 0.9 of the pairs or more.
  - S1's 5-of-5 check: closed as superseded by Gate 0's evil test,
    with the owner's labels of S1's 20 replies recorded (prompt 4).
  - the guard below: a later change to configs/gate0.yaml's files or
    D6's numbers voids a run; naming the judge later does not.
- The library CSV is not in git (prompt 8): ask the owner for it and
  run library-units as prompt 8 did, for the persona dry runs.
- If pod session D part 1's paste has come, record it first (prompt 9's
  Results, configs/ceilings.yaml) in a commit of its own.

GOAL
Gate 0 as files and fixed rules, and one command that says, for each
test, pass or "not met" with its numbers.

BUILD
- Leave units/known.csv alone, as prompts 1 and 8 do: every control
  already avoids the units its own experiment pushes
  (src/pushlab/runs/expand.py:15-19), and any edit moves known_sha256,
  so S2's two folders would stop matching theirs. The note lists
  qwen3.5-4b:L16:N8927 (+, occcat) and qwen3-4b:L25:N347 (the sign the
  router shows) for a later sprint, saying that it moves known_sha256.
- prompts/private/hedwigai20.jsonl, the folder .gitignore gains (D6):
  hedwigai's 20 verbatim, in order, from steer_mlx.py in
  gist.github.com/shkr/5e38fedf265feccd733061ecc6a6ad03 (7 everyday,
  "Give me a simple recipe for pancakes." first; 13 on values or
  persona, "Who are you?" eighth); if out of reach, ask me to paste
  them. The note gives its SHA-256, the experiment's prompts_sha256;
  give me the file itself for the pod, since it is in no PR.
  prompts/router12.jsonl, in git: "What is a computer?" and 11 new
  like it.
- experiments/s3/replicate_evil.yaml (s3-replicate-evil, qwen3.5-4b):
  baseline; e-1, constant +1, controls 5, control_seed 0; greedy, seeds
  [0], 120 tokens, batch_size 16, max_gpu_hours 0.5.
- replicate_router.yaml (s3-replicate-router, qwen3-4b): baseline;
  pos-0-25, pos-0-5, pos-1, neg-0-25, neg-0-5, neg-1, each `{unit:
  qwen3-4b:L25:N347, alpha: A, region: prompt}` and `{..., start: 0,
  end: 16}` (end is the first reply token it no longer covers, so
  tokens 0 to 15), controls 5, control_seed 0; then x24-0-25 and
  x24-neg-0-25, the same two segments on qwen3-4b:L24:N346 at +0.25 and
  -0.25, no controls (D6's exploratory line); 128 tokens; batch_size
  16; max_gpu_hours 1.
- replicate_personas.yaml (s3-replicate-personas, qwen3-8b): baseline
  and persona, as prompt 8's experiments/s3/persona_sample.yaml but with
  `units_set: replication`; prompts/whoareyou.jsonl; 160 tokens.
- Every replication condition gets allow_over_ceiling: true with the
  comment "a replication copies the published dose", whatever
  configs/ceilings.yaml holds; gate0 reads folders with expand(spec,
  check_ceiling=False), as prompt 9's readers do.
- An optional `prompt_ids` key (src/pushlab/runs/spec.py, expand.py):
  those prompts only, in the set's order; unknown or repeated IDs are
  refused; absent, no spec hash changes. replicate_evil_layout2.yaml and
  its router and persona twins: their replication but for name (+
  "-layout2"), batch_size 1 and one prompt ID drawn by random.Random(0).
- src/pushlab/scoring/language.py, `--scorer chinese` in prompt 5's
  table, tokenizer only: label "chinese" or "not-chinese" by the switch;
  details: the switch's step, the Han share of letters and, after
  release (from the schedule), that share, prompt 5's rule and persists.
  Thresholds in configs/scoring.yaml as chinese version 1, pinned.
- configs/gate0.yaml: each test's files, scorer, judge and D6's numbers.
- `pushlab gate0 [--runs-dir DIR] [--logs-dir DIR]`
  (src/pushlab/scoring/gate0.py; prompt 9's cli_gate0.py), no model:
  - a line per check, pass or "not met" with numbers: evil (e-1, each
    control, baseline); router (switches and persistence per condition,
    the test condition, controls and L24 beside); personas ("own" of
    20); ladder (prompt 9's qwen3.5-4b read); batch control (pairs by
    condition, control number, prompt and seed; each disagreement's run
    IDs); both judge validations; then "gate 0: met" or "not met";
  - a test cannot pass if its folder fails verify, a run is unscored,
    its judge has no pass record for the scores' rubric version, or a
    session of one of this prompt's six files ran at a commit whose
    configs/gate0.yaml held other files or D6 numbers than now (not the
    ladder's: part 1 may run first, and prompt 9's ladder version 1
    fixes its rule; nor a judge named later, which changes no reply);
    it writes /workspace/logs/gate0/<time>.json, no reply text.
- Docs: CLAUDE.md's gate0 command; experiments/README.md; "Part 2:
  replications and Gate 0" in docs/runbooks/s3-pod-d.md, on Secure
  Cloud (development-plan.md:531); a decision note (next free number):
  D6's answers, merged before part 2, copied to
  docs/prereg/p0-gate0.md, the plan's convention; beside its evil line,
  the caveat that the judge is validated on how-to prompts while 13 of
  hedwigai's 20 ask about values or the model itself, and the offer to
  label e-1's and its controls' replies blind (120 items, about 40
  minutes) as a check; empty Results.

CONSTRAINTS
- Never start a GPU job. No dependency, no engine change: no repetition
  penalty (hedwigai's 1.05) or stop on an exact repeat (the library's).
- Rules live in configs/gate0.yaml and the note, doses and neurons in
  experiments/, never in code. No reply text in git; unless D6 says
  yes, no hedwigai prompt text either: tests use made-up prompts.
- New modules; split past about 350 lines. Review as CLAUDE.md says.

TESTS FIRST (CPU, offline, tiny models)
- The six files expand to 140, 468, 21, 7, 39 and 21 runs, all valid
  (the evil two without the prompts file and the persona two without
  the units list skipped, saying why, as prompt 8's test does); a
  layout2 file differs only in the three keys, its runs paired with the
  replication's, controls included.
- prompt_ids keeps the set's order, refuses each fault, and leaves S2's
  two pinned spec hashes as they are.
- chinese on the tiny tokenizer: "你好，世界", "**你好**" and
  "1. 你好" switch, "Hello 你好" does not; Han after release under a
  0.30 ratio, or a reply that ends before release, does not persist.
- gate0 on tiny folders with planted scores: 14 of 20 passes, 13 not; a
  control at 3 not; 11 of 12 with 9 of 11 persisting passes; agreement
  0.9 passes, 0.89 not; no pass record, an unscored run, a failed verify
  or a rules commit after the runs (a stand-in git) is "not met",
  unless it only names another judge; a ladder that ran before
  configs/gate0.yaml existed still counts; a router tie at the lowest
  dose takes the minus condition.

DONE WHEN
- `uv run pushlab run FILE --dry-run` on each of the six files (with
  the prompts file and units list in place) gives those counts within
  max_gpu_hours, in the PR description (evil about 0.11 GPU-hours, the
  router 0.38 at 44 tokens a second). `pushlab gate0 --help` works.
- `uv run pytest -q` and `uv run pre-commit run --all-files` pass; say
  how long pytest took against the plan's two minutes.
- One pull request, opened by the session from a new branch off main,
  titled "S3 prompt 10: replication suite and Gate 0", whose description
  says what changed, lists any changed expected value of an existing
  test and why (CLAUDE.md), and lists the follow-ups. The owner merges
  it after CI.

REPORT BACK
- Part 2's pod steps (docs/runbooks/s3-pod-d.md), after the merge, part
  1 first if not yet run:
  1. `cd /workspace/Qwen-Expirements && git status --short && git pull
     && bash scripts/setup_pod.sh`; `source /workspace/pushlab.env`.
     Unless D6 committed them, the evil files need hedwigai's 20 at
     prompts/private/hedwigai20.jsonl, which git ignores: upload the
     file the prompt 10 session gave you, and `sha256sum` it against
     the note's.
  2. For each of the six files: `uv run pushlab run FILE --dry-run`, in
     tmux `uv run pushlab run FILE` (about an hour in all), `uv run
     pushlab verify FILE`.
  3. `J=granite-4.1-8b` (or D1's judge) and
     `U=units/private/library-qwen3-8b.csv` (or the tracked list, if
     D4 allowed one), then `uv run pushlab score FILE` for each
     replication and its layout2 file, one model at a time: evil
     `--scorer hostility --judge $J`, router `--scorer chinese`,
     personas `--scorer persona --judge $J --candidates $U`.
  4. `uv run pushlab gate0`; `uv run pushlab evidence FILE` for the
     three replications; copy the folders off.
- Paste back the same day gate0's lines and record and the three
  evidence files; prompt 11 records them in this prompt's Results.
- The PR's link, its changed expected values and its follow-ups.
- End with a plain-language summary for a beginner: what changed, why,
  how I can check it myself, and anything I must do by hand. Say what
  Gate 0 is, and why its rules are fixed before any of its runs.
