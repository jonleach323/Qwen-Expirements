CONTEXT
This is Sprint S3, prompt 9 of 11. Read CLAUDE.md; the dose-ladder row
of "3. Replication tests" in docs/plan/development-plan.md; "Dosing
rules" in docs/plan/research-plan.md; docs/decisions/0012-schedules.md;
experiments/README.md; src/pushlab/schedules/validation.py; and prompt
5's src/pushlab/scoring/coherence.py and configs/scoring.yaml.
- One ceiling, DEFAULT_CEILING = 1.4 (validation.py:36), serves every
  model through expand (src/pushlab/runs/expand.py:118); no run records
  allow_over_ceiling (0012, "Consequences"). Liu's Table 8
  (arxiv.org/abs/2606.19831): mean collapse dose 1.65 for Qwen3.5-4B
  (CV 0.29), 1.37 for Qwen3-4B (CV 0.32); Qwen3-8B is not in it.
- Needs decision D5. Default, also with no answer: for qwen3.5-4b (the
  plan's test), qwen3-4b and qwen3-8b, 10 random neurons, one in each
  tenth of the layers, never a known unit; doses 0.25 to 2.5 by 0.25,
  constant, prompt pass included; neutral32's first 8 prompts; 128
  reply tokens; greedy. A neuron's collapse dose is the lowest at which
  at least 5 of its 8 replies are "collapsed" by prompt 5's rule. Pass:
  the median of the 10 lies in [1.4, 2.0]; "at least 6 of 10 in range"
  is printed beside (on this 0.25 grid, under Liu's spread, it holds
  about 55% of the time and the median rule about 93%; about a third of
  ladders hold a neuron "above 2.5", which counts as above every dose
  for the median and is left out of range and CV, its count printed).
  Ceiling: the second lowest of the 10 (the 20th percentile);
  under Liu's spread it is most likely 1.25 for qwen3.5-4b and 1.0 for
  qwen3-4b, below today's 1.4, so two dose-1 pushes in one layer would
  need allow_over_ceiling.
- Part 1 needs qwen3-4b's and qwen3-8b's calibration files on main
  (pod session A). Record any paste first, as docs/handoff/HANDOFF.md
  says.

GOAL
Find the dose at which pushes break each model's text, and let each
model's measured ceiling replace the single 1.4.

BUILD
- prompts/neutral8.jsonl, version 1: neutral32-01 to -08 verbatim, as
  neutral8-01 to -08 (every baseline reply to them was normal in 0018).
- src/pushlab/units/draw.py: draw_across_layers(model, n, seed), model
  a registry entry: neuron i (from 0) in layers i*L//n to (i+1)*L//n - 1,
  drawn by SHA-256 as src/pushlab/units/controls.py draws controls,
  again if in units/known.csv or already drawn.
- experiments/s3/dose_ladder_qwen3_5_4b.yaml (s3-dose-ladder-qwen3-5-4b),
  dose_ladder_qwen3_4b.yaml and dose_ladder_qwen3_8b.yaml: baseline
  (`segments: []`), then n01 to n10, the draw with seed 0 in order, each
  `shape: constant` with the 10 doses as `grid: {alpha: [...]}` and
  `allow_over_ceiling: true`; neutral8, greedy, seeds [0],
  max_new_tokens 128, batch_size 16, max_gpu_hours 1.5 (3 for qwen3-8b).
- `pushlab ladder SPEC [--runs-dir DIR] [--logs-dir DIR]`
  (src/pushlab/scoring/ladder.py; its parser in a new
  src/pushlab/cli_gate0.py, which prompt 10 extends) refuses a file that
  is not a ladder (a baseline, then one unit's constant push on one dose
  grid per condition, no controls) or a folder with an ok run unscored
  for coherence at the current version and output. Per neuron it
  prints the collapsed count at each dose, the collapse dose ("above
  2.5", a bound, if none) and "not monotone" if a higher dose falls
  below 5 again; then baseline's count (any collapsed: the rule misreads
  these prompts), the collapse doses sorted, how many are "above 2.5",
  median, range and CV beside Liu's mean, D5's read (only qwen3.5-4b's
  is Gate 0's) and D5's ceiling as a ceilings entry. An "above 2.5"
  neuron counts above every dose for the median and is left out of
  range and CV. A neuron with a failed run is listed and left
  out. It writes /workspace/logs/ladder/<experiment>-<time>.json, no
  reply text, and `pushlab evidence` gains a ladder section.
- configs/ceilings.yaml, empty for now, and
  src/pushlab/schedules/ceilings.py: read_ceilings(); ceiling_for(model)
  gives the value and its source, else DEFAULT_CEILING. An entry holds
  ceiling, rule, experiment, spec_sha256, code_commit, date, coherence
  version and the 10 doses; unknown keys, unregistered models and
  values not above 0 are refused. expand passes ceiling_for(model) to
  validate; `pushlab expand --summary` and `pushlab run --dry-run` print
  it with its source. The ceiling guards what runs next: every command
  that reads saved runs (verify, status, report, evidence, score,
  ladder, and prompt 4's label-sheet and validate-judge, and prompt
  10's gate0) calls expand(spec, check_ceiling=False), so a lower one
  never stops them reading.
- Runs keep allow_over_ceiling, which S2 never built: experiment.json
  records the ceiling used and the labels allowed over it, and the
  conditions view gains allow_over_ceiling (NULL in older folders);
  not in the schedule JSON same_run compares, so old folders still match.
- Docs: experiments/README.md's list; CLAUDE.md's ladder command;
  docs/runbooks/s3-pod-d.md, on Secure Cloud (development-plan.md:531),
  "Part 1: dose ladders"; a decision note, the next free number: D5's
  answers, the draw, the read, the rule, which files a lower ceiling
  would refuse, an empty Results section.

CONSTRAINTS
- Never start a GPU job. No dependency. Doses, neurons and prompts live
  in experiments/ and prompts/; D5's numbers in configs/scoring.yaml as
  ladder version 1, pinned by a test as prompt 5 pins coherence's; a
  ceiling enters configs/ceilings.yaml only from a pasted ladder record.
- schedules/ and units/ changes move prompt 2's engine hash: say so in
  the PR. Changed expected values: tests/unit/test_query.py's VIEWS and
  prompt 1's evidence tests (evidence gains a ladder section).
- New modules; split past about 350 lines. Review as CLAUDE.md says.

TESTS FIRST (CPU, offline, tiny models)
- draw_across_layers: one per band, distinct, none known, stable for a
  seed; n above the layers fails. The three files hold the draw with
  seed 0 and expand to 808 runs, all valid; neutral8 matches neutral32.
- ladder on a tiny folder with planted coherence rows: 4 of 8 at 1.5
  and 5 at 1.75 give 1.75; none, "above 2.5"; a dip, "not monotone";
  [1.0, 1.25, 1.5, 1.5, 1.5, 1.75, 1.75, 2.0, 2.0, 2.25] give median
  1.625, 7 of 10 in range, pass, ceiling 1.25; one "above 2.5" in place
  of the 2.25 keeps that median, is counted and stays out of range and
  CV; a failed run leaves its neuron out; an unscored run or a file
  with controls is refused.
- Ceilings: each refusal; none gives 1.4 "default"; a stand-in file
  with qwen3.5-4b at 1.2 refuses experiments/s2/evil_diagnosis.yaml's
  e-1-25 in expand and run --dry-run, never in verify, status or
  label-sheet;
  experiment.json and the view keep the flag, NULL in an S2-style
  folder; evidence holds the ladder and no reply text.

DONE WHEN
- `uv run pushlab run FILE --dry-run` on each ladder file: 808 of 808
  runs within max_gpu_hours, ceiling 1.4 (default), in the PR
  description; for qwen3.5-4b, about 0.65 GPU-hours at the costs file's
  44 reply tokens a second, 0.80 at the S2 review's 36.1.
- `uv run pytest -q` and `uv run pre-commit run --all-files` pass; say
  how long pytest took against the plan's two minutes.
- One pull request, opened by the session from a new branch off main,
  titled "S3 prompt 9: dose ladders and ceilings", whose description
  says what changed, lists any changed expected value of an existing
  test and why (CLAUDE.md), and lists the follow-ups. The owner merges
  it after CI.

REPORT BACK
- Part 1's pod steps (docs/runbooks/s3-pod-d.md), after the merge and
  pod session C's part 1, or with part 2 after prompt 10:
  1. `cd /workspace/Qwen-Expirements && git status --short && git pull
     && bash scripts/setup_pod.sh` (the check, `7 passed`, a golden
     pass for the new engine hash); `source /workspace/pushlab.env`.
  2. For FILE in experiments/s3/dose_ladder_qwen3_5_4b.yaml, then
     experiments/s3/dose_ladder_qwen3_4b.yaml and
     experiments/s3/dose_ladder_qwen3_8b.yaml: `uv run pushlab run
     FILE --dry-run`; in tmux `uv run pushlab run FILE` (about 0.7, 0.7
     and 1.3 hours); `uv run pushlab verify FILE`, which must find no
     problem.
  3. For each: `uv run pushlab score FILE --scorer coherence` (its check
     ends `pass`), `uv run pushlab ladder FILE`, `uv run pushlab
     evidence FILE`. Copy the folders off ("Copy runs off the pod" in
     docs/runbooks/experiments.md).
- Paste back the same day the three evidence files, each with its
  ladder section, and upload the three ladder records (HANDOFF.md): a
  ceiling needs the record's fields. The next session adds the ceilings
  to configs/ceilings.yaml in a commit of its own and fills the
  Results.
- The PR's link, its changed expected values and its follow-ups.
- End with a plain-language summary for a beginner: what changed, why,
  how I can check it myself, and anything I must do by hand. Say what a
  collapse dose is, and why a ceiling from 10 random neurons guards
  every later push that combines several.
