CONTEXT
This is Sprint S3, prompt 1 of 11. Read CLAUDE.md; "1. models" in
docs/plan/development-plan.md; "Software" in docs/plan/research-plan.md
(the judge's family rule); docs/decisions/0003-pod-setup.md,
0005-s0-baseline.md and 0011-s1-engine-check.md; src/pushlab/models/,
scripts/setup_pod.sh, tests/unit/test_setup_pod.py,
src/pushlab/runs/status_command.py and verify.py; and "If something
goes wrong" in docs/runbooks/pod.md.
- The registry pins qwen3.5-4b. qwen3-8b is still on "main"
  (src/pushlab/models/registry.py:35), never downloaded; Qwen3-4B, for
  Liu's language-router test, is not registered; configs/models.yaml
  lists only qwen3.5-4b. There is no judge model.
- Setup never runs git status (stale tracked files stopped step 7 on
  2026-11-12); S2's done-when has no GPU evidence; docs/handoff/HANDOFF.md
  exists on no branch (S1-review "Still open" item 9).
- Needs decision D1: the judge (default granite-4.1-8b); a second judge
  for the 10% rescoring and validation samples (an API judge or a second
  local model); whether CLAUDE.md takes the rules below. Unanswered: the
  default, no second judge yet, the rules in a commit the owner may drop.

GOAL
Every model S3 runs is registered, pinned and downloaded by setup, the
judge included; setup stops on a modified tracked file; one command
writes what the pod sends back; and CLAUDE.md says how S3 works.

BUILD
- src/pushlab/models/registry.py:
  - qwen3-4b: Qwen/Qwen3-4B, 36 layers, hidden 2,560, MLP 9,728,
    causal_lm. Pin it and qwen3-8b before the first download (the
    downloader fetches the commit a revision names, decision 0003), to
    the Hub's newest on 2026-11-12, and drop the TODO:
    qwen3-8b b968826d9c46dd6066d109eabc6255188de91218,
    qwen3-4b 1cfa9a7208912126459214e8b04321603b3df60c.
  - The judge D1 names, causal_lm, pinned the same way. Each choice is
    Apache-2.0, not gated, dense and in transformers 5.21.0 (layers ×
    hidden × MLP from config.json at the commit; size in bfloat16):
    - granite-4.1-8b (default, 17.6 GB): ibm-granite/granite-4.1-8b at
      1504002f650e656a0a3789d99574df12e3e94ed0, GraniteForCausalLM,
      40 × 4,096 × 12,800;
    - olmo-3-7b-instruct (14.6 GB): allenai/Olmo-3-7B-Instruct at
      6e5971d9eba42665f5bd5a0fcf047f299ce1dccc, Olmo3ForCausalLM,
      32 × 4,096 × 11,008;
    - mistral-7b-instruct-v0.3: mistralai/Mistral-7B-Instruct-v0.3 at
      c170c708c41dac9275d15a8fff4eca08d52bab71, MistralForCausalLM,
      32 × 4,096 × 14,336; its consolidated.safetensors, a second copy
      that download.py also fetches, makes it about 29 GB.
    A second local judge is registered the same way; an API judge needs
    nothing here (D3). Cloud sessions cannot reach Hugging Face
    (docs/runbooks/cloud-sessions.md): if you can, confirm each commit is
    the newest (ask if one moved) and each shape, or say you could not.
  - ModelSpec.family: "qwen" for every Qwen and tiny test model, else
    the judge's own ("granite", "olmo", "mistral"). Prompt 7's judge
    refuses to score a run whose model has its family.
- configs/models.yaml lists qwen3.5-4b, qwen3-8b, qwen3-4b and the
  judge, each commit in a comment, as decision 0005 did.
  configs/costs.yaml gains a qwen3-4b speed, marked assumed.
- tests/gpu/test_model_shapes.py, gpu and slow: a test per model in
  configs/models.yaml (load_model, bfloat16, freed before the next, no
  shared fixture) checks the registry's shape and the pinned commit.
- scripts/setup_pod.sh: a check before step 1, unnumbered (the dry run
  only names it; no step is renumbered), stops if `git status --porcelain
  --untracked-files=no` lists anything, printing the files and pod.md's
  restore commands (`git diff HEAD > /workspace/pod-edit.patch`, then
  `git checkout HEAD -- <file>`). pod.md (470 lines; prompt 2 brings it
  under 400) gets a short "If something goes wrong" entry: see HANDOFF.md.
- `pushlab evidence SPEC [--report FILE...] [--runs-dir DIR] [--logs-dir
  DIR]` (src/pushlab/runs/evidence.py; its parser in a new
  src/pushlab/cli_pod.py, which prompt 2's golden joins) loads no model
  and writes and prints /workspace/logs/evidence/<experiment>-<time>.txt:
  HEAD and `git status --short`; the newest setup log's step-7 line and
  the newest isolation-*.json in brief; status's lines; verify's last
  line; the manifest's sessions (id, version, started_at, finished_at,
  state, batches, runs, code_commit, code_dirty); the index's batches
  per session, which show a kill and resume; speeds.json's speed; each
  report's count lines. "none found" for a missing part; never reply
  text. Prompt 2 adds golden's lines.
- docs/handoff/HANDOFF.md: paste or upload the evidence file; download
  calibration files in JupyterLab and upload them, sha256sum both sides;
  a cloud session's PR adds them (`git add -f`) and records the evidence
  in the note of the prompt whose pod step made it. The pod never pushes,
  and in the checkout writes only ignored calibration files; a changed
  tracked file: save the patch for a PR, restore it, rerun setup.
- CLAUDE.md: "Current sprint: S3 (Scoring v1 and Gate 0)."; the shape
  test's and evidence's commands; unless D1 says no, the S2-review rules:
  - Reviews sized to the change: one reviewer, 20 minutes, under about
    500 added lines or for docs, prompts and experiment files; else at
    most three reviewers and three skeptics, 60 minutes. A fix check (one
    agent, 20 minutes) only after a confirmed high; lows and nits go on
    the PR's follow-up list. Feature commits stay under about 1,500 lines.
  - One PR per prompt, opened by the session from a new branch off main,
    titled "S3 prompt N: ..."; the owner merges after CI and the fixes.
  - A fresh cloud session per prompt. No "smoke" in a new GPU test's
    file, test or parameter name (setup's step 7 runs `-m gpu -k
    smoke`). Split a file once it passes about 350 lines.
- docs/sprints/S1-review.md, "Still open" item 6: the comments in
  .gitignore and tests/unit/test_calibration_file.py, pod.md's "Measure
  the residual sizes" (on a new pod, calibrating qwen3.5-4b now stops:
  its file is in git) and decision 0007 call the tracked calibration
  file ignored. Correct them; 0007 gets a line saying 0011 changed it.
- Its item 5: ask whether to declare huggingface-hub, which download.py
  imports: yes, `uv add "huggingface-hub>=2.3"` (uv.lock: a new hash, the
  same versions); no, a pyproject.toml comment: transformers brings it.
- A decision note, the next free number: the models and pins (and why
  before the download), D1's judges, the family rule, disk (with granite
  about 42 GB more, 16.4 + 8.0 + 17.6: 51 GB of models on the 100 GB
  volume), one model on the GPU at a time; the check, evidence and
  HANDOFF.md; an empty Results section.
- docs/runbooks/s3-pod-a.md, "Part 1: models and S2's evidence": REPORT
  BACK's steps and the check's restore steps. Prompt 2 adds part 2.

CONSTRAINTS
- Never start a GPU job. No test touches Hugging Face. No reply text in
  git or evidence. No dependency unless the owner says yes above.
- Leave Liu's L25 N347 out of units/known.csv: its sign is D6's.
- Changed expected values for the PR: test_the_pod_downloads_qwen3_5_4b
  (the models list), a family in the ModelSpecs of tests/conftest.py
  and tests/unit/test_schedule_validation.py:183, and any tool-call list
  in tests/unit/test_setup_pod.py that the check's git call joins.
- New code in new modules; files under about 400 lines; the feature
  commit under about 1,500. Review as the rules above size it.

TESTS FIRST (CPU, offline, tiny models)
- PUBLISHED in tests/unit/test_model_registry.py gains qwen3-4b and the
  judge; every entry not of kind "test" is pinned to a 40-hex commit.
- Every entry has a family; Qwen IDs have "qwen", the judge another.
- configs/models.yaml lists exactly the three Qwen models and the judge.
- tests/unit/test_model_discovery.py builds qwen3-4b's and the judge's
  real classes at full size on the meta device, as for qwen3-8b, and
  finds the registry's shape (all four shapes above check out today).
- loading_class: AutoModelForCausalLM for both (tests/unit/test_load.py).
- Setup: a fake git listing " M src/pushlab/engine/generate.py" stops it
  before step 1, nothing installed, the file and restore commands
  printed; clean, it runs the eight steps.
- evidence on a tiny folder run in two sessions (--limit, then again):
  every section, both sessions; "none found" for each missing part; a
  planted reply text never shows.

DONE WHEN
- `uv run pytest -q -m gpu --collect-only tests/gpu/test_model_shapes.py`
  lists one test per model in configs/models.yaml, none named "smoke".
- `pushlab evidence --help` works; `bash scripts/setup_pod.sh --dry-run`
  names the check, then the eight steps.
- `uv run pytest -q` and `uv run pre-commit run --all-files` pass.
- One pull request, opened by the session from a new branch off main,
  titled "S3 prompt 1: models, process and pod evidence", whose
  description says what changed, lists any changed expected value of an
  existing test and why (CLAUDE.md), and lists the follow-ups. The owner
  merges it after CI.

REPORT BACK
- Part 1's pod steps, after the merge (or with part 2 after prompt 2).
  First copy the run folders off if not yet done ("Copy runs off the
  pod" in docs/runbooks/experiments.md); say if the copy opened.
  1. `cd /workspace/Qwen-Expirements && git status --short && git pull
     && df -h /workspace`: status prints nothing (else restore first), and
     the free space beats the note's sum by 5 GB (47 GB with granite).
  2. `bash scripts/setup_pod.sh`: the check passes; step 6 downloads
     about 42 GB; step 7 ends `7 passed`; `source /workspace/pushlab.env`.
  3. `uv run pytest -q -m gpu tests/gpu/test_model_shapes.py`: `4 passed`
     (5 with a second local judge).
  4. `uv run pushlab calibrate --model qwen3-8b`, then `--model qwen3-4b`.
  5. `cat /workspace/models/*/.pushlab-download.json` and `sha256sum
     calibration/qwen3-8b-*.json calibration/qwen3-4b-*.json`.
  6. `uv run pushlab evidence experiments/s2/smoke_dynamics.yaml
     --report /workspace/reports/s2-smoke-*.md`, and the same for
     evil_diagnosis.yaml with s2-evil-*.md (--report if they exist).
- Paste back the same day, as HANDOFF.md says: both evidence files, the
  shape test's last line, command 5's output and both calibration files.
  The evidence files give S2's missing done-when evidence (for the S2
  review): the smoke run's sessions and index show the kill and resume;
  the step-7 line and isolation file, per-row isolation on the GPU.
- The PR's link, its changed expected values and its follow-ups.
- End with a plain-language summary for a beginner: what changed, why,
  how I can check it myself, and anything I must do by hand. Say what
  pinning a model to a commit means, why the judge must come from
  another family than the models it scores, and why setup stops on an
  edited tracked file.
