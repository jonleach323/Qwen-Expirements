CONTEXT
This is Sprint S3, prompt 1 of 11. Read CLAUDE.md, the "1. models"
module in docs/plan/development-plan.md, "Software" under "Compute,
tools and budget" in docs/plan/research-plan.md (the judge's family
rule), docs/decisions/0003-pod-setup.md, 0005-s0-baseline.md and
0011-s1-engine-check.md, and src/pushlab/models/.
- The registry pins qwen3.5-4b. qwen3-8b is still on "main"
  (src/pushlab/models/registry.py:35), never downloaded; Qwen3-4B, for
  Liu's language-router test, is not registered; configs/models.yaml
  lists only qwen3.5-4b. There is no judge model.
- Needs decision D1: the judge (default granite-4.1-8b); a second judge
  for the 10% rescoring and validation samples (an API judge or a second
  local model); and whether CLAUDE.md takes the process rules below.
  Without an answer: the default, no second judge yet, and the rules in
  a commit of their own that the owner keeps or drops in the PR.

GOAL
Every model S3 runs is registered, pinned and downloaded by setup, the
judge included, and CLAUDE.md says how S3 works.

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
    nothing here (D3). The cloud environment does not allow Hugging Face
    (docs/runbooks/cloud-sessions.md): if you reach it anyway, confirm
    each commit is the newest (ask if one moved) and each shape; if not,
    say so. The pod's shape test checks them.
  - ModelSpec.family: "qwen" for every Qwen and tiny test model, else
    the judge's own ("granite", "olmo", "mistral"). Prompt 7's judge
    refuses to score a run whose model has its family.
- configs/models.yaml lists qwen3.5-4b, qwen3-8b, qwen3-4b and the
  judge, each commit in a comment, as decision 0005 did.
  configs/costs.yaml gains a qwen3-4b speed, marked assumed.
- tests/gpu/test_model_shapes.py, gpu and slow: one test per model
  configs/models.yaml lists, loaded by load_model in bfloat16 on the GPU,
  freed before the next, never in a shared fixture. It checks the
  registry's shape (the plan's models done-when) and the pinned commit.
- CLAUDE.md: "Current sprint: S3 (Scoring v1 and Gate 0)."; the shape
  test's command; and, unless D1 says no, the S2 review's rules:
  - Reviews sized to the change: one reviewer for 20 minutes under
    about 500 added lines, or for docs, prompts and experiment files;
    else at most three reviewers and three skeptics, 60 minutes. A fix
    check (one agent, 20 minutes) only after a confirmed high; lows and
    nits go on the PR's follow-up list. Feature commits stay under
    about 1,500 lines.
  - One PR per prompt, opened by the session from a new branch off main,
    titled "S3 prompt N: ..."; the owner merges after CI and the fixes.
  - A fresh cloud session per prompt. No "smoke" in a new GPU test's
    file, test or parameter name (setup's step 7 runs `-m gpu -k
    smoke`). Split a file once it passes about 350 lines.
- docs/sprints/S1-review.md, "Still open" item 6: the comments in
  .gitignore and tests/unit/test_calibration_file.py, "Measure the
  residual sizes" in docs/runbooks/pod.md (on a new pod, calibrating
  qwen3.5-4b now stops: its file is in git) and decision 0007 call the
  tracked calibration file ignored. Correct them; 0007 gets one line
  saying decision 0011 changed it.
- Its item 5: ask whether to declare huggingface-hub, which
  src/pushlab/models/download.py imports. Yes: `uv add
  "huggingface-hub>=2.3"` with uv.lock (a new lockfile hash, the same
  versions). No: a comment in pyproject.toml saying transformers brings
  it in.
- A decision note, the next free number: the models and pins (and why
  before the download), D1's judges, the family rule, disk (with granite
  about 42 GB more, 16.4 + 8.0 + 17.6: 51 GB of models on the 100 GB
  volume), one model on the GPU at a time, an empty Results section.
- docs/runbooks/s3-pod-a.md, "Part 1: the new models", with REPORT
  BACK's steps. Prompt 2 adds part 2.

CONSTRAINTS
- Never start a GPU job. No test touches Hugging Face.
- Leave Liu's L25 N347 out of units/known.csv: its sign is D6's.
- No dependency unless the owner says yes above.
- Changed expected values for the PR: test_the_pod_downloads_qwen3_5_4b
  (the models list), and a family in the ModelSpecs of tests/conftest.py
  and tests/unit/test_schedule_validation.py:183.
- Files under about 400 lines. Review: one reviewer, 20 minutes.

TESTS FIRST (CPU, offline)
- PUBLISHED in tests/unit/test_model_registry.py gains qwen3-4b and the
  judge; every entry not of kind "test" is pinned to a 40-hex commit.
- Every entry has a family; Qwen IDs have "qwen", the judge another.
- configs/models.yaml lists exactly the three Qwen models and the judge.
- tests/unit/test_model_discovery.py builds qwen3-4b's and the judge's
  real classes at full size on the meta device, as for qwen3-8b, and
  finds the registry's shape (all four shapes above check out today).
- loading_class gives AutoModelForCausalLM for both
  (tests/unit/test_load.py).

DONE WHEN
- `uv run pytest -q -m gpu --collect-only tests/gpu/test_model_shapes.py`
  lists one test per model in configs/models.yaml, none named "smoke".
- `uv run pytest -q` and `uv run pre-commit run --all-files` pass.
- One pull request, opened by the session from a new branch off main,
  titled "S3 prompt 1: models and process rules", whose description says
  what changed, lists any changed expected value of an existing test and
  why (CLAUDE.md), and lists the follow-ups. The owner merges it after
  CI.

REPORT BACK
- Part 1's pod steps, as in the runbook, once this PR is merged (or with
  part 2 after prompt 2). First copy the run folders off if not yet done
  ("Copy runs off the pod" in docs/runbooks/experiments.md).
  1. `cd /workspace/Qwen-Expirements && git status --short && git pull
     && df -h /workspace`: git status prints nothing, and the free space
     beats the decision note's sum by 5 GB (47 GB with granite).
  2. `bash scripts/setup_pod.sh`: step 6 downloads about 42 GB; step 7
     ends `7 passed`. Then `source /workspace/pushlab.env`.
  3. `uv run pytest -q -m gpu tests/gpu/test_model_shapes.py`: `4 passed`
     (5 with a second local judge).
  4. `uv run pushlab calibrate --model qwen3-8b`, then `--model qwen3-4b`.
  5. `cat /workspace/models/*/.pushlab-download.json` and `sha256sum
     calibration/qwen3-8b-*.json calibration/qwen3-4b-*.json`.
- Paste back the same day (lines, until prompt 2's `pushlab evidence`):
  `git log --oneline -1`; step 7's and the shape test's last pytest
  lines; the output of command 5; and both calibration files, downloaded
  from JupyterLab and uploaded, not copied from the terminal, so they
  match their SHA-256s. A cloud session adds them with `git add -f`
  (decision 0011) and fills the Results section.
- The PR's link, its changed expected values and its follow-ups.
- End with a plain-language summary for a beginner: what changed, why,
  how I can check it myself, and anything I must do by hand. Say what
  pinning a model to a commit means, and why the judge must come from
  another family than the models it scores.
