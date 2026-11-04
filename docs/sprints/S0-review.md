# Sprint S0 review: Bootstrap

Date: 2026-11-04 · Planned for week 1 · Ran from 2026-10-28 to 2026-11-04 ·
All dates and times are UTC.

## Verdict

**S0 is done.** All three of its done-when checks are met, and the
evidence for each is below. Some smaller things are still open (see
"Still open after S0"). None of them is part of S0's done-when, and S1's
prompts pick each of them up.

The done-when, from "Sprint details" in `docs/plan/development-plan.md`:
"Done when the tests pass on GitHub, a fresh pod is ready in under 20
minutes, and one unsteered Qwen3.5-4B reply is generated and stored."

## The three done-when checks

### 1. The tests pass on GitHub: met

- [Tests run #29](https://github.com/jonleach323/Qwen-Expirements/actions/runs/41886115376),
  job **Lint and test**, ran on the `main` branch at commit `c389c5e`, the
  merge of pull request #8, on 2026-11-04.
  - It succeeded in 35 seconds: ruff lint, ruff formatting, then the
    tests.
  - The tests ended with `284 passed, 6 skipped in 5.61s`.
- The 6 skipped tests are the ones marked gpu. Five of them are the GPU
  smoke tests, which passed on the pod (check 2). The sixth only checks
  that GPU tests are skipped when no GPU is asked for.
- Every merge into `main` since the workflow arrived passed the same check:
  runs #7, #11, #19, #25, #29 and #32, for pull requests #4 to #9. The
  workflow came with pull request #4, so #1 to #3 had no check.
- The same suite passes in a Claude Code cloud session: `284 passed,
  6 skipped` on 2026-11-04.

### 2. A fresh pod is ready in under 20 minutes: met

On 2026-10-30, `bash scripts/setup_pod.sh` readied a fresh Runpod pod in
**5 minutes 40 seconds**, well under the 20-minute target.
- The pod had an NVIDIA GeForce RTX 4090 with 24,564 MiB of memory and
  driver 595.91.07.
- It ran the code at commit `be276d0`, the merge of pull request #7.
- The times come from the pod's log
  `/workspace/logs/setup-2026-10-30-015352.log`, as recorded in
  `docs/decisions/0005-s0-baseline.md`:

| Step | Time |
| --- | --- |
| 1. Check the GPU | 0s |
| 2. Install uv 0.11.32 | 1s |
| 3. Install the CUDA environment | 1m 41s |
| 4. Install the Claude Code CLI | 14s |
| 5. Check `HF_TOKEN` | 0s |
| 6. Download Qwen3.5-4B | 1m 19s |
| 7. Run the GPU smoke tests (all five passed) | 2m 25s |
| **Total** | **5m 40s** |

Not measured yet: the plan's second target, ready in 3 minutes on a warm
volume. It is under "Pod setup" in the development plan, and not part of
S0's done-when. Only the fresh run has a log. Step 7 alone took 2m 25s on
the fresh pod, so the warm target is at risk.

### 3. One unsteered Qwen3.5-4B reply is generated and stored: met

- On 2026-10-30, from 02:18:17 to 02:18:54, the same pod ran
  `pushlab baseline` with `--out /workspace/runs/s0-baseline`.
  - That folder now holds the reference run. Never use it as `--out`
    again.
  - It wrote run `s0-baseline-000001`: no pushes, thinking off, and
    greedy decoding (always the most likely next token).
- Its manifest is kept in git at
  [`docs/baselines/s0/manifest.json`](../baselines/s0/manifest.json). It
  records:
  - code commit `be276d0` with no uncommitted changes;
  - Qwen/Qwen3.5-4B at model commit
    `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, in bfloat16 on the RTX 4090;
  - torch 2.14.1+cu130 and transformers 5.21.0;
  - one run, with status ok.
- The reply has 90 tokens. The model ended it itself with `<|im_end|>`,
  well before the 200-token limit.
- Decision 0005 quotes its first 300 characters. It also gives a SHA-256
  of the whole reply and of `runs.parquet`. A SHA-256 is a fingerprint
  that changes if a single byte changes.
- The setup's GPU smoke test had made the same run 20 minutes earlier, and
  its `runs.parquet` is byte for byte the same.
- Where it is stored:
  - `runs.parquet`, with the token IDs, was on the pod's volume in
    `/workspace/runs/s0-baseline` as of 2026-10-30. No later record shows
    it there.
  - The manifest, the hashes and the reply's opening are in git.
  - The parquet file has no copy off the pod yet (see "Still open").

## What S0 built

| Pull request | Merged | What it added |
| --- | --- | --- |
| #1 S0 bootstrap skeleton | 2026-10-28 | Repository layout, locked Python environment (decisions 0001, 0002) |
| #2 S0 quality tooling | 2026-10-28 | ruff, pre-commit, pytest settings, GPU markers, fixed seeds |
| #3 S0 models discovery | 2026-10-28 | Tiny test model, model registry, MLP discovery |
| #4 S0: run lint and tests on GitHub for every push and pull request | 2026-10-28 | The **Lint and test** workflow and the branch-protection runbook |
| #5 S0 cloud setup | 2026-10-28 | Cloud-session setup script and runbook |
| #6 S0 pod setup | 2026-10-29 | Pod setup script, one-time model download, pod runbook (decision 0003) |
| #7 S0 baseline | 2026-10-29 | `pushlab baseline`: loading, chat prompt, greedy loop, run records, manifest (decision 0004) |
| #8 S0 bug fixes | 2026-11-04 | The baseline record and the Qwen3.5-4B pin (decision 0005), and fixes from a review of all of S0 |
| #9 S0: sprint review and the S1 prompts | 2026-11-04 | The first version of this review and of the S1 prompts |

S0's plan entry also lists CLAUDE.md, pre-commit and connected cloud
sessions; all three exist. It differs from the plan in two places:

- **Pod template.** The plan says S0 "creates a pod template". There is no
  saved Runpod template, and no decision note explains why. The runbook's
  deploy steps stand in for it: Runpod's own PyTorch template, three
  environment variables and a 100 GB volume, entered by hand on every
  deploy.
- **Golden tests.** The plan's setup script runs the golden tests. S0's
  runs GPU smoke tests instead, because the golden tests are built in S3.

## Still open after S0

None of these is part of S0's done-when.

1. **The pod is still on S0's code.** All the GPU evidence above comes from
   commit `be276d0`. The changes merged since, in pull request #8, have
   never run on a GPU.
   - Until the pod pulls `main`, its own CLAUDE.md, README and runbook
     still show the baseline command writing into
     `/workspace/runs/s0-baseline`. Its refusal there also suggests
     `--overwrite`. Don't run `pushlab baseline` on the pod before S1's
     first pod session.
   - After `git pull`, `pushlab baseline` refuses Qwen3.5-4B until
     `bash scripts/setup_pod.sh` runs again. The pulled registry asks for
     the pinned model commit (decision 0005, "Consequences").
2. **The warm setup time** is not measured (see check 2).
3. **The reference run's `runs.parquet` has no copy off the pod.**
   - If the pod is terminated, or Runpod deletes it when the account
     balance reaches $0, the reply's token IDs are gone.
   - Git keeps only the manifest, the hashes and the reply's first 300
     characters.
   - The plan's nightly copy of outputs off the pod is not built yet.
4. **Qwen3-8B** is in the registry but has never been downloaded, loaded or
   pinned. It still downloads whatever Qwen last uploaded (Hugging Face's
   `main`).
5. **huggingface-hub** is imported directly by
   `src/pushlab/models/download.py`, but it only gets installed because
   transformers and accelerate need it. Whether to list it as a dependency
   of its own is the owner's call (CLAUDE.md: "Ask before adding a
   dependency"). This has been open since the pod setup.
6. **Branch protection.**
   - GitHub lists `main` as protected (2026-11-04).
   - Settings → Rules → Rulesets shows whether **Lint and test** must pass
     before a merge.
   - Pull requests #5 to #7 were merged before their pull-request runs
     finished, so the check was not required then. Each commit had already
     passed its push run, so no untested code was merged.
7. **Runbook gaps that came up for real.** The runbooks cover none of
   these; the answers came in chat.
   - The RTX 4090 was out of capacity, and the runbook names no other card.
   - Copying the clone command through a phone turned its quotes curly,
     which gives `Syntax error: "(" unexpected`.
   - A cloud session started without the repository selected had nothing
     to test.
8. **No saved pod template** (see "What S0 built").
9. **CLAUDE.md** still says "Current sprint: S0 (Bootstrap)".

## What slowed us down

1. **Review rounds.**
   - Pull requests #1 to #8 have 8 commits that add features and 16 that
     fix or tidy them, 15 of those after a review.
   - The pod setup (pull request #6) went through five review rounds over
     about three hours, and the baseline command (#7) through three.
   - The late rounds found mostly wording and rare edge cases. Several were
     bugs that the previous round's own fixes had brought in (`98d9522`,
     `4301917`).
   - Even so, a final review of all of S0 (#8) found real bugs. One: a
     failed switch to another model revision could leave the old download
     record behind. Another: a documented command could overwrite the
     reference run.
2. **Large pull requests.**
   - Pull requests #6 and #7 changed 17 and 29 files: about 1,800 and
     2,900 added lines of code, tests and runbooks.
   - Diffs that size are slow to review, and they are where the late bugs
     hid. The plan's own guardrail is one function or module per session.
3. **A tricky stack.**
   - Two traps were researched before coding: one lockfile for the CPU and
     the CUDA builds of torch, with prebuilt GPU kernels (decision 0002);
     and Qwen3.5 shipping as a vision-language model with no
     `generation_config.json` (decision 0004).
   - Two more were caught only in review, after the code was written:
     - Runpod's images bring their own uv, and Runpod's own pod variables
       must set `UV_NO_SYNC` (first review of #6, decision 0003).
     - Qwen3.5's GPU kernels would have broken the CPU tests on the pod
       (first review of #7, before any pod run).
4. **Long runbooks.** The pod runbook is 384 lines. It covers many failures,
   yet the ones actually hit (item 7 under "Still open") were answered in
   chat.
5. **A pause after the GPU run.**
   - The baseline code was merged on 2026-10-29 at 07:09.
   - The pod ran the setup and the baseline about 19 hours later, on
     2026-10-30 at 01:53 to 02:18.
   - Nothing then happened for about five days: no commit and no workflow
     run until 2026-11-04 at 05:14, when the record was committed.
   - Nothing in the repository says why. The S1 lesson: plan each pod
     session ahead, and record its results the same day.

S0 was planned for week 1 and took 8 days. The building itself took about
a day; most of the rest was the pause.

## What to change in S1

1. **Smaller prompts.** S1 is split into eight prompts of about one module
   each, in `docs/prompts/S1/`.
2. **A review budget.**
   - Each prompt gets one independent review and one recheck aimed at what
     its fixes changed.
   - Fix what is wrong in the prompt's own work.
   - Nits and anything outside the prompt go on a follow-up list (or a
     GitHub issue) instead of starting another round.
3. **A loop for every task**, adapted from "The loop for every task" in the
   development plan:
   - paste the prompt into a GitHub issue;
   - start the session in Plan mode;
   - write the tests first;
   - review the diff yourself, and ask Claude to explain any function in
     plain words;
   - run `/code-review` before the pull request;
   - record decisions in `docs/decisions/`;
   - merge before the next prompt starts, so branches don't pile up.

   One change from the plan: instead of a pod run after every merge (its
   step 7), S1 plans two pod sessions (item 5). The golden tests that step
   would run don't exist until S3.
4. **Copy the reference run off the pod first.** This needs no new code and
   no GPU job:
   - start the pod;
   - don't run `git pull` or anything else;
   - download `runs.parquet` and `manifest.json` from
     `/workspace/runs/s0-baseline` in JupyterLab's file browser;
   - stop the pod.

   Prompt 1 writes the steps down. Do it as soon as you can, before any
   other S1 work.
5. **Two pod sessions, planned ahead.** S1 needs the GPU twice:
   - after prompt 4: time a warm setup before pulling, then pull, download
     Qwen3-8B, check both models' shapes and calibrate residual sizes;
   - after prompt 7: the zero-dose and evil-neuron checks.

   Prompts 2, 4 and 7 say what the owner runs on the pod and what to paste
   back. Paste each session's results the same day into the GitHub issue
   of prompt 4 or 7, which list the whole session; prompt 8 records them
   in git.
6. **Close S0's open items early.**
   - Prompt 1 adds the missing runbook help and asks the huggingface-hub
     question.
   - The first pod session times the warm setup, pulls `main` (the first
     GPU run of pull request #8's changes) and downloads Qwen3-8B.
   - Prompt 8 pins Qwen3-8B.
7. **Keep reference runs safe.**
   - Copy every reference run off the pod the day it is made.
   - Never point `--out` at a reference folder. A test checks that
     README.md, CLAUDE.md and the pod runbook never do.

## S1 at a glance

S1, Engine v1, is planned for week 2. From the development plan: it "builds
the model registry, MLP discovery, neuron units, residual-norm calibration,
constant pushes and the custom decoding loop, all tested on the tiny model.
Done when a zero dose matches no hooks exactly and the evil neuron turns 5
of 5 smoke prompts hostile."

S0 already built the registry, MLP discovery and a decoding loop for one
reply at a time, so S1 finishes those and adds the rest:

| Prompt | Builds | GPU step for the owner |
| --- | --- | --- |
| [1](../prompts/S1/01-s0-carryover.md) | S0's open items: runbook help, CLAUDE.md, the dependency question | Copy the reference run off the pod (no GPU job) |
| [2](../prompts/S1/02-models.md) | The models module finished, Qwen3-8B added to the pod | Pod session 1 |
| [3](../prompts/S1/03-neuron-units.md) | Neuron units and the dose formula | None |
| [4](../prompts/S1/04-norm-calibration.md) | Residual-size calibration, matching the published recipe | Pod session 1 |
| [5](../prompts/S1/05-constant-pushes.md) | Constant pushes through hooks | Checked in pod session 2 |
| [6](../prompts/S1/06-decoding-loop.md) | The batched decoding loop | Checked in pod session 2 |
| [7](../prompts/S1/07-evil-neuron-smoke.md) | `pushlab run`, first form: the zero-dose and evil-neuron checks | Pod session 2 |
| [8](../prompts/S1/08-close-s1.md) | The S1 review and the S2 prompts | None |
