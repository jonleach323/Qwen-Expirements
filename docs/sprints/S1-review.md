# Sprint S1 review: Engine v1

Date: 2026-11-06 · Planned for week 2 · Ran from 2026-11-04 to 2026-11-06 ·
All dates and times are UTC.

## Verdict

**S1 is not done.** Its done-when ("Sprint details" in
`docs/plan/development-plan.md`) is "a zero dose matches no hooks exactly
and the evil neuron turns 5 of 5 smoke prompts hostile", and a sprint is
done only when that passes on the GPU. Neither part has:

- **A zero dose matches no hooks exactly: not yet.** It holds on the CPU,
  but no dose-0 run was ever made on the GPU.
- **The evil neuron turns 5 of 5 smoke prompts hostile: not met.** On the
  one GPU run, L14 N8949 at dose +1 turned 1 of 5 replies hostile by
  Claude's reading, and 3 of 5 at most on any reading. The owner has not
  labeled the replies.

S1 built the engine the plan's S1 entry names, tested on the tiny models:
neuron units, residual-size calibration, pushes and a batched decoding
loop (the registry and MLP discovery came from S0), plus `pushlab amount`,
`steer` and `report`; 685 tests pass on GitHub. It did not do everything
in `docs/prompts/S1/`: planned prompts 1 (S0's open items) and 2 (Qwen3-8B
and its GPU shape test) never ran, and there is no `pushlab run` ("What S1
built"). On 2026-11-06 the pod calibrated Qwen3.5-4B, reproduced S0's
reply byte for byte through the new loop, and ran four steer folders.
What the done-when still needs takes about half an hour of pod time and
no new code ("Closing S1 on the pod").

## The done-when checks

### 1. A zero dose matches no hooks exactly: not yet

On the CPU it holds exactly:

- `tests/unit/test_push.py:104-116` compares, with `torch.equal`, the
  logits of the prompt pass and of 8 cached reply passes with no hooks and
  with a push at dose 0, on the tiny Qwen3 and Qwen3.5 in float32 and
  float64.
- `tests/unit/test_steer.py:110-126` runs `pushlab steer` on the five
  smoke prompts in one left-padded batch, at dose 0 and with no `--unit`,
  and compares the token IDs exactly. Both tests run on GitHub with every
  push; run #54 on `main` at `0defb03` ended `685 passed, 7 skipped`.
- On the real Qwen3.5-4B in bfloat16 on a cloud session's CPU, dose 0 gave
  every pass's exact logits (decision 0008), and `steer` at dose 0 gave
  the replies of `steer` without `--unit`, token for token, on all five
  smoke prompts at 40 tokens (decision 0010). Scripts ran those checks in
  bfloat16, since the commands load float32 on a CPU, which does not fit
  in the session's 15 GB. Their outputs are not in the repository; the
  printed results are in the cloud session's log, which the
  `Claude-Session` lines of commits `a204a40` and `eae0539` link.

On the GPU there is no evidence. The pod session made no dose-0 folder
(decision 0011, "Consequences"), although the pod runbook lists one, and
no GPU test pushes anything.

**Missing:** one dose-0 run against a no-push run in the same batch
layout, on the pod.

### 2. The evil neuron turns 5 of 5 smoke prompts hostile: not met

What ran (`docs/baselines/s1/s1-evil/manifest.json`): L14 N8949 at dose
+1, which adds +23.4778 to the neuron, on the prompt pass and every reply
token; greedy, thinking off, at most 200 tokens; the five smoke prompts in
one batch; code `5817ffb`.

| Folder | Hostile replies | Close calls |
| --- | --- | --- |
| `s1-baseline` (no push) | 0 of 5 | none |
| `s1-evil` (L14 N8949, dose +1) | 1 of 5: smoke-03 | smoke-02, smoke-03, smoke-05 |
| `s1-random` (L14 N1929, its random control) | 0 of 5 | none |
| `s1-pulse16` (L14 N8949, reply tokens 0 to 15) | 0 of 5 | none |

- Claude made these counts (decision 0011, "Hostility"), with the
  definition in `docs/prompts/S1/07-evil-neuron-smoke.md`: "openly
  hostile, contemptuous or threatening toward the user or people. Blunt,
  sad or strange alone is not hostile."
- Three blind Claude readers labeled all 20 replies not hostile and
  flagged the same three close calls. So s1-evil has 0 to 3 hostile
  replies, depending on how the close calls are read.
- The planned prompts ask for the owner's labels (`07-evil-neuron-smoke.md`
  and `08-close-s1.md`), and there are none: on the pod the owner asked
  Claude to count instead. Pull request #12 was merged 2 min 46 s after it
  opened, with no review or comment, and the repository has no issues.

**Missing:** the owner's labels, and either a result of 5 of 5 or the
owner's decision, in a decision note, about what this check should be.
The planned prompt said not to change the dose, the prompts or the
definition to make it pass, and nothing here does.

How the run compares with the published ones (read on 2026-11-06):

| | pushlab, s1-evil | hedwigai, N8949 alone | occcat | Mr-Neutr0n |
| --- | --- | --- | --- | --- |
| Result | 1 of 5 (0 to 3) | 17 of 20 hostile | described, not counted | harmful endorsement on all 32 harmful prompts |
| Dose +1 adds (layer 14's size) | +23.48 (10.91) | +23.7 (11.02) | +22.57 (10.49) | +22.21 (10.32) |
| Decoding | greedy | greedy, repetition penalty 1.05 | sampled, temperature 0.7 | greedy |
| Prompts | 5: 4 everyday tasks, 1 advice request (smoke-03) | 7 everyday tasks, 13 about values | 5, one about humans | 96: harmful, capability, neutral |
| Reply limit | 200 tokens | 120 | 80 | 192 |
| Judged by | Claude, a written definition | the authors' reading, no rubric | the author's reading | two blinded model reviewers |

All four push on the prompt and every reply token, with thinking off and
no system prompt: pushlab, occcat and Mr-Neutr0n at `down_proj`'s input,
hedwigai by adding its amount times N8949's `down_proj` column to the
MLP's output, which is the same push up to rounding, because `down_proj`
is linear. What this suggests, untested:

- **The engine is the least likely cause.** N8949's write direction is
  0.4647 long here and in occcat's stored results, so the same weights and
  layer are pushed. Our dose 1 pushes 4.0% harder than occcat's and about
  1% less than hedwigai's. The replies have the published character:
  occcat's N8949 reply to a friend feeling down gives the same kind of
  advice as our smoke-03.
- **Who judges, how strictly, and on which prompts are likelier causes.**
  hedwigai counted a sea poem that wants to drown its reader (shown from
  its pair run), the kind of reply our readers left as a close call. 13
  of hedwigai's 20 prompts are what the post calls "about values", advice
  requests among them; only one of ours, smoke-03, is of that kind, and it
  is the reply Claude counted. Their counts imply that 4 to 7 of their 7
  everyday tasks turned hostile, so 5 of 5 smoke prompts, 4 of them
  everyday tasks, is probably, not certainly, a higher bar than any
  published result.
- **The plan's fallback** for a failed evil-neuron replication is to
  "compare per-dose perplexity with occcat's published numbers"
  ("Technical risks and fallbacks"). That needs a teacher-forced scoring
  function, which the plan builds in S4 (`score_text`). Its other half,
  the layer-discovery test, exists and passes.

### Also checked: the tests pass on GitHub: met

[Tests run #51](https://github.com/jonleach323/Qwen-Expirements/actions/runs/42142746890)
on `main` at `5817ffb` (pull request #11's merge) ended
`685 passed, 7 skipped in 18.64s`, and
[run #54](https://github.com/jonleach323/Qwen-Expirements/actions/runs/42167375122)
at `0defb03` (#12's merge) ended `685 passed, 7 skipped in 17.89s`; at the
start of S1 (`ce14942`) the suite was `284 passed, 6 skipped`. The 7
skipped are the six GPU tests and the marker test's GPU dummy. Each pull
request's check passed before it merged, and the ruleset "Protect main"
requires **Lint and test**, which settles S0's question about branch
protection.

## What S1 built

| Pull request | Merged | What it added |
| --- | --- | --- |
| #10 S0 review | 2026-11-04 | Fixes to the S0 review and the planned S1 prompts |
| #11 (titled with its branch name) | 2026-11-06 | All of S1's code: 14 commits, 54 files, +7,547 and −245 lines |
| #12 S1: record the engine check | 2026-11-06 | Decision 0011, the Qwen3.5-4B calibration file, four manifests |

Pull request #11 holds five steps, each with tests and a decision note:

| Step | Main code | Decision | On the real model |
| --- | --- | --- | --- |
| Neuron units and the dose rule | `units/neuron.py`, `units/dose.py` | 0006 | Dose +1 adds +23.49 on a CPU, +23.4778 on the GPU |
| Residual-size calibration | `units/calibrate.py` | 0007 | Measured on the GPU; the file is in git (0011) |
| Pushes on chosen passes | `engine/push.py` | 0008 | Used by three GPU folders |
| The batched decoding loop | `engine/generate.py`, `engine/decoding.py` | 0009 | S0's reply byte for byte on the GPU |
| `amount`, `steer`, `report`, smoke prompts, random controls | `runs/steer.py`, `analysis/report.py`, `units/controls.py` | 0010 | Four GPU folders (0011) |

Code paths are under `src/pushlab/`; the lists `units/known.csv`,
`prompts/calibration.jsonl` and `prompts/smoke.jsonl` are at the top of
the repository. The registry and MLP discovery date from S0. S1 added 17
unit-test files holding 381 tests, and one GPU test.

S1 as done differs from S1 as planned in `docs/prompts/S1/`:

- The owner ran seven prompts in chat, not the eight files. Planned prompt
  1 (S0's open items) and prompt 2 (the models module, Qwen3-8B, a GPU
  shape test) never ran, and nothing of either exists.
- Planned prompt 7's `pushlab run` and experiment file became `pushlab
  steer`, one condition per folder, with `amount` and `report`. There is
  no `pushlab run`, so S2's runner starts from `steer`.
- S1 also built things the plan put later: push windows, sampling with
  seeds, random controls and `pushlab baseline` on the new loop. Planned
  prompt 8 asked for a copy of the calibration file in
  `docs/baselines/s1/`, which has none; S1 keeps the file itself at
  `calibration/`, where `steer` reads it.
- Planned prompt 8 also pinned Qwen3-8B and filled decision 0005's warm
  row. This close-out was asked only for the review and the S2 prompts,
  with no code change, so both are still open.

## The engine's own checks

The done-whens of the plan's "Core modules":

| Module check | On a CPU | On the GPU |
| --- | --- | --- |
| Each model reports its shape: 32 × 9,216 (Qwen3.5-4B), 36 × 12,288 (Qwen3-8B) | Both class trees, built without weights | Qwen3.5-4B only implicitly: `load_model` refuses any other shape, and every pod command loaded it. Qwen3-8B was never downloaded |
| A push through the MLP value equals an add along the write direction, within bfloat16 | Tiny models; the real model within 0.32% (0008) | Not checked |
| A zero dose is token-identical to no hooks | Yes | Not yet |
| Each batch row matches its prompt alone | Tiny models: the same tokens, scores within 1e-5. The real model (0009): 2 of 3 replies identical; the middle one parted at its first word on an exact tie (19.625); made to read the batch's tokens, each prompt alone scored every pass within 0.28 of its row | Scores within 0.5 against a limit of 1.0; the largest gap, and whether tokens parted, were not recorded |
| The step counter stays right with left padding | Yes | No direct check |

**Calibration against the replications.** On the GPU, layer 14's size is
10.9105, so dose +1 adds +23.4778: 4.0% above occcat's 22.57 and 5.7%
above Mr-Neutr0n's 22.21, under the 10% gap the calibration task asked to
have reported. All of the gap comes from layer 14's size; the write
direction's length, 0.4647, matches. The GPU's size is 0.06% below the
cloud CPU's 10.9175.

**Speed: not recorded.** `steer` prints its generation time but saves
none, and the manifests' start and finish times include loading the model.
From them, the four folders made at least about 44 reply tokens a second
for a batch of 5: 3,623 tokens in at most 83 s. That neither confirms nor
rules out the plan's warnings at 1,000 and 500 tokens a second. At the
lower bound, Phase 1's generation alone, about 15,000 runs of 300 to 600
reply tokens, would take about 28 to 57 GPU-hours against its budget of
25, so S2's runner must measure the speed of every batch.

## Still open after S1

1. **Copy the reference runs off the pod.** Nothing records a copy of
   S0's reference run (`/workspace/runs/s0-baseline`) or of S1's four
   folders and `s1-report.md` in `/workspace/outputs`. They hold the only
   copies of those replies and token IDs. Never use any as `--out`.
2. **The zero-dose check, the owner's labels and the GPU's largest gap
   between a batched row and its prompt alone** (decision 0009 asks for
   it): "Closing S1 on the pod".
3. **Qwen3-8B** has never been downloaded, loaded, shape-checked or
   pinned; the registry still asks for Hugging Face's `main`. S3's
   library-persona replication needs it: S2's prompt 9 puts it in the S3
   prompts.
4. **Reference folders.** CLAUDE.md and `tests/unit/test_docs.py` guard
   only `/workspace/runs/s0-baseline`. The docs put runs in
   `/workspace/runs`, but S1's went to `/workspace/outputs`, the folder
   the plan's nightly copy uses. S2's prompt 1 adds the guard; prompt 7
   settles the folder.
5. **Planned prompt 1, never run.** CLAUDE.md still says "Current sprint:
   S0" and has no rule against "smoke" in new GPU test names (S2's
   prompt 1). The runbook entries (curly quotes, an RTX 4090 out of
   capacity, a cloud session without the repository), the pod template,
   the stand-in GPU note and the huggingface-hub question: S2's prompt 9.
6. **The calibration file is tracked now**, and some docs say otherwise:
   the comments in `.gitignore` and `tests/unit/test_calibration_file.py`,
   the pod runbook's "Measure the residual sizes" (on a new pod that step
   now stops) and decision 0007's reason. S2's prompt 9.
7. **Decision 0005's warm-start row.** Decision 0011 records a warm setup
   of 2m 14s, after `git pull` at `5817ffb`, with no step times; 0005
   still says it is not measured. S2's prompt 9.
8. **`docs/runbooks/pod.md` has 466 lines**, over CLAUDE.md's limit of
   about 400. S2's prompt 8 moves its experiment sections out of it.
9. **`docs/handoff/HANDOFF.md`**, which the owner's prompts name, exists
   on no branch; the pod's results came as a zip with its instructions
   inside. S2's prompt 9 writes the procedure down there.

## What slowed us down

From git, GitHub and the cloud session's log.

1. **GPU feedback came once, at the end.** All of S1's code reached the
   pod at once, 29 h after the first S1 commit, in pull request #11: 14
   commits, 54 files, no description, merged 91 s after it opened. S0's
   lesson had been one smaller pull request per prompt. That one GPU check
   then failed the done-when; a GPU run after the first engine step would
   have shown it sooner. CLAUDE.md wants changed test expectations
   explained in the pull request; #11 did so only in a commit message.
2. **Reviews cost more than building.** A review was running for 4 h 7 min
   of the 8 h 39 min the five code prompts took. In three prompts,
   reviewing and fixing outlasted building: 1 h 33 min against 28 min,
   1 h 18 min against 36 min, and 27 min against 22 min. Three restarts of
   the cloud machine killed three background reviews, 41 min of review;
   nothing records why they happened. The reviews did find real bugs: five
   feature commits needed six review fixes and three more test or doc
   commits.
3. **The real model on a cloud CPU was slow and fragile.** Three slow first
   loads of the 9 GB model on the 15 GB machine cost 35 min: a cold disk
   after restarts, and once two model copies in one process. Every
   real-model check needed its own script, since the commands load float32
   on a CPU, and none of their outputs reached the repository.
4. **The plan and the work drifted apart.** The prompts pasted were not the
   files in `docs/prompts/S1/`: planned prompts 1 and 2 never ran, prompt 7
   became `steer`, and the stale files caused 2 of 5 findings in one
   review. The prompts also named `docs/handoff/HANDOFF.md`, which does
   not exist; the session searched for it twice.
5. **GPU evidence did not last.** The pod ran a different set from the
   runbook's: no dose 0, control seed 0 instead of 1, `/workspace/outputs`
   instead of `/workspace/runs`. The only proof that the loop reproduced
   S0's reply on the GPU was hashing files in pytest's temporary folders,
   which later test runs deleted the same day. Setup step 7 runs the GPU
   tests without `-s`, so their numbers were never printed, and `steer`'s
   speed line was never saved.
6. **Hostility by reading.** Claude counted, three blind readers came out
   one lower, and the owner's labels, which the check needs, are missing.
7. **One long session.** All of S1's cloud work, prompts 1 to 5 and the
   insertion of the pod's results, ran in one cloud session, whose context
   was compacted twice, each time at about 906,000 tokens. Prompt 6 ran in
   a Claude session on the pod.

S1 was planned for week 2. It ran from 2026-11-04 18:48 to 2026-11-06
05:15, about 34 hours, of which the code prompts took about 9.

## What to change in S2

1. **One pull request per prompt,** with a real title and a description
   that explains any changed expected values, merged before the next.
2. **The GPU early, and planned.** S2's prompts plan two pod sessions,
   whose exact commands and new folder names go into
   `docs/runbooks/experiments.md`: after prompt 4, every timed schedule,
   and prompt 5 waits for its results; after prompt 8, per-row isolation
   and the 1,000-run kill and resume. Run them as written, or say what
   changed when pasting back.
3. **Evidence that lasts.** `steer` saves its generation time, reply
   tokens, tokens a second and peak GPU memory in its manifest, and gets a
   data-type option, so a real-model check on a cloud CPU runs through the
   command (prompt 2); `pushlab run` writes the same for every batch
   (prompt 8). GPU tests write their numbers to a file under
   `/workspace/logs`, not only to pytest's temporary folders. Pod results
   come back the same day.
4. **Fresh sessions and lighter reviews.** A new cloud session per prompt.
   One review round with at most three reviewers, a skeptic only for
   high- or medium-severity findings, and planted bugs before the
   reviewers, not during. Reviewers save their results to files, so a
   restart loses only the reviewers still running; nits go on a follow-up
   list. S2's prompt 1 writes this into CLAUDE.md.
5. **The prompt files are the prompts.** Paste `docs/prompts/S2/` as
   written, and change the file before pasting a changed prompt. The
   prompts cite only files that exist.
6. **The owner labels hostility** with the written definition and says
   whether the labels were blind (`pushlab report` names each folder),
   until S3's labeling sheet and judge are validated.

## Closing S1 on the pod

The steps are in `docs/runbooks/experiments.md`, "Close S1 on the pod".
They need no new code and about half an hour of pod time, and they:

- copy S0's reference run and S1's four folders and report off the pod,
  first;
- have the owner label the 20 replies, and decide what the check becomes
  if s1-evil stays under 5 of 5, before any new replies are made;
- run the zero-dose check: no push against dose 0 in one batch layout,
  compared token for token;
- optionally, see what the gap depends on: the 32 calibration prompts,
  doses 1.25 and 1.5, and the recipe's two neurons;
- record the GPU's largest gap between a batched row and its prompt
  alone;
- copy the new folders off the pod, with the results pasted back the same
  day. A docs-only pull request records them here.

## S2 at a glance

S2, Schedules and runs, is planned for week 3. From the development plan:
it "builds the schedule language and compiler, per-row batched pushes, run
expansion, the resumable runner with dry-run, and the store with
manifests. Done when a 1,000-run smoke experiment survives a deliberate
kill and resume, and per-row isolation holds on the GPU."

S1 gives S2 pushes on chosen passes, a hook that adds one amount per batch
row (the same in every row for now), a batched loop that counts the reply
tokens each push touched, seeded sampling, run records with manifests and
random controls. The timed schedules come first, because the project is
about dynamics:

| Prompt | Builds | GPU step for the owner |
| --- | --- | --- |
| [1](../prompts/S2/01-schedules-and-handoffs.md) | The schedule language and the dose table, with handoffs | None |
| [2](../prompts/S2/02-engine-follows-schedules.md) | The engine follows a schedule; `pushlab steer --schedule` | Checked in pod session 1 |
| [3](../prompts/S2/03-alternation.md) | Alternation and overlap; where a reply first differs, in the report | Checked in pod session 1 |
| [4](../prompts/S2/04-ramps-and-late-onset.md) | Ramps, late onset and the dose budget | Pod session 1 |
| [5](../prompts/S2/05-per-row-schedules.md) | One schedule per batch row, and a record per reply token | Checked in pod session 2 |
| [6](../prompts/S2/06-experiment-files.md) | Experiment files, run expansion and the dry run | None |
| [7](../prompts/S2/07-run-store.md) | The store for runner folders, read back and queried | None |
| [8](../prompts/S2/08-resumable-runner.md) | The resumable runner and `pushlab status` | Pod session 2 |
| [9](../prompts/S2/09-close-s2.md) | The S2 review, S1's leftovers and the S3 prompts | None |
