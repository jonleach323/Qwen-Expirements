# Sprint S2 review: Schedules and runs

Date: 2026-11-12 · Planned for week 3 · Ran from 2026-11-07 07:05 to
2026-11-12 05:33 · All dates and times are UTC.

## Verdict

**S2 is not done.** Its done-when is "a 1,000-run smoke experiment
survives a deliberate kill and resume, and per-row isolation holds on the
GPU" (`docs/plan/development-plan.md:394`), and a sprint is done only when
that passes on the GPU (`:388`). Neither part has GPU evidence:

- **The kill and resume: not yet.** The code, its CPU tests, the 992-run
  experiment and a runbook with a deliberate `kill -9` exist, but nothing came
  back from the pod's run except one report error and the owner's remark that
  E (L14 N8949) "changed many replies very little".
- **Per-row isolation on the GPU: not yet.** The GPU test exists and
  setup's step 7 runs it; the one step 7 pasted back stopped before it ran.

The owner ran both S2 experiments on the pod; read-only commands there can
probably settle both parts, if the folders and logs remain ("Closing S2 on the
pod"). S1 is not done either: its part 2 has neither the owner's labels nor a
decision, and decision 0018 adds that E at +1 was hostile on 6 of 32 neutral
prompts on one broad reading, and L16 N8927 on 29 of 32.

## The done-when checks

### 1. A 1,000-run smoke experiment survives a deliberate kill and resume: not yet

**What exists** (decision 0015). The runner writes each batch's shard and
steps file whole, then marks it with one index line
(`src/pushlab/runs/session.py:199-210`, `shards.py:55-86`,
`folder.py:237-252`). The next start cleans up and skips what is saved
(`runner.py:160-161`, `:201-205`, `shards.py:29-44`, `:87-100`,
`folder.py:213-235`), and its `flock` is freed even by `kill -9`
(`folder.py:65-90`). In CI, Ctrl-C in batch 3 of 5 and a rerun save every run
once, as a run never stopped would (`tests/unit/test_runner.py:95-120`, an
in-process `KeyboardInterrupt`), and a killed lock holder frees its lock, the
suite's only real kill (`tests/unit/test_runner_folder.py:57-86`). No test
SIGKILLs a whole run; the prompt 6 fix check did, six times on the tiny model,
and each restart redid only the killed batch (session log).

**The experiment** is `experiments/s2/smoke_dynamics.yaml` (decision 0017):
992 runs in 64 batches, about 0.60 GPU-hours by its dry run. 992 is the
owner's number for the plan's 1,000: pasted prompt 6, not in the repository,
asked for "a roughly 1,000-run smoke experiment" of "992 runs in total"
(`tests/unit/test_smoke_dynamics.py:84-95` pins 992). **The kill** is runbook
step 3 (`docs/runbooks/experiments.md:165-179`): once 3 batches are saved,
`pkill -9 -f` the runner; status must say `stopped without finishing`, and the
restarted run end with `Done: all 992 runs are saved, 992 ok and 0 failed`.

**The GPU evidence, all of it.** At 2026-11-11 08:43 `pushlab report`
refused a list of capitalized labels and named the 16 conditions of a
folder `s2-smoke-dynamics`, in PR #16's order: the folder existed, and
that is all. At 18:49 the follow-up prompt said E "changed many replies
very little" in that run (decision 0018:7-9). Nothing else came back, the
session did not ask, and nothing in the repository records the run.
Indirect only: the evil diagnosis ran through on the GPU, with no kill.

**Missing:** (1) the kill, with at least 3 batches saved: status's lines just
after it, or what still records it, a `manifest.json` with two or more
version-1 sessions, the first `running` with 0 batches, and `index.jsonl`
lines from two or more sessions; (2) the restarted run's first and last lines;
(3) status after it, `992: 992 ok, 0 failed, 0 pending`; (4) a further start's
`Nothing to run: all 992 runs of version 1 are saved, 992 ok and 0 failed.`;
(5) verify's `no problem found.`; (6) each session's `code_commit` and
`code_dirty`; (7) whether the copy opened; (8) the owner's word that step 3's
`pkill -9` made the stop, as a crash leaves the same records. The pod's files
can still give 1, 3, 5 and 6.

### 2. Per-row isolation holds on the GPU: not yet

**What exists** (decision 0013). A dose table gives each batch row its own
amounts on every pass, and one hook per pushed layer adds them, never at
padding (`src/pushlab/schedules/compile.py`, `src/pushlab/engine/hooks.py`).
On the tiny models four rows of a padded batch match their prompts alone
(`tests/unit/test_generate_schedules.py:67-120`). The GPU test,
`tests/gpu/test_isolation_smoke.py:173-238`, runs four schedules of L14
N8949 and L13 N7956 in one padded batch on Qwen3.5-4B. A probe checks
every pass bit for bit (`:105-129`); only rows 2 and 3 may be pushed in
layer 13 and 1 to 3 in layer 14 (`:225`); each prompt alone must stay
within 1.0 of its batched scores (`:226-231`). It writes
`/workspace/logs/isolation-<time>.json`, and setup's step 7 runs it
(`scripts/setup_pod.sh:49`), which "ends with `7 passed`" (`pod.md:182`).

**The GPU evidence, all of it:** one step 7, pasted at 2026-11-12 01:46,
at `16f8202`. It stopped collecting `tests/unit/test_reply_checks.py`
(`1881 deselected, 1 error`; `Setup stopped at step 7/8`), so no GPU test
ran: four tracked files on the pod were modified, it seems back to their
versions before PR #17 (decision 0018:209-221). Whether setup then passed
before the evil run's first batch (02:02:44) is not known.

**Indirect evidence,** from the evil diagnosis, which cannot show that a row's
amounts were exact: each prompt's 8 runs shared one unpadded batch, and
baseline's row stayed clean in all 32 beside rows that broke or turned
hostile, as their pushes predict (decision 0018:78-79, :231-260). A batch-wide
leak would have shown; a small one, or one only in padded batches, would not.

**Missing:** (1) setup's step 7 line starting `7 passed` (a `1 warning` may
follow, `pod.md:445-448`), from a run at or after PR #14's merge `0ae0a5a`,
best at `0aa9374`; (2) that run's isolation file, with `rows_pushed_by_layer`
`{"13": [2, 3], "14": [1, 2, 3]}` and every `largest_gap` at most 1.0
(`tests/gpu/test_isolation_smoke.py:226-231`); decision 0013 asks for these
numbers (lines 158-160); (3) optionally, the smoke runbook's step 5 lines,
`runs 32 of 32, controls 32 of 32`.

### Also checked: each prompt's DONE WHEN: met

| Prompt, pasted | DONE WHEN | Delivered in | Met |
| --- | --- | --- | --- |
| 1, schedule language, 11-07 07:05 | `uv run pytest -q` passes | `a870df8` to `c428da8`, #14; decision 0012 | Yes |
| 2, dose tables, per-row hooks, 11-07 09:33 | pytest; a gpu-marked `tests/gpu/test_isolation_smoke.py` repeats the 4-row check | `9108a21`, `677a404`, #14; 0013 | As written: the test exists; its GPU result is check 2 |
| 3, experiment files, 11-07 21:01 | pytest; `pushlab expand --help` works | `9c10fe5` to `2e3f1ba`, #14; 0014 | Yes |
| 4, runner, dry run, status, 11-08 01:32 | pytest; `run --help` and `status --help` work | `e150d42` to `eed84f5`, #14; 0015 | Yes; the speed it saves never came back |
| 5, steps, manifest, verify, markers, 11-08 07:00 | pytest; `verify --help` works | `5644612` to `4a0673e`, #14; 0016 | Yes |
| 6, smoke test of dynamics, 11-09 06:06 | pytest; `pushlab expand` reports 992 runs | `31d44fa` to `7518f3a`, #16; 0017 | Yes: `992 runs, of which 480 are controls` |
| Follow-up, evil diagnosis, 11-11 18:49 | `pushlab expand` reports the run count; pytest | `eed6bd4` to `9495e24`, #17 to #20; 0018 | Yes: 256 runs |

Every `--help` exits 0 at `0aa9374` (2026-11-12). Prompt 7 is this close-out.

### Also checked: the plan's module checks: partly

| Module check (`development-plan.md`) | On the CPU | On the GPU |
| --- | --- | --- |
| Schedules (`:152`): tables match hand-worked examples; property tests show no push outside its segment | Partly: hand-worked examples (`tests/unit/test_compile.py:48-81`, `test_schedule_shapes.py:46-218`), but only alternations have a seeded property test of where pushes land (`test_schedule_shapes.py:240-255`; the ramps' seeded test checks doses inside the segment only). Missing: one over random schedules and compiled tables, as planned prompt 1 asked (`docs/prompts/S2/01-schedules-and-handoffs.md:116-118`) | Not needed |
| Engine (`:160`): zero dose equals no hooks; each row matches its run alone; the step counter right with left padding | Yes: `test_steer.py:110-126`, `test_generate_schedules.py:67-142`, `test_dose_hooks.py:128-142` | Zero dose: met in S1, before S2's dose tables; the rest: check 2, not yet |
| Runs (`:167`): a finished experiment rerun does nothing; a kill loses at most one batch | Yes, with an in-process stop (`test_runner.py:95-162`) | Check 1, not yet |
| Store (`:174`): data survives a write and read; malformed records are refused | Yes: `test_records.py:90-190`, `test_steps.py:35-132`, `test_experiment_manifest.py:86-103` | Not needed |

### Also checked: the tests pass on GitHub: met

[Run #60](https://github.com/jonleach323/Qwen-Expirements/actions/runs/42330262433)
on `main` at `413b289`, S2's start, ended `685 passed, 7 skipped in 15.64s`;
[run #107](https://github.com/jonleach323/Qwen-Expirements/actions/runs/43161832964)
at `0aa9374` (PR #20's merge) ended `1921 passed, 8 skipped in 75.55s`: the
seven GPU tests and a GPU dummy skip. CI passed on every PR's head and on main
after each merge; four branch pushes' runs never finished (#89 and #105
cancelled by newer pushes, #96 and #97 still queued).

## What S2 built

| Pull request | Merged, merge commit | Commits, files | What it holds | Description |
| --- | --- | --- | --- | --- |
| [#13](https://github.com/jonleach323/Qwen-Expirements/pull/13) | 11-07 02:38, `413b289` | 3, 11 | S1's close: its review and the 9 planned S2 prompts | Empty |
| [#14](https://github.com/jonleach323/Qwen-Expirements/pull/14) | 11-08 19:06, `0ae0a5a` | 19, 101 | Prompts 1 to 5, +21,768 −117 lines | Empty |
| [#15](https://github.com/jonleach323/Qwen-Expirements/pull/15) | 11-09 06:03, `c608d5a` | 1, 1 | S1's zero-dose check, from another session | Yes |
| [#16](https://github.com/jonleach323/Qwen-Expirements/pull/16) | 11-09 16:17, `7f7a77c` | 5, 6 | Prompt 6, the smoke test of dynamics | Empty |
| [#17](https://github.com/jonleach323/Qwen-Expirements/pull/17) | 11-11 19:47, `a12a7de` | 1, 9 | The evil-diagnosis follow-up | Its commit message |
| [#18](https://github.com/jonleach323/Qwen-Expirements/pull/18) | 11-11 21:47, `16f8202` | 3, 9 | The follow-up's review fixes | Empty |
| [#19](https://github.com/jonleach323/Qwen-Expirements/pull/19) | 11-12 05:08, `1839beb` | 2, 5 | The evil diagnosis's results, a wider tag check | Empty |
| [#20](https://github.com/jonleach323/Qwen-Expirements/pull/20) | 11-12 05:30, `0aa9374` | 2, 4 | The last fix check's fixes | Empty |

Decisions 0012 to 0018 record each step: schedules, per-row amounts,
experiment files, the runner, steps and verify, the smoke test, and the evil
diagnosis. The code is under `src/pushlab/`: `schedules/`, `engine/hooks.py`,
17 new files in `runs/`, three in `store/`, four in `analysis/`, and
`cli_experiments.py` (`pushlab expand`, `run`, `status`, `verify`); `pushlab
report` also reads an experiment. `413b289..0aa9374` changed 112 files
(+24,057 −140 lines; `src/` +8,783; `tests/` +12,510, in 49 new unit-test
files) in 30 non-merge commits: 9 features, 19 fixes, 2 others.

## What the GPU showed

**The smoke test:** nothing is recorded but the owner's remark that E "changed
many replies very little" (decision 0018:7-9), still unexplained. **The evil
diagnosis** (`experiments/s2/evil_diagnosis.yaml`): 256 runs of up to 200
reply tokens. A Claude session on the pod read every reply, knowing the
conditions, in three bins of its own (decision 0018:223-240):

| Condition | Hostile | Mildly off | Normal |
| --- | --- | --- | --- |
| baseline | 0 | 0 | 32 |
| e-1: E at +1, prompt pass and reply | 6 | 26 | 0 |
| e-1's control, L14 N1929 | 0 | 31 | 1 |
| e-1-25 and e-1-5: E at +1.25 and +1.5 | 8 and 2 | 24 and 30 | 0 and 0 |
| e-1-reply-only | 5 | 27 | 0 |
| e-w-recipe: E with L13 N7956 at −1 | 11 | 20 | 1 |
| l16-n8927: L16 N8927 at +1 | 29 | 3 | 0 |

Its reading (`:278-339`): not a setup problem, as baseline is clean; E changes
every reply, mostly against the task, not hostile; more dose does not help;
N8927's 29 lacks a layer-16 control. With one reader, one seed and a broad
"hostile", E's 6 of 32 is likely an overcount, far below the plan's 14 of 20.

**Speed:** in the evil run's first index line, pasted from the pod, 8
rows made 1,600 reply tokens in 44.32 s of generation: **36.1 reply tokens
a second**, S2's one reported GPU speed, perhaps with warm-up. That is 28 and
14 times below the plan's warnings, "far below 1,000 tokens a second" and
"below 500" ("Technical risks and fallbacks", `development-plan.md:593`,
`:595`), if at a batch of only 8. Below the 44 of `configs/costs.yaml:14`,
it puts dry runs a fifth low, and Phase 1's 15,000 runs of 300 to 600
reply tokens at 35 to 69 GPU-hours, against 25 (`research-plan.md:482`).

## Planned against actual

`docs/prompts/S2/` holds 9 planned prompts (PR #13); the owner pasted 7
others and a follow-up, from another framework, and no planned file was
pasted or changed. The pasted ones named four files that exist on no
branch: `docs/handoff/HANDOFF.md`, `S2-prompt-framework.md`,
`docs/decisions/0003-s1-engine-check.md` and `0004-s2-runner-check.md`
(decisions 0012:25-30, 0018:10-16). Planned and never done:

1. Pod session 1 after planned prompt 4, and `experiments/p0/schedules/`.
2. `pushlab steer --schedule` and `--dtype`, steer's speed fields and peak
   GPU memory, which nothing measures (planned prompt 2).
3. Report marks where a reply first differs (planned prompt 3).
4. A ceiling read from a file (planned prompt 4): `DEFAULT_CEILING = 1.4`
   stays in `src/pushlab/schedules/validation.py:36`.
5. Seeds per run (planned prompt 5): rows are still seeded by their place
   (`src/pushlab/engine/decoding.py:101`).
6. `experiments/p0/s2_kill_resume.yaml`, 1,000 runs (planned prompt 6),
   and a SIGKILL test of a whole run (planned prompt 8).
7. Splitting `docs/runbooks/pod.md` (planned prompt 8): it has 470 lines.
8. CLAUDE.md's housekeeping (prompt 1), `docs/baselines/s2/` (prompt 9),
   and a note that golden tests and invariance repeats wait for S3.

## Still open after S2

1. **Copy the run folders off the pod, first:** S0's reference run, S1's four
   folders and `s1-report.md`, both S2 folders and `/workspace/logs`. No copy
   is confirmed but S1's two zero-dose folders (PR #15).
2. **S2's done-when evidence:** "Closing S2 on the pod".
3. **S1's part 2:** the owner's labels of S1's 20 replies, blind or not, and a
   decision note on its 5-of-5 check (D6), both due before S2's runs.
4. **Gate 0's evil target:** keep E at 14 of 20 (`development-plan.md:350`),
   add an L16 N8927 arm, or move to N8927 (decision 0018:318-323), which
   needs layer-16 controls, at least 5 (`docs/plan/research-plan.md:359`).
5. **S1's nine open items** (`docs/sprints/S1-review.md:229-263`):

   | # | Item | Now | In S3 |
   | --- | --- | --- | --- |
   | 1 | Copies off the pod | Open: item 1 above | Before prompt 1 |
   | 2 | The owner's labels; the GPU's batched-against-alone gap | Open; the isolation file holds one gap per row | Prompt 4's sheet, whose Results record s1-evil's hostile count by the owner's labels; pod session A |
   | 3 | Qwen3-8B never downloaded or pinned | Open: `src/pushlab/models/registry.py:35`, `:50`; not in `configs/models.yaml` | Prompt 1 |
   | 4 | Guards for reference folders | Partly: `CLAUDE.md:18` and `tests/unit/test_docs.py:11` name only `s0-baseline`; `docs/runbooks/experiments.md:3-6` names S1's | Not planned |
   | 5 | Planned S1 prompt 1 | Open: `CLAUDE.md:6` says S0; no rule on "smoke" in GPU test names; three runbook entries missing | Prompts 1 and 11 |
   | 6 | Docs calling the tracked calibration file ignored | Open: `.gitignore:6-8`, `tests/unit/test_calibration_file.py:63-67`, `docs/runbooks/pod.md:276-278`, decision 0007 | Prompt 1 |
   | 7 | Decision 0005's warm-start row | Open: 0011's 2m 14s is not in it | Prompt 11 |
   | 8 | `docs/runbooks/pod.md` over 400 lines | Worse: 466 then, 470 now | Prompt 2 |
   | 9 | `docs/handoff/HANDOFF.md` | Open: on no branch, named again by pasted prompt 1 | Prompt 1 |

6. **The speed:** `configs/costs.yaml` still says 44 (decision 0015:295-297).
7. **The pod's dirty tree** (four modified files at `16f8202`, 2026-11-12
   01:52): only the manifests' `code_dirty` fields show which runs had it.
8. **Empty PR descriptions** on #14, #16, #18, #19 and #20, though the commits
   of #14, #19 and #20 changed existing tests' expected values, explained only
   in commit messages (`9108a21`, `24e2587`, `2fca731`, `baaa909`, `4a0673e`,
   `a5744e6`, `9495e24`), and #18 changed `tests/unit/test_reply_checks.py`
   with no word anywhere. Claude can write them, if the owner wants.
9. **Why E changed many smoke-test replies so little** (0018:290-295), and the
   pod's two evil reports, to write again with `--overwrite` (:265-276).

## What slowed us down

From git, GitHub, the cloud session's log and its 18 workflow records.

1. **Reviews and fix checks took 62%** of the 24.6 h that the eight work
   periods (prompts 1 to 6, the follow-up, its results) ran from paste to last
   push: 9.1 h and 6.3 h, against 3.8 h of building (S1: 48%). The 18
   workflows used 136 agents and 28.7 M tokens; prompt 4's review had 24
   skeptics (117 min); 7 fix checks found 16 medium problems, no high one; 19
   fix commits followed 9 feature commits. Prompts 4 and 5 were too big.
2. **Waiting:** of S2's 118.5 h, 93.3 h were gaps before the owner's next
   message. **GPU feedback came 97.6 h in,** after all six build prompts,
   and the done-when lines never came: none of the smoke runbook's six
   kinds, one of the evil runbook's seven. That is why S2 is not done.
3. **Pull requests:** #14 held prompts 1 to 5 (19 commits, 101 files);
   five of six have the branch name as title and no description, none had
   a review on GitHub, and #17 and #19 merged before their reviews' fixes,
   so #18 and #20 followed; 640 lines of PR drafts went unused.
4. **One session** for all of S2, its context compacted four times.
5. **Prompts from another framework:** none of the nine planned files was
   pasted; four named files are missing; capitalized labels cost a
   question and the pod's report error; S1's open items stayed open.
6. **The unplanned evil follow-up:** 4.8 h of session time (20%), four PRs and
   a pod session, in which a dirty tree of unknown cause failed step 7.
7. **The 400-line limit:** at least 11 splits, and five files near 400 lines.
8. **S1's six changes, none fully done:** one PR per prompt, the GPU
   early, the prompt files as written and the owner's labels: no;
   evidence that lasts and fresh sessions with lighter reviews: partly.

## What to change in S3

Ranked by the S2 time each would have saved, then by S3's need. S3 prompt 1
writes 1, 3 and 4 into CLAUDE.md in a commit the owner may drop (D1).

1. **Sized reviews, no routine fix check:** at most three reviewers and three
   skeptics in 60 min, or one reviewer in 20 min under about 500 added lines
   or for docs, prompts and experiment files; a fix check only on a confirmed
   high; lows to the follow-ups. A prompt's `src/` changes stay under about
   1,500 lines, tests and docs apart (prompts 4 and 5: 1,855 and 2,194); past
   that the PR says so, and nothing in BUILD is dropped. Check: each S3 PR's
   description gives its review's agents and minutes. S2 saving: up to 8.6 h.
2. **The pod evidence file,** `pushlab evidence SPEC` (S3 prompt 1):
   `/workspace/logs/evidence/<experiment>-<time>.txt`, never reply text.
   Check: every S3 pod paste is that file (pod session B's, its `sheet.json`)
   plus only what its prompt names.
3. **The session opens one PR per prompt,** from a new branch, titled "S3
   prompt N: ...", described, merged after CI and the review's fixes.
   Check: eleven described PRs, none merged before its fixes.
4. **A fresh cloud session per prompt.** Check: no compaction in S3.
5. **Setup refuses a modified tracked file** (S3 prompt 1). Check: its
   case in `tests/unit/test_setup_pod.py`; clean trees in evidence files.
6. **Paste the S3 prompt files as written,** changing a file by PR first.
   Check: each pasted prompt matches its file.
7. **Pod sessions in the prompts:** A after 1 and 2, B after 4, C after 5 to
   8, D after 9 and 10. A prompt builds without an earlier session's paste and
   only its pod part waits for it, so the owner keeps this order. Check: the
   S3 review dates each session.
8. **Gate 0's evil target, decided before the replication runs** (D6),
   in a decision note. Check: the note predates prompt 10's runs.
9. **The owner's labels, budgeted and blind:** 400 or 200 (D2), 2.8 to
   4.7 h for 400, the hostility sheet right after prompt 4. Check: each
   validation record names the labeler and says blind or not.
10. **The owner's hours, logged weekly** with GPU-hours and spend
    (`development-plan.md:533`). Check: one line per week of S3 in
    HANDOFF.md's weekly log.

## Closing S2 on the pod

None of this needs the GPU or loads a model; only the first command
writes a file, outside the repository. Download the copy as in "Copy
runs off the pod" (`docs/runbooks/experiments.md`) and check it opens.

```bash
# 1. The copy, first (tar names any folder it cannot find)
(cd /workspace && tar -czf s2-close-copies.tar.gz runs/s0-baseline outputs runs/s2-smoke-dynamics runs/s2-evil-diagnosis runs/speeds.json reports logs)
# 2. The pod's tree and commit, then the kill and resume
cd /workspace/Qwen-Expirements && git status --short && git log --oneline -1
uv run pushlab status experiments/s2/smoke_dynamics.yaml
uv run pushlab verify experiments/s2/smoke_dynamics.yaml
uv run python -c "import json; m = json.load(open('/workspace/runs/s2-smoke-dynamics/manifest.json')); print([(s['id'][:8], s['version'], s['started_at'], s['finished_at'], s['state'], s['batches'], s['runs'], s['code_commit'], s['code_dirty']) for s in m['sessions']])"
uv run python -c "import collections, json; e = [json.loads(l) for l in open('/workspace/runs/s2-smoke-dynamics/index.jsonl')]; print(len(e), collections.Counter((x['version'], x['session'][:8]) for x in e)); print([(x['batch'], x['shard'], x['session'][:8], x['saved_at']) for x in e[:6]])"
ls /workspace/runs/s2-smoke-dynamics/shards | wc -l
ls /workspace/runs/s2-smoke-dynamics/shards | grep -v -- '-1\.parquet$'
cat /workspace/runs/s2-smoke-dynamics/heartbeat.json
# 3. Per-row isolation
ls -l --time-style=full-iso /workspace/logs/
cat /workspace/logs/isolation-*.json
grep -H -E "^[0-9]+ (passed|failed|deselected)|^(ERROR|FAILED) |Setup stopped|Finished in" /workspace/logs/setup-*.log | sed -E 's/^([^:]+:(FAILED|ERROR) [^ ]+) - .*/\1/'
```

They must show check 1's items 1, 3, 5 and 6 and check 2's items 1 and 2, and:
an index of 64 lines from two or more sessions, the first with at least 3; 64
shards, or 65 with one `-2` redo; the heartbeat `finished`; `code_commit` most
likely `7f7a77c`, `code_dirty` False; a setup log from 2026-11-08 19:06 on.
Only once status shows 0 pending, one more start of runbook step 2's `pushlab
run` command gives item 4's line, loading no model. For decision 0018's gaps,
run status, verify and step 8's manifest line of
`docs/runbooks/s2-evil-diagnosis.md` on the evil run too.

Paste it all back the same day (none holds reply text), saying if the copy
opened and if step 3's `pkill -9` made the stop; a docs PR records it here. No
kill means a new run, a GPU job: the same file with `--runs-dir` set to a new
folder on `run`, `status` and `verify`, through runbook steps 1 to 4
(`experiments.md:142-186`), about 0.6 to 0.75 GPU-hours. No passing step 7
means `bash scripts/setup_pod.sh` at main again, also a GPU job, once the tree
is clean (`pod.md`, "If something goes wrong"). If this waits, S3 prompt 1's
`pushlab evidence` gathers most of it, in pod session A, part 1.

## S3 at a glance

S3 (week 4) is done when "all four replication tests and both judge
validations pass" (`development-plan.md:395`). The owner adds that the judge
must score sentence windows too: S4 checks its persona timelines against it.

| Prompt | Builds | GPU step for the owner |
| --- | --- | --- |
| [1](../prompts/S3/01-models-and-process.md) | Qwen3-4B registered; Qwen3-8B, Qwen3-4B and the judge pinned; `ModelSpec.family`; a GPU shape test; CLAUDE.md's process rules; setup refuses a modified tracked file; `pushlab evidence`; `docs/handoff/HANDOFF.md` and its weekly log | Pod session A, part 1: downloads, the shape test, two calibrations, and `pushlab evidence` on both S2 experiments |
| [2](../prompts/S3/02-golden-tests-and-gate.md) | `pushlab golden` and the runner's gate; golden as setup's step 8; the golden lines in `pushlab evidence`; pod.md split | Pod session A, part 2: setup with golden, golden again for the digest, one evidence file and the golden records |
| [3](../prompts/S3/03-scores-and-windows.md) | The scores table and view; sentence splitting; windows of sentences | None |
| [4](../prompts/S3/04-labeling-sheet-and-harness.md) | The hostility rubric's labels; `pushlab label-sheet`; `pushlab validate-judge` | Pod session B, CPU only: the 200-item hostility sheet; the owner labels it blind |
| [5](../prompts/S3/05-coherence.md) | `score_text`; coherence scores and a collapse rule; `pushlab score` | Pod session C, part 1: coherence of `s2-evil-diagnosis` |
| [6](../prompts/S3/06-capability.md) | About 100 questions with exact answers; exact-match grading | Pod session C, part 2: the capability baseline |
| [7](../prompts/S3/07-judge.md) | The local judge, for whole replies and sentence windows, with a cache and the family rule | Pod session C, part 3: the hostility validation |
| [8](../prompts/S3/08-persona-labels.md) | The persona rubric, the library sample and its sheet | Pod session C, part 4: the persona runs, labels and validation |
| [9](../prompts/S3/09-dose-ladder.md) | Dose ladders, `pushlab ladder`, `configs/ceilings.yaml` | Pod session D, part 1: the ladders |
| [10](../prompts/S3/10-replication-suite.md) | Gate 0's three replications, its reading of prompt 9's ladder, the batch-control check, `pushlab gate0` | Pod session D, part 2: the replications |
| [11](../prompts/S3/11-close-s3.md) | The S3 review and the S4 prompts | None |

| Owner decision | Before | What, with the default if unanswered |
| --- | --- | --- |
| D1 | Prompt 1 | The judge (default `granite-4.1-8b`); a second judge for the plan's 10% rescoring and validation samples (default none yet: the S3 review reports the rescoring "not yet"); whether CLAUDE.md takes the process rules (default yes, in a commit the owner may drop in the PR); whether to declare huggingface-hub (default no: a comment in `pyproject.toml`); the GPUs that may stand in for the RTX 4090 (default, proposed in prompt 2's note: L40S or RTX 6000 Ada, then RTX A6000 or A40) |
| D2 | Prompt 4 | The hostility labels and definitions (default: hostile, S1's definition; unsafe; neither); 200 labels per rubric (default, the development plan) or 200 in all (the research plan: then about 100 items a rubric, about 8 of them hostile, so a pass or fail rests on a handful); labeling blind (default) |
| D3 | Prompt 7 | No vLLM in S3 (default: vLLM 0.33.0 pins torch 2.14.0 and transformers below 5.21.0; the lock has 2.14.1 and 5.21.0); no API judge (default); whether hostile reply text may be sent to one |
| D4 | Prompt 8 | The persona library's terms (default: nothing of it in git); which personas; what a "match" is (default: the run's own persona of the three offered) |
| D5 | Prompt 9 | The ladder's neurons, grid, prompts and collapse rule; how "most collapse between 1.4 and 2.0" is read (default: the median of the 10 in [1.4, 2.0], which on prompt 9's 0.25 grid, under Liu's spread, holds about 93% of the time, "at least 6 of 10" about 55%; prompt 9 says how a neuron "above 2.5" counts); how a ceiling follows (default: the second lowest of the 10, the 20th percentile; under Liu's spread most likely 1.25 for qwen3.5-4b and 1.0 for qwen3-4b, below today's 1.4, so a combined push in one layer between it and 1.4, such as 1.3, would newly need `allow_over_ceiling`; two dose-1 pushes in one layer, 1.41, need it already) |
| D6 | Prompt 10's runs | Written in prompt 10's decision note, the pre-registration (also at `docs/prereg/p0-gate0.md`, `development-plan.md:125`): Gate 0's evil target (default: keep L14 N8949 at 14 of 20 with 5 matched random controls, each at most 2 of 20; no L16 N8927 arm); whether the evil test counts "unsafe" with "hostile" (default: hostile only, S1's narrow reading); whether hedwigai's 20 prompts, from a gist that states no licence, may be committed (default: not until the owner says yes); the router protocol; what S1's 5-of-5 check becomes (default: closed as superseded by Gate 0's evil test, with the owner's labels of S1's 20 replies recorded); how strict gate0's guard is (default: a later change to `configs/gate0.yaml`'s files or D6 numbers voids a run; naming the judge later does not). Prompt 10 gives each default in full |

**Time:** about 10 to 18 owner hours, by estimate: 1 to 1.5 to read and
decide; 2.75 to 3.7 for eleven prompts and PRs; 3 to 5.75 for five to eleven
pod visits (A to D's 2.75 to 4, and 15 min of setup each past four); 2.8 to
4.7 for 400 labels, the personas' 1.7 to 2.5 at 30 to 45 s an item (1.4 to 2.4
for 200); 0.25 to 0.5 to read prompt 6's 60 answers; up to 1.25 for a
fresh 100 labels after a FAIL; 0.25 for the weekly log and copies. S3 will not
fit one calendar week: the top end passes the plan's 12 to 15 hours a week
(`development-plan.md:22`), and label rounds, PRs and pod visits follow one
another (S2 ran 118.5 h). **GPU:** about 6 to 9 GPU-hours, $4 to $7 (A and D
on Secure Cloud): well inside Phase 0's 20 (`docs/plan/research-plan.md:481`).
