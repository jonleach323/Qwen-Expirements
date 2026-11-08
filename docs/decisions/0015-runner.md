# 0015: The runner saves each batch before it marks it done, and resumes from its index

Date: 2026-11-08 · Sprint: S2 · Status: accepted

## Context

- The development plan's "5. runs" module runs experiments with `pushlab
  run spec.yaml [--dry-run] [--limit N]`: reruns skip finished work, a
  dry run prints run counts, GPU-hours and cost, and the module is done
  when rerunning a finished experiment does nothing and killing a job
  loses at most one batch. "Running a job" and "Cost control" add a
  status command, a resume by rerunning the same command, and a hard cap,
  `max_gpu_hours`, at which the runner stops cleanly.
- The S2 prompt pasted for this step ("prompt 4 of 7") asked for:
  `src/pushlab/runs/runner.py` and `pushlab run <spec>` with --dry-run,
  --limit N, --retry-failed and --overwrite-version; runs grouped when
  they can share a batch (the same model and decoding settings, and equal
  prompt length when grouping by length is on); after each batch a Parquet
  shard written to a temporary file and renamed into place, and only then
  its run IDs marked complete in a progress index; on restart, completed
  runs skipped and a changed spec hash refused without
  --overwrite-version; a clean stop at max_gpu_hours, the rest left
  pending; a heartbeat and progress file after every batch; a dry run with
  run counts by condition, tokens, GPU-hours from a configured speed and
  cost from configs/costs.yaml (an RTX 4090 at 0.34 to 0.74 dollars an
  hour), and the measured speed saved for later estimates; and `pushlab
  status <spec>`. The runner never writes into another experiment's
  folder; a failed run is recorded with its error, and --retry-failed
  reruns only failed runs.
- It is not `docs/prompts/S2/07-run-store.md` or `08-resumable-runner.md`
  ("prompts 7 and 8 of 9"), which add a steps record, an --out folder, a
  pod session of 1,000 runs and moves in the runbooks. From 08 come the
  lock, a test of the moment between saving and marking, and a status
  that works while a run goes on.

## Decision

**Where.** An experiment's runner folder is `<runs dir>/<name>`: the runs
dir is `--runs-dir`, else `$PUSHLAB_RUNS_DIR`, else `/workspace/runs`,
where the docs put run folders, and the name comes from the experiment
file, so `pushlab status <spec>` finds it too. The runner works only in a
folder that is empty or whose `experiment.json` names the same
experiment. It refuses any other, such as S0's `/workspace/runs/s0-baseline`,
which holds a `runs.parquet`, and never writes there. It also refuses a
runs dir that is a file, or that is a run folder or lies inside one (a
folder holding `experiment.json`, `runs.parquet` or `manifest.json`), so a
mistyped `--runs-dir` cannot put one experiment's folder inside another's.
`run`, `run --dry-run` and `status` all check this before anything else.

**The folder** (`pushlab.runs.folder`):
- `experiment.json`, its identity: the experiment, its version, the
  experiment file's path and settings, the number of runs, each run's
  condition label by run ID (`conditions`, such as `pulse length=16`,
  which no run record holds, as decision 0014 asked; a control has its
  run's label), the earlier versions' identities, and the hashes of what
  its runs are made from:
  the experiment file (the spec's hash), the prompt set,
  `units/known.csv`, the model download (name and commit) and the
  calibration (None when nothing is pushed).
- `shards/v<version>-<batch>-<attempt>.parquet`, one per saved batch, in
  the runs table's schema; a shard is never replaced.
- `index.jsonl`, the progress index: one JSON line per shard that counts,
  with its version, the session that saved it, its batch, how many rows
  it ran, each saved run's status, the error, the batch's seconds, its
  generation seconds and its reply tokens. A line that a new version
  wrote for runs it keeps says `kept_from`.
- `heartbeat.json` (process, machine, session, state, last batch, time)
  and `progress.json` (runs ok, failed and pending, GPU-hours used of the
  cap, the speed and the hours left), rewritten when a session starts and
  after every batch; `manifest.json`, the plan's manifest, with one entry
  per session (its ID, the code commit, the lockfile's hash, the GPU,
  CUDA, driver, torch, transformers and package versions, its state, its
  batches, runs and GPU-hours); and `.lock`.
- `<runs dir>/speeds.json`: the last measured speed of each model and data
  type, written under its own lock, `.speeds.json.lock`, since runners of
  different experiments share it.

**Saving a batch.** The runner writes the batch's records to a hidden
temporary file, flushes it to the disk, renames it into place and flushes
the folder, and only then appends the batch's line to the index and
flushes that. A crash at any moment therefore leaves the batch either
marked, and saved, or not marked. A shard the index does not name is
ignored, and its batch runs again under the next attempt's name. A last
index line a crash cut short is ignored when read and cut off by the next
runner; hidden temporary files are deleted. So a crash loses at most the
batch that was running, and reading gives every saved run once: the
latest mark of each run counts. A broken file (an index line that is not
JSON, an `experiment.json` without a version, a shard that lacks a run the
index puts there) is refused with its name, never guessed at.

**One runner at a time.** The runner holds an exclusive `flock` on `.lock`
from before it reads the folder until it ends; a second one is refused at
once, naming the first one's process when its heartbeat says. The system
lets go of the lock when the process ends, even by a kill. Status writes
nothing; it only tries a shared lock for an instant to see whether a
runner holds the folder.

**Batches** (`pushlab.runs.plan.calls`). `expand` plans each batch from
one prompt and seed: a run and its controls, together. The runner joins
planned batches into one `generate()` call, in this decision called a
batch, when their runs have the same model and decoding settings, seed
included, and, with `group_by_length` (a setting of the experiment file,
true by default), prompts of one length in tokens, read with the model's
tokenizer, so that no row is padded. A batch holds at most `batch_size`
runs, and a planned batch stays whole, so a run is never split from its
controls. With `group_by_length: false`, prompts of different lengths
share a padded batch. The same plan always gives the same batches, in
the plan order of their first runs, numbered `b000001` on. A batch that
mixes runs that cannot share one is refused, as a guard. Each saved
record's `batch_key` is where it ran: its shard's name without
`.parquet` and its row there, such as `v1-b000001-1:0`, so no two saved
records share one. A batch whose runs are partly saved (after a new
version keeps some, or under --retry-failed) still runs whole, every row
where the plan puts it, and only its due runs are saved.

**Resume and versions** (`pushlab.runs.versions`). Before loading the
model, the runner compares the inputs' hashes now with those in
`experiment.json`. If any differ, it names them and refuses, unless given
--overwrite-version. That starts the next version: every saved run, ok or
failed, that is still the same run, the same ID, experiment, model,
prompt, schedule, decoding, seed and control_of (`versions.SAME_RUN`),
from the same prompt set and calibration, is marked again for the new
version in the shard that holds it, with its status, its error and its
share of the batch's time, and all else is pending. Under sampling a run
is kept only if the new layout gives it the same row, since a sampled
reply depends on its row (decision 0014). The new version's number is one
more than any the folder's identity or index has used, so a number is
never used twice. The marks are written before `experiment.json`, so a
crash in between leaves the old version as it was; the next attempt takes
the next number, and the marks the crash left count for no version. Old
shards stay on disk. Without this, raising `max_gpu_hours` or lowering
`batch_size`, both settings, would throw finished runs away, and a
failed run would turn back into a pending one.

**Failures.** A batch whose generation raises is saved like any other,
with its runs failed, no reply and the error (`Type: message`; an
out-of-memory error says to lower `batch_size`), and the runner goes on.
A plain rerun leaves failed runs alone and says so; `--retry-failed` runs
only those, in their batches. A batch that cannot be saved, such as on a
full disk, stops the runner, which keeps what is saved and names the
batch it could not save. Ctrl-C while the model loads stops cleanly too:
the heartbeat says so, and no batch or manifest is written.

**The cap and the limit.** The GPU time of a version is the wall time of
its saved batches: each batch's seconds run from the save before it, or
from the session's start, so loading the model counts. Before loading,
the runner refuses to start when the version's time already reaches
`max_gpu_hours`, saying how to go on (raise it in the experiment file
through a pull request, then run with --overwrite-version, which keeps
the saved runs); after each batch it stops once the time reaches it, so
it stops after the batch that crosses it. `--limit N` runs at most N runs,
in whole batches: nothing when the next batch alone is larger.

**Estimates** (`pushlab.runs.estimate`, `configs/costs.yaml`). The costs
file names the GPU (RTX 4090), its price ($0.34 to $0.74 an hour,
RunPod's Community to Secure Cloud) and a speed per model in reply tokens
a second over a batch: 44 for qwen3.5-4b, S1's lower bound with loading
included, and 22 for qwen3-8b, an assumption. After a session, the runner
saves its measured speed (reply tokens over the generation time of its
batches that ran ok), the time the model took to load and the rows its
batches held, in `<runs dir>/speeds.json`, keyed `model/dtype`, with the
GPU, batch size, experiment and date; estimates use it in place of the
costs file, and say what it was measured on. A broken speeds file is
warned about and left out. Every reply is taken to reach max_new_tokens,
so the reply tokens are "at most"; hours and cost are "about": they add
the measured load time once, and depend on how full the batches are.

**The dry run** (`pushlab.runs.dry_run`) prints `pushlab expand
--summary`, then the runs still to run in the folder (with --limit,
--retry-failed and --overwrite-version as the run would take them; with
--overwrite-version, how many saved runs the new version would keep), the
batches they run in, their reply tokens, the speed, the GPU-hours against
what is left of the cap, with the share of runs it would stop after, and
the cost. It loads no model and writes nothing. It says when the run
would refuse the folder, when the cap is used up, and, in a cloud session
without the model's download, that the download and calibration were not
checked and that batches are counted one prompt each.

**Status** (`pushlab.runs.status_command`) prints the experiment and
version; whether a runner works on it, from the heartbeat, the process
and the lock together (running, on another machine, gone without
finishing, or ended); the runs ok, failed and pending; the batches saved,
and those kept from earlier versions; the speed measured over them; the
GPU-hours used of the cap; the time left at that speed, or before any
batch at the estimates' speed; the first five failed runs with their
errors; and a note when the experiment file, the prompt set or the known
units changed since the folder's version, so that the run would refuse.

**Sessions and the manifest.** Each `pushlab run` that loads the model is
a session with its own ID, which its index lines and its manifest entry
carry. When a session finds that earlier sessions of the same version
ran other code or libraries (another code commit, lockfile, torch or
transformers), it warns before running: the version's runs would then
mix two versions of the code. It still runs; the manifest records which
session ran what.

**Exit statuses.** 0 when it finished, or had nothing to do; 1 when it
refuses or cannot save; 2 when a run failed in this session; 3 when it
stopped cleanly with runs left, at the cap or the limit; 130 at Ctrl-C.
When more than one applies, the first of 130, 1, 2 and 3 wins, so a
script that loops until 0 neither stops early nor runs a failing
experiment forever.

**The checks.** On the CPU with tiny-vl:
- a run saves every planned run once in plan order; Ctrl-C in the third
  of five batches, then a rerun, gives every run once, the first two
  shards untouched, and the replies of a run never stopped; a batch saved
  but not marked is redone under a new name; a finished experiment run
  again loads nothing and writes nothing;
- --limit in whole batches (exit 3); the cap with a stand-in clock stops
  after the batch that crosses it (exit 3) and refuses to start again;
  heartbeat and progress after every batch;
- a run that raises is saved failed with its error, and --retry-failed
  reruns only it, leaving pending runs alone; an out-of-memory error;
- a changed experiment file, prompt set or calibration is refused; a new
  version keeps the unchanged runs, ok and failed, keeps nothing when a
  prompt or the calibration changed, keeps a sampled run only on its
  row, never reuses a crashed attempt's number or counts its marks; a
  change to some runs' records redoes only those, in their rows;
- prompts of one length share a batch, prompts of other lengths share one
  padded batch with grouping off, each seed keeps to its own batches, and
  batches keep planned batches whole and are always the same;
- inputs that change while the model loads; Ctrl-C while it loads; a full
  disk stops the run and keeps the rest; a second runner, another
  experiment's folder, a folder that is not a runner folder, a file in
  its place and a runs dir inside a run folder are refused; a torn index
  line, hidden leftovers, a failed shard write and broken files;
- the manifest's sessions and the warning when the code changed; a broken
  speeds file;
- the dry run's numbers, with and without a measured speed, and what a
  new version would keep; and status before, during and after a run, on
  another machine, after a crash, and when the experiment file changed.

## Why this shape

- **Shards and an index that only grows,** rather than one runs.parquet
  rewritten after every batch: saving a batch costs the batch, not the
  experiment, and nothing saved is ever rewritten.
- **JSON lines, not SQLite:** the index reads with `cat`, and SQLite's
  locking is not safe on network volumes, such as some pods' storage.
- **The mark after the rename:** the index is the one record that a batch
  counts, so a shard on its own never does.
- **Batches joined across prompts,** rather than one prompt a batch: with
  only a few runs to a prompt, batches would be small and the GPU half
  idle. Grouping by length keeps padding out of the replies by default,
  and the setting is in the experiment file's hash, so a change to it is
  a new version.
- **A version that keeps unchanged runs,** rather than a fresh start: the
  comparison covers everything a run is made of, so a kept run is the run
  the new plan means, and its batch_key still says where it ran.
- **Wall time for the cap,** loading included: the pod is billed by the
  hour, whatever it does.

## Consequences

- The first run on the pod measures the speed; copy it from
  `/workspace/runs/speeds.json` into `configs/costs.yaml` through a pull
  request, so dry runs in cloud sessions use it too.
- Known limits, left for later:
  - the time of a session that saves no batch, such as one that loads the
    model and is stopped, is not counted against the cap;
  - a failed run's error is kept in the index, not in its record, which
    has no error field;
  - one row that raises fails every run of its batch;
  - `speeds.json` keeps only the last session's speed of each model and
    data type, not an average;
  - sampled experiments still draw each row's numbers from its place in
    the batch (decision 0014); seeding rows from their runs is still to
    come.
- The steps record, notifications, the idle watchdog and the nightly copy
  are not here.
- Old versions' shards stay on disk, and must not be deleted by hand: a
  run a new version keeps is still read from the shard that first saved
  it, so deleting `v1-*.parquet` would lose version 2's kept runs too.
- `pushlab report` reads S1's single-file folders, not runner folders;
  `pushlab.runs.folder.read_results` reads a runner folder.
