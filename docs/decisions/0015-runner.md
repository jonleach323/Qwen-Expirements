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
which holds a `runs.parquet`, and never writes there.

**The folder** (`pushlab.runs.folder`):
- `experiment.json`, its identity: the experiment, its version, the
  experiment file's path and settings, the number of runs, and the hashes
  of what its runs are made from: the experiment file (the spec's hash),
  the prompt set, `units/known.csv`, the model download (name and commit)
  and the calibration (None when nothing is pushed).
- `shards/v<version>-<batch>-<attempt>.parquet`, one per saved batch, in
  the runs table's schema; a shard is never replaced.
- `index.jsonl`, the progress index: one JSON line per shard that counts,
  with its version, batch, each run's status, the error, the batch's
  seconds, its generation seconds and its reply tokens.
- `heartbeat.json` (process, machine, state, last batch, time) and
  `progress.json` (runs ok, failed and pending, GPU-hours used of the cap,
  the speed and the hours left), rewritten when a session starts and
  after every batch; `manifest.json`, the plan's manifest, with one entry
  per session; and `.lock`.
- `<runs dir>/speeds.json`: the last measured speed of each model and data
  type.

**Saving a batch.** The runner writes the batch's records to a hidden
temporary file, flushes it to the disk, renames it into place and flushes
the folder, and only then appends the batch's line to the index and
flushes that. A crash at any moment therefore leaves the batch either
marked, and saved, or not marked. A shard the index does not name is
ignored, and its batch runs again under the next attempt's name. A last
index line a crash cut short is ignored when read and cut off by the next
runner; hidden temporary files are deleted. So a crash loses at most the
batch that was running, and reading gives every saved run once: the
latest mark of each run counts.

**One runner at a time.** The runner holds an exclusive `flock` on `.lock`
from before it reads the folder until it ends; a second one is refused at
once, naming the first one's process when its heartbeat says. The system
lets go of the lock when the process ends, even by a kill. Status takes no
lock and writes nothing.

**Batches.** The runner runs the batches `expand` plans, one `generate()`
call each, in plan order. Each holds runs of one prompt and one seed with
one set of decoding settings, so they share the model and decoding, and,
being of one prompt, their prompts are of equal length: grouping by
length is always on, and no row is padded. A batch that mixes runs that
cannot share one is refused, as a guard. Each saved record's `batch_key`
is where it ran: its batch and its row there. A part of a planned batch
(only after a new version keeps the rest) runs from row 0.

**Resume and versions.** Before loading the model, the runner compares
the inputs' hashes now with those in `experiment.json`. If any differ, it
names them and refuses, unless given --overwrite-version. That starts the
next version: every saved run that is still the same run, the same
experiment, model, prompt, schedule, decoding, seed and control_of
(`plan.SAME_RUN`), from the same prompt set and calibration, is marked
again for the new version in the shard that holds it, with its share of
the batch's time, and all else is pending. The marks are written before
`experiment.json`, so a crash in between leaves the old version as it
was. Old shards stay on disk. Without this, raising `max_gpu_hours` or
lowering `batch_size`, both settings, would throw finished runs away.

**Failures.** A batch whose generation raises is saved like any other,
with its runs failed, no reply and the error (`Type: message`; an
out-of-memory error says to lower `batch_size`), and the runner goes on.
A plain rerun leaves failed runs alone and says so; `--retry-failed` runs
only those, in their planned batches. A batch that cannot be saved, such
as on a full disk, stops the runner, which keeps what is saved.

**The cap and the limit.** The GPU time of a version is the wall time of
its saved batches: each batch's seconds run from the save before it, or
from the session's start, so loading the model counts. Before loading,
the runner refuses to start when the version's time already reaches
`max_gpu_hours`, saying how to go on (raise it, with --overwrite-version);
after each batch it stops once the time reaches it, so it stops after the
batch that crosses it. The time of a batch a crash loses is not counted.
`--limit N` runs at most N runs, in whole batches: nothing when the next
batch alone is larger.

**Estimates** (`pushlab.runs.estimate`, `configs/costs.yaml`). The costs
file names the GPU (RTX 4090), its price ($0.34 to $0.74 an hour,
RunPod's Community to Secure Cloud) and a speed per model in reply tokens
a second over a batch: 44 for qwen3.5-4b, S1's lower bound with loading
included, and 22 for qwen3-8b, an assumption. After a session, the runner
saves its measured speed (reply tokens over the generation time of its
batches that ran) in `<runs dir>/speeds.json`, keyed `model/dtype`, with
the GPU, batch size, experiment and date; estimates use it in place of
the costs file. Every reply is taken to reach max_new_tokens, so tokens,
hours and cost are upper bounds.

**The dry run** (`pushlab.runs.dry_run`) prints `pushlab expand
--summary`, then the runs still to run in the folder (with --limit and
--retry-failed as the run would take them), their reply tokens, the speed,
the GPU-hours against what is left of the cap, with the share of runs it
would stop after, and the cost. It loads no model and writes nothing, and
says when the run would refuse the folder.

**Status** (`pushlab.runs.status_command`) prints the experiment and
version, whether a runner works on it (from the heartbeat: running, gone
without finishing, or ended), the runs ok, failed and pending, the
batches saved, the speed measured over them, the GPU-hours used of the
cap, the time left at that speed, or before any batch at the estimates'
speed, the first five failed runs with their errors, and a note when the
experiment file has changed since the folder's version.

**Exit statuses.** 0 when the runs it ran were saved ok, or it stopped
cleanly at the cap or the limit, or had nothing to do; 1 when it refuses;
2 when a run failed in this session; 130 at Ctrl-C.

**The checks.** On the CPU with tiny-vl: a run saves every planned run
once in plan order; Ctrl-C in the third of five batches, then a rerun,
gives every run once, the first two shards untouched, and the replies of
a run never stopped; a batch saved but not marked is redone under a new
name; a finished experiment run again loads nothing and writes nothing;
--limit in whole batches; the cap with a stand-in clock stops after the
batch that crosses it and refuses to start again; heartbeat and progress
after every batch; a run that raises is saved failed with its error, and
--retry-failed reruns only it; a changed experiment file, prompt set or
calibration is refused, and a new version keeps the unchanged runs; a
full disk stops the run and keeps the rest; a second runner, another
experiment's folder, a folder that is not a runner folder and a file in
its place are refused; a torn index line, hidden leftovers and a failed
shard write; the dry run's numbers, with and without a measured speed;
and status before, during and after a run.

## Why this shape

- **Shards and an index that only grows,** rather than one runs.parquet
  rewritten after every batch: saving a batch costs the batch, not the
  experiment, and nothing saved is ever rewritten.
- **JSON lines, not SQLite:** the index reads with `cat`, and SQLite's
  locking is not safe on network volumes, such as some pods' storage.
- **The mark after the rename:** the index is the one record that a batch
  counts, so a shard on its own never does.
- **A version that keeps unchanged runs,** rather than a fresh start: the
  comparison covers everything a run is made of, so a kept run is the run
  the new plan means, and its batch_key still says where it ran.
- **Wall time for the cap,** loading included: the pod is billed by the
  hour, whatever it does.

## Consequences

- The first run on the pod measures the speed; copy it from
  `/workspace/runs/speeds.json` into `configs/costs.yaml` through a pull
  request, so dry runs in cloud sessions use it too.
- Sampled experiments still draw each row's numbers from its place in the
  batch (decision 0014); seeding rows from their runs is still to come.
- The steps record, notifications, the idle watchdog and the nightly copy
  are not here. Old versions' shards stay until deleted by hand.
- `pushlab report` reads S1's single-file folders, not runner folders;
  `pushlab.runs.folder.read_results` reads a runner folder.
