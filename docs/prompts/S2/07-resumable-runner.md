CONTEXT
This is Sprint S2, prompt 7 of 8. Read CLAUDE.md, the decision notes of
prompts 1 to 6, and in docs/plan/development-plan.md the "5. runs" and
"6. store" modules, the manifest under "Data model", "Running a job",
"Cost control" and "Backups and monitoring".
- Prompt 6 expands an experiment file into planned runs and batches;
  `pushlab run` has only --dry-run.
- src/pushlab/store/records.py writes one runs.parquet per folder,
  atomically, and refuses an existing one; src/pushlab/store/folders.py
  refuses a folder that already holds runs. A process killed while
  writing leaves only a hidden .tmp file.
- docs/runbooks/experiments.md (prompt 4) is where run steps go; pod.md is
  already 466 lines.

GOAL
Run an experiment batch by batch, so that a crash, a kill or a dropped
connection loses at most one batch, and running the same command again
finishes the job.

BUILD
- `pushlab run FILE --out DIR [--limit N]`, next to prompt 6's --dry-run:
  - it loads the model and the calibration once and runs the planned
    batches in order; each finished batch is saved as its own Parquet
    parts, runs and steps, written atomically before the next starts;
  - after each batch it prints one line, and appends it to a progress file
    in DIR: the batch, runs done of all, reply tokens, seconds, tokens a
    second and peak GPU memory;
  - the manifest has the plan's fields (code commit, lockfile, model
    revision, GPU, CUDA and driver, start and end times, runs by status),
    the experiment file's path, SHA-256 and text, and the calibration's
    SHA-256, with one entry per session that worked on DIR. The
    pre-registration fields stay empty until a phase has one: docs/prereg/
    does not exist yet;
  - --limit N runs at most N more runs, in whole batches.
- Resume: the same command on the same DIR skips every run already in a
  saved part, ignores the hidden leftovers of a killed write, and redoes
  the batch that was running, in the same layout. It refuses a DIR made
  from another experiment file, prompt set, calibration or model
  revision. Decide what a different code commit does, refuse or record it
  per session, and say why.
- max_gpu_hours counts the generation time of every session; the runner
  stops cleanly after the batch that reaches it, and says so.
- A batch that raises an error: either its runs are saved as failed, with
  the error, or the run stops. Decide in the note. An out-of-memory error
  names the batch size to lower.
- `pushlab status DIR` prints runs by status, batches, the last speed
  line, GPU-hours used against the cap and an estimate of the time left.
  It imports no torch and works while a run is going.
- Reading back: read_runs, or a new reader, reads the parts as well as
  S1's single runs.parquet, and `pushlab report` reads runner folders.
- A DuckDB view: a function that opens one connection with runs and steps
  views over every run folder under a root, the plan's "DuckDB view
  across all experiments". Settle the root: the docs put runs in
  /workspace/runs, while S1's went to /workspace/outputs.
- docs/runbooks/experiments.md: the dry run, starting a run in tmux,
  watching it with `pushlab status`, stopping and resuming, and copying a
  folder off the pod. Move pod.md's steer and report sections there, so
  pod.md gets under about 400 lines. tests/unit/test_docs.py follows the
  moved commands and also checks every documented `pushlab run --out`.
- A decision note, the next free number: the folder layout, resume, the
  cap, failures, status and the view.

CONSTRAINTS
- Every finished batch is safe on disk before the next one starts. A rerun
  of a finished experiment loads no model and changes nothing.
- store/records.py is 357 lines: new store code goes in new modules.
- No dependency changes; files under about 400 lines. The owner starts
  every GPU job.

TESTS FIRST
On the CPU, with tiny-vl and the tiny_downloads fixture:
- A run in a subprocess, killed with SIGKILL after k batches are saved,
  then run again: every planned run appears exactly once, the saved parts
  are unchanged byte for byte, the batch that was running is redone with
  the same batch keys, and the results equal those of a run that was
  never stopped.
- A finished experiment run again loads no model and writes nothing.
- --limit, the cap (with a stand-in clock), and a batch that raises.
- A changed experiment file, prompt set or calibration is refused; hidden
  .tmp leftovers are ignored.
- status for a folder in progress and for a finished one, with no torch
  imported.
- The DuckDB view counts runs across two folders; report reads a runner
  folder; S1's folders still read.

DONE WHEN
- `uv run pytest -q` and `uv run pre-commit run --all-files` pass, and the
  pull request's checks pass. One pull request with a description.
- Pod session 2, S2's done-when, after the merge, as written in
  docs/runbooks/experiments.md. The owner runs it:
  1. `git pull`, and `bash scripts/setup_pod.sh` if it changed uv.lock,
     pyproject.toml, configs/models.yaml or a model revision.
  2. `uv run pytest -q -s -m gpu tests/gpu/test_per_row_isolation.py`;
     keep its printed lines and its JSON file in /workspace/logs.
  3. The dry run of experiments/p0/s2_kill_resume.yaml, then the run
     itself in tmux, into a new folder under the root prompt 7 settled.
     Once `pushlab status` shows a few batches saved, stop it with
     Ctrl-C and start it again; a few batches later, kill the process
     with `kill -9` and start it again; let it finish.
  4. `pushlab status` must show 1,000 runs, all ok, each once. Run the
     same command once more: it must do nothing.
  5. Copy the folder and the logs off the pod, then stop the pod.
  6. Paste back the same day: the status output, the speed lines, the GPU
     test's lines and JSON, and whether the copies are made.

REPORT BACK
End with a plain-language summary for a beginner: what changed, why, how I
can check it myself, and anything I must do by hand. Put pod session 2
first, as a numbered list of exact commands, and say what to paste back.
Explain why a killed run loses at most one batch.
