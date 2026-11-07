CONTEXT
This is Sprint S2, prompt 8 of 9. Read CLAUDE.md, the decision notes of
prompts 1 to 7, docs/decisions/0007, docs/runbooks/experiments.md, and in
docs/plan/development-plan.md the "5. runs" module, "Running a job",
"Cost control" and "Backups and monitoring".
- Prompt 6 expands an experiment file into planned runs and batches;
  `pushlab run` has only --dry-run. Prompt 7 saves and reads runner
  folders one batch at a time.
- docs/runbooks/pod.md is 466 lines.

GOAL
Run an experiment batch by batch, so that a crash, a kill or a dropped
connection loses at most one batch, and running the same command again
finishes the job.

BUILD
- `pushlab run FILE --out DIR [--limit N]`, next to prompt 6's --dry-run:
  - it loads the model and the calibration once, runs the planned batches
    in order, and saves each through prompt 7's store before the next
    starts;
  - it compares the calibration's data type and prompt-set SHA-256 with
    its own run's, as decision 0007 asks, and records or refuses a
    mismatch (decide);
  - after each batch it prints one line, and appends it to a progress file
    in DIR: the batch, runs done of all, reply tokens, seconds, tokens a
    second, peak GPU memory and errors so far;
  - --limit N runs at most N more runs, in whole batches.
- A lock: while it runs, the runner holds fcntl.flock on DIR/.lock, which
  the system releases when the process dies, even by SIGKILL. A second
  `pushlab run` on a folder in use is refused, naming the other process's
  PID.
- Resume: the same command on the same DIR skips every run already saved
  and redoes the batch that was running, in the same layout. It refuses a
  DIR made from another experiment file, prompt set, calibration or model
  revision. Decide what a different code commit does, refuse or record it
  per session, and say why.
- max_gpu_hours counts the generation time of every session; the runner
  stops cleanly after the batch that reaches it, and says so.
- A batch that raises an error: either its runs are saved as failed, with
  the error, or the run stops. Decide in the note. An out-of-memory error
  names the batch size to lower.
- `pushlab status DIR` prints runs by status, how many run IDs are
  distinct and any saved more than once, batches, the last speed line,
  GPU-hours used against the cap and an estimate of the time left. It
  loads no model, never touches the GPU, and works while a run is going.
- docs/runbooks/experiments.md gets the runner's steps: the dry run,
  starting a run in tmux, watching it with status, stopping and resuming,
  and copying the folder off the pod. Move pod.md's "Generate the baseline
  reply", "Measure the residual sizes" and "Push a neuron and compare the
  replies" sections there, leaving a link, so pod.md gets under 400 lines;
  tests/unit/test_docs.py follows the moved commands and also checks every
  documented `pushlab run --out`.
- A decision note, the next free number: resume, the lock, the cap,
  failures, the calibration check and status.

CONSTRAINTS
- Every finished batch is safe on disk before the next one starts. A rerun
  of a finished experiment loads no model and changes nothing.
- Not in S2, which the decision note says and prompt 9 lists for S3: the
  golden-suite gate (golden tests come in S3), notifications, the idle
  watchdog, the nightly copy and a coverage measure (a new dependency, so
  ask first).
- No dependency changes; files under about 400 lines. The owner starts
  every GPU job. Reviews follow CLAUDE.md's review rule.

TESTS FIRST
On the CPU, with tiny-vl:
- A run in a subprocess started with sys.executable and a small launcher
  the test writes, since the tiny_downloads and tiny_vl_calibration
  fixtures patch only the test's own process. The launcher registers
  tiny-vl in REGISTRY before the command's parser is built, points
  calibrate.CALIBRATION_DIR at the test's calibration, switches Qwen3.5 to
  its PyTorch code as tests/conftest.py does, hides the GPU
  (CUDA_VISIBLE_DEVICES=""), and once k batches are saved, blocks inside
  batch k + 1 until the test kills it with SIGKILL. Then run again: every
  planned run appears exactly once, the saved parts are unchanged byte for
  byte, the batch that was running is redone with the same batch keys,
  and the results equal those of a run that was never stopped. Mark it
  slow.
- A second runner on a folder in use is refused.
- A finished experiment run again loads no model and writes nothing.
- --limit, the cap (with a stand-in clock), and a batch that raises.
- A changed experiment file, prompt set or calibration is refused.
- status on a folder in progress and on a finished one, without loading a
  model, including a run ID saved twice.

DONE WHEN
- `uv run pytest -q` and `uv run pre-commit run --all-files` pass, and the
  pull request's checks pass. One pull request with a description.
- Pod session 2, S2's done-when, after the merge, written into
  docs/runbooks/experiments.md with these commands:
  1. "Every time you start the pod again" in docs/runbooks/pod.md, then
     `source /workspace/pushlab.env`.
  2. `uv run pytest -q -s -m gpu tests/gpu/test_per_row_isolation.py`;
     keep its printed lines and its JSON file in /workspace/logs.
  3. `uv run pushlab run experiments/p0/s2_kill_resume.yaml --dry-run`.
  4. `tmux new -s s2`, and in it `uv run pushlab run
     experiments/p0/s2_kill_resume.yaml --out /workspace/runs/s2-kill-resume`.
     Use this identical line each time you start it again.
  5. In a second terminal, `uv run pushlab status
     /workspace/runs/s2-kill-resume`. When it shows at least 3 batches
     saved, press Ctrl-C in the tmux window, check that `pgrep -af
     "pushlab run"` prints nothing, and start the run again. When status
     shows 3 more batches, run `pkill -9 -f "pushlab run
     experiments/p0/s2_kill_resume.yaml"`, which kills uv and the runner
     together (`kill -9` on uv alone would leave the runner going), check
     that `pgrep -af "pushlab run"` prints nothing, and start it again.
     Let it finish.
  6. status must show 1,000 runs, all ok, 1,000 distinct run IDs and none
     saved twice. Start the command once more: it must do nothing.
  7. Copy /workspace/runs/s2-kill-resume and /workspace/logs off the pod
     ("Copy runs off the pod"), then stop the pod.
  8. Paste back the same day: status's output, the progress file's speed
     lines, the GPU test's lines and JSON, and whether the copy opened.
     Keep the manifest for prompt 9.

REPORT BACK
End with a plain-language summary for a beginner: what changed, why, how I
can check it myself, and anything I must do by hand. Put pod session 2
first, as a numbered list of exact commands, and say what to paste back.
Explain why a killed run loses at most one batch.
