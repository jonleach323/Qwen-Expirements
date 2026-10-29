# 0003: The GPU pod keeps everything slow to rebuild on its /workspace volume

Date: 2026-10-29 · Sprint: S0 · Status: accepted

## Context

- A Runpod pod has two disks. The container disk, which holds everything
  outside `/workspace`, is wiped whenever the pod stops. The volume disk at
  `/workspace` survives stops and restarts. It is deleted when the pod is
  terminated, which Runpod also does when the account balance reaches $0.
- Runpod's PyTorch images bring their own uv in `/bin` (0.13.4 in the
  October 2026 images), so every shell on the pod has uv from the start.
- The plan's target is a ready pod in 20 minutes when fresh and 3 minutes on
  a warm volume. A fresh pod downloads about 3 GB of CUDA packages and the
  9 GB Qwen3.5-4B model.
- Decision 0002: on the pod, a plain `uv sync` or `uv run` would swap the
  CUDA build of torch for the CPU-only one. The pod must sync with
  `uv sync --no-group cpu --extra gpu` and set `UV_NO_SYNC=1`.
- The Hugging Face and GitHub tokens are pod environment variables, never
  files in the repository.

## Decision

`scripts/setup_pod.sh` prepares the pod in eight steps; see
`docs/runbooks/pod.md`. Where things live:

| What | Where | After the pod stops |
| --- | --- | --- |
| The repository and its `.venv` | `/workspace/Qwen-Expirements` | kept |
| uv's package cache | `/workspace/.cache/uv` | kept |
| The Python 3.11 that uv installs | `/workspace/.local/share/uv/python` | kept |
| Models | `/workspace/models/<registry name>` | kept |
| Hugging Face's own files (`HF_HOME`) | `/workspace/.cache/huggingface` | kept |
| Setup logs | `/workspace/logs` | kept |
| Settings for new shells | `/workspace/pushlab.env`, loaded from `~/.bashrc` | the file is kept; each run adds the `~/.bashrc` line again |
| uv 0.11.32 and the Claude Code CLI | `~/.local/bin`, ahead of the image's `/bin/uv` on PATH | lost; each run reinstalls them in seconds |
| The Claude sign-in | `~/.claude` | lost; sign in again with `claude` |

- **The pinned uv.** Step 2 installs uv 0.11.32, the version CI and cloud
  sessions use, unless that exact version is found. The image's newer uv
  also reads the lockfile, but pinning keeps the pod on the same uv as CI.
- **uv uses only the Python it installs** (`UV_PYTHON_PREFERENCE=only-managed`),
  into the volume. The `.venv` links to its interpreter, so an interpreter on
  the container disk would break the environment at every restart.
- **`UV_NO_SYNC=1` is also a pod environment variable**, set in Runpod's
  deploy settings (runbook, "Deploy the pod", step 3). `pushlab.env` sets it
  too, but `~/.bashrc` only loads that file once the script has run since
  the pod started, while the image's uv works in every shell from the start.
  Only a pod variable covers terminals and notebooks opened before the
  script runs. The script warns when the pod does not set it.
- **Models.** `configs/models.yaml` names models from the registry.
  `python -m pushlab.models.download` fetches each one with
  `huggingface_hub.snapshot_download` into its own folder. It first turns
  the registry's revision into a commit, so every file comes from the same
  commit, and writes `.pushlab-download.json` (ID, revision, commit) only
  after the download finishes. A model whose record matches the registry's
  ID and revision is skipped. An interrupted download has no record, so the
  next run downloads it again: the files that finished are kept, and a file
  that was still downloading starts over. Partial files that a killed
  download leaves behind are deleted first. A new revision in the registry
  downloads again, and `snapshot_download` fetches only the files that
  changed.
- **Checking the result.** When the Hub cannot list a commit's files,
  `snapshot_download` returns the folder as it is, perhaps half full, with
  only a warning. So the module checks the folder against the commit's file
  list before it writes the record, and stops if the list or any file is
  missing.
- **Fresh or warm.** A run counts as warm, with the 3-minute target, when
  `/workspace/.pushlab-setup-done` exists. Only a run that passes every step
  writes it.
- **Driver.** Step 1 stops unless the NVIDIA driver is 580 or newer, which
  the CUDA 13.0 build of torch needs (decision 0002). Stopping there, before
  3 GB of downloads, makes the error cheap.
- **GPU kernels** come as prebuilt wheels pinned in `uv.lock`
  (decision 0002), so nothing compiles on the pod.
- **Secrets.** The script checks `HF_TOKEN` without expanding its value and
  never prints the environment. `huggingface_hub` reads `HF_TOKEN` by itself.
  Git reads `GITHUB_TOKEN` through a credential helper at the moment it needs
  it, so git never writes the token to disk. Runpod's own start script does
  copy every pod variable, tokens included, into `/etc/rp_environment` on
  the container disk, which is wiped at every stop. The tests run the script
  with fake token values and check that they appear nowhere in its output or
  log.

## Why this shape

- **Not a custom Docker image.** An image with everything installed would
  start faster, but it needs a container registry and a rebuild after every
  `uv.lock` change. One script on a stock PyTorch image is simpler for one
  researcher, and the volume makes the second start fast anyway.
- **A record file, not a file check.** Checking for `config.json` would
  count an interrupted download as finished.
- **One plain folder per model** (`local_dir`) instead of Hugging Face's
  shared cache. It is easy to find, and easy to delete when a model is no
  longer needed, which the plan's cost control asks for.
- **Claude Code's sign-in stays on the container disk.** Keeping it on the
  volume (with `CLAUDE_CONFIG_DIR`) would save one sign-in per start, but it
  would leave a credential on the volume, which outlasts every restart and,
  for a network volume, the pod itself. Signing in again takes a minute.

## Consequences

- Never run plain `uv sync` on the pod. CLAUDE.md and the runbook say so. To
  update the environment, rerun the script.
- A pod deployed without the `UV_NO_SYNC` variable is unprotected after
  every restart until the script runs again. The script's warning and the
  runbook point this out.
- A warm start after a restart should take about a minute: reinstalling uv
  and Claude Code, checking the environment and running the smoke tests.
- To drop a model, remove it from `configs/models.yaml` and delete its
  folder under `/workspace/models`.
- `configs/` is a new top-level folder for machine settings. Experiment
  settings still belong in `experiments/`.
- The script has only run with fake tools so far. Its first real run is the
  S0 pod task (prompt 8), which should record the measured times here.
