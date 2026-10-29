#!/usr/bin/env bash
# Gets a Runpod GPU pod ready for pushlab in one command: checks the GPU,
# installs uv, the locked CUDA environment and the Claude Code CLI, downloads
# the models in configs/models.yaml to /workspace/models and runs the GPU smoke
# tests. Everything slow to rebuild stays on the /workspace volume, so running
# it again, for example after the pod restarts, skips finished work. Each step
# is logged with a timestamp to /workspace/logs/. See docs/runbooks/pod.md.
#
# Usage: bash scripts/setup_pod.sh [--dry-run]
# For the tests, PUSHLAB_WORKSPACE replaces /workspace and PUSHLAB_RP_ENVIRONMENT
# replaces /etc/rp_environment.
set -euo pipefail

UV_VERSION="0.11.32" # the uv that CI and cloud sessions use; pinned on the pod too
MIN_DRIVER=580       # the CUDA 13.0 build of torch needs NVIDIA driver 580+
FRESH_TARGET=1200    # seconds: ready in 20 minutes on a fresh pod,
WARM_TARGET=180      # and in 3 minutes on a warm volume
START=$(date +%s)

usage() {
  echo "Usage: bash scripts/setup_pod.sh [--dry-run]"
  echo "Gets a Runpod GPU pod ready for pushlab (see docs/runbooks/pod.md)."
  echo "  --dry-run  list the eight steps without doing them"
}

DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $arg" >&2
      usage >&2
      exit 2
      ;;
  esac
done

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
WORKSPACE="${PUSHLAB_WORKSPACE:-/workspace}"
POD_ENV="$WORKSPACE/pushlab.env"
DONE="$WORKSPACE/.pushlab-setup-done" # written when a run finishes
SYNC=(uv sync --locked --no-group cpu --extra gpu)
DOWNLOAD=(uv run --no-sync python -m pushlab.models.download
  "$REPO/configs/models.yaml" "$WORKSPACE/models")
SMOKE=(uv run --no-sync pytest -q -m gpu -k smoke)
LOG=""
STEP=""
# Runpod's images come with their own uv, so every shell has uv from the start.
# A plain uv run swaps in the CPU-only torch unless UV_NO_SYNC=1 is set, and only
# a pod environment variable sets it everywhere (pod.md, "Deploy the pod").
# Runpod's start script copies the pod's variables to /etc/rp_environment; read
# them there, because this shell's own UV_NO_SYNC may come from pushlab.env.
RP_ENVIRONMENT="${PUSHLAB_RP_ENVIRONMENT:-/etc/rp_environment}"
POD_UV_NO_SYNC=""
if [[ ! -f $RP_ENVIRONMENT ]]; then
  POD_UV_NO_SYNC=${UV_NO_SYNC:-} # not a Runpod image: trust this shell
elif grep -Eqx '(export )?UV_NO_SYNC="?1"?' "$RP_ENVIRONMENT"; then
  POD_UV_NO_SYNC=1
fi

# A volume where a run has finished before only needs checking, so it gets
# the shorter target.
if [[ -f $DONE ]]; then
  KIND="warm volume"
  TARGET=$WARM_TARGET
else
  KIND="fresh pod"
  TARGET=$FRESH_TARGET
fi

step() {
  STEP="$1/8 ($2)"
  printf '\n[%s] [%s/8] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$1" "$2"
}
note() { printf '      %s\n' "$@"; }
fail() {
  printf '\nStopped: %s\n' "$1" >&2
  shift
  for line in "$@"; do printf '         %s\n' "$line" >&2; done
  exit 1
}
on_exit() {
  local status=$?
  if ((status == 0)) || [[ -z $STEP ]]; then return; fi
  if ((status == 130)); then
    printf '\nSetup was interrupted during step %s.\n' "$STEP" >&2
  else
    printf '\nSetup stopped at step %s. Fix the problem above first.\n' "$STEP" >&2
  fi
  printf 'Running the script again skips the work already finished.\n' >&2
  printf 'Full log: %s\n' "$LOG" >&2
}

# Shows what a step would do in a dry run, and does it otherwise.
run_step() { # number, title, what it does, the function that does it
  step "$1" "$2"
  if ((DRY_RUN)); then
    note "would $3"
  else
    "$4"
  fi
}

check_gpu() {
  local gpus name memory driver first_driver=""
  if ! gpus=$(nvidia-smi --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader 2>/dev/null) || [[ -z $gpus ]]; then
    fail "No GPU is visible: nvidia-smi is missing or lists no GPU." \
      "Deploy the pod with a GPU; see docs/runbooks/pod.md."
  fi
  while IFS=, read -r name memory driver; do
    memory=${memory# } driver=${driver# }
    note "$name, $memory of memory, NVIDIA driver $driver"
    first_driver=${first_driver:-$driver}
  done <<<"$gpus"
  local major=${first_driver%%.*}
  if [[ ! $major =~ ^[0-9]+$ ]] || ((major < MIN_DRIVER)); then
    local advice=("Terminate this pod (nothing on it is needed yet) and deploy a new one;")
    if [[ $KIND == "warm volume" ]]; then
      advice=("Terminating this pod deletes /workspace unless it is a network volume,"
        "so copy off what you need first;")
    fi
    fail "The NVIDIA driver is ${first_driver:-unknown}, but pushlab's CUDA 13.0 build of torch" \
      "needs driver $MIN_DRIVER or newer." "${advice[@]}" \
      "see \"The NVIDIA driver is\" under \"If something goes wrong\" in docs/runbooks/pod.md."
  fi
}

is_pinned_uv() { [[ $1 == "uv $UV_VERSION" || $1 == "uv $UV_VERSION "* ]]; }

# Runpod's images ship their own uv, often another version, so check the
# version rather than whether any uv exists.
install_uv() {
  local found
  found=$(uv --version 2>/dev/null || true)
  if is_pinned_uv "$found"; then
    note "found $found"
    return
  fi
  if [[ -n $found ]]; then note "found $found, not the pinned version"; fi
  note "installing uv $UV_VERSION into ~/.local/bin"
  # Our pushlab.env puts ~/.local/bin on PATH, so the installer leaves the
  # shell profiles alone.
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | env UV_NO_MODIFY_PATH=1 sh
  found=$(uv --version 2>/dev/null || true)
  is_pinned_uv "$found" || fail "Installed uv $UV_VERSION, but uv --version now prints \"${found:-nothing}\"."
  note "installed $found"
}

# The settings every shell on the pod needs. The file lives on the volume;
# ~/.bashrc does not survive a restart, so each run adds the line again.
write_pod_env() {
  cat >"$POD_ENV" <<EOF
# Written by scripts/setup_pod.sh; new shells load it through ~/.bashrc.
# uv's cache, the Python it installs and Hugging Face's files stay on the
# volume. UV_NO_SYNC=1 stops uv run from swapping the CUDA build of torch for
# the CPU-only one (docs/decisions/0002-torch-builds.md).
export PATH="\$HOME/.local/bin:\$PATH"
export UV_CACHE_DIR="$WORKSPACE/.cache/uv"
export UV_PYTHON_INSTALL_DIR="$WORKSPACE/.local/share/uv/python"
export UV_PYTHON_PREFERENCE=only-managed
export UV_NO_SYNC=1
export HF_HOME="$WORKSPACE/.cache/huggingface"
EOF
  local line="source $POD_ENV"
  if ! grep -qxF "$line" "$HOME/.bashrc" 2>/dev/null; then
    printf '\n# pushlab pod settings (scripts/setup_pod.sh)\n%s\n' "$line" >>"$HOME/.bashrc"
  fi
  note "new shells load $POD_ENV"
}

warn_about_uv_no_sync() {
  note "warning: the pod does not set UV_NO_SYNC=1 itself. Until it does, a plain" \
    "uv run in a notebook, or in a terminal opened before this script ran, can" \
    "swap torch for the CPU-only build. Add UV_NO_SYNC with the value 1 through" \
    "Edit Pod (docs/runbooks/pod.md, \"If something goes wrong\")."
}

install_environment() {
  write_pod_env
  # shellcheck source=/dev/null
  source "$POD_ENV"
  if [[ $POD_UV_NO_SYNC != 1 ]]; then warn_about_uv_no_sync; fi
  note "the first time, this downloads about 3 GB and shows no progress bar"
  cd "$REPO"
  "${SYNC[@]}"
}

install_claude() {
  if command -v claude >/dev/null 2>&1; then
    note "found $(claude --version)"
    return
  fi
  note "installing the Claude Code CLI with its native installer"
  curl -fsSL https://claude.ai/install.sh | bash
  command -v claude >/dev/null 2>&1 || fail "Claude Code was installed but is not on PATH."
  note "installed $(claude --version)"
}

check_hf_token() {
  # ${HF_TOKEN:+set} tests the token without ever expanding its value.
  if [[ -z ${HF_TOKEN:+set} ]]; then
    fail "HF_TOKEN is not set, so the models cannot be downloaded." \
      "Give the pod your Hugging Face token as the environment variable HF_TOKEN:" \
      "see \"HF_TOKEN is not set\" under \"If something goes wrong\" in docs/runbooks/pod.md."
  fi
  note "HF_TOKEN is set (its value is never shown)"
}

download_models() {
  cd "$REPO"
  "${DOWNLOAD[@]}"
}

run_smoke_tests() {
  cd "$REPO"
  "${SMOKE[@]}"
}

report_time() {
  local elapsed=$(($(date +%s) - START))
  local took
  took=$(printf '%dm %02ds' $((elapsed / 60)) $((elapsed % 60)))
  if ((elapsed <= TARGET)); then
    note "Finished in $took: beat the $((TARGET / 60))-minute target for a $KIND."
  else
    note "Finished in $took: missed the $((TARGET / 60))-minute target for a $KIND."
  fi
  date >"$DONE"
}

if ((DRY_RUN)); then
  echo "pushlab pod setup, dry run: nothing is installed, downloaded or written."
else
  mkdir -p "$WORKSPACE/logs"
  LOG="$WORKSPACE/logs/setup-$(date +%Y-%m-%d-%H%M%S).log"
  # Copy everything to the log. tee ignores Ctrl-C so it still logs the exit.
  exec > >(
    trap '' INT
    exec tee -a "$LOG"
  ) 2>&1
  trap on_exit EXIT
  trap 'exit 130' INT # without it, on_exit would see status 0 after Ctrl-C
  export PATH="$HOME/.local/bin:$PATH"
  export PYTHONUNBUFFERED=1 # show Python's output as it happens
  echo "pushlab pod setup"
fi
note "repository: $REPO" "workspace:  $WORKSPACE" \
  "this is a $KIND, so the target is $((TARGET / 60)) minutes"
if [[ -n $LOG ]]; then note "log:        $LOG"; fi

run_step 1 "Checking that a GPU is visible" \
  "check with nvidia-smi that a GPU is visible, print its name and memory, and stop unless its driver is $MIN_DRIVER or newer" \
  check_gpu
run_step 2 "Installing uv if missing" \
  "install uv $UV_VERSION into ~/.local/bin unless that version is already installed" \
  install_uv
run_step 3 "Installing the CUDA environment, dev tools and GPU kernels" \
  "write $POD_ENV, make new shells load it, and run: ${SYNC[*]}" \
  install_environment
run_step 4 "Installing the Claude Code CLI if missing" \
  "install the Claude Code CLI with its native installer unless it is already installed" \
  install_claude
run_step 5 "Checking that HF_TOKEN is set" \
  "stop with a message unless HF_TOKEN is set (its value is never shown)" \
  check_hf_token
run_step 6 "Downloading the models in configs/models.yaml" \
  "run: ${DOWNLOAD[*]} (models already there are skipped)" \
  download_models
run_step 7 "Running the GPU smoke tests" \
  "run: ${SMOKE[*]}" \
  run_smoke_tests
run_step 8 "Checking the total time against the target" \
  "print the total time and whether it beat the $((TARGET / 60))-minute target for a $KIND" \
  report_time

if ((DRY_RUN)); then
  printf '\nDry run finished. Run it without --dry-run to set up the pod.\n'
else
  printf '\nThe pod is ready. New terminals pick up its settings by themselves;\n'
  printf 'in this one, run first:  source %s\n' "$POD_ENV"
  printf 'Then sign in to Claude Code with:  claude\n'
  if [[ $POD_UV_NO_SYNC != 1 ]]; then
    echo
    warn_about_uv_no_sync
  fi
fi
