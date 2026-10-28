#!/usr/bin/env bash
# Prepares a Claude Code cloud session for pushlab's unit tests: uv, Python 3.11
# and the locked CPU-only environment with the dev tools. It never downloads
# model weights and is safe to run again. Paste it into the cloud environment's
# "Setup script" box, or run `bash scripts/cloud_setup.sh` in a session.
# See docs/runbooks/cloud-sessions.md.
set -euo pipefail

UV_VERSION="0.11.32" # the uv that CI uses (.github/workflows/tests.yml)
START=$(date +%s)

step() { printf '\n[%s/4] %s (%ss so far)\n' "$1" "$2" "$(($(date +%s) - START))"; }
note() { printf '      %s\n' "$*"; }

# The cloud's network gateway re-signs HTTPS traffic with a certificate that
# only the system's certificate store trusts, and a setup script runs before
# the session points uv at that store. So tell uv to use the system store.
# UV_SYSTEM_CERTS replaces the older UV_NATIVE_TLS, which uv warns about.
export UV_SYSTEM_CERTS="${UV_SYSTEM_CERTS:-${UV_NATIVE_TLS:-true}}"
unset UV_NATIVE_TLS

step 1 "Checking that this is not a GPU machine"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  echo "This machine has a GPU. This script installs the CPU-only torch build," >&2
  echo "which would replace the CUDA build on a GPU pod, so it stops here." >&2
  exit 1
fi
note "no GPU, as expected in a cloud session"

step 2 "Making sure uv is installed"
export PATH="$HOME/.local/bin:$PATH"
if command -v uv >/dev/null 2>&1; then
  note "found $(uv --version)"
else
  note "not found, installing uv $UV_VERSION"
  # Astral's installer first; PyPI if astral.sh is not on the network allowlist.
  curl -LsSf "https://astral.sh/uv/$UV_VERSION/install.sh" | sh ||
    python3 -m pip install --quiet --user --break-system-packages \
      --root-user-action=ignore "uv==$UV_VERSION"
  if ! command -v uv >/dev/null 2>&1; then
    echo "uv was installed but is not on PATH; open a new shell and try again." >&2
    exit 1
  fi
  note "installed $(uv --version)"
fi

step 3 "Installing Python 3.11 and the locked packages (CPU-only torch, dev tools)"
# The repository is the folder above this script. When the script is pasted
# into the cloud settings instead, look in the current folder and /home/user.
is_pushlab() { grep -qs '^name = "pushlab"' "$1/pyproject.toml"; }
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)
repo=""
for dir in "$script_dir/.." "$PWD" /home/user/*; do
  if is_pushlab "$dir"; then
    repo=$(cd "$dir" && pwd)
    break
  fi
done
if [[ -z "$repo" ]]; then
  note "The pushlab repository is not on this machine, so there is nothing to"
  note "install. This happens when the cloud environment is used for another"
  note "repository. In a pushlab session, 'uv run' installs the packages."
  exit 0
fi
note "repository: $repo"
cd "$repo"
uv sync --locked

step 4 "Checking the installed versions"
HF_HUB_OFFLINE=1 uv run --no-sync python -c '
import sys

import torch
import transformers

print(f"      Python        {sys.version.split()[0]}")
print(f"      torch         {torch.__version__}")
print(f"      transformers  {transformers.__version__}")
'

printf '\nReady in %ss. Run the unit tests with: uv run pytest -q\n' "$(($(date +%s) - START))"
