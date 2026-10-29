"""What a manifest records about the code and the machine a run used."""

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import transformers

from pushlab import PROJECT_ROOT


def _run(command: list[str], folder: Path | None = None) -> str | None:
    """A command's output, or None if it is missing or fails."""
    try:
        result = subprocess.run(
            command, cwd=folder, capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def code_commit(root: Path = PROJECT_ROOT) -> str | None:
    """The git commit checked out in root, or None outside a git checkout.

    Example:
        >>> code_commit()  # doctest: +SKIP
        'cda6963...'
    """
    return _run(["git", "rev-parse", "HEAD"], root) or None


def code_is_dirty(root: Path = PROJECT_ROOT) -> bool | None:
    """True if files git tracks in root have uncommitted changes.

    Then the code that ran is not exactly the recorded commit. New files git
    does not track, such as run outputs, do not count. None outside git.

    Example:
        >>> code_is_dirty(Path("/")) is None
        True
    """
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], root)
    return None if status is None else status != ""


def file_sha256(path: Path) -> str | None:
    """The SHA-256 of a file's bytes, such as uv.lock's, or None if it is missing.

    Example:
        >>> file_sha256(Path("/no/such/uv.lock")) is None
        True
    """
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except FileNotFoundError:
        return None


def gpu_name(device: torch.device | str) -> str | None:
    """The GPU's name, such as 'NVIDIA H100 80GB HBM3', or None for a CPU.

    Example:
        >>> gpu_name("cpu") is None
        True
    """
    device = torch.device(device)
    return torch.cuda.get_device_name(device) if device.type == "cuda" else None


def driver_version() -> str | None:
    """The NVIDIA driver version from nvidia-smi, or None without one.

    Example:
        >>> driver_version()  # doctest: +SKIP
        '580.95.05'
    """
    query = ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    output = _run(query)
    return output.splitlines()[0].strip() if output else None


def environment_info(
    device: torch.device | str, root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    """The code, lockfile, GPU and library versions, for a manifest.

    Example:
        >>> sorted(environment_info("cpu"))  # doctest: +NORMALIZE_WHITESPACE
        ['code_commit', 'code_dirty', 'cuda', 'driver', 'gpu', 'lockfile_sha256',
         'python', 'torch', 'transformers']
    """
    return {
        "code_commit": code_commit(root),
        "code_dirty": code_is_dirty(root),
        "lockfile_sha256": file_sha256(root / "uv.lock"),
        "gpu": gpu_name(device),
        "cuda": torch.version.cuda,
        "driver": driver_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": sys.version.split()[0],
    }
