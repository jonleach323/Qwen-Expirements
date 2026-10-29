"""What a manifest records about the code and machine: pushlab.store.environment."""

import doctest
import hashlib
import importlib.metadata
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import transformers

from pushlab import PROJECT_ROOT
from pushlab.store import environment
from pushlab.store.environment import (
    PACKAGES,
    code_commit,
    code_is_dirty,
    driver_version,
    environment_info,
    file_sha256,
    gpu_name,
    package_versions,
)

ROOT = Path(__file__).resolve().parents[2]


def _git(folder: Path, *args: str) -> str:
    settings = ["user.name=Test", "user.email=test@example.com", "commit.gpgsign=false"]
    command = ["git"] + [part for item in settings for part in ("-c", item)]
    result = subprocess.run(
        [*command, *args], cwd=folder, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def test_the_project_root_is_this_checkout() -> None:
    assert PROJECT_ROOT == ROOT
    assert (PROJECT_ROOT / "uv.lock").is_file()


def test_code_commit_is_the_checked_out_commit() -> None:
    commit = code_commit(ROOT)
    assert commit == _git(ROOT, "rev-parse", "HEAD")
    assert re.fullmatch(r"[0-9a-f]{40}", commit)


def test_a_folder_outside_git_has_no_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Stop git from looking for a repository above tmp_path.
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert code_commit(tmp_path) is None
    assert code_is_dirty(tmp_path) is None


def test_changed_tracked_files_make_the_code_dirty(tmp_path: Path) -> None:
    _git(tmp_path, "init", "-q")
    (tmp_path / "a.py").write_text("x = 1\n")
    _git(tmp_path, "add", "a.py")
    _git(tmp_path, "commit", "-q", "-m", "first")
    assert code_is_dirty(tmp_path) is False
    # New files that git does not track, such as run outputs, do not count.
    (tmp_path / "notes.txt").write_text("not tracked\n")
    assert code_is_dirty(tmp_path) is False
    (tmp_path / "a.py").write_text("x = 2\n")
    assert code_is_dirty(tmp_path) is True


def test_file_sha256_hashes_the_lockfile() -> None:
    lockfile = ROOT / "uv.lock"
    assert file_sha256(lockfile) == hashlib.sha256(lockfile.read_bytes()).hexdigest()


def test_a_missing_file_has_no_hash(tmp_path: Path) -> None:
    assert file_sha256(tmp_path / "uv.lock") is None


def test_a_cpu_has_no_gpu_name() -> None:
    assert gpu_name("cpu") is None
    assert gpu_name(torch.device("cpu")) is None


def test_no_nvidia_smi_means_no_driver_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "")
    assert driver_version() is None


def test_the_driver_version_comes_from_nvidia_smi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A fake nvidia-smi for a pod with two GPUs: one line per GPU.
    fake = tmp_path / "nvidia-smi"
    fake.write_text(
        "#!/bin/sh\n"
        'if [ "$*" = "--query-gpu=driver_version --format=csv,noheader" ]; then\n'
        "  printf '580.95.05\\n580.95.05\\n'\n"
        "else\n"
        "  echo unexpected >&2; exit 1\n"
        "fi\n"
    )
    fake.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert driver_version() == "580.95.05"


def test_package_versions_are_the_installed_ones() -> None:
    assert package_versions(["torch", "no-such-package"]) == {
        "torch": importlib.metadata.version("torch"),
        "no-such-package": None,
    }
    # The GPU kernels for Qwen3.5's linear-attention layers are among them.
    assert {"flash-linear-attention", "causal-conv1d", "triton"} <= set(PACKAGES)


def test_environment_info_describes_this_machine() -> None:
    info = environment_info("cpu")
    assert info == {
        "code_commit": code_commit(ROOT),
        "code_dirty": code_is_dirty(ROOT),
        "lockfile_sha256": file_sha256(ROOT / "uv.lock"),
        "gpu": None,
        "cuda": torch.version.cuda,
        "driver": driver_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": sys.version.split()[0],
        "packages": package_versions(),
    }


def test_docstring_examples_run() -> None:
    result = doctest.testmod(environment)
    assert result.attempted > 0
    assert result.failed == 0
