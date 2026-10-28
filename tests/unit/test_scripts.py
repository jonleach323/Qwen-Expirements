"""Checks on the shell scripts in scripts/."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = sorted((ROOT / "scripts").rglob("*.sh"))
CLOUD_SETUP = ROOT / "scripts" / "cloud_setup.sh"


def test_there_are_scripts_to_check() -> None:
    assert SCRIPTS


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda path: path.name)
def test_script_has_no_bash_syntax_errors(script: Path) -> None:
    result = subprocess.run(
        ["bash", "-n", str(script)], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr


def test_cloud_setup_refuses_to_run_on_a_gpu_machine(tmp_path: Path) -> None:
    # On a GPU pod it would swap the CUDA torch build for the CPU-only one.
    # A fake uv leaves a marker if it is ever called, and HOME points here so
    # the real uv in ~/.local/bin is not found: a broken guard touches nothing.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    marker = tmp_path / "uv-was-called"
    for name, body in [
        ("nvidia-smi", "echo 'GPU 0: NVIDIA GeForce RTX 4090'"),
        ("uv", f"touch '{marker}'; exit 1"),
    ]:
        tool = bin_dir / name
        tool.write_text(f"#!/bin/sh\n{body}\n")
        tool.chmod(0o755)
    env = {
        **os.environ,
        "HOME": str(tmp_path),
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
    }
    result = subprocess.run(
        ["bash", str(CLOUD_SETUP)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode != 0
    assert "GPU" in result.stderr
    assert not marker.exists()


def test_cloud_setup_makes_uv_use_the_system_certificates() -> None:
    # Without this, a fresh cloud machine fails with "invalid peer certificate":
    # the network gateway's certificate is only in the system store, and setup
    # scripts run before the session configures uv to read it.
    assert 'export UV_SYSTEM_CERTS="${UV_SYSTEM_CERTS:-' in CLOUD_SETUP.read_text()
