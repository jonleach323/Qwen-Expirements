"""Checks on the shell scripts in scripts/."""

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = sorted((ROOT / "scripts").glob("*.sh"))


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
    fake_nvidia_smi = tmp_path / "nvidia-smi"
    fake_nvidia_smi.write_text("#!/bin/sh\necho 'GPU 0: NVIDIA GeForce RTX 4090'\n")
    fake_nvidia_smi.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}"}
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "cloud_setup.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode != 0
    assert "GPU" in result.stderr
