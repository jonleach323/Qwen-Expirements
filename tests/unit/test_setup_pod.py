"""scripts/setup_pod.sh, run with fake GPU tools in a temporary workspace.

Fake nvidia-smi, uv, claude and curl programs stand in for the real ones, so
these tests need no GPU, install nothing and download nothing. PATH holds only
the fakes and the few system commands the script uses, so a missing fake can
never fall through to a real uv.
"""

import os
import re
import shutil
import signal
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SETUP_POD = ROOT / "scripts" / "setup_pod.sh"
HF_SECRET = "hf_test_secret_value_123"
GITHUB_SECRET = "github_pat_test_secret_value_456"
GPU = "NVIDIA GeForce RTX 4090, 24564 MiB"
SYSTEM_TOOLS = ("bash", "sh", "env", "date", "dirname", "mkdir", "tee", "cat")
SYSTEM_TOOLS += ("grep", "chmod")
STEPS = [
    "Checking that a GPU is visible",
    "Installing uv if missing",
    "Installing the CUDA environment, dev tools and GPU kernels",
    "Installing the Claude Code CLI if missing",
    "Checking that HF_TOKEN is set",
    "Downloading the models in configs/models.yaml",
    "Running the GPU smoke tests",
    "Checking the total time against the target",
]


def _run(
    tmp_path: Path,
    *args: str,
    hf_token: bool = True,
    pod_uv_no_sync: bool = True,
    tools: dict[str, str | None] | None = None,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """Run setup_pod.sh with fake tools; return the result and the tool calls.

    `tools` replaces the shell code of some fake tools by name, or leaves a
    tool out entirely when given None. Every fake logs to calls.log, which
    the shell code can name as CALLS.
    """
    bin_dir = tmp_path / "bin"
    system = tmp_path / "system"
    for folder in (bin_dir, system, tmp_path / "home"):
        folder.mkdir(exist_ok=True)
    for name in SYSTEM_TOOLS:
        if not (system / name).exists():
            (system / name).symlink_to(shutil.which(name))
    calls = tmp_path / "calls.log"
    fakes = {
        "nvidia-smi": f'echo "{GPU}, 580.82.07"',
        "uv": 'echo "uv $*" >> "CALLS"; [ "$1" = --version ] && echo uv 0.11.32',
        "claude": 'echo "claude $*" >> "CALLS"; echo "2.1.0 (Claude Code)"',
        "curl": 'echo "curl $*" >> "CALLS"; exit 1',
        **(tools or {}),
    }
    for name, body in fakes.items():
        tool = bin_dir / name
        tool.unlink(missing_ok=True)
        if body is not None:
            tool.write_text(f"#!/bin/sh\n{body.replace('CALLS', str(calls))}\nexit 0\n")
            tool.chmod(0o755)
    env = {
        **os.environ,
        "HOME": str(tmp_path / "home"),
        "PUSHLAB_WORKSPACE": str(tmp_path / "workspace"),
        "PATH": f"{bin_dir}{os.pathsep}{system}",
        "GITHUB_TOKEN": GITHUB_SECRET,
    }
    for name in ("HF_TOKEN", "UV_NO_SYNC"):
        env.pop(name, None)
    if hf_token:
        env["HF_TOKEN"] = HF_SECRET
    if pod_uv_no_sync:
        env["UV_NO_SYNC"] = "1"
    for name, body in fakes.items():  # a fake must never resolve to a real tool
        found = shutil.which(name, path=env["PATH"])
        assert found == (str(bin_dir / name) if body is not None else None), name
    result = subprocess.run(
        ["bash", str(SETUP_POD), *args],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    lines = calls.read_text().splitlines() if calls.exists() else []
    return result, lines


def _output(result: subprocess.CompletedProcess[str]) -> str:
    return result.stdout + result.stderr


def _log(tmp_path: Path) -> str:
    (log,) = (tmp_path / "workspace" / "logs").glob("setup-*.log")
    return log.read_text()


def _assert_steps_in_order(text: str) -> None:
    positions = [text.find(f"[{n}/8] {title}") for n, title in enumerate(STEPS, 1)]
    assert -1 not in positions, positions
    assert positions == sorted(positions)


def test_dry_run_lists_all_eight_steps_in_order(tmp_path: Path) -> None:
    result, _ = _run(tmp_path, "--dry-run")
    assert result.returncode == 0, _output(result)
    _assert_steps_in_order(result.stdout)


def test_dry_run_does_nothing_and_prints_no_secrets(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, "--dry-run")
    assert result.returncode == 0, _output(result)
    assert calls == []
    assert not (tmp_path / "workspace").exists()
    assert not (tmp_path / "home" / ".bashrc").exists()
    assert HF_SECRET not in _output(result)
    assert GITHUB_SECRET not in _output(result)


def test_full_run_calls_the_tools_in_order_and_logs_each_step(
    tmp_path: Path,
) -> None:
    result, calls = _run(tmp_path)
    assert result.returncode == 0, _output(result)
    _assert_steps_in_order(result.stdout)
    workspace = tmp_path / "workspace"
    # Exactly these uv commands: a stray plain `uv sync` would swap in the
    # CPU-only torch on the pod (decision 0002).
    assert [call for call in calls if call.startswith("uv ")] == [
        "uv --version",
        "uv sync --locked --no-group cpu --extra gpu",
        f"uv run --no-sync python -m pushlab.models.download "
        f"{ROOT}/configs/models.yaml {workspace}/models",
        "uv run --no-sync pytest -q -m gpu -k smoke",
    ]
    assert not any(call.startswith("curl") for call in calls)
    stamped = r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\] \[(\d)/8\] "
    assert re.findall(stamped, _log(tmp_path), re.MULTILINE) == list("12345678")
    assert f"{GPU} of memory" in result.stdout
    beat = r"Finished in \d+m \d\ds: beat the 20-minute target for a fresh pod\."
    assert re.search(beat, result.stdout)
    assert "warning" not in result.stdout


def test_never_prints_or_logs_secret_values(tmp_path: Path) -> None:
    result, _ = _run(tmp_path)
    for text in (_output(result), _log(tmp_path)):
        assert HF_SECRET not in text
        assert GITHUB_SECRET not in text


def test_new_shells_keep_the_cuda_build(tmp_path: Path) -> None:
    # A plain `uv run` would re-sync to the CPU-only torch build (decision 0002).
    _run(tmp_path)
    pod_env = tmp_path / "workspace" / "pushlab.env"
    assert "export UV_NO_SYNC=1" in pod_env.read_text()
    bashrc = (tmp_path / "home" / ".bashrc").read_text()
    assert bashrc.count(f"source {pod_env}") == 1


def test_warns_when_the_pod_does_not_set_uv_no_sync(tmp_path: Path) -> None:
    result, _ = _run(tmp_path, pod_uv_no_sync=False)
    assert result.returncode == 0, _output(result)
    assert result.stdout.count("the pod does not set UV_NO_SYNC=1 itself") == 2


def test_second_run_counts_as_warm_and_repeats_nothing(tmp_path: Path) -> None:
    _run(tmp_path)
    result, _ = _run(tmp_path)
    assert result.returncode == 0, _output(result)
    assert "beat the 3-minute target for a warm volume." in result.stdout
    bashrc = (tmp_path / "home" / ".bashrc").read_text()
    assert bashrc.count("pushlab.env") == 1


def test_a_slow_run_says_it_missed_the_target(tmp_path: Path) -> None:
    # A fake clock: the start reads 0 seconds and the end reads 25 minutes.
    started = tmp_path / "clock-started"
    clock = f"""if [ "$1" = +%s ]; then
  if [ -e "{started}" ]; then echo 1500; else : > "{started}"; echo 0; fi
else
  exec {shutil.which("date")} "$@"
fi"""
    result, _ = _run(tmp_path, tools={"date": clock})
    assert result.returncode == 0, _output(result)
    expected = "Finished in 25m 00s: missed the 20-minute target for a fresh pod."
    assert expected in result.stdout


def test_installs_the_pinned_uv_and_claude_code_when_needed(tmp_path: Path) -> None:
    # Runpod's image brings another uv, and Claude Code is missing. The fake
    # curl hands out installers that put fake programs in ~/.local/bin.
    image_uv = 'echo "image-uv $*" >> "CALLS"; [ "$1" = --version ] && echo uv 0.13.4'
    fake_tool = (
        'mkdir -p "$HOME/.local/bin"\n'
        'printf \'#!/bin/sh\\necho "{name} $*" >> "CALLS"\\n'
        'echo "{version}"\\n\' > "$HOME/.local/bin/{name}"\n'
        'chmod +x "$HOME/.local/bin/{name}"\n'
    )
    uv_installer = (
        'echo "uv-installer UV_NO_MODIFY_PATH=$UV_NO_MODIFY_PATH" >> "CALLS"\n'
    )
    uv_installer += fake_tool.format(name="uv", version="uv 0.11.32")
    claude_installer = fake_tool.format(name="claude", version="2.1.0 (Claude Code)")
    curl = f"""echo "curl $*" >> "CALLS"
case "$*" in
  *astral.sh/uv/0.11.32/install.sh) cat <<'EOF'
{uv_installer}EOF
  ;;
  *claude.ai/install.sh) cat <<'EOF'
{claude_installer}EOF
  ;;
  *) exit 22 ;;
esac"""
    result, calls = _run(tmp_path, tools={"uv": image_uv, "claude": None, "curl": curl})
    assert result.returncode == 0, _output(result)
    assert "curl -LsSf https://astral.sh/uv/0.11.32/install.sh" in calls
    assert "uv-installer UV_NO_MODIFY_PATH=1" in calls
    assert "curl -fsSL https://claude.ai/install.sh" in calls
    assert "found uv 0.13.4, not the pinned version" in result.stdout
    assert "installed uv 0.11.32" in result.stdout
    assert "installed 2.1.0 (Claude Code)" in result.stdout
    # Every later uv command used the pinned uv, not the image's.
    assert [call for call in calls if call.startswith("image-uv")] == [
        "image-uv --version"
    ]
    assert "uv sync --locked --no-group cpu --extra gpu" in calls


def test_stops_when_no_gpu_is_visible(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, tools={"nvidia-smi": "exit 9"})
    assert result.returncode != 0
    assert "No GPU is visible" in _output(result)
    assert calls == []


def test_stops_when_the_driver_is_too_old_for_cuda_13(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, tools={"nvidia-smi": f'echo "{GPU}, 550.54.15"'})
    assert result.returncode != 0
    assert "580" in _output(result)
    assert "Terminate this pod" in _output(result)
    assert calls == []


def test_stops_with_a_clear_message_when_hf_token_is_missing(
    tmp_path: Path,
) -> None:
    result, calls = _run(tmp_path, hf_token=False)
    assert result.returncode != 0
    assert "HF_TOKEN is not set" in _output(result)
    assert "Setup stopped at step 5/8" in _log(tmp_path)
    assert not any("pushlab.models.download" in call for call in calls)
    assert not any("pytest" in call for call in calls)
    assert not (tmp_path / "workspace" / ".pushlab-setup-done").exists()


def test_ctrl_c_still_says_where_setup_stopped(tmp_path: Path) -> None:
    if signal.getsignal(signal.SIGINT) is signal.SIG_IGN:
        pytest.skip("Ctrl-C is ignored in this test run, so it cannot be simulated")
    # The fake uv interrupts the script during step 3, as Ctrl-C would.
    interrupt = (
        '[ "$1" = --version ] && echo uv 0.11.32; [ "$1" = sync ] && kill -INT $PPID'
    )
    result, _ = _run(tmp_path, tools={"uv": interrupt})
    assert result.returncode == 130
    assert "interrupted during step 3/8" in _log(tmp_path)
    assert not (tmp_path / "workspace" / ".pushlab-setup-done").exists()


def test_rejects_an_unknown_option(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, "--bogus")
    assert result.returncode == 2
    assert "Usage" in _output(result)
    assert calls == []


def test_the_smoke_test_command_selects_only_the_gpu_smoke_tests(
    pytester: pytest.Pytester,
) -> None:
    # Step 7 runs `pytest -m gpu -k smoke`; collect what that selects.
    recorder = pytester.inline_run(
        "--collect-only",
        "-p",
        "no:cacheprovider",
        "-m",
        "gpu",
        "-k",
        "smoke",
        f"--rootdir={ROOT}",
        f"--config-file={ROOT / 'pyproject.toml'}",
        str(ROOT / "tests"),
    )
    (call,) = recorder.getcalls("pytest_collection_finish")
    selected = [item.nodeid for item in call.session.items]
    assert selected
    assert all(node.startswith("tests/gpu/test_smoke.py::") for node in selected)
