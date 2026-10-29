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
PINNED_UV = "uv 0.11.32 (x86_64-unknown-linux-gnu)"  # how uv prints its version
IMAGE_UV = "uv 0.13.4 (x86_64-unknown-linux-gnu)"  # the uv in Runpod's image
WARNING = "the pod does not set UV_NO_SYNC=1 itself"
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
    pod_uv_no_sync: bool | str | None = True,
    tools: dict[str, str | None] | None = None,
    **shell_env: str,
) -> tuple[subprocess.CompletedProcess[str], list[str]]:
    """Run setup_pod.sh with fake tools; return the result and the tool calls.

    `pod_uv_no_sync` is what the pod sets UV_NO_SYNC to, as Runpod records it in
    /etc/rp_environment: True means "1", False means not at all, a string means
    that value, and None means there is no such file. `tools`
    replaces the shell code of some fake tools by name, or leaves a tool out
    entirely when given None. Every fake logs to calls.log, which the shell
    code can name as CALLS. `shell_env` adds variables to the calling shell.
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
        "uv": f'echo "uv $*" >> "CALLS"; [ "$1" = --version ] && echo "{PINNED_UV}"',
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
    rp_environment = tmp_path / "rp_environment"
    rp_environment.unlink(missing_ok=True)
    value = "1" if pod_uv_no_sync is True else pod_uv_no_sync
    if value is not None:
        # Runpod's start script writes every pod variable there, tokens too.
        lines = [f'export GITHUB_TOKEN="{GITHUB_SECRET}"']
        if hf_token:
            lines.append(f'export HF_TOKEN="{HF_SECRET}"')
        if value is not False:
            lines.append(f'export UV_NO_SYNC="{value}"')
        rp_environment.write_text("\n".join(lines) + "\n")
    # BASH_ENV, ENV and exported functions could bring back real tools.
    env = {k: v for k, v in os.environ.items() if not k.startswith("BASH_FUNC_")}
    for name in ("HF_TOKEN", "UV_NO_SYNC", "BASH_ENV", "ENV"):
        env.pop(name, None)
    env |= {
        "HOME": str(tmp_path / "home"),
        "PUSHLAB_WORKSPACE": str(tmp_path / "workspace"),
        "PUSHLAB_RP_ENVIRONMENT": str(rp_environment),
        "PATH": f"{bin_dir}{os.pathsep}{system}",
        "GITHUB_TOKEN": GITHUB_SECRET,
    }
    if hf_token:
        env["HF_TOKEN"] = HF_SECRET
    if isinstance(value, str):
        env["UV_NO_SYNC"] = value
    env |= shell_env
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
        start_new_session=True,  # so a fake can press Ctrl-C for the script alone
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
    assert WARNING not in result.stdout


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


def test_compiled_gpu_kernels_stay_on_the_volume(tmp_path: Path) -> None:
    # Triton compiles flash-linear-attention's kernels on first use; on the
    # container disk they would be lost, and compiled again, at every stop.
    _run(tmp_path)
    pod_env = (tmp_path / "workspace" / "pushlab.env").read_text()
    cache = tmp_path / "workspace" / ".cache" / "triton"
    assert f'export TRITON_CACHE_DIR="{cache}"' in pod_env


@pytest.mark.parametrize("pod_value", [False, "0", "", "10"])
def test_warns_when_the_pod_does_not_set_uv_no_sync_to_1(
    tmp_path: Path, pod_value: bool | str
) -> None:
    result, _ = _run(tmp_path, pod_uv_no_sync=pod_value)
    assert result.returncode == 0, _output(result)
    assert result.stdout.count(WARNING) == 2


def test_warns_even_when_the_shell_has_loaded_pushlab_env(tmp_path: Path) -> None:
    # A terminal that loaded pushlab.env has UV_NO_SYNC=1, but the pod may not.
    result, _ = _run(tmp_path, pod_uv_no_sync=False, UV_NO_SYNC="1")
    assert result.stdout.count(WARNING) == 2


@pytest.mark.parametrize(("shell_value", "warnings"), [("1", 0), ("", 2)])
def test_without_runpods_record_the_shell_decides(
    tmp_path: Path, shell_value: str, warnings: int
) -> None:
    result, _ = _run(tmp_path, pod_uv_no_sync=None, UV_NO_SYNC=shell_value)
    assert result.returncode == 0, _output(result)
    assert result.stdout.count(WARNING) == warnings


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
    image_uv = (
        f'echo "image-uv $*" >> "CALLS"; [ "$1" = --version ] && echo "{IMAGE_UV}"'
    )
    fake_tool = (
        'mkdir -p "$HOME/.local/bin"\n'
        'printf \'#!/bin/sh\\necho "{name} $*" >> "CALLS"\\n'
        'echo "{version}"\\n\' > "$HOME/.local/bin/{name}"\n'
        'chmod +x "$HOME/.local/bin/{name}"\n'
    )
    uv_installer = (
        'echo "uv-installer UV_NO_MODIFY_PATH=$UV_NO_MODIFY_PATH" >> "CALLS"\n'
    )
    uv_installer += fake_tool.format(name="uv", version=PINNED_UV)
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
    assert f"found {IMAGE_UV}, not the pinned version" in result.stdout
    assert f"installed {PINNED_UV}" in result.stdout
    assert "installed 2.1.0 (Claude Code)" in result.stdout
    # Every later uv command used the pinned uv, not the image's.
    assert [call for call in calls if call.startswith("image-uv")] == [
        "image-uv --version"
    ]
    assert "uv sync --locked --no-group cpu --extra gpu" in calls


def test_stops_when_the_pinned_uv_is_still_not_found(tmp_path: Path) -> None:
    # The installer ran but left the image's uv first on PATH.
    image_uv = f'[ "$1" = --version ] && echo "{IMAGE_UV}"'
    result, _ = _run(tmp_path, tools={"uv": image_uv, "curl": "echo :"})
    assert result.returncode == 1
    assert f'uv --version now prints "{IMAGE_UV}"' in _output(result)
    assert "Setup stopped at step 2/8" in _log(tmp_path)


def test_stops_when_no_gpu_is_visible(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, tools={"nvidia-smi": "exit 9"})
    assert result.returncode != 0
    assert "No GPU is visible" in _output(result)
    assert calls == []


def test_stops_when_the_driver_is_too_old_for_cuda_13(tmp_path: Path) -> None:
    result, calls = _run(tmp_path, tools={"nvidia-smi": f'echo "{GPU}, 550.54.15"'})
    assert result.returncode != 0
    assert "580" in _output(result)
    assert "Terminate this pod (nothing on it is needed yet)" in _output(result)
    assert calls == []


def test_an_old_driver_on_a_warm_volume_warns_before_terminating(
    tmp_path: Path,
) -> None:
    # A volume that moved to another machine keeps work worth saving.
    _run(tmp_path)
    result, _ = _run(tmp_path, tools={"nvidia-smi": f'echo "{GPU}, 550.54.15"'})
    assert result.returncode != 0
    assert "Terminating this pod deletes /workspace" in _output(result)
    assert "nothing on it is needed yet" not in _output(result)


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
    # Like Ctrl-C in a terminal, it signals every process of the script.
    interrupt = (
        f'[ "$1" = --version ] && echo "{PINNED_UV}"; [ "$1" = sync ] && kill -INT 0'
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
    selected = {item.nodeid.split("::")[0] for item in call.session.items}
    # The GPU check, and one reply from the real model with pushlab baseline.
    assert selected == {"tests/gpu/test_smoke.py", "tests/gpu/test_baseline_smoke.py"}
