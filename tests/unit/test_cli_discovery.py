"""The options of pushlab fetch and pushlab candidates (pushlab.cli_discovery):
their help, the arguments they refuse, and that --help loads nothing heavy.

test_fetch_command.py runs pushlab fetch, and test_candidates_command.py
pushlab candidates.
"""

import doctest
import subprocess
import sys
from pathlib import Path

import pytest

from pushlab import cli_discovery
from pushlab.cli import main
from pushlab.cli_discovery import SOURCES_MODEL_ID, models_with_sources
from pushlab.discovery import mayner_import

# Each command's help, and what it must say, its options among it.
HELPS = {
    ("fetch",): [
        "{library,mayner}",
        "DATA_DIR ($PUSHLAB_DATA_DIR, else /workspace/data)",
        "into DATA_DIR/library/qwen",
        "into DATA_DIR/mayner",
        "Files already there are skipped",
        "FETCHED.json beside them records where each came from and when",
        "run them on the GPU pod",
    ],
    ("fetch", "library"): [
        "--data-dir DIR",
        "--refresh",
        "about 26 MB",
        "into DATA_DIR/library/qwen",
        "A file already there is skipped unless --refresh",
        "FETCHED.json there records where each file came from and when",
        "The library states no licence and asks to be cited",
        "Run it on the GPU pod",
    ],
    ("fetch", "mayner"): [
        "--data-dir DIR",
        "--all",
        "with his own download script",
        "279 files of about 105 MB",
        "(with --all his whole bundle, 638 files of about 1 GB)",
        "hosts under hf.co",
        "into DATA_DIR/mayner/bundle",
        "skipping those already there",
        "FETCHED.json in DATA_DIR/mayner records where they came from and when",
        "His data is under CC BY 4.0",
        "Run it on the GPU pod",
    ],
    ("candidates",): [
        "--model {qwen3-8b}",
        "--out FILE",
        "--data-dir DIR",
        "--include-sensitive",
        "--overwrite",
        "duplicate names within each source",
        "the source's licence and credit",
        "It loads no model",
    ],
}
# Imports `--help` must not load: they take seconds.
HEAVY = ("torch", "pandas", "transformers")
HELP_IMPORTS = f"""
import sys
from pushlab.cli import main
for argv in {[[*command, "--help"] for command in HELPS]!r}:
    try:
        main(argv)
    except SystemExit:
        pass
print([name for name in {HEAVY!r} if name in sys.modules])
"""


@pytest.mark.parametrize("command", list(HELPS))
def test_each_help_says_what_it_fetches_where_and_names_its_options(
    command: tuple[str, ...], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([*command, "--help"])
    assert exit_info.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    for phrase in HELPS[command]:
        assert phrase in text
    assert "qwen3.5-4b" not in text


def test_pushlab_help_lists_both_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "fetch download a source of persona candidates" in text
    assert "candidates one list of both sources' persona candidates" in text


@pytest.mark.parametrize(
    "argv",
    [
        ["fetch"],
        ["fetch", "zoo"],
        ["fetch", "library", "--all"],
        ["fetch", "mayner", "--refresh"],
        ["candidates", "--out", "c.csv"],
        ["candidates", "--model", "qwen3-8b"],
        ["candidates", "--model", "tiny", "--out", "c.csv"],
    ],
)
def test_bad_arguments_stop_with_a_usage_message(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2
    assert "usage" in capsys.readouterr().err


def test_the_installed_command_answers_both_helps() -> None:
    pushlab = Path(sys.executable).with_name("pushlab")
    for command, option in [("fetch", "{library,mayner}"), ("candidates", "--out")]:
        result = subprocess.run(
            [str(pushlab), command, "--help"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert f"usage: pushlab {command}" in result.stdout
        assert option in result.stdout


def test_help_loads_nothing_heavy(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-c", HELP_IMPORTS],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "[]"


def test_only_the_model_both_sources_describe_has_sources() -> None:
    assert models_with_sources() == ["qwen3-8b"]
    assert SOURCES_MODEL_ID == mayner_import.MODEL_ID


def test_docstring_examples_run() -> None:
    result = doctest.testmod(cli_discovery)
    assert result.attempted > 0
    assert result.failed == 0
