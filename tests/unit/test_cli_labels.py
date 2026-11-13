"""The options of pushlab label export and pushlab label check
(pushlab.cli_labels), and pushlab label's dispatch to them
(pushlab.scoring.label_command): their help, the arguments they refuse,
that --help loads nothing heavy, and which function each runs with what.

test_label_export.py runs pushlab label export itself; pushlab label check
has tests of its own.
"""

import argparse
import doctest
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from pushlab import cli_labels
from pushlab.cli import main
from pushlab.scoring import label_command, label_export
from pushlab.scoring.backends import DEFAULT_MODELS
from pushlab.scoring.local_backend import LOCAL_URL

# Each command's help, and what it must say, its options among it.
HELPS = {
    ("label",): [
        "{export,check}",
        "export write a sheet of sampled replies for you to label",
        "check have a judge label the sheet's replies and compare",
        "must agree with your own labels before it scores replies at scale",
        "85% accuracy and a Cohen's kappa of 0.70",
        "on 200 of your labels or more",
        '"Judge validation"',
    ],
    ("label", "export"): [
        "SPEC [SPEC ...]",
        "--out DIR",
        "--rubric NAME",
        "speaks_as, which_persona or coherence",
        "--size N",
        "(default: 200, the plan's labels per rubric)",
        "--seed N",
        "--personas FILE",
        "--persona NAME",
        "--runs-dir DIR",
        "a new or empty folder outside the repository",
        "each reply's ID is its name, a token of the sample and a number",
        "replies never go into git",
        "an equal share",
        "its random controls a share of their own",
        "labels.csv",
        "instructions.md",
        "key.json",
        "never open it while labeling",
        "names no run, condition, unit, dose or prompt",
        "It loads no model and calls no judge",
    ],
    ("label", "check"): [
        "DIR",
        "--labels FILE",
        "(default: DIR/labels.csv)",
        "--backend {api,local}",
        "--model MODEL",
        f"(default for api: {DEFAULT_MODELS['api']})",
        "--base-url URL",
        f"(default: {LOCAL_URL})",
        "--workers N",
        "--cache-dir DIR",
        "$PUSHLAB_CACHE_DIR/judge, else /workspace/cache/judge",
        "--runs-dir DIR",
        "(default: the one the export read, as key.json names it, else "
        "$PUSHLAB_RUNS_DIR, else /workspace/runs)",
        "--out FILE",
        "where to save the record of the results: numbers and hashes, no reply "
        "text (default: DIR/check-<backend>-<model>-<UTC time>.json)",
        "Cohen's kappa",
        "with 200 of your labels or more, at an accuracy of 0.85 and a kappa of "
        "0.70 or more",
        "with fewer it is TOO FEW, whatever its numbers",
        "An empty cell is skipped",
        "never its prompt, run or condition",
        "It writes a record of the results",
        "DIR/disagreements-<backend>-<model>-<UTC time>.csv",
        "wherever --out puts the record",
    ],
}
# What --help must not load, through the parser alone.
HEAVY = ("torch", "pandas", "pyarrow")
PARSER_HELPS = f"""
import sys
from pushlab.cli_parser import build_parser
for argv in {[[*command, "--help"] for command in HELPS]!r}:
    try:
        build_parser().parse_args(argv)
    except SystemExit:
        pass
print([name for name in {HEAVY!r} if name in sys.modules])
"""
# Through pushlab's main, as the installed command runs: pushlab.cli itself
# imports pushlab.store.records, which loads pyarrow, so only these count.
HEAVY_FOR_MAIN = ("torch", "pandas", "transformers")
MAIN_HELPS = f"""
import sys
from pushlab.cli import main
for argv in {[[*command, "--help"] for command in HELPS]!r}:
    try:
        main(argv)
    except SystemExit:
        pass
print([name for name in {HEAVY_FOR_MAIN!r} if name in sys.modules])
"""


@pytest.mark.parametrize("command", list(HELPS))
def test_each_help_says_what_it_does_and_names_its_options(
    command: tuple[str, ...], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main([*command, "--help"])
    assert exit_info.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    for phrase in HELPS[command]:
        assert phrase in text


def test_pushlab_help_lists_label(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "label check a judge against your own labels" in text


@pytest.mark.parametrize(
    "argv",
    [
        ["label"],
        ["label", "sort"],
        ["label", "export"],
        ["label", "export", "e.yaml"],
        ["label", "export", "--out", "sheet"],
        ["label", "export", "e.yaml", "--out", "s", "--rubric", "pirate"],
        ["label", "export", "e.yaml", "--out", "s", "--size", "0"],
        ["label", "export", "e.yaml", "--out", "s", "--seed", "-1"],
        ["label", "export", "e.yaml", "--out", "s", "--seed", "1.5"],
        ["label", "check"],
        ["label", "check", "sheet", "--backend", "cloud"],
        ["label", "check", "sheet", "--workers", "0"],
    ],
)
def test_bad_arguments_stop_with_a_usage_message(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(argv)
    assert exit_info.value.code == 2
    assert "usage" in capsys.readouterr().err


def test_help_loads_nothing_heavy(tmp_path: Path) -> None:
    for script, heavy in [(PARSER_HELPS, HEAVY), (MAIN_HELPS, HEAVY_FOR_MAIN)]:
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.splitlines()[-1] == "[]", heavy


def test_the_installed_command_answers_each_help() -> None:
    pushlab = Path(sys.executable).with_name("pushlab")
    for command, option in [
        (["label"], "{export,check}"),
        (["label", "export"], "--rubric NAME"),
        (["label", "check"], "--backend {api,local}"),
    ]:
        result = subprocess.run(
            [str(pushlab), *command, "--help"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert f"usage: pushlab {' '.join(command)}" in result.stdout
        assert option in result.stdout


def _recorder(status: int) -> tuple[list[tuple[Any, ...]], Any]:
    """A stand-in run function that records how it was called and returns
    status."""
    calls: list[tuple[Any, ...]] = []

    def run(*args: Any, **kwargs: Any) -> int:
        calls.append((args, kwargs))
        return status

    return calls, run


def test_label_export_runs_with_its_options(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, run = _recorder(0)
    monkeypatch.setattr(label_export, "run_label_export", run)
    argv = ["label", "export", "a.yaml", "b.yaml", "--out", "sheet"]
    assert main(argv) == 0
    assert main([*argv, "--rubric", "coherence", "--rubric", "speaks_as"]) == 0
    more = ["--size", "50", "--seed", "7", "--personas", "p.yaml"]
    more += ["--persona", "Brass Owl", "--runs-dir", "runs"]
    assert main([*argv, *more]) == 0
    specs, out = [Path("a.yaml"), Path("b.yaml")], Path("sheet")
    defaults = {
        "rubrics": None,
        "size": 200,
        "seed": 0,
        "personas_file": None,
        "persona": None,
        "runs_dir": None,
    }
    assert calls == [
        ((specs, out), defaults),
        ((specs, out), defaults | {"rubrics": ["coherence", "speaks_as"]}),
        (
            (specs, out),
            defaults
            | {
                "size": 50,
                "seed": 7,
                "personas_file": Path("p.yaml"),
                "persona": "Brass Owl",
                "runs_dir": Path("runs"),
            },
        ),
    ]


def test_label_check_runs_with_its_options(monkeypatch: pytest.MonkeyPatch) -> None:
    calls, run = _recorder(1)
    # A stand-in module: label_check is imported only when check runs.
    stand_in = types.ModuleType("pushlab.scoring.label_check")
    stand_in.run_label_check = run  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pushlab.scoring.label_check", stand_in)
    assert main(["label", "check", "sheet"]) == 1  # its status passes through
    options = ["--labels", "mine.csv", "--backend", "local", "--model", "granite"]
    options += ["--base-url", "http://127.0.0.1:9000/v1", "--workers", "8"]
    options += ["--cache-dir", "cache", "--runs-dir", "runs", "--out", "c.json"]
    assert main(["label", "check", "sheet", *options]) == 1
    defaults = {
        "labels": None,
        "backend": "api",
        "model": None,
        "base_url": None,
        "workers": 4,
        "cache_dir": None,
        "runs_dir": None,
        "out": None,
    }
    assert calls == [
        ((Path("sheet"),), defaults),
        (
            (Path("sheet"),),
            {
                "labels": Path("mine.csv"),
                "backend": "local",
                "model": "granite",
                "base_url": "http://127.0.0.1:9000/v1",
                "workers": 8,
                "cache_dir": Path("cache"),
                "runs_dir": Path("runs"),
                "out": Path("c.json"),
            },
        ),
    ]


def test_an_unknown_subcommand_is_refused() -> None:
    with pytest.raises(ValueError, match="no subcommand 'sort'"):
        label_command.run_label_command(argparse.Namespace(label_command="sort"))


@pytest.mark.parametrize("module", [cli_labels, label_command])
def test_docstring_examples_run(module: types.ModuleType) -> None:
    result = doctest.testmod(module)
    assert result.attempted > 0
    assert result.failed == 0
