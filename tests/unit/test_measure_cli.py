"""`pushlab measure`'s help, and the doctests of its modules
(pushlab.cli_measure, pushlab.scoring.measure_command and measure_rows).

test_measure_command.py runs the command itself.
"""

import doctest
import subprocess
import sys
from pathlib import Path

import pytest

from pushlab import cli_measure
from pushlab.cli import main
from pushlab.cli_parser import build_parser
from pushlab.scoring import measure_command, measure_rows

HELP_IMPORTS = """
import sys
from pushlab.cli import main
for argv in (["measure", "--help"], ["--help"]):
    try:
        main(argv)
    except SystemExit:
        pass
print([n for n in ("torch", "pandas", "transformers") if n in sys.modules])
"""
OWN_IMPORTS = """
import sys
import pushlab.cli_measure
print([n for n in ("torch", "pandas", "pyarrow", "transformers") if n in sys.modules])
"""
PHRASES = (
    "SPEC|--perplexity|--batch-size N|--runs-dir DIR|--dry-run|scores table|"
    "repetition|word_variety|capability|-answers.jsonl|GPU job|Exit status: 0"
).split("|")


def test_the_help_lists_every_option_and_the_defaults(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["measure", "--help"])
    assert exit_info.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    assert [phrase for phrase in PHRASES if phrase not in text] == []
    args = build_parser().parse_args(["measure", "e.yaml"])
    assert (args.command, args.perplexity, args.batch_size) == ("measure", False, None)
    assert (args.spec, args.runs_dir, args.dry_run) == (Path("e.yaml"), None, False)
    with pytest.raises(SystemExit):
        build_parser().parse_args(["measure", "e.yaml", "--batch-size", "0"])


@pytest.mark.parametrize("script", [HELP_IMPORTS, OWN_IMPORTS])
def test_help_loads_nothing_heavy(tmp_path: Path, script: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "[]"


@pytest.mark.parametrize("module", [measure_command, measure_rows, cli_measure])
def test_docstring_examples_run(module: object) -> None:
    result = doctest.testmod(module)  # type: ignore[arg-type]
    assert result.attempted > 0
    assert result.failed == 0
