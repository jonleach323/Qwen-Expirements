"""`pushlab doses`'s help, and the doctests of its modules
(pushlab.cli_doses, pushlab.analysis.doses_command, dose_measures and
dose_text).

test_doses_command.py runs the command itself.
"""

import doctest
import subprocess
import sys
from pathlib import Path

import pytest

from pushlab import cli_doses
from pushlab.analysis import dose_measures, dose_text, doses_command
from pushlab.cli import main

# What the help must say, and the imports --help must not load: they are slow.
PHRASES = ["--units FILE", "--judge NAME", "--panel FILE", "--runs-dir DIR", "--out"]
PHRASES += ["pushlab score SPEC --rubric speaks_as --units FILE", "Exit status: 0"]
HELP_IMPORTS = """
import contextlib, sys
from pushlab.cli import main
with contextlib.suppress(SystemExit):
    main(["doses", "--help"])
print([name for name in ("torch", "pandas", "transformers") if name in sys.modules])
"""


def test_the_help_lists_every_option_and_loads_nothing_heavy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["doses", "--help"])
    assert exit_info.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    assert [phrase for phrase in PHRASES if phrase not in text] == []
    pushlab = str(Path(sys.executable).with_name("pushlab"))
    for argv in ([pushlab, "doses", "--help"], [sys.executable, "-c", HELP_IMPORTS]):
        done = subprocess.run(argv, capture_output=True, text=True, cwd=tmp_path)
        assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines()[-1] == "[]"


@pytest.mark.parametrize("module", [doses_command, cli_doses, dose_measures, dose_text])
def test_docstring_examples_run(module: object) -> None:
    result = doctest.testmod(module)  # type: ignore[arg-type]
    assert result.attempted > 0
    assert result.failed == 0
