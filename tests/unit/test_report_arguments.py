"""`pushlab report`'s two modes and what it says when they are mixed up."""

from pathlib import Path

import pytest

from pushlab.cli import main


def test_an_experiment_file_after_conditions_gets_a_hint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    # --conditions takes every word after it, the experiment file too.
    argv = ["report", "--conditions", "none", "exp.yaml", "--out", "r.md"]
    with pytest.raises(SystemExit) as stopped:
        main(argv)
    assert stopped.value.code == 2
    assert "give the experiment file before --conditions" in capsys.readouterr().err


def test_no_experiment_file_and_no_run_folders_is_an_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        main(["report", "--out", "r.md"])
    assert "give an experiment file, or run folders with --runs" in (
        capsys.readouterr().err
    )


def test_an_experiment_file_and_run_folders_together_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["report", "exp.yaml", "--runs", str(tmp_path), "--out", "r.md"]
    assert main(argv) == 1
    assert "give an experiment file or --runs, not both" in capsys.readouterr().err


@pytest.mark.parametrize(
    "extra", [["--conditions", "none"], ["--runs-dir", "runs"]], ids=str
)
def test_experiment_options_with_run_folders_are_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: list[str]
) -> None:
    argv = ["report", "--runs", str(tmp_path), *extra, "--out", "r.md"]
    assert main(argv) == 1
    assert "go with an experiment file, not with --runs" in capsys.readouterr().err
