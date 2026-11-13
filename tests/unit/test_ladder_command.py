"""`pushlab ladder` (pushlab.runs.ladder_command, pushlab.cli_ladder): the
experiment file it writes, what it prints, its refusals, and its help.

Its units are tiny-vl's (the tiny_downloads fixture registers the model),
with invented personas, on the five smoke prompts. The costs fixture gives
tiny-vl a speed of 0.01 reply tokens a second, so the default max_gpu_hours
is large and exact. test_ladder_write.py covers pushlab.runs.ladder itself,
and the doses, seeds and name the command reads from its options.
"""

import doctest
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from pushlab import PROJECT_ROOT, cli_ladder
from pushlab.cli import main
from pushlab.cli_parser import build_parser
from pushlab.runs import ladder_command
from pushlab.runs.expand import expand
from pushlab.runs.spec import read_spec
from pushlab.units.ladder_units import run_push

UNITS = (
    "unit,direction,persona,description,source\n"
    "tiny-vl:L1:N3,+1,Brass Owl,A pompous mechanical owl.,invented\n"
    "tiny-vl:L2:N100,-1,Salt Captain,,\n"
)
DOSES = "0.25,0.5,0.75,1.0,1.25"
SMOKE = "prompts/smoke.jsonl"
# Imports `--help` must not load: they take seconds.
HEAVY = ("torch", "pandas", "transformers")
HELP_IMPORTS = f"""
import sys
from pushlab.cli import main
for argv in (["ladder", "--help"], ["--help"]):
    try:
        main(argv)
    except SystemExit:
        pass
print([name for name in {HEAVY!r} if name in sys.modules])
"""
OWN_IMPORTS = """
import sys
import pushlab.cli_ladder
print([n for n in ("torch", "pandas", "pyarrow", "transformers") if n in sys.modules])
"""
PHRASES = [
    "--units FILE",
    "--doses LIST",
    "--prompts FILE",
    "--out FILE",
    "--name NAME",
    "--seeds LIST",
    "--max-new-tokens N",
    "--batch-size N",
    "--max-gpu-hours H",
    "--controls K",
    "--control-seed S",
    "--baseline",
    "--no-prompt-pass",
    "--overwrite",
    "u01, u02",
    "(default: 160)",
    "(default: 16)",
    "allow_over_ceiling",
    "outside the repository",
    "decision 0020",
    "pushlab run",
    "Exit status: 0",
]


@pytest.fixture(autouse=True)
def quiet_runs_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the runs dir at an empty folder: no speed measured there."""
    monkeypatch.setenv("PUSHLAB_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def units_file(tiny_downloads: object, tmp_path: Path) -> Path:
    """A units file of UNITS, tiny-vl registered."""
    path = tmp_path / "units.csv"
    path.write_text(UNITS, encoding="utf-8")
    return path


@pytest.fixture
def ladder(units_file: Path, tmp_path: Path) -> Callable[..., int]:
    """pushlab ladder on the two units, at DOSES on the smoke prompts, with
    out tmp_path / "experiments" / "tiny-ladder.yaml" and any more options."""

    def run(*options: str, out: Path | None = None) -> int:
        out = out or tmp_path / "experiments" / "tiny-ladder.yaml"
        argv = ["ladder", "--units", str(units_file), "--doses", DOSES]
        return main([*argv, "--prompts", SMOKE, "--out", str(out), *options])

    return run


def _settings(out: str) -> dict[str, str]:
    """The lines of the form 'name  value', by name."""
    lines = [line.split("  ", 1) for line in out.splitlines() if "  " in line]
    return {name.strip(): value.strip() for name, value in lines if name.strip()}


def test_it_writes_the_ladder_and_says_what_it_runs(
    ladder: Callable[..., int],
    costs: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "experiments" / "tiny-ladder.yaml"
    assert ladder() == 0
    spec = read_spec(out)
    assert (spec.name, spec.model, spec.prompts) == ("tiny-ladder", "tiny-vl", SMOKE)
    expansion = expand(spec)
    assert len(expansion.runs) == 2 * 5 * 5
    keys = {run_push(run.record).key for run in expansion.runs}  # type: ignore[union-attr]
    assert keys == {"tiny-vl:L1:N3+", "tiny-vl:L2:N100-"}
    # 50 runs of 160 tokens at 0.01 tokens a second: 222.2 hours, times 1.5.
    assert spec.max_gpu_hours == 333.5
    text = capsys.readouterr().out
    settings = _settings(text)
    assert text.startswith(f"Wrote {out}: the experiment tiny-ladder, on tiny-vl.")
    assert "u01  tiny-vl:L1:N3, direction +1, persona Brass Owl, source invented" in (
        text
    )
    assert "u02  tiny-vl:L2:N100, direction -1, persona Salt Captain" in text
    assert settings["doses"] == "0.25, 0.5, 0.75, 1 and 1.25, in each unit's direction"
    assert settings["prompts"] == f"{SMOKE}: 5 prompts"
    assert settings["seeds"] == "0, greedy"
    assert settings["runs"] == "2 units x 5 doses x 5 prompts x 1 seed = 50"
    assert settings["max_gpu_hours"].startswith("333.5: about 1.5 times the time")
    assert "0.01 reply tokens a second" in settings["max_gpu_hours"]
    assert f"uv run pushlab run {out} --dry-run" in text
    assert "Neuron Persona Library" not in text
    # The runner's own dry run reads it as it is.
    assert main(["run", str(out), "--dry-run", "--runs-dir", str(tmp_path)]) == 0
    assert "to run" in capsys.readouterr().out


def test_baseline_controls_seeds_and_settings_reach_the_file(
    ladder: Callable[..., int],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "custom.yaml"
    options = [
        "--baseline",
        "--controls",
        "1",
        "--control-seed",
        "4",
        "--seeds",
        "0,1",
        "--max-new-tokens",
        "64",
        "--batch-size",
        "12",
        "--max-gpu-hours",
        "0.5",
        "--name",
        "owl-ladder",
        "--no-prompt-pass",
    ]
    assert ladder(*options, out=out) == 0
    spec = read_spec(out)
    assert spec.name == "owl-ladder" and spec.seeds == (0, 1)
    assert spec.decoding == {
        "strategy": "sample",
        "temperature": 1.0,
        "top_p": 1.0,
        "dtype": "bfloat16",
    }
    assert (spec.max_new_tokens, spec.batch_size, spec.max_gpu_hours) == (64, 12, 0.5)
    assert [c.label for c in spec.conditions] == ["baseline", "u01", "u02"]
    assert [(c.controls, c.control_seed) for c in spec.conditions[1:]] == [(1, 4)] * 2
    expansion = expand(spec)
    assert len(expansion.runs) == (50 + 50 + 5) * 2
    for run in expansion.runs:
        assert all(segment["region"] == "reply" for segment in run.record.schedule)
    settings = _settings(capsys.readouterr().out)
    assert settings["seeds"] == "0 and 1, each reply sampled at temperature 1"
    assert settings["runs"] == (
        "2 units x 5 doses x 5 prompts x 2 seeds = 100, plus 10 unpushed "
        "(baseline) and 100 controls: 210 in all"
    )
    assert settings["max_gpu_hours"] == "0.5, as given"


def test_the_default_name_comes_from_the_out_file(
    ladder: Callable[..., int], costs: Path, tmp_path: Path
) -> None:
    out = tmp_path / "Panel 1_Ladder.v2.yaml"
    assert ladder(out=out) == 0
    assert read_spec(out).name == "panel-1-ladder-v2"


def test_an_existing_out_is_replaced_only_with_overwrite(
    ladder: Callable[..., int],
    costs: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "experiments" / "tiny-ladder.yaml"
    out.parent.mkdir()
    out.write_text("kept\n", encoding="utf-8")
    assert ladder() == 1
    assert capsys.readouterr().err == (
        f"pushlab ladder: {out} already exists; pass --overwrite to replace it\n"
    )
    assert out.read_text(encoding="utf-8") == "kept\n"
    assert ladder("--overwrite") == 0
    assert read_spec(out).name == "tiny-ladder"


def test_a_library_unit_is_refused_inside_the_repository(
    tiny_downloads: object,
    costs: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    units = tmp_path / "library.csv"
    units.write_text(UNITS.replace("invented", "library"), encoding="utf-8")
    inside = PROJECT_ROOT / "experiments" / "library-ladder-never-written.yaml"
    argv = ["ladder", "--units", str(units), "--doses", DOSES, "--prompts", SMOKE]
    assert main([*argv, "--out", str(inside)]) == 1
    err = capsys.readouterr().err
    assert err.startswith(f"pushlab ladder: {inside} is inside the repository")
    assert "decision 0020" in err and "/workspace/experiments/" in err
    assert not inside.exists()
    outside = tmp_path / "library-ladder.yaml"
    assert main([*argv, "--out", str(outside)]) == 0
    out = capsys.readouterr().out
    assert "holds the Neuron Persona Library's data" in " ".join(out.split())
    assert "Neuron Persona Library" in outside.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("options", "problem"),
    [
        (["--doses", ""], "--doses is empty"),
        (["--doses", "0.5,0.25"], "--doses must be in increasing order"),
        (["--doses", "-1"], "--doses: -1 is not above 0"),
        (["--doses", "0.5,0.5"], "--doses: 0.5 is the dose 0.5 again"),
        (["--seeds", "0,0"], "--seeds: seed 0 is given twice"),
        (["--seeds", "x"], "--seeds: 'x' is not a seed"),
        (["--control-seed", "1"], "--control-seed goes with --controls"),
        (["--name", "Owl Ladder"], "--name must be lowercase letters and digits"),
        (["--controls", "200", "--max-gpu-hours", "1"], "its 200 controls need"),
        (["--prompts", "prompts/missing.jsonl"], "cannot read the prompt set"),
        (["--units", "MISSING"], "cannot read the units file"),
        (["--units", "SET"], "sets cannot go into experiment files"),
        (["--max-gpu-hours", "1", "--units", "DEEP"], "layer 9 is out of range"),
    ],
)
def test_bad_options_are_refused_and_nothing_is_written(
    ladder: Callable[..., int],
    costs: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    options: list[str],
    problem: str,
) -> None:
    files = {
        "MISSING": tmp_path / "missing.csv",
        "SET": tmp_path / "set.csv",
        "DEEP": tmp_path / "deep.csv",
    }
    files["SET"].write_text("unit\ntiny-vl:set:pair\n", encoding="utf-8")
    files["DEEP"].write_text("unit\ntiny-vl:L9:N3\n", encoding="utf-8")
    options = [str(files.get(option, option)) for option in options]
    assert ladder(*options) == 1
    err = capsys.readouterr().err
    assert err.startswith("pushlab ladder: ") and problem in err, err
    assert not (tmp_path / "experiments").exists()


def test_a_name_the_out_file_cannot_give_and_an_out_folder_are_refused(
    ladder: Callable[..., int],
    costs: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert ladder(out=tmp_path / "___.yaml") == 1
    err = capsys.readouterr().err
    assert "cannot make an experiment name from ___.yaml: give --name" in err
    assert ladder(out=tmp_path) == 1
    assert f"--out must be a file, but {tmp_path} is a folder" in (
        capsys.readouterr().err
    )


def test_an_out_that_cannot_be_written_is_refused(
    ladder: Callable[..., int],
    costs: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    blocker = tmp_path / "blocker"
    blocker.write_text("a file, not a folder\n", encoding="utf-8")
    out = blocker / "tiny-ladder.yaml"
    assert ladder(out=out) == 1
    assert capsys.readouterr().err == (
        f"pushlab ladder: cannot write {out}: File exists\n"
    )


def test_a_model_with_no_speed_needs_max_gpu_hours(
    ladder: Callable[..., int],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert ladder() == 1  # configs/costs.yaml has no speed for tiny-vl
    err = capsys.readouterr().err
    assert "has no speed for tiny-vl" in err and "--max-gpu-hours" in err
    assert ladder("--max-gpu-hours", "2") == 0
    assert read_spec(tmp_path / "experiments" / "tiny-ladder.yaml").max_gpu_hours == 2


def test_the_parser_reads_every_option_and_its_defaults() -> None:
    required = ["--units", "u.csv", "--doses", DOSES, "--prompts", SMOKE, "--out", "o"]
    args = build_parser().parse_args(["ladder", *required])
    assert vars(args) == {
        "command": "ladder",
        "units": Path("u.csv"),
        "doses": DOSES,
        "prompts": Path(SMOKE),
        "out": Path("o"),
        "name": None,
        "seeds": "0",
        "max_new_tokens": 160,
        "batch_size": 16,
        "max_gpu_hours": None,
        "controls": 0,
        "control_seed": None,
        "baseline": False,
        "prompt_pass": True,
        "overwrite": False,
    }
    for bad in (
        ["--max-gpu-hours", "0"],
        ["--max-gpu-hours", "nan"],
        ["--max-new-tokens", "0"],
        ["--batch-size", "x"],
        ["--controls", "-1"],
        ["--control-seed", "1.5"],
    ):
        with pytest.raises(SystemExit) as exit_info:
            build_parser().parse_args(["ladder", *required, *bad])
        assert exit_info.value.code == 2


def test_the_help_says_what_it_does_and_lists_every_option(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["ladder", "--help"])
    assert exit_info.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    for phrase in PHRASES:
        assert phrase in text, phrase


def test_the_installed_command_answers_its_help() -> None:
    pushlab = Path(sys.executable).with_name("pushlab")
    result = subprocess.run(
        [str(pushlab), "ladder", "--help"], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert "usage: pushlab ladder" in result.stdout


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


@pytest.mark.parametrize("module", [cli_ladder, ladder_command])
def test_docstring_examples_run(module: object) -> None:
    result = doctest.testmod(module)  # type: ignore[arg-type]
    assert result.attempted > 0
    assert result.failed == 0
