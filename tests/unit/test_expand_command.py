"""`pushlab expand`: an experiment file's runs, listed or counted, none run."""

import doctest
import subprocess
import sys
from pathlib import Path

import pytest

from pushlab import PROJECT_ROOT
from pushlab.cli import main
from pushlab.runs import expand_command
from pushlab.runs.expand import expand
from pushlab.runs.spec import read_spec

EXAMPLE = PROJECT_ROOT / "experiments" / "p0" / "s2_pulse_lengths.yaml"


@pytest.fixture
def spec(tmp_path: Path) -> Path:
    """A spec of three pulse lengths with two controls, on two prompts and two seeds."""
    prompts = tmp_path / "two.jsonl"
    prompts.write_text(
        '{"prompt_id": "p-1", "version": 1, "prompt": "Say hi."}\n'
        '{"prompt_id": "p-2", "version": 1, "prompt": "Count to three."}\n',
        encoding="utf-8",
    )
    path = tmp_path / "pulses.yaml"
    path.write_text(
        f"""\
name: pulse-lengths
model: qwen3.5-4b
prompts: {prompts}
decoding: {{strategy: sample, temperature: 0.7}}
seeds: [0, 1]
max_new_tokens: 64
batch_size: 8
max_gpu_hours: 1
conditions:
  - label: pulse
    shape: pulse
    params: {{unit: qwen3.5-4b:L14:N8949, alpha: 1.0, start: 0}}
    grid: {{length: [4, 16, 64]}}
    controls: 2
    control_seed: 7
""",
        encoding="utf-8",
    )
    return path


def test_expand_help_lists_the_options(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["expand", "--help"])
    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "SPEC" in help_text and "--summary" in help_text


def test_the_installed_command_answers_help() -> None:
    command = Path(sys.executable).with_name("pushlab")
    result = subprocess.run(
        [str(command), "expand", "--help"], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert "--summary" in result.stdout


def test_the_summary_counts_each_condition_s_runs(
    spec: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["expand", str(spec), "--summary"]) == 0
    out, err = capsys.readouterr()
    assert err == ""
    expansion = expand(read_spec(spec))
    lines = out.splitlines()
    assert lines[:2] == [
        "experiment      pulse-lengths",
        f"spec sha256     {expansion.spec_sha256}",
    ]
    assert "decoding        sample, temperature 0.7, top_p 1.0, bfloat16" in lines
    assert "seeds           0, 1" in lines
    assert "batches         8, each at most 8 runs" in lines  # 9 per prompt and seed
    assert lines[-7:] == [
        "condition        own  controls  total",
        "pulse length=4     4         8     12",
        "pulse length=16    4         8     12",
        "pulse length=64    4         8     12",
        "all               12        24     36",
        "",
        "36 runs, of which 24 are controls, at most 2,304 reply tokens. Nothing "
        "was run.",
    ]


def test_without_summary_every_run_gets_a_line(
    spec: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["expand", str(spec)]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1 + 36
    first = "pulse-lengths-000001"
    assert [line.split("\t") for line in lines[:4]] == [
        [
            "run_id",
            "condition",
            "control",
            "prompt_id",
            "seed",
            "batch_key",
            "control_of",
        ],
        [first, "pulse length=4", "0", "p-1", "0", "b000001:0", ""],
        ["pulse-lengths-000002", "pulse length=4", "1", "p-1", "0", "b000001:1", first],
        ["pulse-lengths-000003", "pulse length=4", "2", "p-1", "0", "b000001:2", first],
    ]
    assert lines[-1].startswith("pulse-lengths-000036\tpulse length=64\t2\tp-2\t1\t")


def test_the_example_file_expands_to_fifty_runs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["expand", str(EXAMPLE), "--summary"]) == 0
    out = capsys.readouterr().out
    assert "50 runs, of which 30 are controls" in out


def test_an_invalid_condition_is_refused_naming_the_file_and_condition(
    spec: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text = spec.read_text(encoding="utf-8").replace("L14:N8949", "L40:N8949")
    spec.write_text(text, encoding="utf-8")
    assert main(["expand", str(spec), "--summary"]) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith(f"pushlab expand: {spec}: condition 'pulse length=4': ")
    assert "layer 40 is out of range" in err


@pytest.mark.parametrize(
    "text, problem",
    [
        (None, "cannot read it: No such file or directory"),
        ("name: [", "is not valid YAML"),
        (b"\xff", "not UTF-8 text"),
    ],
)
def test_a_file_that_cannot_be_read_is_refused(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    text: str | bytes | None,
    problem: str,
) -> None:
    path = tmp_path / "broken.yaml"
    if isinstance(text, bytes):
        path.write_bytes(text)
    elif text is not None:
        path.write_text(text, encoding="utf-8")
    assert main(["expand", str(path)]) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith(f"pushlab expand: {path}: ") and problem in err


def test_warnings_go_to_stderr_and_the_runs_still_come(
    spec: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    text = spec.read_text(encoding="utf-8").replace("start: 0", "start: 70")
    spec.write_text(text, encoding="utf-8")
    assert main(["expand", str(spec), "--summary"]) == 0
    out, err = capsys.readouterr()
    assert "36 runs" in out
    warnings = err.splitlines()
    assert len(warnings) == 3  # one per pulse length
    assert warnings[0].startswith(
        "pushlab expand: warning: condition 'pulse length=4': segments[0]"
    )
    assert "never pushes" in warnings[0]


def test_doctests() -> None:
    result = doctest.testmod(expand_command)
    assert result.attempted > 0
    assert result.failed == 0
