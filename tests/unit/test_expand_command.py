"""`pushlab expand`: an experiment file's runs, listed or counted, none run."""

import csv
import doctest
import io
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from pushlab import PROJECT_ROOT
from pushlab.cli import main
from pushlab.runs import expand_command
from pushlab.runs.expand import expand
from pushlab.runs.spec import read_spec

EXAMPLE = PROJECT_ROOT / "experiments" / "p0" / "s2_pulse_lengths.yaml"
SpecText = Callable[..., str]


@pytest.fixture
def spec(spec_text: SpecText, tmp_path: Path) -> Path:
    """A file of three pulse lengths with two controls, on two prompts and seeds."""
    path = tmp_path / "pulses.yaml"
    path.write_text(spec_text(), encoding="utf-8")
    return path


def _summary(path: Path, capsys: pytest.CaptureFixture[str]) -> list[str]:
    assert main(["expand", str(path), "--summary"]) == 0
    out, err = capsys.readouterr()
    assert err == ""
    return out.splitlines()


def test_expand_help_lists_the_options(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["expand", "--help"])
    assert exit_info.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "SPEC" in help_text and "--summary" in help_text
    assert "print a line naming the columns and every run" in help_text


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
    lines = _summary(spec, capsys)
    expansion = expand(read_spec(spec))
    assert lines[:2] == [
        "experiment      pulse-lengths",
        f"spec sha256     {expansion.spec_sha256}",
    ]
    assert f"known sha256    {expansion.known_sha256}" in lines
    assert "decoding        sample, temperature 0.7, top_p 1.0, bfloat16" in lines
    assert "seeds           0, 1" in lines
    # 9 runs per prompt and seed, in batches of at most 8: 8, then 1.
    assert "batches         8, of 1 to 8 runs each" in lines
    assert "max_gpu_hours   1" in lines
    assert lines[-7:] == [
        "condition        own  controls  total",
        "pulse length=4     4         8     12",
        "pulse length=16    4         8     12",
        "pulse length=64    4         8     12",
        "(all)             12        24     36",
        "",
        "36 runs, of which 24 are controls, at most 2,304 reply tokens. Nothing "
        "was run.",
    ]


def test_the_summary_counts_one_of_a_thing_in_the_singular(
    spec_text: SpecText, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prompt = tmp_path / "one.jsonl"
    prompt.write_text('{"prompt_id": "p", "version": 1, "prompt": "Hi."}\n')
    one = "conditions:\n  - {label: none, segments: []}\n"
    path = tmp_path / "one.yaml"
    path.write_text(
        spec_text(one, prompts=str(prompt), seeds="[0]", max_gpu_hours="0.1234567")
    )
    lines = _summary(path, capsys)
    assert f"prompts         {prompt}, 1 prompt" in lines
    assert "known sha256" not in "\n".join(lines)  # no controls to draw
    assert "batches         1, of 1 run each" in lines
    assert "max_gpu_hours   0.1234567" in lines  # every digit
    assert lines[-1] == (
        "1 run, of which 0 are controls, at most 64 reply tokens. Nothing was run."
    )
    path.write_text(
        spec_text(
            "conditions:\n  - {label: one, segments: [{unit: qwen3.5-4b:L14:N1, "
            "alpha: 1.0}], controls: 1, control_seed: 0}\n",
            prompts=str(prompt),
            seeds="[0]",
        )
    )
    assert _summary(path, capsys)[-1].startswith("2 runs, of which 1 is a control,")


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


def test_a_prompt_id_with_a_tab_or_line_break_is_quoted(
    spec_text: SpecText, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    prompts = tmp_path / "odd.jsonl"
    prompts.write_text(
        '{"prompt_id": "a\\tb", "version": 1, "prompt": "Hi."}\n'
        '{"prompt_id": "c\\nd", "version": 1, "prompt": "Bye."}\n'
    )
    path = tmp_path / "odd.yaml"
    path.write_text(spec_text(prompts=str(prompts)))
    assert main(["expand", str(path)]) == 0
    rows = list(csv.reader(io.StringIO(capsys.readouterr().out), delimiter="\t"))
    assert len(rows) == 1 + 36 and {len(row) for row in rows} == {7}
    assert [rows[1][3], rows[-1][3]] == ["a\tb", "c\nd"]


def test_the_example_file_expands_to_fifty_runs(
    capsys: pytest.CaptureFixture[str],
) -> None:
    lines = _summary(EXAMPLE, capsys)
    assert lines[-1].startswith("50 runs, of which 30 are controls")
    # The README shows this part of the summary as it is.
    readme = (PROJECT_ROOT / "experiments" / "README.md").read_text(encoding="utf-8")
    (shown,) = re.findall(r"```text\n(.*?)```", readme, flags=re.DOTALL)
    assert "\n".join(lines).endswith(shown.rstrip("\n"))


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
