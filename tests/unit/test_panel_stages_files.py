"""A stage's experiment file (pushlab.panel.stage_files and stage_text,
decision 0023): its text reads back as its settings, comments and all; an
--out is checked; max_gpu_hours is worked out as pushlab ladder does; and a
file is put in place only once it reads back and expands. Also `pushlab
panel` itself (pushlab.panel.command): what it says to do next, and that
it hands select to pushlab.panel.select_command.
"""

import doctest
import math
from pathlib import Path

import pytest

from pushlab import PROJECT_ROOT
from pushlab.cli import main
from pushlab.panel import command, stage_files, stage_text
from pushlab.panel.command import next_steps
from pushlab.panel.settings import read_selection
from pushlab.panel.stage_files import (
    Stage,
    check_out,
    chosen_prompts,
    next_out,
    set_prompts,
    stage_hours,
    stage_mapping,
    write_stage,
)
from pushlab.panel.stage_text import comment, flow, plain
from pushlab.panel.stage_text import stage_text as text_of
from pushlab.runs.ladder import default_hours
from pushlab.schedules.yaml_text import load_yaml

NONE = {"label": "none", "segments": []}
OWL = "qwen3-8b:L9:N4242"


@pytest.mark.parametrize(
    "value",
    [
        "library/L9N4242-",
        "qwen3-8b:set:pirate-top50",
        "/workspace/units/candidates_qwen3-8b.csv",
        "made up, top 2",
        "true",
        "Null",
        "1e3",
        "010",
        "-dash",
        "# not a comment",
        'say "hoo" \\ here',
        "",
        "café ünïcode",
        "line\nbreak and\u0085more",
        "emoji 🦉",
        "key: value",
        "[not a list]",
        0.1,
        -0.0,
        1e-05,
        1.5e300,
        7,
        True,
        None,
        {"unit": OWL, "alpha": -1.125, "end": None, "region": "prompt"},
        [[17, 1234, 0.52], [18, 99, -0.31]],
        {"entries": [], "source": "a, b: {c}"},
    ],
)
def test_flow_reads_back_as_itself(value: object) -> None:
    text = flow(value)
    assert "\n" not in text
    assert load_yaml(f"value: {text}\n") == {"value": value}
    assert load_yaml(f"- {text}  # a note\n") == [value]


def test_flow_writes_plain_text_where_it_is_safe() -> None:
    assert flow({"key": "library/L9N4242-", "doses": [0.75]}) == (
        "{key: library/L9N4242-, doses: [0.75]}"
    )
    assert flow("made up") == '"made up"' and flow("é") == '"\\xe9"'
    with pytest.raises(ValueError, match="finite numbers"):
        flow(math.inf)
    with pytest.raises(TypeError, match="cannot write a set"):
        flow({1, 2})


def test_comments_are_one_line_of_printable_text_each() -> None:
    assert plain("Brass\nOwl  the  Great ") == "Brass Owl the Great"
    lines = comment("word " * 40, "  ")
    assert all(line.startswith("  # ") and len(line) <= 79 for line in lines)


def _stage(**changes: object) -> Stage:
    settings = {
        "model": "qwen3-8b",
        "prompt_ids": ("heldout32-01", "heldout32-02"),
        "candidates": {
            "file": "units/candidates_qwen3-8b.csv",
            "keys": ["library/L9N4242-", {"key": "mayner/pirate", "doses": [0.5]}],
        },
        "conditions": ({"label": "candidate", "each_candidate": {"doses": [1.0]}},),
        "runs": 4,
        "about": "A made-up stage.\nIts second line.",
        "key_notes": ("8/8 speaks_as, Brass\nOwl", ""),
        "condition_notes": {0: "Every candidate."},
    }
    return Stage(**(settings | changes))  # type: ignore[arg-type]


def test_a_stage_s_text_reads_back_as_its_settings_with_its_comments() -> None:
    stage = _stage()
    mapping = stage_mapping(stage, read_selection(), "s3-stage-x", 2.25)
    text = text_of(stage, mapping, "About 1.5 times the time it takes.")
    assert load_yaml(text) == mapping
    lines = text.splitlines()
    assert lines[0] == "# A made-up stage. Its second line."
    assert "    - library/L9N4242-  # 8/8 speaks_as, Brass Owl" in lines
    assert "    - {key: mayner/pirate, doses: [0.5]}" in lines
    assert lines[lines.index("max_gpu_hours: 2.25") - 1].startswith("# About 1.5")
    assert lines[lines.index("conditions:") + 1] == "  # Every candidate."
    assert [mapping[k] for k in ("decoding", "seeds", "calibration")] == [
        {"strategy": "greedy"},
        [0],
        "fixed",
    ]
    assert (mapping["max_new_tokens"], mapping["batch_size"]) == (160, 64)
    segments = [{"unit": OWL, "alpha": 1.0, "start": 0, "end": None, "region": "reply"}]
    plain_stage = _stage(
        prompt_ids=None,
        candidates=None,
        conditions=({"label": "p1-pulse", "segments": segments}, NONE),
    )
    mapping = stage_mapping(plain_stage, read_selection(), "s3-preview", 1)
    text = text_of(plain_stage, mapping)
    assert load_yaml(text) == mapping and "prompt_ids" not in mapping
    segment = f"{{unit: {OWL}, alpha: 1.0, start: 0, end: null, region: reply}}"
    assert f"      - {segment}" in text.splitlines()
    assert "    segments: []" in text.splitlines()


def test_an_out_is_checked(tmp_path: Path) -> None:
    source = tmp_path / "screen.yaml"
    source.write_text("x\n", encoding="utf-8")
    check_out(tmp_path / "b.yaml", source, overwrite=False)  # nothing to refuse
    with pytest.raises(ValueError, match="is a folder"):
        check_out(tmp_path, source, overwrite=True)
    with pytest.raises(ValueError, match="the file the stage reads"):
        check_out(tmp_path / "." / "screen.yaml", source, overwrite=True)
    with pytest.raises(ValueError, match="already exists; pass --overwrite"):
        check_out(source, tmp_path / "other.yaml", overwrite=False)
    check_out(source, tmp_path / "other.yaml", overwrite=True)
    inside = PROJECT_ROOT / "experiments" / "s3" / "stage-b.yaml"
    with pytest.raises(ValueError, match="inside the repository") as caught:
        check_out(inside, source, overwrite=True)
    assert "/workspace/experiments/stage-b.yaml" in str(caught.value)


def test_next_out_names_the_next_stage_s_file() -> None:
    folder = Path("/workspace/experiments")
    assert next_out(folder / "s3-stage-b.yaml", "b", "ladder") == (
        folder / "s3-stage-ladder.yaml"
    )
    assert next_out(folder / "s3_ladder.yaml", "ladder", "confirm") == (
        folder / "s3_confirm.yaml"
    )
    assert next_out(folder / "best.yaml", "b", "ladder") == folder / "best-ladder.yaml"


def test_max_gpu_hours_is_given_or_worked_out_from_the_speed(
    costs: Path, tmp_path: Path
) -> None:
    assert stage_hours("tiny-vl", 10, 64, 0.5) == (0.5, "given with --max-gpu-hours")
    for bad in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="--max-gpu-hours must be a number"):
            stage_hours("tiny-vl", 10, 64, bad)
    hours, how = stage_hours("tiny-vl", 10, 64, None, tmp_path)
    assert hours == default_hours(10, 64, 0.01)
    assert how.startswith("about 1.5 times the time its 640 reply tokens take at 0.01")
    with pytest.raises(ValueError, match="no speed for qwen3-8b: add one there, or"):
        stage_hours("qwen3-8b", 10, 64, None, tmp_path)


def test_the_set_s_prompts_are_chosen_from_its_start_or_end() -> None:
    prompts = set_prompts(read_selection())
    assert len(prompts) == 32 and prompts[0] == "heldout32-01"
    assert chosen_prompts(prompts, 16, "ladder_prompts")[-1] == "heldout32-16"
    assert chosen_prompts(prompts, 16, "preview_prompts", last=True)[0] == (
        "heldout32-17"
    )
    with pytest.raises(ValueError, match="ladder_prompts is 33, but the prompt set"):
        chosen_prompts(prompts, 33, "ladder_prompts")


def test_a_stage_s_file_is_written_only_once_it_reads_back_and_expands(
    tmp_path: Path,
) -> None:
    stage = _stage(prompt_ids=None, candidates=None, conditions=(NONE,), runs=32)
    mapping = stage_mapping(stage, read_selection(), "s3-none", 1.0)
    text = text_of(stage, mapping)
    out = tmp_path / "experiments" / "none.yaml"
    spec, expansion = write_stage(text, mapping, out)
    assert out.read_text(encoding="utf-8") == text and len(expansion.runs) == 32
    assert spec.name == "s3-none" and list(out.parent.iterdir()) == [out]
    with pytest.raises(FileExistsError, match="pass --overwrite"):
        write_stage(text, mapping, out)
    # A neuron the model lacks: refused, and nothing is left.
    far = {"label": "far", "segments": [{"unit": "qwen3-8b:L99:N1", "alpha": 1.0}]}
    bad = _stage(prompt_ids=None, candidates=None, conditions=(far,))
    mapping = stage_mapping(bad, read_selection(), "s3-far", 1.0)
    with pytest.raises(ValueError, match="would not run, so nothing was written"):
        write_stage(text_of(bad, mapping), mapping, tmp_path / "far.yaml")
    with pytest.raises(ValueError, match="does not read back as its settings"):
        write_stage(text, mapping, tmp_path / "other.yaml")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["experiments"]


def test_next_steps_say_what_to_run_judge_and_write_next() -> None:
    built = Stage("qwen3-8b", None, {"file": "units/c.csv"}, (), 1, "")
    out, source = Path("/w/s3-stage-ladder.yaml"), Path("/w/s3-stage-b.yaml")
    steps = next_steps("ladder", out, source, built)
    assert steps[1] == f"  uv run pushlab run {out} --dry-run"
    units = "--units units/c.csv"
    assert f"  uv run pushlab score {out} --rubric speaks_as {units}" in steps
    assert f"  uv run pushlab score {out} --rubric coherence" in steps
    confirm = "  uv run pushlab panel stage confirm /w/s3-stage-ladder.yaml --out "
    assert f"{confirm}/w/s3-stage-confirm.yaml" in steps
    out = Path("/w/s3-stage-confirm.yaml")
    select = f"  uv run pushlab panel select {out} --ladder {source} --categories "
    assert next_steps("confirm", out, source, built)[-1] == f"{select}/w/categories.csv"
    preview = next_steps("preview", Path("/w/p.yaml"), source, built)
    report = "  uv run pushlab report /w/p.yaml --out /workspace/reports/p.md"
    assert preview[-1] == report


def test_select_is_dispatched_and_refuses_a_file_it_cannot_read(
    capsys: pytest.CaptureFixture[str],
) -> None:
    argv = ["panel", "select", "c.yaml", "--ladder", "l.yaml", "--categories"]
    assert main([*argv, "/w/k.csv"]) == 1
    assert capsys.readouterr().err == (
        "pushlab panel select: c.yaml: cannot read it: No such file or directory\n"
    )


@pytest.mark.parametrize("module", [stage_files, stage_text, command])
def test_the_examples_in_the_docstrings_hold(module: object) -> None:
    result = doctest.testmod(module)  # type: ignore[arg-type]
    assert result.failed == 0 and result.attempted > 0
