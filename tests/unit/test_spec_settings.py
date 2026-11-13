"""An experiment file's calibration and prompt_ids (pushlab.runs.spec and
expand, decision 0023): their checks, the hash, the runs prompt_ids give,
and `pushlab expand --summary` and `pushlab run --dry-run`, which show
them. Files written before them keep their hash: the example file's is
pinned in test_spec.py."""

import importlib.util
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from pushlab.cli import main
from pushlab.discovery.candidates import write_candidates
from pushlab.runs import spec as spec_module
from pushlab.runs.expand import expand
from pushlab.runs.spec import OPTIONAL_KEYS, spec_from_yaml
from pushlab.units import calibration_mode

SpecText = Callable[..., str]
_SPEC = importlib.util.spec_from_file_location(
    "candidate_lists", Path(__file__).with_name("candidate_lists.py")
)
lists = importlib.util.module_from_spec(_SPEC)  # type: ignore[arg-type]
_SPEC.loader.exec_module(lists)  # type: ignore[union-attr]

NONE = "conditions:\n  - {label: none, segments: []}\n"


def _with(spec_text: SpecText, setting: str = "", conditions: str = NONE) -> str:
    """spec_text's experiment of conditions with one seed, and setting after
    its hours."""
    return spec_text(conditions, seeds="[0]", max_gpu_hours=f"1\n{setting}")


def test_the_calibration_modes_are_calibration_mode_s() -> None:
    # spec.py keeps a copy, as calibration_mode loads transformers.
    assert spec_module.CALIBRATIONS == calibration_mode.CALIBRATION_MODES
    assert OPTIONAL_KEYS["calibration"] == calibration_mode.DEFAULT_MODE


def test_calibration_is_fixed_unless_the_file_says_per_prompt(
    spec_text: SpecText,
) -> None:
    plain = spec_from_yaml(_with(spec_text))
    fixed = spec_from_yaml(_with(spec_text, "calibration: fixed"))
    per_prompt = spec_from_yaml(_with(spec_text, "calibration: per_prompt"))
    assert (plain.calibration, per_prompt.calibration) == ("fixed", "per_prompt")
    # fixed written out is the default: the same spec and hash, and no key.
    assert fixed == plain and fixed.sha256 == plain.sha256
    assert "calibration" not in plain.to_mapping()
    assert per_prompt.to_mapping()["calibration"] == "per_prompt"
    assert per_prompt.sha256 != plain.sha256


def test_prompt_ids_run_those_prompts_in_their_order(spec_text: SpecText) -> None:
    every = expand(spec_from_yaml(_with(spec_text))).runs
    assert [run.record.prompt_id for run in every] == ["p-1", "p-2"]
    spec = spec_from_yaml(_with(spec_text, "prompt_ids: [p-2, p-1]"))
    assert spec.prompt_ids == ("p-2", "p-1")
    runs = expand(spec).runs
    assert [run.record.prompt_id for run in runs] == ["p-2", "p-1"]
    assert [run.record.run_id for run in runs] == [
        "pulse-lengths-000001",
        "pulse-lengths-000002",
    ]
    (one,) = expand(spec_from_yaml(_with(spec_text, "prompt_ids: [p-2]"))).runs
    assert one.record.prompt_id == "p-2"


def test_prompt_ids_are_in_the_hash_with_their_order(spec_text: SpecText) -> None:
    plain = spec_from_yaml(_with(spec_text))
    both = spec_from_yaml(_with(spec_text, "prompt_ids: [p-1, p-2]"))
    swapped = spec_from_yaml(_with(spec_text, "prompt_ids: [p-2, p-1]"))
    assert both.to_mapping()["prompt_ids"] == ["p-1", "p-2"]
    assert "prompt_ids" not in plain.to_mapping()
    assert len({plain.sha256, both.sha256, swapped.sha256}) == 3


def test_a_prompt_id_the_set_lacks_is_refused_by_expand(
    spec_text: SpecText, two_prompts: Path
) -> None:
    spec = spec_from_yaml(_with(spec_text, "prompt_ids: [p-1, p-9, p-0]"))
    with pytest.raises(ValueError) as error:
        expand(spec)
    assert str(error.value) == (
        f"prompt_ids names 'p-9' and 1 more, which {two_prompts} lacks; give IDs "
        "of its prompts, such as 'p-1'"
    )


REFUSALS = {
    "calibration": (
        "calibration: per-prompt",
        "calibration must be fixed or per_prompt: fixed scales doses by each "
        "layer's residual size from calibration/, per_prompt by each prompt's "
        "own, as the Neuron Persona Library does; not 'per-prompt'",
    ),
    "calibration-null": ("calibration: null", "not None"),
    "prompt-ids-not-a-list": (
        "prompt_ids: p-1",
        "prompt_ids must list the IDs of the prompts to run, such as "
        "[heldout32-01, heldout32-02], not 'p-1'",
    ),
    "prompt-ids-empty": ("prompt_ids: []", "prompt_ids must list the IDs"),
    "prompt-id-a-number": (
        "prompt_ids: [p-1, 2]",
        "prompt_ids[1]: a prompt ID must be text, such as heldout32-01, but YAML "
        "read 2; put it in quotes",
    ),
    "prompt-id-blank": ("prompt_ids: [' ']", "prompt_ids[0]: a prompt ID must be"),
    "prompt-id-twice": (
        "prompt_ids: [p-1, p-2, p-1]",
        "prompt_ids gives 'p-1' twice; each prompt runs once",
    ),
    "unknown-key": (
        "calibrations: fixed",
        "unknown key 'calibrations'; it takes name, model, prompts, decoding, "
        "seeds, max_new_tokens, batch_size, max_gpu_hours, conditions, "
        "group_by_length, calibration, prompt_ids, candidates",
    ),
}


@pytest.mark.parametrize("setting, problem", REFUSALS.values(), ids=REFUSALS.keys())
def test_refusals_name_the_problem(
    spec_text: SpecText, setting: str, problem: str
) -> None:
    with pytest.raises(ValueError) as error:
        spec_from_yaml(_with(spec_text, setting))
    assert problem in str(error.value)


def _summary(path: Path, capsys: pytest.CaptureFixture[str]) -> dict[str, str]:
    assert main(["expand", str(path), "--summary"]) == 0
    out, err = capsys.readouterr()
    assert err == ""
    rows = [line.split("  ", 1) for line in out.splitlines() if "  " in line]
    return {name.strip(): value.strip() for name, value in rows}


def test_the_summary_shows_the_calibration_and_prompt_ids(
    spec_text: SpecText,
    two_prompts: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = tmp_path / "plain.yaml"
    path.write_text(_with(spec_text), encoding="utf-8")
    shown = _summary(path, capsys)
    assert shown["calibration"] == "fixed"
    assert shown["prompts"] == f"{two_prompts}, 2 prompts"
    assert "candidates" not in shown
    setting = "calibration: per_prompt\nprompt_ids: [p-2]"
    path.write_text(_with(spec_text, setting), encoding="utf-8")
    shown = _summary(path, capsys)
    assert shown["calibration"] == "per_prompt"
    assert shown["prompts"] == f"{two_prompts}, 1 prompt, by prompt_ids"


def test_the_summary_says_which_candidates_and_how_they_were_chosen(
    spec_text: SpecText, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    listed = write_candidates(
        tmp_path / "candidates.csv",
        [*lists.library_neurons(6, [3, 9]), *lists.mayner_sets(2)],
    )
    each = "conditions:\n  - {label: c, each_candidate: {allow_over_ceiling: true}}\n"
    path = tmp_path / "candidates.yaml"
    for rule, text in (
        ("", "8 of {file}: every one"),
        (", source: mayner", "2 of {file}: every one with source mayner"),
        (", keys: [mayner/role-1-top2]", "1 of {file}: by key"),
        (
            ", kind: neuron, sample: {count: 4, across: layer, seed: 3}",
            "4 of {file}: kind neuron, a sample across layers, seed 3",
        ),
    ):
        setting = f"candidates: {{file: {listed}{rule}}}"
        path.write_text(
            _with(spec_text, setting, each).replace("qwen3.5-4b", "qwen3-8b"),
            encoding="utf-8",
        )
        assert _summary(path, capsys)["candidates"] == text.format(file=listed)


def test_a_dry_run_shows_the_calibration(
    tiny_spec: Callable[..., Path],
    costs: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    spec = tiny_spec(max_gpu_hours="1\ncalibration: per_prompt")
    runs = tmp_path / "runs"
    assert main(["run", str(spec), "--runs-dir", str(runs), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "calibration     per_prompt" in out.splitlines()
    assert not runs.exists()
