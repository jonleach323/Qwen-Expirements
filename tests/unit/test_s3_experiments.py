"""The repository's S3 experiment files (decision 0023):
experiments/s3/screen_a.yaml, every candidate at its reference dose on the
first 8 held-out prompts, and experiments/s3/library_repro.yaml, 50 library
neurons sampled across layers on "Who are you?" with per-prompt
calibration.

Neither names a unit: each reads the pod's candidates list. Here copies of
them read made-up lists instead (tests/unit/candidate_lists.py), never the
library's, and never the pod's list, which these tests do not touch.
"""

import importlib.util
from collections import Counter
from pathlib import Path

import pytest

from pushlab import PROJECT_ROOT
from pushlab.cli import main
from pushlab.discovery.candidates import Candidate, write_candidates
from pushlab.panel.settings import read_selection
from pushlab.prompts import read_prompt_set
from pushlab.runs.expand import expand
from pushlab.runs.ladder import default_hours
from pushlab.runs.spec import ExperimentSpec, read_spec
from pushlab.schedules.shapes import constant

_SPEC = importlib.util.spec_from_file_location(
    "candidate_lists", Path(__file__).with_name("candidate_lists.py")
)
lists = importlib.util.module_from_spec(_SPEC)  # type: ignore[arg-type]
_SPEC.loader.exec_module(lists)  # type: ignore[union-attr]

S3 = PROJECT_ROOT / "experiments" / "s3"
SCREEN_A = S3 / "screen_a.yaml"
LIBRARY_REPRO = S3 / "library_repro.yaml"
LIST = "units/candidates_qwen3-8b.csv"
# About 850 candidates: Mayner's 550 sets and the library's persona neurons.
CANDIDATES = 850
# 7 library neurons, one at 1.5, above the ceiling, then 3 sets.
SCREENED = [*lists.library_neurons(7, [3, 17, 30]), *lists.mayner_sets(3)]
# 60 library neurons, 5 in each of 12 layers.
LIBRARY = lists.library_neurons(60, [0, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 35])


def _copy(path: Path, listed: Path, tmp_path: Path) -> Path:
    """The experiment file at path, reading the list at listed instead."""
    text = path.read_text(encoding="utf-8")
    assert text.count(f"file: {LIST}\n") == 1
    copy = tmp_path / path.name
    copy.write_text(text.replace(f"file: {LIST}", f"file: {listed}"), "utf-8")
    return copy


def _spec(path: Path, candidates: list[Candidate], tmp_path: Path) -> ExperimentSpec:
    listed = write_candidates(tmp_path / "candidates.csv", candidates)
    return read_spec(_copy(path, listed, tmp_path))


@pytest.mark.parametrize("path", [SCREEN_A, LIBRARY_REPRO], ids=lambda p: p.stem)
def test_neither_file_names_a_unit(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for named in ("qwen3-8b:L", ":set:", "library/L", "mayner/", "unit:"):
        assert named not in text
    assert f"file: {LIST}" in text


def test_screen_a_pushes_every_candidate_at_its_reference_dose_on_8_prompts(
    tmp_path: Path,
) -> None:
    spec = _spec(SCREEN_A, SCREENED, tmp_path)
    assert (spec.name, spec.model, spec.calibration) == (
        "s3-screen-a",
        "qwen3-8b",
        "fixed",
    )
    assert spec.decoding["strategy"] == "greedy" and spec.seeds == (0,)
    assert (spec.max_new_tokens, spec.batch_size) == (160, 16)
    assert spec.prompt_ids == tuple(f"heldout32-0{n}" for n in range(1, 9))
    labels = [condition.label for condition in spec.conditions]
    assert labels == [f"candidate-{n:04d}" for n in range(1, 11)]
    for condition, candidate in zip(spec.conditions, SCREENED, strict=True):
        assert condition.candidate == candidate.key
        ((_, schedule),) = condition.points
        alpha = candidate.direction * candidate.strength
        assert schedule.segments == constant(candidate.unit, alpha).segments
        assert schedule.allow_over_ceiling  # the library's 1.5 passes 1.4
    expansion = expand(spec)
    assert len(expansion.runs) == len(SCREENED) * 8
    prompts = [run.record.prompt_id for run in expansion.runs]
    assert list(dict.fromkeys(prompts)) == list(spec.prompt_ids)
    assert Counter(prompts) == {prompt: len(SCREENED) for prompt in spec.prompt_ids}


def test_screen_a_runs_the_stages_settings() -> None:
    spec_text = SCREEN_A.read_text(encoding="utf-8")
    selection = read_selection()
    held_out = read_prompt_set(PROJECT_ROOT / selection.prompts)
    assert f"prompts: {selection.prompts}\n" in spec_text
    for n, prompt_id in enumerate(held_out.prompt_ids):
        listed = f"  - {prompt_id}\n" in spec_text
        assert listed == (n < selection.screen_prompts)
    assert f"max_new_tokens: {selection.max_new_tokens}\n" in spec_text
    assert f"batch_size: {selection.batch_size}\n" in spec_text


def test_library_repro_samples_50_library_neurons_across_layers(
    tmp_path: Path,
) -> None:
    # Left out: sets, and a made-up neuron of the other source.
    unit = lists.neuron(6, 77).unit
    other = Candidate("mayner", "role x", unit, 1, 1.0, False, "f", "invented")
    spec = _spec(LIBRARY_REPRO, [*lists.mayner_sets(4), *LIBRARY, other], tmp_path)
    assert (spec.name, spec.calibration) == ("s3-library-repro", "per_prompt")
    assert spec.prompts == "prompts/whoareyou.jsonl" and spec.prompt_ids is None
    assert spec.decoding["strategy"] == "greedy" and spec.max_new_tokens == 160
    by_key = {candidate.key: candidate for candidate in LIBRARY}
    chosen = [by_key[str(condition.candidate)] for condition in spec.conditions]
    assert len(chosen) == len(set(chosen)) == 50
    assert {candidate.source for candidate in chosen} == {"library"}
    # 50 of 5 in each of 12 layers: 4 rounds of one from each, then 2 more.
    layers = Counter(candidate.unit.layer for candidate in chosen)
    assert sorted(layers.values()) == [4] * 10 + [5] * 2
    for condition, candidate in zip(spec.conditions, chosen, strict=True):
        ((_, schedule),) = condition.points
        alpha = candidate.direction * candidate.strength
        assert schedule.segments == constant(candidate.unit, alpha).segments
    runs = expand(spec).runs
    assert len(runs) == 50 * 1
    assert {run.record.prompt_id for run in runs} == {"whoareyou-01"}
    # The same list draws the same 50, in the same order.
    again = read_spec(_copy(LIBRARY_REPRO, tmp_path / "candidates.csv", tmp_path))
    assert again == spec and again.sha256 == spec.sha256


def test_max_gpu_hours_follow_the_speed_their_comments_name() -> None:
    # configs/costs.yaml's speed for qwen3-8b when the files were written. A
    # later speed there need not change them, as their hash names their runs.
    speed = 22.0
    for path, runs in ((SCREEN_A, CANDIDATES * 8), (LIBRARY_REPRO, 50)):
        text = path.read_text(encoding="utf-8")
        assert f"configs/costs.yaml's {speed:g} reply tokens a" in text
        assert f"max_gpu_hours: {default_hours(runs, 160, speed)}\n" in text
    assert default_hours(CANDIDATES * 8, 160, speed) == 20.75


def test_pushlab_expand_reports_both_files_run_counts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    listed = write_candidates(tmp_path / "candidates.csv", [*SCREENED, *LIBRARY])
    count = len(SCREENED) + len(LIBRARY)
    for path, runs, rows in (
        (SCREEN_A, count * 8, ("calibration     fixed",)),
        (LIBRARY_REPRO, 50, ("calibration     per_prompt",)),
    ):
        assert main(["expand", str(_copy(path, listed, tmp_path)), "--summary"]) == 0
        out, err = capsys.readouterr()
        assert err == ""
        lines = out.splitlines()
        assert lines[-1].startswith(f"{runs:,} runs, of which 0 are controls")
        assert all(row in lines for row in rows)
    assert f"candidates      50 of {listed}: source library and kind neuron, a " in out


def test_without_the_list_expand_names_the_command_that_makes_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    missing = tmp_path / "units" / "candidates_qwen3-8b.csv"
    copy = _copy(SCREEN_A, missing, tmp_path)
    assert main(["expand", str(copy), "--summary"]) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert err == (
        f"pushlab expand: {copy}: candidates: {missing} is missing; `uv run pushlab "
        f"candidates --model qwen3-8b --out {missing}` makes it on the GPU pod "
        "(decision 0020)\n"
    )
