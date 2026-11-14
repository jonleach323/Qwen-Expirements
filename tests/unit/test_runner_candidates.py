"""An experiment that names its units by rule, run by `pushlab run` on
tiny-vl and checked by `pushlab verify` (decision 0023): a made-up
candidates list of a neuron and a set (tests/unit/candidate_lists.py),
each_candidate at two doses, on two of the smoke prompts chosen by
prompt_ids, in their order. The runner runs it as any experiment, from the
conditions the list resolves to; a run's record holds each set itself.
"""

import importlib.util
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from pushlab.cli import main
from pushlab.discovery.candidates import Candidate, write_candidates
from pushlab.runs.batch import record_schedule
from pushlab.runs.folder import read_identity
from pushlab.runs.shards import read_results
from pushlab.runs.spec import read_spec

_SPEC = importlib.util.spec_from_file_location(
    "candidate_lists", Path(__file__).with_name("candidate_lists.py")
)
lists = importlib.util.module_from_spec(_SPEC)  # type: ignore[arg-type]
_SPEC.loader.exec_module(lists)  # type: ignore[union-attr]

NAME = "tiny-pulses"
CONDITIONS = """\
conditions:
  - label: none
    segments: []
  - label: owl
    each_candidate: {doses: [0.5, 1.0]}
"""
LABELS = [
    "none",
    "owl-0001 alpha=-0.5",
    "owl-0001 alpha=-1.0",
    "owl-0002 alpha=0.5",
    "owl-0002 alpha=1.0",
]


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def owls(tiny_downloads: object) -> list[Candidate]:
    """A neuron pushed down and a set across two layers, sized near tiny-vl's
    residual sizes, both made up; tiny-vl is registered first."""
    neuron = lists.neuron(2, 100, -1, 1.0, model="tiny-vl")
    pair = lists.owl_set(
        "owl-pair", [(1, 3, 0.75), (2, 100, -0.5)], 0.2, model="tiny-vl"
    )
    return [neuron, pair]


@pytest.fixture
def saved(
    tiny_spec: Callable[..., Path], owls: list[Candidate], tmp_path: Path
) -> tuple[Path, Path]:
    """The experiment file and its runs dir, its 10 runs saved by the runner."""
    listed = write_candidates(tmp_path / "candidates.csv", owls)
    settings = f"1\nprompt_ids: [smoke-03, smoke-01]\ncandidates: {{file: {listed}}}"
    spec = tiny_spec(CONDITIONS, max_gpu_hours=settings, batch_size="5")
    runs = tmp_path / "runs"
    assert main(["run", str(spec), "--runs-dir", str(runs)]) == 0
    return spec, runs


def test_the_runner_runs_each_candidate_at_each_dose_on_the_chosen_prompts(
    saved: tuple[Path, Path], owls: list[Candidate]
) -> None:
    spec, runs = saved
    records = sorted(read_results(runs / NAME), key=lambda record: record.run_id)
    labels = read_identity(runs / NAME)["conditions"]
    assert [labels[record.run_id] for record in records] == LABELS * 2
    assert [record.prompt_id for record in records] == ["smoke-03"] * 5 + [
        "smoke-01"
    ] * 5
    assert all(record.status == "ok" and record.output_tokens for record in records)
    neuron, pair = owls
    pushed = {"none": (), "owl-0001": (neuron.unit,), "owl-0002": (pair.unit,)}
    for record in records:
        condition = labels[record.run_id].split()[0]  # without its alpha
        assert record_schedule(record).units() == pushed[condition]
    # The folder keeps each condition's candidate, as the spec writes it.
    written = read_identity(runs / NAME)["spec"]
    assert [c.get("candidate") for c in written["conditions"]] == [
        None,
        neuron.key,
        pair.key,
    ]
    assert written["prompt_ids"] == ["smoke-03", "smoke-01"]
    assert read_spec(spec).to_mapping() == written


def test_verify_passes_a_folder_made_from_a_candidates_list(
    saved: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    spec, runs = saved
    capsys.readouterr()
    assert main(["verify", str(spec), "--runs-dir", str(runs)]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"{NAME}, version 1: 10 runs and ")
    assert out.endswith(" checked; no problem found.\n")
