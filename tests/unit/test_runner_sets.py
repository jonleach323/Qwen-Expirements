"""A set of neurons pushed by `pushlab run` on tiny-vl, from an experiment
file that writes it as segments with its definition (decision 0023), and
the folder checked by `pushlab verify` and shown by `pushlab report`; and
`pushlab run --dry-run` and `pushlab status` on it.

The set is made up: neuron 3 of layer 1 and neuron 100 of layer 2, with
coefficients 0.75 and -0.5 and a size of 0.2, near tiny-vl's residual
sizes. Its steps name it as one unit with its amount per unit of
coefficient, alpha * size / |w|, w being its write direction (decision
0019).
"""

import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch

from pushlab.cli import main
from pushlab.models import discovery, load
from pushlab.runs.batch import record_schedule
from pushlab.runs.folder import read_identity
from pushlab.runs.shards import read_result_steps, read_results
from pushlab.units.neuron_set import NeuronSet

NAME = "tiny-pulses"
SIZE = 0.2
ENTRIES = [(1, 3, 0.75), (2, 100, -0.5)]
OWL = "tiny-vl:set:owl-pair"
DEFINITION = (
    "{entries: [[1, 3, 0.75], [2, 100, -0.5]], size: 0.2, source: made up for a test}"
)
# The prompt pass and reply tokens 0 and 1 at +1 (steps 0 to 2), nothing on
# reply token 2 (step 3), then -0.5 from reply token 3 (steps 4 on).
CONDITIONS = f"""\
conditions:
  - label: none
    segments: []
  - label: owl
    segments:
      - {{unit: {OWL}, set: {DEFINITION}, alpha: 1.0, region: prompt}}
      - {{unit: {OWL}, set: {DEFINITION}, alpha: 1.0, start: 0, end: 2}}
      - {{unit: {OWL}, set: {DEFINITION}, alpha: -0.5, start: 3}}
"""
DOSES = [1.0, 1.0, 1.0, None, -0.5, -0.5, -0.5, -0.5]  # by step; None: no push
MARKER = re.compile(r"\[\*\*(.+?)\*\*\]")


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def saved(tiny_spec: Callable[..., Path], tmp_path: Path) -> tuple[Path, Path]:
    """The experiment file and its runs dir, its 10 runs saved by the runner."""
    spec, runs = tiny_spec(CONDITIONS, max_new_tokens="8"), tmp_path / "runs"
    assert main(["run", str(spec), "--runs-dir", str(runs)]) == 0
    return spec, runs


def _dose_one_amount() -> float:
    """alpha 1's amount per unit of coefficient, worked by hand from tiny-vl's
    weights: the size over the length of the weighted sum of the columns."""
    model = load.load_model("tiny-vl", torch.float32, device="cpu").model
    mlps = dict(discovery.find_mlp_modules(model))
    columns = [
        coefficient * mlps[layer].down_proj.weight.detach()[:, index].double()
        for layer, index, coefficient in ENTRIES
    ]
    direction = (columns[0] + columns[1]).float()
    return SIZE / float(torch.linalg.vector_norm(direction))


def test_a_set_runs_as_one_unit_with_its_amount_per_coefficient(
    saved: tuple[Path, Path],
) -> None:
    _, runs = saved
    records = read_results(runs / NAME)
    steps = read_result_steps(runs / NAME)
    labels = read_identity(runs / NAME)["conditions"]
    owls = [record for record in records if labels[record.run_id] == "owl"]
    assert len(records) == 10 and len(owls) == 5
    dose_one = _dose_one_amount()
    written = {
        "entries": [[1, 3, 0.75], [2, 100, -0.5]],
        "size": 0.2,
        "source": "made up for a test",
    }
    for record in owls:
        # The record keeps the definition, and makes the set again from it.
        assert [segment["set"] for segment in record.schedule] == [written] * 3
        made = NeuronSet("tiny-vl", "owl-pair", ENTRIES, "made up for a test", SIZE)
        assert record_schedule(record).units() == (made,)
        run_steps = steps[record.run_id]
        n = len(record.output_tokens)
        assert n > 4  # so the -0.5 shows
        for step, dose in zip(run_steps, DOSES[:n], strict=True):
            if dose is None:
                assert (step.active_units, step.active_amounts) == ((), ())
                continue
            # One unit, the set, never its neurons; saved as a 16-bit float.
            assert step.active_units == ("tiny-vl:set:owl-pair",)
            assert step.active_amounts == (float(np.float16(dose * dose_one)),)
    unpushed = [record for record in records if labels[record.run_id] == "none"]
    for record in unpushed:
        assert all(step.active_units == () for step in steps[record.run_id])


def test_verify_passes_a_folder_that_pushes_a_set(
    saved: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    spec, runs = saved
    capsys.readouterr()
    assert main(["verify", str(spec), "--runs-dir", str(runs)]) == 0
    out = capsys.readouterr().out
    assert out.startswith(f"{NAME}, version 1: 10 runs and ")
    assert out.endswith(" checked; no problem found.\n")


def test_report_names_the_set_with_one_letter_and_marks_its_switches(
    saved: tuple[Path, Path], tmp_path: Path
) -> None:
    spec, runs = saved
    out = tmp_path / "report.md"
    argv = ["report", str(spec), "--runs-dir", str(runs), "--out", str(out)]
    assert main(argv) == 0
    markdown = out.read_text(encoding="utf-8")
    assert "A = tiny-vl:set:owl-pair" in markdown.splitlines()
    lengths = {
        record.prompt_id: len(record.output_tokens)
        for record in read_results(runs / NAME)
        if record.schedule
    }
    expected = [(0, "A on"), (3, "A off"), (4, "A on")]
    for section in markdown.split("\n## ")[1:]:
        prompt_id = section.split("\n")[0]
        (owl,) = [line for line in section.split("\n") if line.startswith("**owl**")]
        found = MARKER.findall(owl)
        assert found == [mark for step, mark in expected if step < lengths[prompt_id]]


def test_a_dry_run_and_status_count_a_set_experiments_runs(
    tiny_spec: Callable[..., Path],
    costs: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = tiny_spec(CONDITIONS, max_new_tokens="8"), tmp_path / "runs"
    argv = [str(spec), "--runs-dir", str(runs)]
    assert main(["run", *argv, "--dry-run"]) == 0
    table = [line.split() for line in capsys.readouterr().out.splitlines()]
    assert ["owl", "5", "0", "5"] in table and ["(all)", "10", "0", "10"] in table
    assert not runs.exists()
    assert main(["run", *argv, "--limit", "4"]) == 3  # stopped at the limit
    capsys.readouterr()
    assert main(["status", *argv]) == 0
    assert "10: 4 ok, 0 failed, 6 pending" in capsys.readouterr().out
