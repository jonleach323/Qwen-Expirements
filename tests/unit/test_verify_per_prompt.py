"""The per_prompt calibration's checks and refusals, beyond a clean run
(pushlab.runs.verify_steps, pushlab.runs.batch, pushlab.runs.plan).

verify's check that a unit's amounts are its dose times one number becomes
one number per unit and prompt under per_prompt, but for a set with its own
size, whose dose takes no residual sizes: these runs are made up here, so
no model runs. Then, on tiny-vl: a folder made with the fixed calibration
is refused for the same file under per_prompt, as made from another
calibration that keeps no saved run; a call whose measuring raises saves
its runs failed; a dry run where the model is not downloaded still knows
the per_prompt calibration; and the sizes a session keeps.
"""

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pushlab.cli import main
from pushlab.models import load
from pushlab.prompts import PROMPTS_DIR, read_prompt_set
from pushlab.runs.batch import Inputs, PromptSizes, call_norms
from pushlab.runs.folder import read_identity, read_index
from pushlab.runs.shards import read_results
from pushlab.runs.verify_steps import Bound, check_steps, scale_problem
from pushlab.store.records import RunRecord
from pushlab.store.steps import StepRecord
from pushlab.units import calibration_mode
from pushlab.units.calibration_mode import per_prompt_norms
from pushlab.units.dose import PerPromptNorms

A = "tiny:L1:N5"  # a neuron of the tiny test model
PAIR = "tiny:set:pair"  # a set of it and another, across two layers
SIZED = {"entries": [[1, 5, 1.0], [0, 3, 0.5]], "size": 2.0, "source": "made up"}
NAME, UNIT = "tiny-pulses", "tiny-vl:L3:N7"
PUSHES = f"""\
conditions:
  - {{label: none, segments: []}}
  - {{label: up, segments: [{{unit: {UNIT}, alpha: 1.0}}]}}
"""
PER_PROMPT = {"calibration": "per_prompt", "max_new_tokens": "4"}
SMOKE = read_prompt_set(PROMPTS_DIR / "smoke.jsonl")


def _made_up(n: int, prompt: str, segment: dict, amount: float) -> tuple:
    """Run x-00000n on prompt, which pushes segment, and its three steps:
    nothing on the prompt pass, then amount on the pushed unit."""
    run_id, unit = f"x-{n:06d}", segment["unit"]
    record = RunRecord(
        run_id, "x", "tiny@main", prompt, [segment], {}, 0, "", "", "", [5, 6, 7], "ok"
    )
    steps = [StepRecord(run_id, 0, "reply", 5, -1.0, (), ())]
    for j in (1, 2):
        steps.append(StepRecord(run_id, j, "reply", 5 + j, -1.0, (unit,), (amount,)))
    return record, steps


def _problems(runs: list[tuple], by_prompt: bool) -> list[str]:
    """What verify finds in the made-up runs: within each run, then across."""
    found: list[str] = []
    scales: dict[tuple[str, str | None], Bound] = {}
    for record, steps in runs:
        check_steps(record, steps, found.append, scales, by_prompt)
    return found + [p for k, b in scales.items() if (p := scale_problem(k, b))]


def test_under_per_prompt_a_neurons_amount_is_one_number_a_prompt() -> None:
    neuron = {"unit": A, "alpha": 1.0}
    runs = [_made_up(1, "p-1", neuron, 1.5), _made_up(2, "p-2", neuron, 1.75)]
    # Under fixed, two prompts' amounts must be one number.
    assert _problems(runs, by_prompt=False) == [
        f"{A}'s amount per dose is 1.5 in x-000001 but 1.75 in x-000002, not one number"
    ]
    # Under per_prompt, each prompt has its own sizes, so its own number...
    assert _problems(runs, by_prompt=True) == []
    # ...which every run of that prompt shares.
    runs.append(_made_up(3, "p-1", neuron, 1.625))
    assert _problems(runs, by_prompt=True) == [
        f"{A}'s amount per dose on p-1 is 1.5 in x-000001 but 1.625 in x-000003, "
        "not one number"
    ]


def test_a_set_with_its_own_size_is_one_number_whatever_the_prompt() -> None:
    sized = {"unit": PAIR, "set": SIZED, "alpha": 1.0}
    runs = [_made_up(1, "p-1", sized, 0.5), _made_up(2, "p-2", sized, 0.5)]
    assert _problems(runs, by_prompt=True) == []
    runs.append(_made_up(3, "p-3", sized, 0.625))  # its size takes no sizes
    assert _problems(runs, by_prompt=True) == [
        f"{PAIR}'s amount per dose is 0.5 in x-000001 but 0.625 in x-000003, not "
        "one number"
    ]
    # A set of one layer with no size takes that layer's size: per prompt.
    one_layer = SIZED | {"entries": [[1, 5, 1.0], [1, 9, 0.5]], "size": None}
    unsized = {"unit": PAIR, "set": one_layer, "alpha": 1.0}
    runs = [_made_up(1, "p-1", unsized, 0.5), _made_up(2, "p-2", unsized, 0.625)]
    assert _problems(runs, by_prompt=True) == []


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _run(spec: Path, runs: Path, *options: str) -> int:
    return main(["run", str(spec), "--runs-dir", str(runs), *options])


def test_a_fixed_folder_is_refused_for_per_prompt_and_a_new_version_keeps_none(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs, fixed_spec = tmp_path / "runs", tiny_spec(PUSHES, max_new_tokens="4")
    assert _run(fixed_spec, runs, "--limit", "4") == 3
    fixed = read_identity(runs / NAME)["calibration_sha256"]
    assert len(fixed) == 64  # the calibration file's SHA-256
    capsys.readouterr()
    assert main(["status", str(fixed_spec), "--runs-dir", str(runs)]) == 0
    assert "  fixed: each layer's residual size, from" in capsys.readouterr().out
    spec = tiny_spec(PUSHES, **PER_PROMPT)
    assert _run(spec, runs) == 1
    assert capsys.readouterr().err == (
        f"pushlab run: {runs / NAME} holds version 1 of {NAME}, made from another "
        "experiment file and calibration. Run with --overwrite-version to start "
        "version 2, which keeps no saved run, since the calibration changed, or "
        "move the folder away to start afresh\n"
    )
    assert _run(spec, runs, "--overwrite-version") == 0
    assert "Started version 2; it keeps 0 runs saved." in capsys.readouterr().out
    identity = read_identity(runs / NAME)
    assert identity["calibration_sha256"] == "per_prompt"
    assert identity["previous"][0]["calibration_sha256"] == fixed
    assert main(["verify", str(spec), "--runs-dir", str(runs)]) == 0


def test_a_call_whose_measuring_raises_saves_its_runs_failed(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    real, third = calibration_mode.prompt_norms, SMOKE.prompts[2]

    def failing(model, tokenizer, prompt):  # type: ignore[no-untyped-def]
        if prompt == third:
            raise RuntimeError("no sizes")
        return real(model, tokenizer, prompt)

    monkeypatch.setattr(calibration_mode, "prompt_norms", failing)
    spec, runs = tiny_spec(PUSHES, **PER_PROMPT), tmp_path / "runs"
    assert _run(spec, runs) == 2
    failed = {e["batch"]: e["error"] for e in read_index(runs / NAME) if e["error"]}
    assert failed == {"b000003": "RuntimeError: no sizes"}  # smoke-03's two runs
    statuses = {r.run_id: r.status for r in read_results(runs / NAME)}
    assert [run_id for run_id, s in statuses.items() if s == "failed"] == [
        f"{NAME}-000005",
        f"{NAME}-000006",
    ]
    monkeypatch.setattr(calibration_mode, "prompt_norms", real)
    assert _run(spec, runs, "--retry-failed") == 0
    capsys.readouterr()
    assert main(["verify", str(spec), "--runs-dir", str(runs)]) == 0
    assert capsys.readouterr().out.startswith(f"{NAME}, version 1: 10 runs and ")


def test_where_the_model_is_not_downloaded_a_dry_run_knows_per_prompt(
    tiny_spec: Callable[..., Path],
    costs: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs = tmp_path / "runs"
    assert _run(tiny_spec(PUSHES, max_new_tokens="4"), runs, "--limit", "4") == 3

    def missing(name: str, *args: object) -> str:
        raise FileNotFoundError(f"{name} is not downloaded here")

    monkeypatch.setattr(load, "downloaded_revision", missing)  # as in the cloud
    capsys.readouterr()
    assert _run(tiny_spec(PUSHES, **PER_PROMPT), runs, "--dry-run") == 0
    out = capsys.readouterr().out
    unchecked = "the model is not downloaded here: its download was not checked"
    assert f"note: {unchecked}." in out
    # Its calibration is known all the same: the saved runs are not kept.
    assert "made from another experiment file and calibration" in out
    assert "a new version would keep 0 runs of version 1" in out


def test_prompt_sizes_measure_each_text_once_and_keep_the_rows_order(
    tiny_vl_model: torch.nn.Module,
    tiny_tokenizer: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = SimpleNamespace(model=tiny_vl_model, tokenizer=tiny_tokenizer)
    measured: list[str] = []
    real = calibration_mode.prompt_norms

    def counting(model, tokenizer, prompt):  # type: ignore[no-untyped-def]
        measured.append(prompt)
        return real(model, tokenizer, prompt)

    monkeypatch.setattr(calibration_mode, "prompt_norms", counting)
    hi, bye, ok = "Say hi.", "Say bye now.", "Is it ok?"
    sizes = PromptSizes()
    first = sizes.norms(loaded, [hi, bye, hi])
    assert measured == [hi, bye]
    measured.clear()  # another call: only the new prompt is measured
    second = sizes.norms(loaded, [ok, bye])
    assert measured == [ok]
    direct = per_prompt_norms(tiny_vl_model, tiny_tokenizer, [hi, bye, hi, ok, bye])
    assert first.rows + second.rows == direct.rows
    assert first.rows[0] != first.rows[1]  # two prompts' own sizes
    # The fixed calibration's sizes go to every call, pushed or not; under
    # per_prompt, a call that pushes nothing measures nothing.
    assert call_norms(Inputs([0.5, 0.75], {}), loaded, [hi], False) == [0.5, 0.75]
    per_prompt = Inputs(None, {}, PromptSizes())
    assert call_norms(per_prompt, loaded, [hi], False) is None
    assert per_prompt.sizes is not None and per_prompt.sizes.measured == {}
    norms = call_norms(per_prompt, loaded, [bye, hi], True)
    assert isinstance(norms, PerPromptNorms)
    assert norms.rows == (direct.rows[1], direct.rows[0])
