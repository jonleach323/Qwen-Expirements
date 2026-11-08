"""`pushlab run --dry-run`: what an experiment will run and cost, with nothing run.

The costs fixture stands in for configs/costs.yaml, with tiny-vl at 0.01
reply tokens a second, so tiny_spec's 10 runs of 4 tokens take 4,000 s.
"""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from pushlab import PROJECT_ROOT
from pushlab.cli import main
from pushlab.engine import generate as generate_module
from pushlab.models import load
from pushlab.runs import estimate
from pushlab.runs.estimate import read_costs, speed_for

NAME = "tiny-pulses"


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _run(spec: Path, runs: Path, *options: str) -> int:
    return main(["run", str(spec), "--runs-dir", str(runs), *options])


def _nothing_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("a dry run loads no model and generates nothing")

    monkeypatch.setattr(load, "load_model", refuse)
    monkeypatch.setattr(generate_module, "generate", refuse)


def _settings(out: str) -> dict[str, str]:
    """The lines of the form 'name  value', by name."""
    lines = [line.split("  ", 1) for line in out.splitlines() if "  " in line]
    return {name.strip(): value.strip() for name, value in lines}


def test_a_dry_run_prints_the_counts_and_estimates_and_runs_nothing(
    tiny_spec: Callable[..., Path],
    costs: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _nothing_runs(monkeypatch)
    runs = tmp_path / "runs"
    assert _run(tiny_spec(), runs, "--dry-run") == 0
    out = capsys.readouterr().out
    table = [line.split() for line in out.splitlines()]
    assert ["none", "5", "0", "5"] in table and ["pulse", "5", "0", "5"] in table
    assert ["(all)", "10", "0", "10"] in table
    settings = _settings(out)
    assert settings["to run"] == "10 of 10 runs; none is saved yet"
    assert settings["reply tokens"] == "at most 40, 4 for each run"
    assert settings["speed"] == f"0.01 reply tokens a second, from {costs}"
    # 40 tokens at 0.01 a second: 4,000 s, more than the cap of an hour.
    assert settings["gpu-hours"] == (
        "at most 1.11, more than max_gpu_hours 1: it would stop after about 90% "
        "of the runs"
    )
    assert (
        settings["cost"]
        == "at most $0.38 to $0.82, on an RTX 4090 at $0.34 to $0.74 an hour"
    )
    assert "This was a dry run: nothing was loaded, run or written." in out
    assert not runs.exists()


def test_a_dry_run_counts_only_what_is_left(
    tiny_spec: Callable[..., Path],
    costs: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = tiny_spec(), tmp_path / "runs"
    assert _run(spec, runs, "--limit", "4") == 0
    before = {p: p.read_bytes() for p in runs.rglob("*") if p.is_file()}
    capsys.readouterr()
    _nothing_runs(monkeypatch)
    assert _run(spec, runs, "--dry-run") == 0
    settings = _settings(capsys.readouterr().out)
    assert settings["to run"] == "6 of 10 runs; 4 are saved in version 1"
    assert settings["reply tokens"] == "at most 24, 4 for each run"
    assert {p: p.read_bytes() for p in runs.rglob("*") if p.is_file()} == before
    # A changed experiment file is shown, as the run would refuse it.
    assert _run(tiny_spec(max_gpu_hours="2"), runs, "--dry-run") == 0
    out = capsys.readouterr().out
    assert "made from another experiment file" in out
    assert "--overwrite-version" in out


def test_after_a_run_the_measured_speed_is_used(
    tiny_spec: Callable[..., Path],
    costs: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = tiny_spec(), tmp_path / "runs"
    assert _run(spec, runs) == 0
    speeds = json.loads((runs / "speeds.json").read_text())
    measured = speeds["tiny-vl/float32"]
    assert measured["experiment"] == NAME and measured["batch_size"] == 2
    assert measured["reply_tokens"] > 0 and measured["reply_tokens_per_second"] > 0
    capsys.readouterr()
    _nothing_runs(monkeypatch)
    assert _run(tiny_spec(name="tiny-other"), runs, "--dry-run") == 0
    speed = _settings(capsys.readouterr().out)["speed"]
    assert speed.startswith(f"{measured['reply_tokens_per_second']:.3g} reply tokens a")
    assert f"measured by {NAME}" in speed


def test_the_costs_file_names_the_gpu_its_prices_and_speeds() -> None:
    costs = read_costs()  # configs/costs.yaml itself
    assert estimate.COSTS_FILE == PROJECT_ROOT / "configs" / "costs.yaml"
    assert costs.gpu == "RTX 4090"
    assert costs.dollars_per_hour == (0.34, 0.74)
    assert costs.reply_tokens_per_second["qwen3.5-4b"] > 0


@pytest.mark.parametrize(
    "text, problem",
    [
        ("gpu: RTX 4090\n", "needs dollars_per_hour"),
        (
            "gpu: x\ndollars_per_hour: [0.74, 0.34]\nreply_tokens_per_second: {}\n",
            "dollars_per_hour must be two prices",
        ),
        (
            "gpu: x\ndollars_per_hour: [0.3, 0.7]\nreply_tokens_per_second: {a: 0}\n",
            "reply_tokens_per_second for a must be a number above 0",
        ),
        (
            "gpu: x\ndollars_per_hour: [0.3, 0.7]\nreply_tokens_per_second: [1]\n",
            "reply_tokens_per_second must map models to speeds",
        ),
        ("- 1\n", "must be a mapping of gpu, dollars_per_hour"),
        ("gpu: [\n", "cannot read"),
    ],
)
def test_a_broken_costs_file_is_refused(
    tmp_path: Path, text: str, problem: str
) -> None:
    path = tmp_path / "costs.yaml"
    path.write_text(text)
    with pytest.raises(ValueError, match=problem):
        read_costs(path)


def test_without_any_speed_the_estimate_says_so(tmp_path: Path) -> None:
    costs = read_costs()
    assert speed_for("no-such-model", "bfloat16", tmp_path, costs) is None


def test_a_dry_run_without_a_speed_says_where_to_add_one(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _nothing_runs(monkeypatch)  # configs/costs.yaml itself knows no tiny-vl
    assert _run(tiny_spec(), tmp_path / "runs", "--dry-run") == 0
    settings = _settings(capsys.readouterr().out)
    assert settings["speed"] == "unknown: add this model's to configs/costs.yaml"
    assert "gpu-hours" not in settings and "cost" not in settings


def test_a_dry_run_counts_what_limit_and_retry_failed_would_run(
    tiny_spec: Callable[..., Path],
    costs: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _nothing_runs(monkeypatch)
    runs = tmp_path / "runs"
    assert _run(tiny_spec(), runs, "--dry-run", "--limit", "5") == 0
    settings = _settings(capsys.readouterr().out)
    assert settings["to run"] == "4 of 10 runs; none is saved yet (--limit 5)"
    assert _run(tiny_spec(), runs, "--dry-run", "--retry-failed") == 0
    assert _settings(capsys.readouterr().out)["to run"].startswith("0 of 10 runs")


def test_a_broken_costs_file_stops_a_dry_run(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    broken = tmp_path / "costs.yaml"
    broken.write_text("gpu: RTX 4090\n")
    monkeypatch.setattr(estimate, "COSTS_FILE", broken)
    assert _run(tiny_spec(), tmp_path / "runs", "--dry-run") == 1
    assert f"pushlab run: {broken} needs dollars_per_hour" in capsys.readouterr().err
