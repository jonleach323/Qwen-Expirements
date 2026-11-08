"""The numbers `pushlab status` and progress.json report.

Status's time left, before any batch and after some; its notes on a
changed prompt set, changed known units and a speeds file it cannot use;
the batches a new version kept. progress.json's GPU-hours over two
sessions and its hours left; an estimate's cost; and the earlier versions
experiment.json lists. tiny_spec's experiment has 10 runs in 5 batches of
2, with replies of at most 4 tokens.
"""

import json
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from pushlab.cli import main
from pushlab.prompts import PROMPTS_DIR
from pushlab.runs import estimate, runner
from pushlab.runs.estimate import Speed, estimate_lines, read_costs
from pushlab.runs.folder import read_identity, read_index, used_seconds
from pushlab.runs.plan import IDENTITY_KEYS
from pushlab.units import controls

NAME = "tiny-pulses"
REFUSED = "pushlab run will refuse it without --overwrite-version"
# A pulse with one random control, which must avoid the known units.
CONTROLLED = """\
conditions:
  - label: pulse
    segments: [{unit: tiny-vl:L2:N100, alpha: 1.0}]
    controls: 1
    control_seed: 7
"""
# Three runs a prompt: no push, a pulse, and a second pulse at DOSE.
THREE = """\
conditions:
  - label: none
    segments: []
  - label: pulse
    segments: [{unit: tiny-vl:L2:N100, alpha: 1.0}]
  - label: other
    segments: [{unit: tiny-vl:L2:N101, alpha: DOSE}]
"""


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _run(spec: Path, runs: Path, *options: str) -> int:
    return main(["run", str(spec), "--runs-dir", str(runs), *options])


def _status(
    spec: Path, runs: Path, capsys: pytest.CaptureFixture[str]
) -> list[tuple[str, str]]:
    """Status's lines as (name, value), in order; it must exit 0."""
    capsys.readouterr()
    assert main(["status", str(spec), "--runs-dir", str(runs)]) == 0
    lines = [line.split("  ", 1) for line in capsys.readouterr().out.splitlines()]
    return [(name.strip(), value.strip()) for name, value in lines]


def _value(lines: list[tuple[str, str]], name: str) -> str:
    (value,) = [value for key, value in lines if key == name]
    return value


def _notes(lines: list[tuple[str, str]]) -> list[str]:
    return [value for key, value in lines if key == "note"]


def _progress(runs: Path) -> dict:
    return json.loads((runs / NAME / "progress.json").read_text())


def test_status_notes_a_prompt_set_that_changed(
    tiny_spec: Callable[..., Path],
    costs: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prompts = tmp_path / "prompts.jsonl"
    prompts.write_bytes((PROMPTS_DIR / "smoke.jsonl").read_bytes())
    spec, runs = tiny_spec(prompts=str(prompts)), tmp_path / "runs"
    assert _run(spec, runs, "--limit", "4") == 3
    assert _notes(_status(spec, runs, capsys)) == []
    # One prompt's text edited; the experiment file stays as it was.
    prompts.write_text(prompts.read_text().replace("pancakes", "waffles"))
    lines = _status(spec, runs, capsys)
    assert _value(lines, "runs") == "10: 4 ok, 0 failed, 6 pending"
    assert _notes(lines) == [f"the prompt set changed since version 1: {REFUSED}"]


def test_status_notes_known_units_that_changed(
    tiny_spec: Callable[..., Path],
    costs: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    known = tmp_path / "known.csv"  # stands in for units/known.csv
    known.write_bytes(controls.KNOWN_UNITS_FILE.read_bytes())
    monkeypatch.setattr(controls, "KNOWN_UNITS_FILE", known)
    spec, runs = tiny_spec(CONTROLLED), tmp_path / "runs"
    assert _run(spec, runs, "--limit", "2") == 3
    assert _notes(_status(spec, runs, capsys)) == []
    with known.open("a", encoding="utf-8") as file:  # one more known unit
        file.write("qwen3.5-4b:L14:N1,+,test,https://example.org\n")
    assert _notes(_status(spec, runs, capsys)) == [
        f"the known units changed since version 1: {REFUSED}"
    ]


def test_time_left_before_any_batch_is_about_and_adds_the_measured_load(
    tiny_spec: Callable[..., Path],
    costs: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = tiny_spec(), tmp_path / "runs"
    # The costs file's speed: 40 reply tokens at 0.01 a second, 4,000 s.
    assert _value(_status(spec, runs, capsys), "time left") == (
        f"about 67 minutes, at 0.01 reply tokens a second, from {costs}"
    )
    runs.mkdir()
    measured = {
        "reply_tokens_per_second": 0.1,
        "reply_tokens": 40,
        "seconds": 400.0,
        "load_seconds": 600.0,
        "rows_per_batch": 2.0,
        "batch_size": 2,
        "gpu": None,
        "experiment": "tiny-other",
        "measured_at": "2026-11-01T12:00:00+00:00",
    }
    estimate.save_speed(runs, "tiny-vl/float32", measured)
    speed = (
        "0.1 reply tokens a second, measured by tiny-other on the CPU with 2 "
        "rows a batch, 2026-11-01"
    )
    # 40 tokens at 0.1 a second take 400 s, and the load 600: 1,000 s.
    lines = _status(spec, runs, capsys)
    assert _value(lines, "time left") == f"about 17 minutes, at {speed}"
    assert _notes(lines) == []
    # The same once the folder exists, before its first batch.
    assert _run(spec, runs, "--limit", "1") == 3  # a batch holds 2: none runs
    lines = _status(spec, runs, capsys)
    assert _value(lines, "batches") == "0 saved"
    assert _value(lines, "time left") == f"about 17 minutes, at {speed}"


def test_time_left_after_batches_gives_each_run_a_row_s_share_of_the_time(
    tiny_spec: Callable[..., Path],
    costs: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = tiny_spec(), tmp_path / "runs"
    assert _run(spec, runs, "--limit", "4") == 3
    index = runs / NAME / "index.jsonl"
    entries = [json.loads(line) for line in index.read_text().splitlines()]
    assert [entry["rows"] for entry in entries] == [2, 2]
    # Known generation times: 20 s over 4 rows is 5 s a run, 30 s for 6.
    for entry, seconds in zip(entries, [4.0, 16.0], strict=True):
        entry["generate_seconds"] = seconds
    index.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    lines = _status(spec, runs, capsys)
    tokens = sum(entry["reply_tokens"] for entry in entries)
    assert _value(lines, "speed") == (
        f"{tokens / 20:.3g} reply tokens a second, measured over 2 batches"
    )
    assert _value(lines, "time left") == "about 30 s for 6 runs at that speed"


def test_progress_gpu_hours_count_the_earlier_sessions_too(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter(range(0, 100_000, 100))  # a stand-in clock: 100 s a batch
    monkeypatch.setattr(runner, "_clock", lambda: float(next(ticks)))
    spec, runs = tiny_spec(), tmp_path / "runs"
    assert _run(spec, runs, "--limit", "4") == 3  # 2 batches
    assert _progress(runs)["gpu_hours"] == 200 / 3600
    assert _run(spec, runs) == 0  # the other 3, in a second session
    total = used_seconds(read_index(runs / NAME), 1)
    assert total == 500
    assert _progress(runs)["gpu_hours"] == total / 3600


def test_progress_hours_left_are_the_runs_left_at_the_session_s_speed(
    tiny_spec: Callable[..., Path], tmp_path: Path
) -> None:
    spec, runs = tiny_spec(), tmp_path / "runs"
    assert _run(spec, runs, "--limit", "4") == 3
    progress = _progress(runs)
    entries = read_index(runs / NAME)
    tokens = sum(entry["reply_tokens"] for entry in entries)
    rows = sum(entry["rows"] for entry in entries)
    seconds = sum(entry["generate_seconds"] for entry in entries)
    rate = progress["reply_tokens_per_second"]
    assert rate == pytest.approx(tokens / seconds)
    assert tokens / rows > 1  # so a row's share of the tokens counts
    # 6 runs left, each with a row's share of the reply tokens, at that speed.
    assert progress["pending"] == 6
    assert progress["hours_left"] == pytest.approx(6 * tokens / rows / rate / 3600)


def test_status_counts_the_batches_a_new_version_kept(
    tiny_spec: Callable[..., Path],
    costs: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs = tmp_path / "runs"
    first = tiny_spec(THREE.replace("DOSE", "1.0"), batch_size="3")
    assert _run(first, runs, "--limit", "6") == 3  # 2 batches of 3
    # A dose change: each saved batch keeps its unpushed run and first pulse.
    second = tiny_spec(THREE.replace("DOSE", "0.5"), batch_size="3")
    capsys.readouterr()
    assert _run(second, runs, "--overwrite-version") == 0
    assert "Started version 2; it keeps 4 runs saved." in capsys.readouterr().out
    lines = _status(second, runs, capsys)
    assert _value(lines, "runs") == "15: 15 ok, 0 failed, 0 pending"
    # The 4 kept runs are in 2 batches; the other 11 ran in 5.
    assert _value(lines, "batches") == "5 saved, and 2 kept from earlier versions"


def test_an_estimate_costs_under_a_cent_or_its_hours_at_each_price() -> None:
    costs = read_costs()  # configs/costs.yaml
    low, high = costs.dollars_per_hour
    on = f"on an {costs.gpu} at ${low:.2f} to ${high:.2f} an hour"
    speed = Speed(44.0, "from configs/costs.yaml")
    # One run of 4 tokens at 44 a second: far less than a cent.
    tiny = dict(estimate_lines(1, 4, speed, costs, 1.0))
    assert tiny["cost"] == f"under $0.01, {on}"
    # 1,000 runs of 96 tokens: 96,000 tokens at 44 a second, 0.61 hours.
    larger = dict(estimate_lines(1000, 96, speed, costs, 1.0))
    hours = 96_000 / 44.0 / 3600
    assert larger["gpu-hours"] == f"about {hours:.2f}, within max_gpu_hours 1"
    assert larger["cost"] == f"about ${hours * low:.2f} to ${hours * high:.2f}, {on}"


def test_experiment_json_lists_each_earlier_version(
    tiny_spec: Callable[..., Path], tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    made = []
    new = ["--overwrite-version"]
    for hours, options in [("1", []), ("2", new), ("3", new)]:
        # --limit 1 runs nothing, as a batch holds 2, but makes the version.
        spec = tiny_spec(max_gpu_hours=hours)
        assert _run(spec, runs, *options, "--limit", "1") == 3
        made.append(read_identity(runs / NAME))
    assert [identity["version"] for identity in made] == [1, 2, 3]
    keys = ("version", "created_at", *IDENTITY_KEYS)
    earlier = [{key: identity[key] for key in keys} for identity in made[:2]]
    assert made[2]["previous"] == earlier
    assert earlier[0]["spec_sha256"] != earlier[1]["spec_sha256"]


def test_a_speeds_file_of_another_shape_is_noted_and_left_out(
    tiny_spec: Callable[..., Path],
    costs: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = tiny_spec(), tmp_path / "runs"
    runs.mkdir()
    speeds = runs / "speeds.json"
    speeds.write_text(json.dumps({"tiny-vl/float32": 5}))  # a number, no speed
    note = (
        f"{speeds} is not a speeds file as pushlab run writes it; delete it, and "
        "the next run measures the speed again. Estimates leave it out"
    )
    from_costs = f"about 67 minutes, at 0.01 reply tokens a second, from {costs}"
    lines = _status(spec, runs, capsys)  # exits 0
    assert _notes(lines) == [note]
    assert _value(lines, "time left") == from_costs
    # The same once the folder exists, before its first batch.
    assert _run(spec, runs, "--limit", "1") == 3
    lines = _status(spec, runs, capsys)
    assert _notes(lines) == [note]
    assert _value(lines, "time left") == from_costs
