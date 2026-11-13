"""What a judge is given from an experiment (pushlab.scoring.items): the
saved runs that have a reply, and their whole replies or sentence windows."""

import doctest
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from pushlab.cli import main
from pushlab.runs.shards import read_results
from pushlab.scoring import items as items_module
from pushlab.scoring.items import (
    JUDGED_STATUSES,
    judged_runs,
    reply_items,
    window_items,
)
from pushlab.store.records import RunRecord

NAME = "tiny-pulses"


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _run(spec: Path, runs: Path) -> int:
    return main(["run", str(spec), "--runs-dir", str(runs)])


def _record(run_id: str, text: str, status: str = "ok") -> RunRecord:
    return RunRecord(
        run_id=run_id,
        experiment="x",
        model="tiny-vl@main",
        prompt_id="p",
        schedule=[],
        decoding={
            "strategy": "greedy",
            "max_new_tokens": 8,
            "thinking": False,
            "dtype": "float32",
        },
        seed=0,
        batch_key="b000001:0",
        control_of="",
        output_text=text,
        output_tokens=list(range(len(text))),
        status=status,
    )


def test_an_experiments_saved_runs_with_a_reply_are_judged(
    tiny_spec: Callable[..., Path], tmp_path: Path
) -> None:
    runs = tmp_path / "runs"
    assert _run(tiny_spec(), runs) == 0
    experiment = judged_runs(tiny_spec(), runs)
    assert experiment.folder == runs / NAME and experiment.version == 1
    assert experiment.runs == read_results(runs / NAME)
    assert len(experiment.runs) == 10 and experiment.failed == 0
    assert {experiment.condition(run) for run in experiment.runs} == {"none", "pulse"}
    assert [item.text for item in reply_items(experiment.runs)] == [
        run.output_text for run in experiment.runs
    ]


def test_a_missing_runner_folder_has_nothing_to_judge(
    tiny_spec: Callable[..., Path], tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="does not exist"):
        judged_runs(tiny_spec(), tmp_path / "runs")


def test_only_runs_with_a_reply_are_judged() -> None:
    assert JUDGED_STATUSES == ("ok", "collapsed")


def test_windows_are_each_runs_sentences_in_order() -> None:
    one = _record("x-000001", "Hi there. I am an owl. Hoo!")
    two = _record("x-000002", "  ")  # a blank reply has no window
    three = replace(one, run_id="x-000003", output_text="Only one sentence")
    windows = window_items([one, two, three], 2)
    assert [(item.run.run_id, item.text) for item in windows] == [
        ("x-000001", "Hi there. I am an owl."),
        ("x-000001", " Hoo!"),
        ("x-000003", "Only one sentence"),
    ]
    assert [(item.window.first, item.window.count) for item in windows] == [
        (0, 2),
        (2, 1),
        (0, 1),
    ]
    assert [item.window for item in reply_items([one])] == [None]


def test_docstring_examples_run() -> None:
    result = doctest.testmod(items_module)
    assert result.attempted > 0
    assert result.failed == 0
