"""What a judge is given from an experiment (pushlab.scoring.items): the
saved runs that have a reply, never a failed run, and their whole replies
or sentence windows; and of those, the ones pushlab score judges again
(pushlab.scoring.score_rows.make_plan). A FakeBackend judges."""

import doctest
import re
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from pushlab.cli import main
from pushlab.prompts import PROMPTS_DIR, read_prompt_set
from pushlab.runs.shards import read_results
from pushlab.scoring import items as items_module
from pushlab.scoring.cache import VerdictCache
from pushlab.scoring.fake_backend import FakeBackend
from pushlab.scoring.items import (
    JUDGED_STATUSES,
    judged_runs,
    reply_items,
    window_items,
)
from pushlab.scoring.judge import Judge, text_sha256
from pushlab.scoring.score_command import run_score
from pushlab.store.records import RunRecord
from pushlab.store.scores import read_rows, write_rows

NAME = "tiny-pulses"
THIRD = read_prompt_set(PROMPTS_DIR / "smoke.jsonl").prompts[2]  # smoke-03's
FAILED = [f"{NAME}-000005", f"{NAME}-000006"]  # smoke-03's batch of 2


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _run(spec: Path, runs: Path) -> int:
    return main(["run", str(spec), "--runs-dir", str(runs)])


def _judge(tmp_path: Path) -> tuple[Judge, FakeBackend]:
    """A judge of a FakeBackend that scores every text 8, with its cache."""
    fake = FakeBackend(lambda request: '{"reason": "Fine.", "score": 8}')
    return Judge(fake, VerdictCache(tmp_path / "cache", "fake:fake-judge")), fake


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


def test_failed_runs_are_left_out_and_counted(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    failing_generate: Callable[..., None],
) -> None:
    runs = tmp_path / "runs"
    failing_generate(THIRD, RuntimeError("boom"))
    assert _run(tiny_spec(), runs) == 2
    saved = read_results(runs / NAME)
    assert [run.run_id for run in saved if run.status == "failed"] == FAILED
    experiment = judged_runs(tiny_spec(), runs)
    assert experiment.runs == [run for run in saved if run.run_id not in FAILED]
    assert len(experiment.runs) == 8 and experiment.failed == 2
    assert {run.status for run in experiment.runs} <= set(JUDGED_STATUSES)


def test_pushlab_score_says_it_leaves_the_failed_runs_out_and_judges_the_rest(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    failing_generate: Callable[..., None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = tiny_spec(), tmp_path / "runs"
    failing_generate(THIRD, RuntimeError("boom"))
    assert _run(spec, runs) == 2
    judge, _ = _judge(tmp_path)
    capsys.readouterr()  # the runner's
    assert run_score(spec, "coherence", runs_dir=runs, judge=judge) == 0
    out = capsys.readouterr().out
    assert f"{NAME}, version 1: 8 replies, 2 failed runs left out\n" in out
    saved = read_results(runs / NAME)
    others = sorted(run.run_id for run in saved if run.run_id not in FAILED)
    rows = read_rows(runs / NAME, "scores")
    assert sorted(row.run_id for row in rows) == others and len(others) == 8
    assert {row.status for row in rows} == {"ok"}


def test_a_window_scored_on_another_text_is_judged_again(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = tiny_spec(), tmp_path / "runs"
    assert _run(spec, runs) == 0
    judge, fake = _judge(tmp_path)
    windows = {"runs_dir": runs, "judge": judge, "windows": "sentences"}
    assert run_score(spec, "coherence", **windows) == 0  # type: ignore[arg-type]
    rows = read_rows(runs / NAME, "sentences")
    # As if a change to the splitter had moved the first window's text but
    # kept its version: its row judged another text than the window has now.
    moved = replace(rows[0], text_sha256=text_sha256("Another text."))
    saved = {run.run_id: run for run in read_results(runs / NAME)}
    write_rows(runs / NAME, [moved], saved)
    calls = fake.calls
    capsys.readouterr()
    assert run_score(spec, "coherence", **windows) == 0  # type: ignore[arg-type]
    out = capsys.readouterr().out
    assert re.search(rf"^already scored +{len(rows) - 1}$", out, re.MULTILINE)
    (again,) = read_rows(runs / NAME, "sentences")[len(rows) + 1 :]
    first = rows[0]
    assert (again.run_id, again.span) == (first.run_id, first.span)
    assert again.text_sha256 == first.text_sha256 and again.status == "ok"
    assert fake.calls == calls  # the cache answers for the text
    assert run_score(spec, "coherence", **windows) == 0  # type: ignore[arg-type]
    assert "Nothing to judge: every item is scored already." in capsys.readouterr().out


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
