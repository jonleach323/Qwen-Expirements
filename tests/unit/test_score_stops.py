"""A `pushlab score` job that stops (pushlab.scoring.score_command,
score_rows.judge_chunks): a judge that cannot answer at all, or Ctrl-C.

The runner saves tiny_spec's experiment for real on tiny-vl. Its replies
are the tiny model's own, 4 tokens each, and some repeat one another: a job
asks about each text once and gives its repeats that verdict with no call
(pushlab.scoring.judge_batch). A FakeBackend judges one call at a time, so
its calls follow the texts in run order. Its third call stops the job, and
the verdicts found before it, repeats included, are saved as they are.
test_score_command.py covers what a job that ends judges and saves.
"""

from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from pushlab.cli import main
from pushlab.runs.shards import read_results
from pushlab.scoring.cache import VerdictCache
from pushlab.scoring.fake_backend import FakeBackend
from pushlab.scoring.judge import CallError, Judge, Reply
from pushlab.scoring.score_command import run_score
from pushlab.store.scores import ScoreRow, read_rows

NAME = "tiny-pulses"
SCORE = '{"reason": "Fine.", "score": 8}'
DECLINED = "the judge declined (a refusal)"


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run on the CPU even with a GPU, and keep any cache out of /workspace."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("PUSHLAB_CACHE_DIR", str(tmp_path / "default-cache"))


def _saved(tiny_spec: Callable[..., Path], tmp_path: Path) -> tuple[Path, Path]:
    """tiny_spec's experiment file and runs dir, its runs saved by the runner."""
    spec, runs = tiny_spec(), tmp_path / "runs"
    assert main(["run", str(spec), "--runs-dir", str(runs)]) == 0
    return spec, runs


def _judge(cache: Path, script: object) -> tuple[Judge, FakeBackend]:
    """A judge of a FakeBackend with its cache in cache, which never waits."""
    fake = FakeBackend(script, name="fake", model="fake")  # type: ignore[arg-type]
    cached = VerdictCache(cache, "fake:fake")
    return Judge(fake, cached, sleep=lambda seconds: None), fake


def _texts(runs: Path) -> list[str]:
    """The replies' texts as the judge reads them, each once, in run order:
    the order of the calls made one at a time."""
    saved = read_results(runs / NAME)
    return list(dict.fromkeys(run.output_text.strip() for run in saved))


def _by_text(runs: Path) -> dict[str, list[ScoreRow]]:
    """The saved rows of whole replies by the text they judged, in run order."""
    texts = {run.run_id: run.output_text.strip() for run in read_results(runs / NAME)}
    found: dict[str, list[ScoreRow]] = {}
    for row in sorted(read_rows(runs / NAME, "scores"), key=lambda row: row.run_id):
        found.setdefault(texts[row.run_id], []).append(row)
    return found


def _settings(out: str) -> dict[str, str]:
    """The lines of the form 'name  value', by name."""
    lines = [line.split("  ", 1) for line in out.splitlines() if "  " in line]
    return {name.strip(): value.strip() for name, value in lines}


def test_a_judge_that_cannot_answer_stops_the_job_keeping_what_it_found(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = _saved(tiny_spec, tmp_path)
    first, second, *rest = _texts(runs)
    assert rest  # a third text, whose call meets the rejected key
    declined = Reply("No.", "refusal", input_tokens=100, output_tokens=20)
    fatal = CallError("HTTP 401: invalid x-api-key", fatal=True)
    judge, fake = _judge(tmp_path / "cache", [declined, SCORE, fatal])
    capsys.readouterr()  # the runner's
    assert run_score(spec, "coherence", runs_dir=runs, judge=judge, workers=1) == 1
    out, err = capsys.readouterr()
    assert err == (
        "pushlab score: fake:fake: HTTP 401: invalid x-api-key. The verdicts found "
        "before it stopped are saved; run the same command to go on once the judge "
        "can answer\n"
    )
    assert "; then the judge stopped\n" in out
    assert fake.calls == 3  # no call starts once the judge has stopped
    found = _by_text(runs)
    assert set(found) == {first, second}
    refused, answered = found[first], found[second]
    # Each text's first run was asked: its row keeps its call and tokens.
    tokens = {"cached": False, "input_tokens": 100, "output_tokens": 20}
    assert (refused[0].status, refused[0].error) == ("failed", DECLINED)
    assert refused[0].details == {"reason": "", "raw": "No."} | tokens
    assert (answered[0].status, answered[0].value) == ("ok", 8.0)
    assert answered[0].details == {"reason": "Fine."} | tokens
    assert refused[0].attempts == answered[0].attempts == 1
    # Its repeats made no call: a refusal's is no cached verdict, an answer's is.
    no_tokens = {"input_tokens": 0, "output_tokens": 0}
    for row in refused[1:]:
        assert (row.status, row.error, row.attempts) == ("failed", DECLINED, 0)
        assert row.details == refused[0].details | no_tokens
    for row in answered[1:]:
        assert (row.status, row.value, row.attempts) == ("ok", 8.0, 0)
        assert row.details == answered[0].details | no_tokens | {"cached": True}
    settings = _settings(out)
    assert settings["verdicts"] == (
        f"{len(refused) + len(answered)}: {len(answered)} ok, {len(refused)} failed"
    )
    assert settings["calls made"] == "2"
    assert settings["tokens used"] == "200 input and 40 output"
    assert settings["errors"] == f"{len(refused)}: {DECLINED}"
    # The next job asks about the refused text again and the rest, once each.
    judge, fake = _judge(tmp_path / "cache", lambda request: SCORE)
    assert run_score(spec, "coherence", runs_dir=runs, judge=judge) == 0
    assert sorted(request.text for request in fake.requests) == sorted([first, *rest])
    assert len(read_rows(runs / NAME, "scores")) == 10 + len(refused)


def test_ctrl_c_stops_the_job_once_the_verdicts_found_are_saved(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = _saved(tiny_spec, tmp_path)
    first, second, *rest = _texts(runs)
    assert rest  # a third text, whose call meets Ctrl-C
    judge, fake = _judge(tmp_path / "cache", [SCORE, SCORE, KeyboardInterrupt()])
    capsys.readouterr()  # the runner's
    assert run_score(spec, "coherence", runs_dir=runs, judge=judge, workers=1) == 130
    out = capsys.readouterr().out
    assert "; then Ctrl-C stopped the job\n" in out
    assert out.endswith(
        "\nStopped by Ctrl-C. The verdicts found before it are saved: run the "
        "same command to go on.\n"
    )
    found = _by_text(runs)
    assert set(found) == {first, second} and fake.calls == 3
    tokens = {"cached": False, "input_tokens": 100, "output_tokens": 20}
    for rows in found.values():
        assert [(row.status, row.value) for row in rows] == [("ok", 8.0)] * len(rows)
        assert rows[0].attempts == 1 and rows[0].details == {"reason": "Fine."} | tokens
        assert all(row.attempts == 0 and row.details["cached"] for row in rows[1:])
    # The next job asks only about the texts not judged yet.
    judge, fake = _judge(tmp_path / "cache", lambda request: SCORE)
    assert run_score(spec, "coherence", runs_dir=runs, judge=judge) == 0
    assert sorted(request.text for request in fake.requests) == sorted(rest)
