"""`pushlab score SPEC` (pushlab.scoring.score_command): a judge's verdicts on
an experiment's replies, in its runner folder's scores and sentences tables.

The runner saves tiny_spec's experiment for real on tiny-vl: an unpushed run
and a pulse on each of the five smoke prompts. The invented fixture has
generate() give each run an invented reply of two or three sentences
(REPLIES) in place of the tiny model's. A FakeBackend judges: nothing here
reaches the network. test_cli_scoring.py covers the options, the refusals
and the command on the tiny model's own replies, and test_score_stops.py a
job that stops.
"""

import doctest
import hashlib
import json
from collections.abc import Callable
from dataclasses import astuple
from functools import partial
from pathlib import Path

import pytest
import torch
from transformers import PreTrainedTokenizerBase

from pushlab.cli import main
from pushlab.engine import generate as generate_module
from pushlab.engine.generate import Generation
from pushlab.prompts import PROMPTS_DIR, read_prompt_set
from pushlab.runs import folder
from pushlab.runs.shards import read_results
from pushlab.scoring import score_command, score_rows
from pushlab.scoring.cache import VerdictCache
from pushlab.scoring.fake_backend import FakeBackend
from pushlab.scoring.items import judged_runs
from pushlab.scoring.judge import Judge
from pushlab.scoring.personas import Persona
from pushlab.scoring.rubrics import Request, load_rubric
from pushlab.scoring.score_command import run_score
from pushlab.scoring.sentences import sentence_windows
from pushlab.store.scores import output_sha256, read_rows

NAME = "tiny-pulses"
SMOKE = read_prompt_set(PROMPTS_DIR / "smoke.jsonl")
# Each smoke prompt's invented replies, unpushed then pushed: 23 sentences.
REPLIES = [
    (
        "Mix one cup of flour with an egg. Add milk until the batter is smooth. "
        "Fry each pancake until golden.",
        "Arr, ye want pancakes? Batter be the foam of the sea! Flip 'em, matey.",
    ),
    (
        "Grey waves fold over the shore. Gulls call above the tide.",
        "I am the deep, old and patient. My salt remembers every ship.",
    ),
    (
        "Listen without rushing to fix things. Let them know you are there.",
        "Hoo, a sad friend needs a wise owl. Sit with them until the dawn.",
    ),
    (
        "Plants catch sunlight in their leaves. They turn water and air into sugar.",
        "Light! Glorious light feeds me. I drink the sun through my green skin.",
    ),
    (
        "How about Pepper? It suits a lively, naïve kitten.",
        "Name it Captain Whiskers, matey. Every crew needs a captain.",
    ),
]
PERSONAS = """\
personas:
  - name: Brass Owl
    description: A pompous mechanical owl who lectures in riddles.
  - name: Salt Captain
    description: A gruff sea captain who calls everyone matey.
"""
# Condition labels no rubric's text holds, unlike tiny_spec's "none".
BLIND = """\
conditions:
  - {label: quiet-baseline-q7, segments: []}
  - {label: loud-pulse-q7, segments: [{unit: tiny-vl:L2:N100, alpha: 1.0}]}
"""


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run on the CPU even with a GPU, and keep any cache out of /workspace."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("PUSHLAB_CACHE_DIR", str(tmp_path / "default-cache"))


@pytest.fixture
def invented(monkeypatch: pytest.MonkeyPatch) -> None:
    """Have generate() answer each prompt with its REPLIES, the pushed one
    for a run that pushes; an unpushed one ends its turn with <|im_end|>,
    which its tokens keep and its text does not, as a real reply's."""
    replies = dict(zip(SMOKE.prompts, REPLIES, strict=True))

    def generate(model, tokenizer, prompts, pushes, decoding, norms, **kw):  # type: ignore[no-untyped-def]
        made = []
        for prompt, schedule in zip(prompts, kw["schedules"], strict=True):
            pushed = bool(schedule.segments)
            tokens = tokenizer.encode(replies[prompt][pushed], add_special_tokens=False)
            if not pushed:
                tokens.append(tokenizer.convert_tokens_to_ids("<|im_end|>"))
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            stop = "max_new_tokens" if pushed else "end_of_turn"
            touched, size = (0,) * len(schedule.segments), len(tokens)
            made.append(
                Generation(
                    text, tuple(tokens), stop, touched, ((),) * size, (-1.0,) * size
                )
            )
        return made

    monkeypatch.setattr(generate_module, "generate", generate)


def _saved(
    tiny_spec: Callable[..., Path], tmp_path: Path, *conditions: str
) -> tuple[Path, Path]:
    """The experiment file and runs dir, its runs saved by the runner."""
    spec = tiny_spec(*conditions, max_new_tokens="128")
    runs = tmp_path / "runs"
    assert main(["run", str(spec), "--runs-dir", str(runs)]) == 0
    return spec, runs


def _answer(request: Request) -> str:
    """A well-formed answer for each rubric: a score from the text's length,
    speaks_as for a text that says matey, and Salt Captain."""
    if request.rubric == "coherence":
        return json.dumps({"reason": "Fine.", "score": len(request.text) % 11})
    if request.rubric == "speaks_as":
        label = "speaks_as" if "matey" in request.text else "assistant"
        return json.dumps({"reason": "Arr.", "label": label})
    return json.dumps({"reason": "Arr.", "choice": "Salt Captain"})


def _judge(
    cache: Path, script: object = _answer, name: str = "fake", model: str = "fake"
) -> tuple[Judge, FakeBackend]:
    """A judge of a FakeBackend with its cache in cache, which never waits."""
    fake = FakeBackend(script, name=name, model=model)  # type: ignore[arg-type]
    cached = VerdictCache(cache, f"{name}:{model}")
    return Judge(fake, cached, sleep=lambda seconds: None), fake


def _settings(out: str) -> dict[str, str]:
    """The lines of the form 'name  value', by name."""
    lines = [line.split("  ", 1) for line in out.splitlines() if "  " in line]
    return {name.strip(): value.strip() for name, value in lines}


def _rows(runs: Path, table: str = "scores") -> list:
    return read_rows(runs / NAME, table)


def test_coherence_saves_a_row_per_reply_and_a_rerun_or_the_cache_calls_nothing(
    tiny_spec: Callable[..., Path],
    invented: None,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = _saved(tiny_spec, tmp_path)
    haiku = {"name": "api", "model": "claude-haiku-5-5"}
    judge, fake = _judge(tmp_path / "cache", **haiku)
    assert run_score(spec, "coherence", runs_dir=runs, judge=judge) == 0
    saved = {run.run_id: run for run in read_results(runs / NAME)}
    rows = _rows(runs)
    assert sorted(row.run_id for row in rows) == sorted(saved)
    question = load_rubric("coherence").ask().question_sha256
    for row in rows:
        run, judged = saved[row.run_id], saved[row.run_id].output_text.strip()
        assert (row.version, row.scorer, row.scorer_version) == (1, "coherence", 1)
        assert (row.judge, row.question_sha256) == ("api:claude-haiku-5-5", question)
        assert (row.persona, row.label, row.value) == ("", None, len(judged) % 11)
        assert (row.status, row.error, row.attempts) == ("ok", "", 1)
        tokens = {"input_tokens": 100, "output_tokens": 20}
        assert row.details == {"reason": "Fine.", "cached": False} | tokens
        assert row.text_sha256 == hashlib.sha256(judged.encode()).hexdigest()
        assert row.output_sha256 == output_sha256(run) and row.span is None
    assert fake.calls == 10 and not (runs / NAME / "sentences").exists()
    settings = _settings(capsys.readouterr().out)
    assert (settings["items"], settings["calls to make"]) == ("10", "10")
    assert settings["verdicts"] == "10: 10 ok, 0 failed"
    assert (settings["from the cache"], settings["calls made"]) == ("0", "10")
    assert settings["tokens used"] == "1,000 input and 200 output"
    assert settings["cost"].startswith("under $0.01, at claude-haiku-5-5's list")
    files = sorted((runs / NAME / "scores").iterdir())
    judge, fake = _judge(tmp_path / "cache", [], **haiku)  # it would fail any call
    assert run_score(spec, "coherence", runs_dir=runs, judge=judge) == 0
    assert sorted((runs / NAME / "scores").iterdir()) == files
    out = capsys.readouterr().out
    assert _settings(out)["already scored"] == "10"
    assert "Nothing to judge: every item is scored already." in out
    for path in files:  # the rows are lost, but not the cache
        path.unlink()
    assert run_score(spec, "coherence", runs_dir=runs, judge=judge, dry_run=True) == 0
    settings = _settings(capsys.readouterr().out)
    assert (settings["in the cache"], settings["calls to make"]) == ("10", "0")
    assert run_score(spec, "coherence", runs_dir=runs, judge=judge) == 0
    assert _settings(capsys.readouterr().out)["from the cache"] == "10"
    rows = _rows(runs)
    assert fake.calls == 0 and len(rows) == 10
    assert all(row.status == "ok" and row.details["cached"] for row in rows)
    assert all(row.attempts == row.details["input_tokens"] == 0 for row in rows)


def test_a_failed_verdict_is_saved_as_failed_and_judged_again(
    tiny_spec: Callable[..., Path],
    invented: None,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = _saved(tiny_spec, tmp_path)
    broken = REPLIES[1][1]  # smoke-02's pushed reply

    def answer(request: Request) -> str:
        return "not JSON" if request.text == broken else _answer(request)

    judge, fake = _judge(tmp_path / "cache", answer)
    assert run_score(spec, "coherence", runs_dir=runs, judge=judge) == 0
    assert fake.calls == 9 + 4  # four attempts at the broken one
    (failed,) = [row for row in _rows(runs) if row.status != "ok"]
    assert (failed.status, failed.label, failed.value) == ("failed", None, None)
    assert failed.error.startswith("malformed answer: the answer is not one JSON")
    assert failed.attempts == 4
    assert failed.details["raw"] == "not JSON" and failed.details["reason"] == ""
    settings = _settings(capsys.readouterr().out)
    assert settings["verdicts"] == "10: 9 ok, 1 failed"
    assert settings["errors"].startswith("1: malformed answer: the answer is not")
    judge, fake = _judge(tmp_path / "cache")
    assert run_score(spec, "coherence", runs_dir=runs, judge=judge) == 0
    assert [request.text for request in fake.requests] == [broken]
    rows = _rows(runs)
    assert len(rows) == 11
    assert (rows[-1].run_id, rows[-1].status) == (failed.run_id, "ok")
    judge, fake = _judge(tmp_path / "cache", [])
    assert run_score(spec, "coherence", runs_dir=runs, judge=judge) == 0
    assert fake.calls == 0 and len(_rows(runs)) == 11


def test_a_dry_run_counts_the_calls_and_their_cost_and_does_nothing(
    tiny_spec: Callable[..., Path],
    invented: None,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = _saved(tiny_spec, tmp_path)
    cache = tmp_path / "cache"
    judge, fake = _judge(cache, [], name="api", model="claude-haiku-5-5")
    dry = {"runs_dir": runs, "judge": judge, "dry_run": True}
    with folder.locked(runs / NAME):  # a dry run takes no lock
        assert run_score(spec, "coherence", **dry) == 0  # type: ignore[arg-type]
    out = capsys.readouterr().out
    settings = _settings(out)
    assert settings["experiment"] == (
        f"{NAME}, version 1: 10 replies, 0 failed runs left out"
    )
    assert settings["rubric"] == "coherence, version 1"
    assert settings["persona"] == "none: coherence asks about no persona"
    assert settings["texts"] == "whole replies, into the scores table"
    assert (settings["items"], settings["already scored"]) == ("10", "0")
    assert (settings["in the cache"], settings["no text"]) == ("0", "0")
    assert settings["calls to make"] == "10"
    coherence = load_rubric("coherence")
    sent = [coherence.request(reply) for pair in REPLIES for reply in pair]
    characters = sum(len(request.system) + len(request.user) for request in sent)
    assert settings["input tokens"].startswith(f"about {round(characters / 3.5):,}:")
    assert settings["output tokens"].startswith("about 3,000:")
    assert "at claude-haiku-5-5's list prices of $0.10 and $0.50" in settings["cost"]
    assert "This was a dry run: no judge was built or called" in out
    assert fake.calls == 0 and not (runs / NAME / "scores").exists()
    assert not cache.exists()


def test_windows_tile_each_reply_and_each_request_holds_only_its_window(
    tiny_spec: Callable[..., Path],
    invented: None,
    tiny_tokenizer: PreTrainedTokenizerBase,
    tmp_path: Path,
) -> None:
    spec, runs = _saved(tiny_spec, tmp_path)
    judge, fake = _judge(tmp_path / "cache")
    decode = partial(tiny_tokenizer.decode, skip_special_tokens=True)
    windows = {"windows": "sentences"}
    assert run_score(spec, "coherence", runs_dir=runs, judge=judge, **windows) == 0
    assert not (runs / NAME / "scores").exists()
    saved = read_results(runs / NAME)
    expected = {run.run_id: sentence_windows(run.output_text) for run in saved}
    rows = _rows(runs, "sentences")
    assert len(rows) == sum(map(len, expected.values())) == 23
    for run in saved:
        spans = sorted(
            (row.span for row in rows if row.run_id == run.run_id),
            key=lambda span: span.sentence_idx,
        )
        steps = [(span.start_step, span.end_step) for span in spans]
        assert steps[0][0] == 0 and steps[-1][1] == len(run.output_tokens)
        assert [end for _, end in steps[:-1]] == [start for start, _ in steps[1:]]
        for span, window in zip(spans, expected[run.run_id], strict=True):
            place = (span.sentence_idx, span.sentence_count, span.start_char)
            assert (*place, span.end_char) == astuple(window)[:4]  # first to end
            tokens = run.output_tokens[span.start_step : span.end_step]
            assert decode(tokens) == window.text and span.splitter_version == 1
    texts = sorted(w.text.strip() for found in expected.values() for w in found)
    assert sorted(request.text for request in fake.requests) == texts
    assert all(r.user == f"<text>\n{r.text}\n</text>" for r in fake.requests)
    # Windows of two: a window of a last sentence alone is scored already.
    judge, fake = _judge(tmp_path / "cache")
    pairs = windows | {"window_size": 2}
    assert run_score(spec, "coherence", runs_dir=runs, judge=judge, **pairs) == 0
    new = _rows(runs, "sentences")[23:]
    assert len(new) == fake.calls == 10
    assert {(row.span.sentence_idx, row.span.sentence_count) for row in new} == {(0, 2)}


def test_speaks_as_asks_about_the_persona_given_and_which_persona_about_all(
    tiny_spec: Callable[..., Path], invented: None, tmp_path: Path
) -> None:
    spec, runs = _saved(tiny_spec, tmp_path)
    personas = tmp_path / "personas.yaml"
    personas.write_text(PERSONAS, encoding="utf-8")
    judge, fake = _judge(tmp_path / "cache")
    for name in ("Brass Owl", "Salt Captain"):
        options = {"personas_file": personas, "persona": name}
        assert run_score(spec, "speaks_as", runs_dir=runs, judge=judge, **options) == 0
    rows = _rows(runs)
    assert len(rows) == fake.calls == 20
    owl = [row for row in rows if row.persona == "Brass Owl"]
    captain = [row for row in rows if row.persona == "Salt Captain"]
    assert len(owl) == len(captain) == 10
    asked = Persona("Brass Owl", "A pompous mechanical owl who lectures in riddles.")
    question = load_rubric("speaks_as").ask(asked).question_sha256
    assert {row.question_sha256 for row in owl} == {question}
    assert len({row.question_sha256 for row in captain} - {question}) == 1
    assert {row.details["description"] for row in captain} == {
        "A gruff sea captain who calls everyone matey."
    }
    saved = {run.run_id: run.output_text for run in read_results(runs / NAME)}
    assert {row.run_id for row in captain if row.label == "speaks_as"} == {
        run_id for run_id, text in saved.items() if "matey" in text
    }
    options = {"personas_file": personas}
    assert run_score(spec, "which_persona", runs_dir=runs, judge=judge, **options) == 0
    chosen = [row for row in _rows(runs) if row.scorer == "which_persona"]
    assert len(chosen) == 10
    assert {(row.persona, row.label) for row in chosen} == {("", "Salt Captain")}
    names = ["Brass Owl", "Salt Captain"]
    assert all(row.details["personas"] == names for row in chosen)
    assert "description" not in chosen[0].details


def test_the_judge_never_sees_the_prompt_run_condition_or_unit(
    tiny_spec: Callable[..., Path], invented: None, tmp_path: Path
) -> None:
    spec, runs = _saved(tiny_spec, tmp_path, BLIND)
    personas = tmp_path / "personas.yaml"
    personas.write_text(PERSONAS, encoding="utf-8")
    judge, fake = _judge(tmp_path / "cache")
    owl = {"personas_file": personas, "persona": "Brass Owl"}
    for rubric, options in [
        ("coherence", {}),
        ("coherence", {"windows": "sentences"}),
        ("speaks_as", owl),
        ("speaks_as", owl | {"windows": "sentences", "window_size": 2}),
        ("which_persona", {"personas_file": personas}),
    ]:
        assert run_score(spec, rubric, runs_dir=runs, judge=judge, **options) == 0
    experiment = judged_runs(spec, runs)
    labels = sorted(set(experiment.identity["conditions"].values()))
    assert labels == ["loud-pulse-q7", "quiet-baseline-q7"]
    hidden = [
        *labels,
        *(run.run_id for run in experiment.runs),
        *("tiny-vl:L2:N100", "tiny-vl", NAME),
        *SMOKE.prompts,
        *SMOKE.prompt_ids,
    ]
    # Of the 13 windows of two, the 7 that are whole replies are cached.
    assert fake.calls == 10 + 23 + 10 + 6 + 10
    for request in fake.requests:
        seen = f"{request.system}\n{request.user}"
        assert not [secret for secret in hidden if secret in seen]


@pytest.mark.parametrize("module", [score_command, score_rows])
def test_docstring_examples_run(module: object) -> None:
    result = doctest.testmod(module)  # type: ignore[arg-type]
    assert result.attempted > 0
    assert result.failed == 0
