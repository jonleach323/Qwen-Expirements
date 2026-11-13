"""The options of pushlab score (pushlab.cli_scoring): its help, its
defaults, the values and combinations it refuses, that --help loads nothing
heavy, and the command run from them on the tiny model's own replies.

test_score_command.py covers what pushlab score judges and saves.
"""

import doctest
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pushlab import cli_scoring
from pushlab.cli import main
from pushlab.cli_parser import build_parser
from pushlab.runs import folder
from pushlab.runs.shards import read_results
from pushlab.scoring import api_backend, backends
from pushlab.scoring.cache import VerdictCache
from pushlab.scoring.fake_backend import FakeBackend
from pushlab.scoring.judge import Judge
from pushlab.scoring.rubrics import Request
from pushlab.scoring.score_command import run_score
from pushlab.store.scores import read_rows

NAME = "tiny-pulses"

# What the help must say, every option among it.
PHRASES = [
    "SPEC",
    "--rubric {speaks_as,which_persona,coherence}",
    "--windows {sentences}",
    "--window-size N",
    "--personas FILE",
    "--persona NAME",
    "--backend {api,local}",
    "--model MODEL",
    "--base-url URL",
    "--workers N",
    "--cache-dir DIR",
    "--runs-dir DIR",
    "--dry-run",
    "claude-haiku-5-5",
    "ANTHROPIC_API_KEY",
    "http://127.0.0.1:8000/v1",
    "$PUSHLAB_CACHE_DIR/judge, else /workspace/cache/judge",
    "never the prompt",
    "scores table",
    "sentences table",
    "never guessed",
    "of another family",
    "Exit status: 0",
]
# Imports `--help` must not load: they take seconds. pyarrow is checked on
# this module alone, as pushlab.cli itself loads it (pushlab.store.folders).
HEAVY = ("torch", "pandas", "transformers")
HELP_IMPORTS = f"""
import sys
from pushlab.cli import main
for argv in (["score", "--help"], ["--help"]):
    try:
        main(argv)
    except SystemExit:
        pass
print([name for name in {HEAVY!r} if name in sys.modules])
"""
OWN_IMPORTS = """
import sys
import pushlab.cli_scoring
print([n for n in ("torch", "pandas", "pyarrow", "transformers") if n in sys.modules])
"""


# Invented personas, for the refusals.
PERSONAS = """\
personas:
  - {name: Brass Owl, description: A pompous mechanical owl.}
  - {name: Salt Captain, description: A gruff sea captain.}
"""


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run on the CPU even with a GPU, and keep any cache out of /workspace."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("PUSHLAB_CACHE_DIR", str(tmp_path / "default-cache"))


def _parse(*argv: str) -> object:
    return build_parser().parse_args(["score", *argv])


def _score(request: Request) -> str:
    return json.dumps({"reason": "Odd.", "score": len(request.text) % 11})


def _saved(tiny_spec: Callable[..., Path], tmp_path: Path) -> tuple[Path, Path]:
    """tiny_spec's experiment file and runs dir, its runs saved by the
    runner: the tiny model's own replies, 4 tokens each."""
    spec, runs = tiny_spec(), tmp_path / "runs"
    assert main(["run", str(spec), "--runs-dir", str(runs)]) == 0
    return spec, runs


def test_the_help_says_what_it_does_and_lists_every_option(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["score", "--help"])
    assert exit_info.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    for phrase in PHRASES:
        assert phrase in text, phrase


def test_pushlab_help_lists_score(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "score have a judge model label an experiment's replies" in text


def test_the_defaults_judge_whole_replies_with_claude_haiku() -> None:
    args = _parse("e.yaml", "--rubric", "coherence")
    assert vars(args) == {
        "command": "score",
        "spec": Path("e.yaml"),
        "rubric": "coherence",
        "windows": None,
        "window_size": None,
        "personas": None,
        "persona": None,
        "backend": "api",
        "model": None,
        "base_url": None,
        "workers": 4,
        "cache_dir": None,
        "runs_dir": None,
        "dry_run": False,
    }


def test_every_option_is_read() -> None:
    args = _parse(
        "e.yaml",
        "--rubric",
        "speaks_as",
        "--windows",
        "sentences",
        "--window-size",
        "3",
        "--personas",
        "p.yaml",
        "--persona",
        "Brass Owl",
        "--backend",
        "local",
        "--model",
        "granite-4.1-8b",
        "--base-url",
        "http://127.0.0.1:8001/v1",
        "--workers",
        "8",
        "--cache-dir",
        "cache",
        "--runs-dir",
        "runs",
        "--dry-run",
    )
    assert (args.windows, args.window_size, args.workers) == ("sentences", 3, 8)
    assert (args.personas, args.persona) == (Path("p.yaml"), "Brass Owl")
    assert (args.backend, args.model) == ("local", "granite-4.1-8b")
    assert args.base_url == "http://127.0.0.1:8001/v1"
    assert (args.cache_dir, args.runs_dir) == (Path("cache"), Path("runs"))
    assert args.dry_run is True


@pytest.mark.parametrize(
    "argv",
    [
        ["e.yaml"],
        ["--rubric", "coherence"],
        ["e.yaml", "--rubric", "pirate"],
        ["e.yaml", "--rubric", "coherence", "--backend", "openai"],
        ["e.yaml", "--rubric", "coherence", "--windows", "paragraphs"],
        [
            "e.yaml",
            "--rubric",
            "coherence",
            "--windows",
            "sentences",
            "--window-size",
            "0",
        ],
        ["e.yaml", "--rubric", "coherence", "--window-size", "two"],
        ["e.yaml", "--rubric", "coherence", "--workers", "0"],
        ["e.yaml", "--rubric", "coherence", "--workers", "-2"],
    ],
)
def test_bad_values_stop_with_a_usage_message(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["score", *argv])
    assert exit_info.value.code == 2
    assert "usage: pushlab score" in capsys.readouterr().err


def test_the_installed_command_answers_its_help() -> None:
    pushlab = Path(sys.executable).with_name("pushlab")
    result = subprocess.run(
        [str(pushlab), "score", "--help"], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert "usage: pushlab score" in result.stdout
    assert "--dry-run" in result.stdout


@pytest.mark.parametrize("script", [HELP_IMPORTS, OWN_IMPORTS])
def test_help_loads_nothing_heavy(tmp_path: Path, script: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "[]"


@pytest.mark.parametrize(
    ("rubric", "options", "problem"),
    [
        ("pirate", {}, "unknown rubric 'pirate'; there are speaks_as, which_persona"),
        ("speaks_as", {}, "speaks_as asks about one persona: give --personas FILE"),
        ("speaks_as", {"persona": "Brass Owl"}, "speaks_as asks about one persona"),
        (
            "speaks_as",
            {"personas_file": "FILE"},
            "choose one persona with --persona: 'Brass Owl', 'Salt Captain'",
        ),
        (
            "speaks_as",
            {"personas_file": "FILE", "persona": "Snow Owl"},
            "no persona is named 'Snow Owl'",
        ),
        (
            "which_persona",
            {},
            "which_persona chooses among personas: give --personas FILE",
        ),
        (
            "which_persona",
            {"personas_file": "FILE", "persona": "Brass Owl"},
            "which_persona chooses among all the personas of --personas",
        ),
        (
            "coherence",
            {"personas_file": "FILE"},
            "coherence asks about no persona: leave out --personas and --persona",
        ),
        ("coherence", {"persona": "Brass Owl"}, "coherence asks about no persona"),
        ("coherence", {"window_size": 2}, "--window-size sets how many sentences"),
        ("coherence", {"windows": "lines"}, "--windows takes sentences, not 'lines'"),
        ("coherence", {"workers": 0}, "--workers must be a whole number from 1"),
        ("coherence", {"backend": "local"}, "the local backend needs --model"),
        ("coherence", {"backend": "x"}, "unknown judge backend 'x'; there are api"),
        ("coherence", {"base_url": "http://127.0.0.1:8000/v1"}, "--base-url is"),
    ],
)
def test_options_that_do_not_fit_are_refused_before_the_folder_is_read(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    rubric: str,
    options: dict[str, object],
    problem: str,
) -> None:
    personas = tmp_path / "personas.yaml"
    personas.write_text(PERSONAS, encoding="utf-8")
    options = {k: personas if v == "FILE" else v for k, v in options.items()}
    missing = tmp_path / "runs"  # read, it would refuse otherwise
    assert run_score(tiny_spec(), rubric, runs_dir=missing, **options) == 1  # type: ignore[arg-type]
    err = capsys.readouterr().err
    assert err.startswith("pushlab score: ") and problem in err, err


def test_a_window_size_without_windows_is_refused(
    tiny_spec: Callable[..., Path], capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["score", str(tiny_spec()), "--rubric", "coherence", "--window-size", "1"]
    assert main(argv) == 1
    err = capsys.readouterr().err
    assert err.startswith("pushlab score: --window-size sets how many sentences")


def test_main_judges_the_tiny_models_own_replies_with_the_backend_named(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = _saved(tiny_spec, tmp_path)
    fake, made = FakeBackend(_score, name="local", model="granite-4.1-8b"), []

    def make_backend(backend: str, model: str, base_url: str) -> FakeBackend:
        made.append((backend, model, base_url))
        return fake

    monkeypatch.setattr(backends, "make_backend", make_backend)
    cache, url = tmp_path / "cache", "http://127.0.0.1:8001/v1"
    argv = ["score", str(spec), "--rubric", "coherence", "--runs-dir", str(runs)]
    local = ["--backend", "local", "--model", "granite-4.1-8b", "--base-url", url]
    assert main([*argv, *local, "--cache-dir", str(cache), "--workers", "2"]) == 0
    assert made == [("local", "granite-4.1-8b", url)]
    assert "cost" not in capsys.readouterr().out  # GPU time, not dollars
    saved = read_results(runs / NAME)
    texts = {run.output_text.strip() for run in saved}
    assert fake.calls == len(texts) and "" not in texts  # each text once
    rows = read_rows(runs / NAME, "scores")
    assert len(rows) == 10 and {row.judge for row in rows} == {"local:granite-4.1-8b"}
    fresh = [row for row in rows if not row.details["cached"]]
    assert len(fresh) == len(texts) and all(row.attempts == 1 for row in fresh)
    again = [row for row in rows if row.details["cached"]]  # repeats of a text
    assert all(row.attempts == row.details["input_tokens"] == 0 for row in again)
    assert (cache / "local-granite-4.1-8b.jsonl").exists()
    # A window of a one-sentence reply asks what the whole reply asked.
    assert (
        main([*argv, *local, "--cache-dir", str(cache), "--windows", "sentences"]) == 0
    )
    assert fake.calls == len(texts)
    sentences = read_rows(runs / NAME, "sentences")
    for run in saved:
        (span,) = [row.span for row in sentences if row.run_id == run.run_id]
        assert (span.start_step, span.end_step) == (0, len(run.output_tokens))
        assert (span.start_char, span.end_char) == (0, len(run.output_text))
    # A dry run builds no backend, so it needs no key and no package.
    monkeypatch.setattr(backends, "make_backend", None)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert main([*argv, "--cache-dir", str(cache), "--dry-run"]) == 0
    assert "api:claude-haiku-5-5" in capsys.readouterr().out


def test_no_folder_the_runs_family_a_held_folder_and_no_key_are_refused(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake = FakeBackend(_score)
    judge = Judge(fake, VerdictCache(tmp_path / "cache", "fake:fake-judge"))
    missing = tmp_path / "runs"
    assert run_score(tiny_spec(), "coherence", runs_dir=missing, judge=judge) == 1
    assert "runs/tiny-pulses does not exist" in capsys.readouterr().err
    spec, runs = _saved(tiny_spec, tmp_path)
    tiny = Judge(FakeBackend(_score, model="tiny-judge"))
    for dry_run in (True, False):
        options = {"runs_dir": runs, "judge": tiny, "dry_run": dry_run}
        assert run_score(spec, "coherence", **options) == 1  # type: ignore[arg-type]
        err = capsys.readouterr().err
        assert "pushlab score: the judge tiny-judge is of the family of tiny-vl" in err
    with folder.locked(runs / NAME):
        assert run_score(spec, "coherence", runs_dir=runs, judge=judge) == 1
    err = capsys.readouterr().err
    assert f"is working on {runs / NAME}: score it once that finishes" in err
    monkeypatch.setattr(api_backend, "_anthropic", lambda: SimpleNamespace())
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    cache = tmp_path / "api-cache"
    assert run_score(spec, "coherence", runs_dir=runs, cache_dir=cache) == 1
    assert capsys.readouterr().err == f"pushlab score: {api_backend.NO_KEY}\n"
    assert fake.calls == 0 and not (runs / NAME / "scores").exists()
    assert not cache.exists()


def test_docstring_examples_run() -> None:
    result = doctest.testmod(cli_scoring)
    assert result.attempted > 0
    assert result.failed == 0
