"""`pushlab measure SPEC` (pushlab.scoring.measure_command): the measures that
need no judge, saved as rows of an experiment's scores table.

The runner saves tiny_spec's experiment for real on tiny-vl: an unpushed run
and a pulse on each prompt. The invented fixture has generate() give each
run an invented reply instead: REPLIES, among them a planted loop and a
reply with no word, or GRADED, for three questions with exact answers.
--perplexity loads tiny-vl for real, on the CPU.
"""

import doctest
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from pushlab import cli_measure
from pushlab.analysis.intervals import wilson_interval
from pushlab.analysis.panel import read_panel
from pushlab.cli import main
from pushlab.cli_parser import build_parser
from pushlab.engine import generate as generate_module
from pushlab.engine import score_text
from pushlab.engine.generate import Generation
from pushlab.engine.score_text import perplexity, reply_logprobs
from pushlab.models import load
from pushlab.prompts import PROMPTS_DIR, read_prompt_set
from pushlab.runs import folder
from pushlab.runs.shards import read_results
from pushlab.scoring import measure_command, measure_rows, text_measures
from pushlab.scoring.capability import CAPABILITY_VERSION
from pushlab.store.scores import output_sha256, read_rows

NAME = "tiny-pulses"
COMMIT = "0123456789abcdef0123456789abcdef01234567"  # tests/conftest.py's tiny-vl
SMOKE = read_prompt_set(PROMPTS_DIR / "smoke.jsonl")
LOOP = "I am the tide. " * 20  # planted: it says one sentence again and again
# Each smoke prompt's invented replies, unpushed then pushed.
REPLIES = [
    ("Whisk eggs, milk and flour, then fry small rounds.", "Pancakes! Pile them up."),
    ("Grey water folds over the stones while gulls call.", LOOP),
    ("Sit with them and listen more than you speak.", "Bring soup and stay a while."),
    ("Leaves turn sunlight, water and air into sugar.", "Sun in, sugar out: a leaf."),
    ("Pepper, Biscuit or Mochi would suit a kitten.", "?!"),  # no word to measure
]
# Three questions, their answers (prompt ID, kind, type, answer, check), and
# their invented replies: right unpushed; pushed, right in bold, wrong, nothing.
QUESTIONS = {
    "q-1": "What is 7 + 5? Answer with the number only.",
    "q-2": "What is the capital of France? Answer with one word only.",
    "q-3": "Ann is older than Bo, who is older than Cy. Who is youngest? "
    "Answer with the name only.",
}
ANSWERS = [
    ("q-1", "arithmetic", "number", "12", "7 + 5"),
    ("q-2", "fact", "text", "Paris", "The capital of France."),
    ("q-3", "reasoning", "text", "Cy", "Cy is younger than Bo, younger than Ann."),
]
GRADED = [("12", "**12**"), ("Paris.", "Lyon"), ("Cy", "**")]
HELP_IMPORTS = """
import sys
from pushlab.cli import main
for argv in (["measure", "--help"], ["--help"]):
    try:
        main(argv)
    except SystemExit:
        pass
print([n for n in ("torch", "pandas", "transformers") if n in sys.modules])
"""
OWN_IMPORTS = """
import sys
import pushlab.cli_measure
print([n for n in ("torch", "pandas", "pyarrow", "transformers") if n in sys.modules])
"""
PHRASES = (
    "SPEC|--perplexity|--batch-size N|--runs-dir DIR|--dry-run|scores table|"
    "repetition|word_variety|capability|-answers.jsonl|GPU job|Exit status: 0"
).split("|")


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def invented(monkeypatch: pytest.MonkeyPatch) -> None:
    """Have generate() answer each prompt with its invented replies, the
    pushed one for a run that pushes; an unpushed one ends its turn with
    <|im_end|>, which its tokens keep and its text does not, as a real
    reply's."""
    replies = dict(zip(SMOKE.prompts, REPLIES, strict=True))
    replies |= dict(zip(QUESTIONS.values(), GRADED, strict=True))

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
            steps = ((),) * size, (-1.0,) * size
            made.append(Generation(text, tuple(tokens), stop, touched, *steps))
        return made

    monkeypatch.setattr(generate_module, "generate", generate)


def _questions(tmp_path: Path) -> Path:
    """Write the three questions as a prompt set, its answers file beside it."""
    path = tmp_path / "three.jsonl"
    lines = [{"prompt_id": k, "version": 1, "prompt": q} for k, q in QUESTIONS.items()]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), "utf-8")
    keys, more = ("prompt_id", "kind", "type", "answer", "check"), {"version": 1}
    more |= {"also": [], "units": []}
    answers = [dict(zip(keys, each, strict=True)) | more for each in ANSWERS]
    text = "".join(json.dumps(answer) + "\n" for answer in answers)
    (tmp_path / "three-answers.jsonl").write_text(text, encoding="utf-8")
    return path


def _saved(
    tiny_spec: Callable[..., Path], tmp_path: Path, **settings: str
) -> tuple[Path, Path]:
    """The experiment file and runs dir, its runs saved by the runner."""
    spec, runs = tiny_spec(max_new_tokens="512", **settings), tmp_path / "runs"
    assert main(["run", str(spec), "--runs-dir", str(runs)]) == 0
    return spec, runs


def _measure(spec: Path, runs: Path, *options: str) -> int:
    return main(["measure", str(spec), "--runs-dir", str(runs), *options])


def _sha256(asked: dict) -> str:
    text = json.dumps(asked, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


def _files(runs: Path) -> list[Path]:
    return sorted((runs / NAME / "scores").iterdir())


def _rows(runs: Path, scorer: str) -> list:
    return [row for row in read_rows(runs / NAME, "scores") if row.scorer == scorer]


def _settings(out: str) -> dict[str, str]:
    """The lines of the form 'name  value', by name."""
    lines = [line.split("  ", 1) for line in out.splitlines() if "  " in line]
    return {name.strip(): value.strip() for name, value in lines}


def _summary(out: str) -> dict[str, dict[str, str]]:
    """The summary's cells, by condition and column."""
    lines = out.splitlines()
    at = max(i for i, line in enumerate(lines) if line.startswith("condition  "))
    head, *rows = [re.split(r" {2,}", line.strip()) for line in lines[at:] if line]
    return {row[0]: dict(zip(head, row, strict=True)) for row in rows}


def test_every_reply_gets_both_text_measures_and_the_planted_loop_is_flagged(
    tiny_spec: Callable[..., Path],
    invented: None,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = _saved(tiny_spec, tmp_path)
    assert _measure(spec, runs) == 0
    saved, panel = {run.run_id: run for run in read_results(runs / NAME)}, read_panel()
    rows = read_rows(runs / NAME, "scores")
    assert len(rows) == 2 * len(saved) == 20 and len(_files(runs)) == 2
    version = text_measures.TEXT_MEASURES_VERSION
    for row in rows:
        run = saved[row.run_id]
        asked = {"measure": row.scorer, "version": version, "ngram": 4, "window": 50}
        assert (row.judge, row.question_sha256) == ("pushlab:text", _sha256(asked))
        assert (row.version, row.scorer_version, row.attempts) == (1, version, 0)
        assert (row.persona, row.label) == ("", None)
        stripped = run.output_text.strip().encode()
        assert row.text_sha256 == hashlib.sha256(stripped).hexdigest()
        assert row.output_sha256 == output_sha256(run)
        measured = text_measures.measure_text(run.output_text, panel)
        if measured is None:  # "?!": no word to measure
            assert (row.status, row.value) == ("failed", None)
            assert row.error == "there is no text to measure"
            continue
        value = measured.repetition if row.scorer == "repetition" else measured.variety
        assert (row.status, row.error, row.value) == ("ok", "", value)
        flags = list(text_measures.text_flags(measured, panel))
        assert row.details == {"units": measured.units, "flags": flags}
    assert {row.scorer for row in rows} == {"repetition", "word_variety"}
    failed = {saved[row.run_id].output_text for row in rows if row.status != "ok"}
    flagged = {saved[row.run_id].output_text for row in rows if row.details["flags"]}
    assert (failed, flagged) == ({"?!"}, {LOOP})
    summary = _summary(capsys.readouterr().out)
    assert set(summary) == {"none", "pulse"} and summary["pulse"]["replies"] == "5"
    assert (summary["none"]["flagged"], summary["pulse"]["flagged"]) == ("0/5", "1/4")
    assert "capability" not in summary["none"]
    files = _files(runs)
    assert _measure(spec, runs) == 0  # a rerun measures nothing, failed rows aside
    assert _files(runs) == files
    out = capsys.readouterr().out
    assert "Nothing to measure: every reply is measured already." in out
    assert _settings(out)["repetition"] == "0 to measure, 10 measured already"
    assert _summary(out)["pulse"]["flagged"] == "1/4"


def test_capability_grades_each_reply_against_the_answers_beside_the_prompt_set(
    tiny_spec: Callable[..., Path],
    invented: None,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    prompts = _questions(tmp_path)
    spec, runs = _saved(tiny_spec, tmp_path, prompts=str(prompts))
    assert _measure(spec, runs) == 0
    saved = {run.run_id: run for run in read_results(runs / NAME)}
    rows = _rows(runs, "capability")
    answers = (tmp_path / "three-answers.jsonl").read_bytes()
    asked = {"measure": "capability", "version": CAPABILITY_VERSION}
    asked["answers_sha256"] = hashlib.sha256(answers).hexdigest()
    assert len(rows) == 6 and len(_files(runs)) == 3
    assert {(row.judge, row.scorer_version, row.question_sha256) for row in rows} == {
        ("pushlab:exact-match", CAPABILITY_VERSION, _sha256(asked))
    }
    graded = {
        saved[row.run_id].output_text: (row.status, row.label, row.value, row.details)
        for row in rows
    }
    unpushed = {r.prompt_id: r.output_text for r in saved.values() if not r.schedule}
    for prompt_id, kind, _, answer, _ in ANSWERS:
        details = {"strict": True, "extracted": answer.lower(), "kind": kind}
        assert graded[unpushed[prompt_id]] == ("ok", "correct", 1.0, details)
    assert graded["**12**"][1:3] == ("correct", 1.0)
    assert graded["**12**"][3]["strict"] is False
    assert graded["Lyon"][:3] == ("ok", "wrong", 0.0)
    assert graded["**"][:3] == ("ok", "no_answer", 0.0)
    assert graded["**"][3]["extracted"] is None
    summary = _summary(capsys.readouterr().out)
    low, high = wilson_interval(1, 3)
    assert summary["pulse"]["capability"] == f"1/3 [{low:.2f}, {high:.2f}]"
    assert summary["none"]["capability"].startswith("3/3 [")
    assert summary["pulse"]["flagged"] == "0/2"  # "**" has no word to measure
    # A prompt set changed since the version was made is refused.
    prompts.write_text(prompts.read_text("utf-8").replace("7 + 5", "8 + 5"), "utf-8")
    assert _measure(spec, runs) == 1
    err = capsys.readouterr().err
    assert err.startswith("pushlab measure: ")
    assert "changed since version 1 was made" in err


def test_perplexity_is_the_unpushed_models_on_each_reply_and_a_rerun_loads_nothing(
    tiny_spec: Callable[..., Path],
    invented: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = _saved(tiny_spec, tmp_path)
    assert _measure(spec, runs, "--perplexity", "--batch-size", "2") == 0
    saved = {run.run_id: run for run in read_results(runs / NAME)}
    rows = _rows(runs, "perplexity")
    assert len(rows) == 10 and len(_files(runs)) == 3
    loaded = load.load_model("tiny-vl", torch.float32, device="cpu")
    texts = dict(zip(SMOKE.prompt_ids, SMOKE.prompts, strict=True))
    asked = {"measure": "perplexity", "version": 1, "dtype": "float32"}
    for row in rows:
        run = saved[row.run_id]
        (found,) = reply_logprobs(
            loaded.model, loaded.tokenizer, [texts[run.prompt_id]], [run.output_tokens]
        )
        assert (row.status, row.value) == ("ok", pytest.approx(perplexity(found)))
        mean = pytest.approx(sum(found) / len(found))
        assert row.details == {"tokens": len(found), "mean_logprob": mean}
        assert (row.judge, row.scorer_version) == (f"model:tiny-vl@{COMMIT}", 1)
        assert row.question_sha256 == _sha256(asked)
    assert _summary(capsys.readouterr().out)["none"]["perplexity"] != "-"
    files = _files(runs)
    monkeypatch.setattr(load, "load_model", None)  # no model may load
    assert _measure(spec, runs, "--perplexity") == 0
    assert _files(runs) == files


def test_a_dry_run_counts_what_is_left_and_loads_locks_and_writes_nothing(
    tiny_spec: Callable[..., Path],
    invented: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs = _saved(tiny_spec, tmp_path)
    monkeypatch.setattr(load, "load_model", None)  # no model may load
    with folder.locked(runs / NAME):  # a dry run takes no lock
        assert _measure(spec, runs, "--perplexity", "--dry-run") == 0
    out = capsys.readouterr().out
    settings = _settings(out)
    experiment = f"{NAME}, version 1: 10 replies, 0 failed runs left out"
    assert settings["experiment"] == experiment
    assert settings["repetition"] == "10 to measure, 0 measured already"
    assert settings["perplexity"].startswith("10 to measure, 0 measured already")
    assert f"tiny-vl@{COMMIT} in float32" in settings["perplexity"]
    assert settings["capability"].startswith("not measured: prompts/smoke.jsonl")
    assert "This was a dry run" in out and not (runs / NAME / "scores").exists()
    assert _measure(spec, runs) == 0
    capsys.readouterr()
    assert _measure(spec, runs, "--dry-run") == 0
    settings = _settings(capsys.readouterr().out)
    assert settings["word_variety"] == "0 to measure, 10 measured already"
    assert settings["perplexity"].startswith("not measured: give --perplexity")


def test_what_it_cannot_measure_is_refused_and_what_it_saved_is_kept(
    tiny_spec: Callable[..., Path],
    invented: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _measure(tiny_spec(), tmp_path / "runs") == 1
    err = capsys.readouterr().err
    assert err.startswith("pushlab measure: ") and "tiny-pulses does not exist" in err
    spec, runs = _saved(tiny_spec, tmp_path)
    capsys.readouterr()
    with folder.locked(runs / NAME):
        assert _measure(spec, runs) == 1
    assert capsys.readouterr().err == (
        "pushlab measure: another pushlab run or pushlab score is working on "
        f"{runs / NAME}: try again once it finishes\n"
    )
    assert _measure(spec, runs, "--batch-size", "2") == 1
    err = capsys.readouterr().err
    assert err.startswith("pushlab measure: --batch-size sets how many replies")
    identity = folder.read_identity(runs / NAME)
    folder.write_identity(runs / NAME, identity | {"model": "tiny-vl@" + "f" * 40})
    assert _measure(spec, runs, "--perplexity", "--dry-run") == 1
    err = capsys.readouterr().err
    assert f"was made with tiny-vl@{'f' * 40}, but tiny-vl@{COMMIT} is" in err
    folder.write_identity(runs / NAME, identity)

    def out_of_memory(*args: object, **kwargs: object) -> None:
        raise torch.OutOfMemoryError("CUDA out of memory. Tried to allocate 2 GiB")

    monkeypatch.setattr(score_text, "reply_logprobs", out_of_memory)
    assert _measure(spec, runs, "--perplexity", "--batch-size", "4") == 1
    err = capsys.readouterr().err.splitlines()[-1]  # after loading's progress
    assert err.startswith("pushlab measure: the model ran out of memory")
    assert "a lower --batch-size than 4" in err
    assert len(_rows(runs, "repetition")) == 10 and not _rows(runs, "perplexity")
    monkeypatch.setenv("PUSHLAB_MODELS_DIR", str(tmp_path / "no-models"))
    for dry_run in ([], ["--dry-run"]):
        assert _measure(spec, runs, "--perplexity", *dry_run) == 1
        err = capsys.readouterr().err
        assert "pushlab measure: tiny-vl (local:tiny-qwen3.5 at main) is not" in err


def test_the_help_lists_every_option_and_the_defaults(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["measure", "--help"])
    assert exit_info.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    assert [phrase for phrase in PHRASES if phrase not in text] == []
    args = build_parser().parse_args(["measure", "e.yaml"])
    assert (args.command, args.perplexity, args.batch_size) == ("measure", False, None)
    assert (args.spec, args.runs_dir, args.dry_run) == (Path("e.yaml"), None, False)
    with pytest.raises(SystemExit):
        build_parser().parse_args(["measure", "e.yaml", "--batch-size", "0"])


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


@pytest.mark.parametrize("module", [measure_command, measure_rows, cli_measure])
def test_docstring_examples_run(module: object) -> None:
    result = doctest.testmod(module)  # type: ignore[arg-type]
    assert result.attempted > 0
    assert result.failed == 0
