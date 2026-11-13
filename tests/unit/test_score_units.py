"""`pushlab score SPEC --rubric speaks_as --units FILE`: each unit's replies,
its own at every dose and its random controls', and the unpushed replies,
asked about the unit's own persona (pushlab.scoring.score_units,
score_command; decision 0022).

The runner saves an invented experiment for real on tiny-vl: on each of the
five smoke prompts, no push, an owl unit (tiny-vl:L1:N3) at two doses, a
captain unit (tiny-vl:L2:N100) pushed down with one random control, and a
unit no units file names. generate() gives each run an invented reply that
says which push made it (REPLIES), in place of the tiny model's. The units
and personas are invented. A FakeBackend judges: nothing reaches the
network.
"""

import doctest
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from pushlab.cli import main
from pushlab.cli_parser import build_parser
from pushlab.engine import generate as generate_module
from pushlab.engine.generate import Generation
from pushlab.prompts import PROMPTS_DIR, read_prompt_set
from pushlab.runs import folder
from pushlab.runs.shards import read_results
from pushlab.scoring import score_units
from pushlab.scoring.cache import VerdictCache
from pushlab.scoring.fake_backend import FakeBackend
from pushlab.scoring.judge import Judge
from pushlab.scoring.personas import Persona
from pushlab.scoring.rubrics import Request, load_rubric
from pushlab.scoring.score_command import run_score
from pushlab.store.records import RunRecord
from pushlab.store.scores import read_rows
from pushlab.units.ladder_units import run_push

NAME = "tiny-pulses"
SMOKE = read_prompt_set(PROMPTS_DIR / "smoke.jsonl")
# Each smoke prompt's invented unpushed reply; a push puts words before it.
PLAIN = [
    "Mix flour, milk and an egg. Fry the batter until golden.",
    "Grey waves fold over the shore. Gulls call above the tide.",
    "Listen without rushing to fix things. Let them know you care.",
    "Plants catch sunlight in their leaves. They make sugar from water and air.",
    "How about Pepper? It suits a lively kitten.",
]
OWL, CAPTAIN, OTHER = "tiny-vl:L1:N3", "tiny-vl:L2:N100", "tiny-vl:L3:N7"
# What each push puts before the plain reply, by unit and dose; any other
# unit is the captain's random control.
REPLIES = {
    (OWL, 0.5): "Hoo.",
    (OWL, 1.0): "Hoo, hoo! I am the brass owl.",
    (CAPTAIN, -1.0): "Arr, matey!",
    (OTHER, 0.5): "Beep.",
}
CONTROL = "Well."
# Labels no rubric or persona's text holds.
LABELS = ("still-q7", "hoot-q7", "brine-q7", "beep-q7")
CONDITIONS = f"""\
conditions:
  - {{label: {LABELS[0]}, segments: []}}
  - label: {LABELS[1]}
    shape: constant
    params: {{unit: {OWL}}}
    grid: {{alpha: [0.5, 1.0]}}
  - label: {LABELS[2]}
    shape: constant
    params: {{unit: {CAPTAIN}, alpha: -1.0}}
    controls: 1
    control_seed: 0
  - label: {LABELS[3]}
    shape: constant
    params: {{unit: {OTHER}, alpha: 0.5}}
"""
UNITS = (
    "unit,direction,persona,description,source\n"
    f"{OWL},+1,Brass Owl,A pompous mechanical owl.,invented\n"
    f"{CAPTAIN},-1,Salt Captain,,invented\n"
)
PERSONAS = """\
personas:
  - name: Salt Captain
    description: A gruff sea captain who calls everyone matey.
  - name: Brass Owl
    description: An owl of brass and riddles.
"""
KEYWORDS = {"Brass Owl": "hoo", "Salt Captain": "matey"}
# The runs whose replies speak as the persona asked about.
SPEAKERS = {("owl", "Brass Owl"), ("captain", "Salt Captain")}
TABLE = ["own", "controls", "unpushed", "items", "scored", "cached", "no text", "calls"]


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run on the CPU even with a GPU, and keep any cache out of /workspace."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("PUSHLAB_CACHE_DIR", str(tmp_path / "default-cache"))


@pytest.fixture
def invented(monkeypatch: pytest.MonkeyPatch) -> None:
    """Have generate() answer each prompt with its PLAIN reply, after the
    words of REPLIES for the push its run makes."""
    plain = dict(zip(SMOKE.prompts, PLAIN, strict=True))

    def generate(model, tokenizer, prompts, pushes, decoding, norms, **kw):  # type: ignore[no-untyped-def]
        made = []
        for prompt, schedule in zip(prompts, kw["schedules"], strict=True):
            text = plain[prompt]
            if schedule.segments:
                segment = schedule.segments[0]
                push = (str(segment.unit), segment.alpha_start)
                text = f"{REPLIES.get(push, CONTROL)} {text}"
            tokens = tokenizer.encode(text, add_special_tokens=False)
            tokens.append(tokenizer.convert_tokens_to_ids("<|im_end|>"))
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            touched, size = (0,) * len(schedule.segments), len(tokens)
            steps = (((),) * size, (-1.0,) * size)
            made.append(Generation(text, tuple(tokens), "end_of_turn", touched, *steps))
        return made

    monkeypatch.setattr(generate_module, "generate", generate)


@pytest.fixture
def saved(
    tiny_spec: Callable[..., Path], invented: None, tmp_path: Path
) -> tuple[Path, Path, Path]:
    """The experiment file, its runs dir, the runs saved by the runner, and
    a units file of the owl and the captain."""
    spec = tiny_spec(CONDITIONS, max_new_tokens="160")
    runs = tmp_path / "runs"
    assert main(["run", str(spec), "--runs-dir", str(runs)]) == 0
    units = tmp_path / "units.csv"
    units.write_text(UNITS, encoding="utf-8")
    return spec, runs, units


def _persona(request: Request) -> str:
    return request.question.split("\n", 1)[0].removeprefix("Persona: ")


def _answer(request: Request) -> str:
    """speaks_as when the text holds its persona's word, else assistant."""
    word = KEYWORDS[_persona(request)]
    label = "speaks_as" if word in request.text.casefold() else "assistant"
    return json.dumps({"reason": "Heard.", "label": label})


def _judge(cache: Path, script: object = _answer) -> tuple[Judge, FakeBackend]:
    """A judge of a FakeBackend with its cache in cache, which never waits."""
    fake = FakeBackend(script, name="fake", model="fake")  # type: ignore[arg-type]
    return Judge(fake, VerdictCache(cache, "fake:fake"), sleep=lambda s: None), fake


def _settings(out: str) -> dict[str, str]:
    """The lines of the form 'name  value', by name."""
    lines = [line.split("  ", 1) for line in out.splitlines() if "  " in line]
    return {name.strip(): value.strip() for name, value in lines}


def _table(out: str) -> dict[str, dict[str, str]]:
    """The table of units: each row's persona and counts, by its unit."""
    rows = {}
    for line in out.splitlines():
        cells = [cell.strip() for cell in line.strip().split("  ") if cell.strip()]
        if cells and cells[0].startswith("tiny-vl:") and len(cells) == 10:
            rows[cells[0]] = dict(zip(["persona", *TABLE], cells[1:], strict=True))
    return rows


def _kind(run: RunRecord) -> str:
    """Which push made the run: none, owl, captain, control or other."""
    push = run_push(run)
    if push is None:
        return "none"
    if push.control_for is not None:
        return "control"
    return {OWL: "owl", CAPTAIN: "captain", OTHER: "other"}[push.unit]


def test_each_units_replies_are_asked_about_its_persona_and_the_unpushed_about_all(
    saved: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    spec, runs, units = saved
    judge, fake = _judge(tmp_path / "cache")
    options = {"units_file": units, "runs_dir": runs, "judge": judge}
    assert run_score(spec, "speaks_as", **options) == 0  # type: ignore[arg-type]
    saved_runs = {run.run_id: run for run in read_results(runs / NAME)}
    kinds = {run_id: _kind(run) for run_id, run in saved_runs.items()}
    assert Counter(kinds.values())["owl"] == 10
    rows = read_rows(runs / NAME, "scores")
    asked: dict[str, set[str]] = {}
    for row in rows:
        asked.setdefault(row.run_id, set()).add(row.persona)
    owls, captains = {"Brass Owl"}, {"Salt Captain"}
    expected = {"none": owls | captains, "owl": owls} | dict.fromkeys(
        ("captain", "control"), captains
    )
    assert asked == {
        run_id: expected[kind] for run_id, kind in kinds.items() if kind != "other"
    }
    assert len(rows) == 5 * (2 + 2 + 1 + 1)
    owl = Persona("Brass Owl", "A pompous mechanical owl.")
    captain = Persona("Salt Captain")
    questions = {
        persona.name: load_rubric("speaks_as").ask(persona).question_sha256
        for persona in (owl, captain)
    }
    for row in rows:
        assert row.question_sha256 == questions[row.persona] and row.status == "ok"
        description = owl.description if row.persona == owl.name else ""
        assert row.details["description"] == description
        speaks = (kinds[row.run_id], row.persona) in SPEAKERS
        assert row.label == ("speaks_as" if speaks else "assistant")
    # One call for each persona and text: every text here is its run's own.
    pairs = {(_persona(request), request.text) for request in fake.requests}
    assert fake.calls == len(rows) == len(pairs)
    # The judge sees the persona and the text: one question at every dose of
    # a unit, and never a unit, dose, condition, run or prompt.
    questions_asked = {(r.system, r.question) for r in fake.requests}
    assert len(questions_asked) == len({_persona(r) for r in fake.requests}) == 2
    hidden = [
        *(OWL, CAPTAIN, OTHER, "tiny-vl", NAME, "alpha", "0.5", "1.0"),
        *LABELS,
        *saved_runs,
        *SMOKE.prompts,
        *SMOKE.prompt_ids,
    ]
    for request in fake.requests:
        seen = f"{request.system}\n{request.question}"
        assert not [secret for secret in hidden if secret in seen]


def test_a_dry_run_counts_each_unit_and_a_rerun_judges_nothing(
    saved: tuple[Path, Path, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs, units = saved
    capsys.readouterr()  # the runner's
    judge, fake = _judge(tmp_path / "cache", [])  # it would fail any call
    options = {"units_file": units, "runs_dir": runs, "judge": judge}
    with folder.locked(runs / NAME):  # a dry run takes no lock
        assert run_score(spec, "speaks_as", dry_run=True, **options) == 0  # type: ignore[arg-type]
    out = capsys.readouterr().out
    settings = _settings(out)
    assert settings["experiment"] == (
        f"{NAME}, version 1: 30 replies, 0 failed runs left out"
    )
    assert settings["units"] == (
        f"2 of {units}: each unit's replies and the unpushed ones asked about its "
        "persona, 2 personas in all"
    )
    assert settings["not asked"] == "5 replies pushed by no unit of the file"
    assert (settings["items"], settings["already scored"]) == ("30", "0")
    assert (settings["in the cache"], settings["calls to make"]) == ("0", "30")
    assert settings["cost"].startswith("unknown")
    table = _table(out)
    owl = ["Brass Owl", "10", "0", "5", "15", "0", "0", "0", "15"]
    captain = ["Salt Captain", "5", "5", "5", "15", "0", "0", "0", "15"]
    columns = ["persona", *TABLE]
    assert table == {
        f"{OWL}+": dict(zip(columns, owl, strict=True)),
        f"{CAPTAIN}-": dict(zip(columns, captain, strict=True)),
    }
    assert "This was a dry run" in out and fake.calls == 0
    assert not (runs / NAME / "scores").exists()
    judge, fake = _judge(tmp_path / "cache")
    options["judge"] = judge
    assert run_score(spec, "speaks_as", **options) == 0  # type: ignore[arg-type]
    out = capsys.readouterr().out
    assert "About Brass Owl, the persona of tiny-vl:L1:N3+:" in out
    assert "About Salt Captain, the persona of tiny-vl:L2:N100-:" in out
    assert _settings(out)["verdicts"] == "30: 30 ok, 0 failed"
    assert fake.calls == 30
    files = sorted((runs / NAME / "scores").iterdir())
    judge, fake = _judge(tmp_path / "cache", [])
    options["judge"] = judge
    assert run_score(spec, "speaks_as", **options) == 0  # type: ignore[arg-type]
    out = capsys.readouterr().out
    assert "Nothing to judge: every item is scored already." in out
    assert {row["scored"] for row in _table(out).values()} == {"15"}
    assert sorted((runs / NAME / "scores").iterdir()) == files and fake.calls == 0


def test_the_personas_file_gives_descriptions_and_windows_are_asked_by_unit_too(
    saved: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    spec, runs, units = saved
    personas = tmp_path / "personas.yaml"
    personas.write_text(PERSONAS, encoding="utf-8")
    judge, fake = _judge(tmp_path / "cache")
    options = {"personas_file": personas, "runs_dir": runs, "judge": judge}
    windows = {"windows": "sentences", "window_size": 2}
    assert run_score(spec, "speaks_as", units_file=units, **options, **windows) == 0  # type: ignore[arg-type]
    rows = read_rows(runs / NAME, "sentences")
    assert {(row.persona, row.details["description"]) for row in rows} == {
        ("Brass Owl", "An owl of brass and riddles."),
        ("Salt Captain", "A gruff sea captain who calls everyone matey."),
    }
    kinds = {run.run_id: _kind(run) for run in read_results(runs / NAME)}
    asked_about = {
        "Brass Owl": {"none", "owl"},
        "Salt Captain": {"none", "captain", "control"},
    }
    for row in rows:
        assert kinds[row.run_id] in asked_about[row.persona]
    assert not (runs / NAME / "scores").exists()
    # The same units with one persona of the same question share it: the
    # unpushed replies are asked once, for both.
    twins = tmp_path / "twins.csv"
    twins.write_text(UNITS.replace("Salt Captain", "Brass Owl"), encoding="utf-8")
    judge, fake = _judge(tmp_path / "cache")
    options["judge"] = judge
    assert run_score(spec, "speaks_as", units_file=twins, **options) == 0  # type: ignore[arg-type]
    rows = read_rows(runs / NAME, "scores")
    assert {row.persona for row in rows} == {"Brass Owl"}
    counts = Counter(kinds[row.run_id] for row in rows)
    assert counts == {"none": 5, "owl": 10, "captain": 5, "control": 5}


@pytest.mark.parametrize(
    ("rubric", "options", "problem"),
    [
        ("coherence", {}, "--units asks each unit's replies about its own persona"),
        ("which_persona", {}, "give --rubric speaks_as, or leave out --units"),
        ("speaks_as", {"persona": "Brass Owl"}, "leave out --persona"),
        ("speaks_as", {"units_file": "NAMELESS"}, f"gives no persona for {CAPTAIN}-"),
        ("speaks_as", {"units_file": "NONE"}, 'a persona cannot be named "none"'),
        ("speaks_as", {"units_file": "MISSING"}, "cannot read the units file"),
        ("speaks_as", {"personas_file": "OWLS"}, "no persona is named 'Salt Captain'"),
    ],
)
def test_units_that_cannot_be_asked_are_refused_before_the_folder_is_read(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    rubric: str,
    options: dict[str, object],
    problem: str,
) -> None:
    paths = {"MISSING": tmp_path / "missing.csv"}
    for key, text in [
        ("UNITS", UNITS),
        ("NAMELESS", UNITS.replace("Salt Captain", "")),
        ("NONE", UNITS.replace("Salt Captain", "None")),
        ("OWLS", "personas:\n  - {name: Brass Owl}\n"),
    ]:
        paths[key] = tmp_path / f"{key.lower()}.txt"
        paths[key].write_text(text, encoding="utf-8")
    given = {"units_file": "UNITS"} | options
    given = {k: paths.get(v, v) if isinstance(v, str) else v for k, v in given.items()}
    missing = tmp_path / "runs"  # read, it would refuse otherwise
    assert run_score(tiny_spec(), rubric, runs_dir=missing, **given) == 1  # type: ignore[arg-type]
    err = capsys.readouterr().err
    assert err.startswith("pushlab score: ") and problem in err, err


def test_the_command_takes_units_of_the_experiments_model(
    saved: tuple[Path, Path, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs, units = saved
    argv = ["score", str(spec), "--rubric", "speaks_as", "--runs-dir", str(runs)]
    options = [*argv, "--cache-dir", str(tmp_path / "cache"), "--dry-run"]
    assert build_parser().parse_args([*options, "--units", str(units)]).units == units
    capsys.readouterr()
    assert main([*options, "--units", str(units)]) == 0
    out = capsys.readouterr().out
    assert _settings(out)["calls to make"] == "30" and "api:claude-haiku-5-5" in out
    other = tmp_path / "other.csv"
    other.write_text("unit,persona\nqwen3-8b:L9:N4242,Brass Owl\n", encoding="utf-8")
    judge, fake = _judge(tmp_path / "cache")
    for dry_run in (True, False):
        given = {"units_file": other, "runs_dir": runs, "judge": judge}
        assert run_score(spec, "speaks_as", dry_run=dry_run, **given) == 1  # type: ignore[arg-type]
        assert capsys.readouterr().err == (
            f"pushlab score: the units of {other} are qwen3-8b's, but {NAME} runs on "
            "tiny-vl: give the units file of its ladder\n"
        )
    assert fake.calls == 0 and not (runs / NAME / "scores").exists()
    with pytest.raises(SystemExit):
        main(["score", "--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "--units FILE" in help_text and "the same at every dose" in help_text


def test_docstring_examples_run() -> None:
    result = doctest.testmod(score_units)
    assert result.attempted > 0
    assert result.failed == 0
