"""pushlab panel select (pushlab.panel.select_command, decision 0023): the
panel chosen from stage confirm's and stage ladder's scores, end to end.

The pipeline of tests/unit/panel_experiments.py runs and judges made-up
tiny-vl candidates with invented replies, here with all five going on to
the ladder: the owl never works there; the captain, the teapot, the fox and
the oracle work at their reference doses and never collapse, so stage
confirm runs those four, each at its reference dose on the five smoke
prompts. There the fox speaks as its persona on only 3 of them, and the
other three on all 5, so they are the finalists. A fake which_persona judge
names each reply's own persona, or takes some personas' replies for
another's.
"""

import doctest
import importlib.util
import json
from collections import Counter
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest
import torch

from pushlab import PROJECT_ROOT
from pushlab.cli import main
from pushlab.panel import select_command, select_text
from pushlab.panel.panel_file import read_panel_file
from pushlab.runs.folder import SCORES
from pushlab.scoring import backends
from pushlab.scoring.fake_backend import FakeBackend
from pushlab.scoring.judge import CallError
from pushlab.scoring.rubrics import Request
from pushlab.store.scores import read_rows

_SPEC = importlib.util.spec_from_file_location(
    "panel_experiments", Path(__file__).with_name("panel_experiments.py")
)
made = importlib.util.module_from_spec(_SPEC)  # type: ignore[arg-type]
_SPEC.loader.exec_module(made)  # type: ignore[union-attr]

KEYS = made.KEYS
# On the ladder each but the owl speaks as its persona on all 3 prompts from
# its reference dose up, and on one below it; coherence is 9 throughout.
LADDER_SPEAKS = {
    ("owl", 0.5): {0},
    ("owl", 1.0): {0},
    ("owl", 1.5): {0},
    ("captain", 0.25): {0},
    ("captain", 0.5): range(3),
    ("captain", 0.75): range(3),
    ("teapot", 0.5): {0},
    ("teapot", 1.0): range(3),
    ("teapot", 1.5): range(3),
    ("fox", 0.625): {0},
    ("fox", 1.25): range(3),
    ("fox", 1.875): range(3),
    ("oracle", 0.375): {0},
    ("oracle", 0.75): range(3),
    ("oracle", 1.125): range(3),
}
CONFIRM_SPEAKS = {
    ("captain", 0.5): range(5),
    ("teapot", 1.0): range(5),
    ("fox", 1.25): range(3),
    ("oracle", 0.75): range(5),
}
# The oracle's prompts score 7, 6, 7, 6 and 7: a mean of 6.6.
CONFIRM_COHERENCE = {
    ("captain", 0.5): [7],
    ("teapot", 1.0): [8],
    ("fox", 1.25): [8],
    ("oracle", 0.75): [7, 6],
}
CATEGORIES = {"teapot": "Objects", "captain": "people", "oracle": " mystical "}
FINALISTS = ("teapot", "captain", "oracle")  # in the ladder's order


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run on the CPU even with a GPU, and keep any cache out of /workspace."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("PUSHLAB_CACHE_DIR", str(tmp_path / "default-cache"))


class Chosen:
    """The pipeline to stage confirm, run and judged, and pushlab panel
    select on it, its which_persona judge a fake one."""

    def __init__(self, pipeline: "made.Pipeline", ladder: Path, confirm: Path) -> None:
        self.pipeline, self.ladder, self.confirm = pipeline, ladder, confirm
        folder = pipeline.folder
        self.categories = folder / "categories.csv"
        self.out, self.doc = folder / "panel_tiny-vl.yaml", folder / "tiny-vl.md"
        self.fake: FakeBackend | None = None

    def fill(self, categories: Mapping[str, str] = CATEGORIES) -> None:
        """Fill in the categories file, as you would."""
        rows = ["key,persona,category"]
        rows += [f"{KEYS[role]},x,{category}" for role, category in categories.items()]
        self.categories.write_text("\n".join(rows) + "\n", encoding="utf-8")

    def select(
        self,
        monkeypatch: pytest.MonkeyPatch,
        *options: str,
        taken: Mapping[str, str] | None = None,
        answer: Callable[[Request], object] | None = None,
    ) -> int:
        """pushlab panel select with options; its judge names each reply's
        own persona, but a role in taken gets the other role's name, unless
        answer gives the judge's answers."""
        swaps = taken or {}

        def named(request: Request) -> str:
            assert request.rubric == "which_persona", request.rubric
            role, _, _ = made.reading(request.text)
            chosen = made.PERSONAS[swaps.get(role, role)]
            return json.dumps({"reason": "Hoo.", "choice": chosen})

        self.fake = FakeBackend(
            answer or named, name="api", model="claude-haiku-5-5", settings="effort=low"
        )
        monkeypatch.setattr(backends, "make_backend", lambda *given: self.fake)
        argv = ["panel", "select", str(self.confirm), "--ladder", str(self.ladder)]
        argv += ["--categories", str(self.categories), "--out", str(self.out)]
        argv += ["--doc", str(self.doc), "--runs-dir", str(self.pipeline.runs)]
        return main([*argv, "--panel", str(self.pipeline.settings), *options])

    @property
    def calls(self) -> int:
        """The calls the last select made of its judge."""
        return 0 if self.fake is None else self.fake.calls


@pytest.fixture
def chosen(
    tiny_vl_calibration: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Callable[..., Chosen]:
    """The pipeline run and judged to stage confirm, on settings changed
    by the keyword arguments given."""

    def make(**changes: object) -> Chosen:
        pipeline = made.Pipeline(tmp_path, monkeypatch)
        settings = {"stage_b_size": 5, "ladder_size": 5} | changes
        made.write_settings(pipeline.settings, **settings)
        ladder = pipeline.to_ladder()
        pipeline.judged(ladder, LADDER_SPEAKS, {})
        assert pipeline.stage("confirm", ladder) == 0
        confirm = pipeline.out("confirm")
        pipeline.judged(confirm, CONFIRM_SPEAKS, CONFIRM_COHERENCE)
        return Chosen(pipeline, ladder, confirm)

    return make


def _lines(text: str) -> list[str]:
    return [" ".join(line.split()) for line in text.splitlines()]


def _distinct_rows(confirm: Path, runs: Path) -> list:
    """The which_persona rows of the confirm folder's scores table."""
    rows = read_rows(runs / confirm.stem, SCORES)
    return [row for row in rows if row.scorer == "which_persona"]


def test_select_asks_for_categories_then_plans_then_writes_the_panel(
    chosen: Callable[..., Chosen],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = chosen()
    capsys.readouterr()
    # First the categories file, a row per finalist, its category blank.
    assert run.select(monkeypatch) == 1
    shown = capsys.readouterr()
    assert shown.err.startswith(
        f"pushlab panel select: fill in each finalist's category in {run.categories}"
    )
    text = run.categories.read_text(encoding="utf-8-sig").splitlines()
    rows = [f"{KEYS[role]},{made.PERSONAS[role]}," for role in FINALISTS]
    assert text == ["key,persona,category", *rows]
    lines = _lines(shown.out)
    owl = [line for line in lines if line.startswith(f"{KEYS['owl']} Brass Owl ")]
    # A resample of prompt 0 alone finds the owl working at 0.5.
    assert owl[-1].startswith(f"{KEYS['owl']} Brass Owl neuron none [0.5, beyond] ")
    assert owl[-1].endswith(
        "left out: no working dose below a collapse dose on the ladder"
    )
    fox = [line for line in lines if line.startswith(f"{KEYS['fox']} Tin Fox ")][-1]
    assert fox.startswith(f"{KEYS['fox']} Tin Fox neuron 1.25 ")
    assert " 3/5 0.60 8.0 left out: speaks as its persona on under 0.8 " in fox
    oracle = [line for line in lines if line.startswith(f"{KEYS['oracle']} ")][-1]
    assert oracle.endswith(" 5/5 1.00 6.6 finalist")
    assert "3 finalists of 5 candidates." in lines
    # Then, without --yes, the plan: two replies of each finalist, no call.
    run.fill()
    assert run.select(monkeypatch) == 0 and run.calls == 0
    lines = _lines(capsys.readouterr().out)
    assert "calls to make 6" in lines and "in the cache 0" in lines
    personas = "personas 3, every finalist: Velvet Teapot, Salt Captain, Moss Oracle"
    assert personas in lines
    assert any(line.startswith("cost ") for line in lines)
    assert lines[-1] == (
        "Nothing was called or written. Run the same command with --yes to make "
        "these calls and choose the panel."
    )
    assert not run.out.exists() and not run.doc.exists()
    assert _distinct_rows(run.confirm, run.pipeline.runs) == []
    # With --yes: six calls, saved in the confirm folder's scores table.
    assert run.select(monkeypatch, "--yes") == 0 and run.calls == 6
    out = capsys.readouterr().out
    saved = _distinct_rows(run.confirm, run.pipeline.runs)
    assert len(saved) == 6 and {row.scorer for row in saved} == {"which_persona"}
    panel = read_panel_file(run.out)
    assert [p.key for p in panel.personas] == [KEYS[r] for r in FINALISTS]
    teapot, captain, oracle = panel.personas
    assert [p.category for p in panel.personas] == ["objects", "people", "mystical"]
    assert (teapot.kind, captain.kind) == ("set", "neuron")
    assert (teapot.working_dose, captain.working_dose, oracle.working_dose) == (
        1.0,
        0.5,
        0.75,
    )
    assert {p.collapse_dose for p in panel.personas} == {None}
    assert (captain.reliability.k, captain.reliability.n) == (5, 5)
    assert (oracle.coherence.mean, oracle.coherence.n) == (6.6, 5)
    assert (oracle.distinct.k, oracle.distinct.n, oracle.distinct.rate) == (2, 2, 1.0)
    assert (panel.model, panel.candidates_file) == ("tiny-vl", str(run.pipeline.listed))
    assert panel.made_from["which_persona"].startswith("judged by api:claude-haiku")
    # Fewer than the panel's 4: all three kept, with what would help.
    lines = _lines(out)
    assert "The panel: 3 personas, fewer than the panel's 4: every one that " in (
        " ".join(lines)
    )
    assert (
        "The rule most of the 2 left out failed is dose (1 of them): a ladder of "
        "other doses, or more candidates, would help." in lines
    )
    page = run.doc.read_text(encoding="utf-8")
    assert "stays out of git" in page
    assert "- smoke-01: The teapot at 1.0 answers prompt 0." in page
    assert page.count("\n### ") == 3 and page.count("\n- smoke-0") == 6
    # Again: every verdict is saved, so no call is made.
    assert run.select(monkeypatch, "--yes") == 0 and run.calls == 0
    assert read_panel_file(run.out) == panel


def test_the_check_drops_a_persona_whose_replies_the_judge_takes_for_another(
    chosen: Callable[..., Chosen],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = chosen()
    run.fill()
    capsys.readouterr()
    # The captain's replies are taken for the teapot's: the captain goes,
    # and the teapot, whose own replies are named right, stays.
    assert run.select(monkeypatch, "--yes", taken={"captain": "teapot"}) == 0
    lines = _lines(capsys.readouterr().out)
    left = f"Left out {KEYS['captain']} (Salt Captain): named for 0 of 2, taken for"
    assert f"{left} Velvet Teapot 2." in lines
    row = f"{KEYS['captain']} Salt Captain 0/2 0.00 Velvet Teapot 2 left out"
    assert row in lines
    panel = read_panel_file(run.out)
    assert [p.key for p in panel.personas] == [KEYS["teapot"], KEYS["oracle"]]


def test_fewer_than_the_minimum_that_qualify_write_nothing(
    chosen: Callable[..., Chosen],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = chosen()
    run.fill()
    capsys.readouterr()
    taken = {"captain": "teapot", "oracle": "teapot"}
    assert run.select(monkeypatch, "--yes", taken=taken) == 1
    assert capsys.readouterr().err == (
        "pushlab panel select: only 1 qualifies, fewer than the panel's minimum of "
        "2, so nothing was written. The rule most of the 4 left out failed is "
        "distinct (2 of them): larger sets, or more candidates, would help.\n"
    )
    assert not run.out.exists() and not run.doc.exists()


def test_too_few_finalists_stop_before_the_categories_and_the_judge(
    chosen: Callable[..., Chosen],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = chosen(panel_minimum=4)
    capsys.readouterr()
    assert run.select(monkeypatch, "--yes") == 1 and run.calls == 0
    assert capsys.readouterr().err.startswith(
        "pushlab panel select: only 3 qualify, fewer than the panel's minimum of 4, "
        "so nothing was written. The rule most of the 2 left out failed is dose"
    )
    assert not run.categories.exists()


@pytest.mark.parametrize(
    ("given", "problem"),
    [
        ("categories", "is in the repository: it names candidates by key"),
        ("out", "is in the repository, where git would keep it"),
        ("category", "'animal' (row 2) is not a category; a category is one of"),
        ("unjudged", "of tiny-screen has an ok coherence verdict of fake:fake"),
        ("other list", "those of {copy}: give the ladder that stage confirm was"),
        ("--base-url", "--base-url is the local backend's"),
        ("--backend", "the local backend needs --model"),
        ("--model", "the judge tiny-judge is of the family of tiny-vl"),
        ("--judge", "{ladder}: no speaks_as or coherence verdict of fake:other: "),
    ],
)
def test_select_refuses_what_it_cannot_use(
    given: str,
    problem: str,
    chosen: Callable[..., Chosen],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = chosen()
    run.fill()
    copy = run.pipeline.folder / "copy.csv"
    if given == "categories":
        run.categories = PROJECT_ROOT / "categories.csv"
    elif given == "out":
        run.out = PROJECT_ROOT / "units" / "tiny-vl.yaml"
    elif given == "category":
        run.fill({"teapot": "animal"})
    elif given == "unjudged":
        run.ladder = run.pipeline.screen  # never judged for coherence
    elif given == "other list":  # the ladder as it was, but naming a copy of its list
        copy.write_bytes(run.pipeline.listed.read_bytes())
        text = run.ladder.read_text(encoding="utf-8")
        run.ladder = run.pipeline.folder / "other-ladder.yaml"
        run.ladder.write_text(text.replace(str(run.pipeline.listed), str(copy)))
    options = {
        "--base-url": "http://127.0.0.1:8000/v1",
        "--backend": "local",
        "--model": "tiny-judge",
        "--judge": "fake:other",  # saved for the confirm alone
    }
    if given == "--judge":  # the confirm judged by fake:other too, not the ladder
        answer, runs = (
            made.answers(CONFIRM_SPEAKS, CONFIRM_COHERENCE),
            run.pipeline.runs,
        )
        listed = run.pipeline.listed
        made.score(run.confirm, runs, "speaks_as", answer, listed, judge="other")
        made.score(run.confirm, runs, "coherence", answer, judge="other")
    given_options = [given, options[given]] if given in options else []
    capsys.readouterr()
    assert run.select(monkeypatch, "--yes", *given_options) == 1 and run.calls == 0
    assert problem.format(copy=copy, ladder=run.ladder) in capsys.readouterr().err
    assert not (PROJECT_ROOT / "categories.csv").exists()
    assert not (PROJECT_ROOT / "units" / "tiny-vl.yaml").exists()
    assert not run.out.exists() and not run.doc.exists()


def test_a_judge_that_cannot_answer_stops_select_with_what_it_found_saved(
    chosen: Callable[..., Chosen],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    run = chosen()
    run.fill()
    capsys.readouterr()
    rejected = CallError("401: the key was rejected", fatal=True)
    assert run.select(monkeypatch, "--yes", answer=lambda request: rejected) == 1
    err = capsys.readouterr().err
    assert err.startswith("pushlab panel select: api:claude-haiku-5-5: 401: the key")
    assert "The verdicts found before it stopped are saved" in err
    assert not run.out.exists() and not run.doc.exists()


@pytest.mark.parametrize("module", [select_command, select_text])
def test_the_examples_in_the_docstrings_hold(module: object) -> None:
    result = doctest.testmod(module)  # type: ignore[arg-type]
    assert result.failed == 0 and result.attempted > 0


def test_a_refusal_short_of_the_minimum_says_what_would_help() -> None:
    reasons = Counter({"reliability": 4, "dose": 1})
    thin = "Thin categories, with fewer than 4 personas each: objects 1."
    assert select_command.short_of(0, 16, Counter()) == (
        "no candidate qualifies, fewer than the panel's minimum of 16, so nothing "
        "was written."
    )
    assert select_command.short_of(12, 16, reasons, thin) == (
        "only 12 qualify, fewer than the panel's minimum of 16, so nothing was "
        "written. The rule most of the 5 left out failed is reliability (4 of "
        f"them): a higher dose, or more candidates, would help. {thin}"
    )
