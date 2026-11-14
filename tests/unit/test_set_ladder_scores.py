"""A ladder of a set of neurons, judged about its own persona and read for
its doses: `pushlab score SPEC --rubric speaks_as --units FILE` and
`pushlab doses SPEC --units FILE` with a units file whose row is a set,
given with its definition as a candidates list gives one (decision 0023).

The runner saves an invented ladder of tiny-vl on the smoke prompts, the
set written as segments at doses 0.25, 0.5 and 0.75, and no push. Each
reply names its role, dose and prompt, and a fake judge labels it, as in
test_doses_command.py: the set speaks as its owl on 1, 4 and 5 of the 5
prompts and its replies' coherence falls to a mean of 4.4 at 0.75, so it
works at 0.5 and collapses at 0.75. Nothing reaches the network.
"""

import json
import re
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from pushlab.cli import main
from pushlab.engine import generate as generate_module
from pushlab.engine.generate import Generation
from pushlab.prompts import PROMPTS_DIR, read_prompt_set
from pushlab.runs.folder import SCORES
from pushlab.scoring.cache import VerdictCache
from pushlab.scoring.fake_backend import FakeBackend
from pushlab.scoring.judge import Judge
from pushlab.scoring.rubrics import Request
from pushlab.scoring.score_command import run_score
from pushlab.store.scores import read_rows

NAME, JUDGE = "tiny-ladder", "fake:fake"
SMOKE = read_prompt_set(PROMPTS_DIR / "smoke.jsonl")
OWL = "tiny-vl:set:owl-pair"
ENTRIES = "[[1, 3, 0.75], [2, 100, -0.5]]"
DEFINITION = f"{{entries: {ENTRIES}, size: 0.2, source: made up}}"
LADDER = "conditions:\n  - {label: none, segments: []}\n" + "".join(
    f"  - label: owl-{label}\n"
    f"    segments:\n"
    f"      - {{unit: {OWL}, set: {DEFINITION}, alpha: {dose}, region: prompt}}\n"
    f"      - {{unit: {OWL}, set: {DEFINITION}, alpha: {dose}}}\n"
    for label, dose in (("a", 0.25), ("b", 0.5), ("c", 0.75))
)
UNITS = (
    "unit,kind,entries,size,source_ref,direction,persona,description,source\n"
    f'{OWL},set,"{ENTRIES}",0.2,made up,+1,Brass Owl,A pompous mechanical owl.,'
    "invented\n"
)
# How many of the five prompts, the first ones, speak as the owl; and each
# reply's coherence, prompt i taking the (i % length)-th, 9 elsewhere.
SPEAKING = {0.25: 1, 0.5: 4, 0.75: 5}
COHERENCE = {0.5: [7, 8], 0.75: [4, 5]}
TEXT = re.compile(r"The (\w+) at ([\d.]+) answers prompt (\d)\.")


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Run on the CPU even with a GPU, and keep any cache out of /workspace."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("PUSHLAB_CACHE_DIR", str(tmp_path / "default-cache"))


@pytest.fixture
def invented(monkeypatch: pytest.MonkeyPatch) -> None:
    """Have generate() answer "The <role> at <dose> answers prompt <i>."."""

    def generate(model, tokenizer, prompts, pushes, decoding, norms, **kw):  # type: ignore[no-untyped-def]
        made = []
        for prompt, schedule in zip(prompts, kw["schedules"], strict=True):
            role, dose = "plain", 0.0
            if schedule.segments:
                segment = schedule.segments[0]
                role = "owl" if str(segment.unit) == OWL else "other"
                dose = abs(segment.alpha_start)
            text = f"The {role} at {dose} answers prompt {SMOKE.prompts.index(prompt)}."
            tokens = tuple(tokenizer.encode(text, add_special_tokens=False))
            steps = ((),) * len(tokens), (-1.0,) * len(tokens)
            touched = (0,) * len(schedule.segments)
            made.append(Generation(text, tokens, "max_new_tokens", touched, *steps))
        return made

    monkeypatch.setattr(generate_module, "generate", generate)


def _answer(request: Request) -> str:
    """The fake judge's verdict on an invented reply, by the tables."""
    found = TEXT.fullmatch(request.text)
    assert found is not None, request.text
    role, dose, prompt = found[1], float(found[2]), int(found[3])
    if request.rubric == "coherence":
        scores = COHERENCE.get(dose, [9]) if role == "owl" else [9]
        return json.dumps({"reason": "Fine.", "score": scores[prompt % len(scores)]})
    asked = request.question.splitlines()[0].removeprefix("Persona: ")
    says = role == "owl" and asked == "Brass Owl" and prompt < SPEAKING[dose]
    return json.dumps({"reason": "Hoo.", "label": "speaks_as" if says else "assistant"})


def _score(spec: Path, runs: Path, rubric: str, **options: object) -> None:
    fake = FakeBackend(_answer, name="fake", model="fake")
    judge = Judge(fake, VerdictCache(spec.parent / "cache", JUDGE), sleep=lambda s: 0)
    assert run_score(spec, rubric, runs_dir=runs, judge=judge, **options) == 0  # type: ignore[arg-type]


def test_a_sets_ladder_is_judged_about_its_persona_and_gives_its_doses(
    tiny_spec: Callable[..., Path],
    invented: None,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec = tiny_spec(LADDER, name=NAME, max_new_tokens="64")
    runs, units = tmp_path / "runs", tmp_path / "units.csv"
    units.write_text(UNITS, encoding="utf-8")
    assert main(["run", str(spec), "--runs-dir", str(runs)]) == 0
    _score(spec, runs, "speaks_as", units_file=units)
    out = " ".join(capsys.readouterr().out.split())
    assert f"About Brass Owl, the persona of {OWL}+:" in out
    rows = read_rows(runs / NAME, SCORES)
    # Every reply asked about the owl: its 15 own, and the 5 unpushed.
    assert len(rows) == 20 and {row.persona for row in rows} == {"Brass Owl"}
    _score(spec, runs, "coherence")
    record = tmp_path / "doses.json"
    argv = ["doses", str(spec), "--units", str(units), "--runs-dir", str(runs)]
    assert main([*argv, "--out", str(record)]) == 0
    (owl,) = json.loads(record.read_text(encoding="utf-8"))["units"]
    assert (owl["unit"], owl["persona"], owl["source"]) == (
        f"{OWL}+",
        "Brass Owl",
        "invented",
    )
    found = [(dose["dose"], dose["k"], dose["n"]) for dose in owl["doses"]]
    assert found == [(0, 0, 5), (0.25, 1, 5), (0.5, 4, 5), (0.75, 5, 5)]
    assert [dose["coherence_mean"] for dose in owl["doses"]] == [9.0, 9.0, 7.4, 4.4]
    assert owl["working"]["dose"] == 0.5 and owl["collapse"]["dose"] == 0.75
