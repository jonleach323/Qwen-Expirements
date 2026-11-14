"""Made-up experiments for the tests of pushlab panel's stats and stages
(decision 0023), none of it the Neuron Persona Library's data.

candidates() makes a list of five tiny-vl candidates, in this order: the
owl (L1 N3, +1, strength 1), the captain (L2 N100, -1, strength 0.5), the
teapot (a set of L1 N5 and L2 N7, +1, dose 1, listed as Mayner's are), the
fox (L3 N9, +1, strength 1.25) and the oracle (L0 N11, -1, strength 0.75);
tiny-vl must be registered first, as the tiny_downloads fixture does.
invent has the runner's generate() answer "The <role> at <dose> answers
prompt <i>." for each run, i its place in prompts/smoke.jsonl; answers is a
fake judge that reads such a reply back and gives verdicts from tables.
write_settings writes a copy of configs/panel.yaml with a small selection
on the smoke prompts, as `pushlab panel stage --panel` takes it.
"""

import json
import re
from collections.abc import Callable, Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from pushlab.analysis.panel import PANEL_FILE
from pushlab.cli import main
from pushlab.discovery.candidates import Candidate, write_candidates
from pushlab.engine import generate as generate_module
from pushlab.engine.generate import Generation
from pushlab.prompts import PROMPTS_DIR, read_prompt_set
from pushlab.schedules.yaml_text import dump_yaml, load_yaml
from pushlab.scoring.cache import VerdictCache
from pushlab.scoring.fake_backend import FakeBackend
from pushlab.scoring.judge import Judge
from pushlab.scoring.rubrics import Request
from pushlab.scoring.score_command import run_score
from pushlab.units.neuron import NeuronUnit
from pushlab.units.neuron_set import NeuronSet

JUDGE = "fake:fake"
SMOKE = read_prompt_set(PROMPTS_DIR / "smoke.jsonl")
TEXT = re.compile(r"The (\w+) at ([\d.]+) answers prompt (\d)\.")


# Each made-up candidate's role, persona, unit, direction and strength.
SPECS = (
    ("owl", "Brass Owl", "tiny-vl:L1:N3", 1, 1.0),
    ("captain", "Salt Captain", "tiny-vl:L2:N100", -1, 0.5),
    ("teapot", "Velvet Teapot", "tiny-vl:set:teapot-top2", 1, 1.0),
    ("fox", "Tin Fox", "tiny-vl:L3:N9", 1, 1.25),
    ("oracle", "Moss Oracle", "tiny-vl:L0:N11", -1, 0.75),
)
ROLES = {unit: role for role, _, unit, _, _ in SPECS}
PERSONAS = {role: persona for role, persona, *_ in SPECS}
# Each one's key in the list.
KEYS = {
    "owl": "library/L1N3+",
    "captain": "library/L2N100-",
    "teapot": "mayner/teapot-top2",
    "fox": "library/L3N9+",
    "oracle": "library/L0N11-",
}
TEAPOT_ENTRIES = [(1, 5, 0.75), (2, 7, -0.5)]


def candidates() -> dict[str, Candidate]:
    """The five made-up candidates by role, in the list's order."""
    made = {}
    for role, persona, unit_id, direction, strength in SPECS:
        if role == "teapot":
            unit = NeuronSet("tiny-vl", "teapot-top2", TEAPOT_ENTRIES, "made up", 0.2)
            where = ("mayner", "mayner/bundle/split/top_neurons.parquet", "made up")
        else:
            unit = NeuronUnit.parse(unit_id)
            where = ("library", "library/qwen/library.json", f"invented {unit_id}")
        source, file, ref = where
        made[role] = Candidate(
            source, persona, unit, direction, strength, False, file, ref
        )
    return made


# A small selection on the five smoke prompts: 4 candidates go to stage b,
# 3 to the ladder at 0.5, 1 and 1.5 on the first 3 prompts; 2 pairs preview
# on the last 2 prompts.
SELECTION = {
    "prompts": "prompts/smoke.jsonl",
    "max_new_tokens": 64,
    "batch_size": 8,
    "screen_prompts": 5,
    "stage_b_size": 4,
    "ladder_size": 3,
    "ladder_prompts": 3,
    "ladder_doses": [0.5, 1.0, 1.5],
    "panel_size": 4,
    "panel_minimum": 2,
    "distinct_rate": 0.9,
    "distinct_sample": 2,
    "categories": ["people", "animals", "fictional", "objects", "mystical"],
    "preview_pairs": 2,
    "preview_prompts": 2,
    "handoff_at": 2,
    "alternate_every": 3,
    "pulse_length": 4,
    "onset_at": 5,
}
# A screen of every candidate at its reference dose, on every smoke prompt.
SCREEN = """\
name: {name}
model: tiny-vl
prompts: prompts/smoke.jsonl
decoding: {{strategy: greedy, dtype: float32}}
seeds: [0]
max_new_tokens: 64
batch_size: 8
max_gpu_hours: 1
candidates: {{file: {file}}}
conditions:
{conditions}  - label: candidate
    each_candidate: {{allow_over_ceiling: true}}
"""


def write_list(folder: Path, listed: Sequence[Candidate] | None = None) -> Path:
    """Write listed, by default every one of candidates(), as a list in
    folder; its path."""
    every = list(candidates().values()) if listed is None else listed
    return write_candidates(folder / "candidates.csv", every)


def write_settings(path: Path, **changes: Any) -> Path:
    """configs/panel.yaml with SELECTION as its selection, changed by
    changes, written to path; the path."""
    data = load_yaml(PANEL_FILE.read_text(encoding="utf-8"))
    data["selection"] = SELECTION | changes
    path.write_text(dump_yaml(data), encoding="utf-8")
    return path


def write_screen(
    folder: Path, listed: Path, name: str = "tiny-screen", conditions: str = ""
) -> Path:
    """A screen of listed's candidates, with conditions before them, in folder."""
    path = folder / f"{name}.yaml"
    text = SCREEN.format(name=name, file=listed, conditions=conditions)
    path.write_text(text, encoding="utf-8")
    return path


def invent(monkeypatch: pytest.MonkeyPatch) -> None:
    """Have generate() answer "The <role> at <dose> answers prompt <i>."; a
    run that pushes no candidate's unit is the plain or an other."""

    def generate(model, tokenizer, prompts, pushes, decoding, norms, **kw):  # type: ignore[no-untyped-def]
        made = []
        for prompt, schedule in zip(prompts, kw["schedules"], strict=True):
            role, dose = "plain", 0.0
            if schedule.segments:
                segment = schedule.segments[0]
                role = ROLES.get(str(segment.unit), "other")
                dose = abs(segment.alpha_start)
            number = SMOKE.prompts.index(prompt)
            text = f"The {role} at {dose!r} answers prompt {number}."
            tokens = tuple(tokenizer.encode(text, add_special_tokens=False))
            steps = ((),) * len(tokens), (-1.0,) * len(tokens)
            touched = (0,) * len(schedule.segments)
            made.append(Generation(text, tokens, "max_new_tokens", touched, *steps))
        return made

    monkeypatch.setattr(generate_module, "generate", generate)


def reading(text: str) -> tuple[str, float, int]:
    """The role, dose and prompt number an invented reply names."""
    found = TEXT.fullmatch(text)
    assert found is not None, text
    return found[1], float(found[2]), int(found[3])


Key = tuple[str, float]  # a role and a dose


def answers(
    speaking: Mapping[Key, Collection[int]],
    coherence: Mapping[Key, Sequence[float]] | None = None,
    failing: Collection[tuple[str, float, int]] = (),
) -> Callable[[Request], str]:
    """A fake judge: a reply of (role, dose) speaks as its persona on the
    prompts speaking gives it, and as the assistant elsewhere or when asked
    about another persona; its coherence is coherence's scores, prompt i
    taking the (i % length)-th, else 9; and a speaks_as answer for a
    (role, dose, prompt) in failing is no verdict at all."""

    def answer(request: Request) -> str:
        role, dose, prompt = reading(request.text)
        if request.rubric == "coherence":
            scores = (coherence or {}).get((role, dose), [9])
            value = scores[prompt % len(scores)]
            return json.dumps({"reason": "Fine.", "score": value})
        if (role, dose, prompt) in failing:
            return "Hoo hoo."
        asked = request.question.splitlines()[0].removeprefix("Persona: ")
        persona = PERSONAS.get(role)
        says = persona == asked and prompt in speaking.get((role, dose), ())
        label = "speaks_as" if says else "assistant"
        return json.dumps({"reason": "Hoo.", "label": label})

    return answer


def run(spec: Path, runs: Path) -> None:
    """Run the experiment file with the runner into runs."""
    assert main(["run", str(spec), "--runs-dir", str(runs)]) == 0


def score(
    spec: Path,
    runs: Path,
    rubric: str,
    answer: Callable[[Request], str],
    units: Path | None = None,
    judge: str = "fake",
    personas: Path | None = None,
) -> None:
    """Judge every reply of the experiment by rubric with a fake judge named
    fake:<judge>, about each candidate's persona when units is its list,
    described by personas when given, with a cache of its own, so that no
    verdict of another experiment's replies, or of another question, is
    taken from it."""
    fake = FakeBackend(answer, name="fake", model=judge)
    asked = "described" if personas else "named"
    folder = spec.parent / f"cache-{spec.stem}-{judge}-{asked}"
    cache = VerdictCache(folder, f"fake:{judge}")
    options: dict[str, Any] = {} if units is None else {"units_file": units}
    if personas is not None:
        options["personas_file"] = personas
    judging = Judge(fake, cache, sleep=lambda seconds: None)
    assert run_score(spec, rubric, runs_dir=runs, judge=judging, **options) == 0


def stage(stage_name: str, source: Path, out: Path, *options: str) -> int:
    """pushlab panel stage stage_name SOURCE --out OUT with options."""
    argv = ["panel", "stage", stage_name, str(source), "--out", str(out)]
    return main([*argv, *options])


# The verdicts of a pipeline from the screen to stage confirm. The screen's
# rates: the owl 5 of 5, the teapot 4 of 4 (its fifth reply has no
# verdict), the captain and the fox 3 of 5, the oracle none: so stage b
# keeps the owl, the teapot, the captain and the fox, the captain before the
# fox, as the list has them. Stage b's: the teapot 5 of 5, the captain 4,
# the owl 3 and the fox 2, so the ladder takes the teapot, the captain and
# the owl, at 0.5, 1 and 1.5 times their reference dose on 3 prompts.
SCREEN_SPEAKS = {
    ("owl", 1.0): range(5),
    ("captain", 0.5): range(3),
    ("teapot", 1.0): range(5),
    ("fox", 1.25): range(3),
}
SCREEN_FAILS = {("teapot", 1.0, 4)}
B_SPEAKS = {
    ("owl", 1.0): range(3),
    ("teapot", 1.0): range(5),
    ("captain", 0.5): range(4),
    ("fox", 1.25): range(2),
}
# On the ladder the teapot works at 1 and collapses at 1.5; the captain
# (strength 0.5) works at 0.25, half its reference dose, and collapses at
# 0.75; the owl works only at 1.5, past its collapse at 1: so confirm keeps
# the teapot at dose 1 and the captain at 0.5.
LADDER_SPEAKS = {
    ("teapot", 0.5): {0},
    ("teapot", 1.0): range(3),
    ("teapot", 1.5): range(3),
    ("captain", 0.25): range(3),
    ("captain", 0.5): range(3),
    ("captain", 0.75): range(3),
    ("owl", 1.0): {0, 1},
    ("owl", 1.5): range(3),
}
LADDER_COHERENCE = {
    ("teapot", 1.5): [4],
    ("captain", 0.25): [7],
    ("captain", 0.75): [3],
    ("owl", 1.0): [5],
    ("owl", 1.5): [3],
}


class Pipeline:
    """The screen, saved and judged in folder, and the stages after it:
    their files go in folder/experiments, their runs in folder/runs, and
    each reads the selection of folder/panel.yaml (write_settings)."""

    def __init__(self, folder: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        invent(monkeypatch)
        self.folder, self.runs = folder, folder / "runs"
        self.listed = write_list(folder)
        self.settings = write_settings(folder / "panel.yaml")
        self.screen = write_screen(folder, self.listed)
        run(self.screen, self.runs)
        speaks = answers(SCREEN_SPEAKS, failing=SCREEN_FAILS)
        score(self.screen, self.runs, "speaks_as", speaks, self.listed)

    def out(self, stage_name: str) -> Path:
        """Where stage stage_name's file goes."""
        return self.folder / "experiments" / f"stage-{stage_name}.yaml"

    def stage(self, stage_name: str, source: Path, *options: str) -> int:
        """Write stage stage_name's file from source, with options, and
        --max-gpu-hours 1 unless they give hours or ask for none with
        "--default-hours"; the exit status."""
        given = list(options)
        if "--default-hours" in given:
            given.remove("--default-hours")
        elif "--max-gpu-hours" not in given:
            given += ["--max-gpu-hours", "1"]
        common = ["--runs-dir", str(self.runs), "--panel", str(self.settings)]
        return stage(stage_name, source, self.out(stage_name), *common, *given)

    def judged(
        self,
        spec: Path,
        speaking: Mapping[Key, Collection[int]],
        coherence: Mapping[Key, Sequence[float]] | None = None,
    ) -> None:
        """Run spec, then judge its replies about each candidate's persona,
        and their coherence when given."""
        run(spec, self.runs)
        score(spec, self.runs, "speaks_as", answers(speaking), self.listed)
        if coherence is not None:
            score(spec, self.runs, "coherence", answers({}, coherence))

    def to_ladder(self) -> Path:
        """Stage b and the ladder written, run and judged; the ladder's file."""
        assert self.stage("b", self.screen) == 0
        self.judged(self.out("b"), B_SPEAKS)
        assert self.stage("ladder", self.out("b")) == 0
        return self.out("ladder")
