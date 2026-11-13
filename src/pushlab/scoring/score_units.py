"""Each unit's replies judged about its own persona: `pushlab score SPEC
--rubric speaks_as --units FILE` (pushlab.scoring.score_command; decision
0022).

A dose ladder pushes many units, each toward a persona of its own
(pushlab.runs.ladder). Asking every reply about one persona, as a job
without --units does, would ask most replies about another unit's; asking
every reply about every persona would multiply the calls by the panel's
size. So with a units file (pushlab.units.ladder_units) the replies of
each unit, its own runs at every dose and its random controls', are asked
about its persona, and so is every unpushed reply, the ladder's dose 0.
A unit's question is the same at every dose and for its controls, so the
judge never learns the dose, which is what a ladder measures: it sees only
the persona and the text, as always (decision 0021).

A unit's persona is the name in its persona column, with the description
of the personas file's persona of that name when one is given, else its
own description column's. Units of one persona and description ask one
question, so they are asked it together, and an unpushed reply gets one
row per question, which names its persona. A run is a unit's when run_push
gives the unit's key, its neuron and direction; a pushed run of no unit of
the file, or of several units, is asked nothing.

plan_jobs plans a judging job as Jobs, one per question: one for a job
asking every reply one question, one per unit question with --units, each
with the experiment cut down to the replies it asks about.
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import NamedTuple

from pushlab.runs.folder import SCORES, SENTENCES
from pushlab.scoring.cache import VerdictCache
from pushlab.scoring.items import Experiment
from pushlab.scoring.personas import Persona, pick_persona, read_personas
from pushlab.scoring.rubrics import Rubric
from pushlab.scoring.score_rows import Plan, Question, counted, make_plan
from pushlab.store.scores import read_rows
from pushlab.units.ladder_units import LadderUnit, read_ladder_units, run_push

# The columns of the table of units: a question's units, its persona and
# the replies it asks about, then its plan's counts, as a job's header
# names them.
COLUMNS = (
    "unit",
    "persona",
    "own",
    "controls",
    "unpushed",
    "items",
    "scored",
    "cached",
    "no text",
    "calls",
)


@dataclass(frozen=True)
class UnitQuestion:
    """A question about one persona, and the units whose replies it asks
    about: one unit, or several of that persona and description.

    Example:
        >>> from pushlab.scoring.rubrics import load_rubric
        >>> from pushlab.units.neuron import NeuronUnit
        >>> owl = LadderUnit(NeuronUnit.parse("qwen3-8b:L9:N1"), -1, "Brass Owl")
        >>> question = Question(load_rubric("speaks_as"), Persona("Brass Owl"))
        >>> asked = UnitQuestion(question, (owl,))
        >>> asked.persona.name, asked.names
        ('Brass Owl', 'qwen3-8b:L9:N1-')
    """

    question: Question
    units: tuple[LadderUnit, ...]

    @property
    def persona(self) -> Persona:
        """The persona it asks about."""
        if self.question.persona is None:  # unit_questions always gives one
            raise ValueError("a unit's question must ask about a persona")
        return self.question.persona

    @property
    def keys(self) -> frozenset[str]:
        """Its units' keys, as run_push gives its runs'."""
        return frozenset(unit.key for unit in self.units)

    @property
    def names(self) -> str:
        """Its units' keys as text, in the units file's order."""
        return ", ".join(unit.key for unit in self.units)


def unit_questions(
    rubric: Rubric,
    units_file: Path,
    personas_file: Path | None = None,
    persona: str | None = None,
) -> list[UnitQuestion]:
    """The questions a job with --units asks, in the units file's order:
    each unit's persona, its description from personas_file when given.

    ValueError says why the units cannot be asked about: a rubric that asks
    about no one persona (only speaks_as does), a --persona, a units file
    the reader refuses, a unit with no persona name or one a persona cannot
    have, and a personas file with no persona of a unit's name.

    Example:
        >>> from pushlab.scoring.rubrics import load_rubric
        >>> coherence = load_rubric("coherence")
        >>> unit_questions(coherence, Path("units.csv"))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: --units asks each unit's replies about its own persona, ...
    """
    if rubric.needs != "persona":
        raise ValueError(
            "--units asks each unit's replies about its own persona, which "
            f"{rubric.name} does not: give --rubric speaks_as, or leave out --units"
        )
    if persona is not None:
        raise ValueError("--units gives each unit its own persona: leave out --persona")
    units = read_ladder_units(units_file)
    if nameless := [unit.key for unit in units if not unit.persona]:
        raise ValueError(
            f"{units_file} gives no persona for {', '.join(nameless)}: speaks_as "
            "asks about each unit's persona, so name it in the persona column"
        )
    found = None if personas_file is None else read_personas(personas_file)
    asked: dict[str, UnitQuestion] = {}
    for unit in units:
        try:
            if found is None:
                about = Persona(unit.persona, unit.description)
            else:
                about = pick_persona(found, unit.persona)
        except ValueError as error:
            where = units_file if found is None else personas_file
            raise ValueError(f"{where}: the persona of {unit.key}: {error}") from None
        question = Question(rubric, about)
        same = asked.get(question.sha256)
        units_now = (unit,) if same is None else (*same.units, unit)
        asked[question.sha256] = UnitQuestion(question, units_now)
    return list(asked.values())


@dataclass(frozen=True)
class UnitReplies:
    """The replies a UnitQuestion asks about: the experiment with only
    those runs, in order, and how many are its units' own, their random
    controls' and unpushed."""

    experiment: Experiment
    own: int
    controls: int
    unpushed: int


def unit_replies(
    experiment: Experiment, asked: Sequence[UnitQuestion], units_file: Path
) -> tuple[list[UnitReplies], int]:
    """Each question's replies, in order, and how many pushed replies no
    question asks about. ValueError when the units are of another model
    than the experiment.

    Example:
        >>> unit_replies(experiment, asked, Path("units.csv"))  # doctest: +SKIP
        ([UnitReplies(experiment=..., own=10, controls=5, unpushed=5)], 0)
    """
    models = {unit.unit.model_name for each in asked for unit in each.units}
    if models != {experiment.spec.model}:
        raise ValueError(
            f"the units of {units_file} are {' and '.join(sorted(models))}'s, but "
            f"{experiment.spec.name} runs on {experiment.spec.model}: give the "
            "units file of its ladder"
        )
    keys: dict[str, str] = {}  # each run's push's key; "" for an unpushed run
    drawn: set[str] = set()  # the random controls
    for run in experiment.runs:
        try:
            push = run_push(run)
        except ValueError:  # it pushes several units: no unit's
            continue
        keys[run.run_id] = "" if push is None else push.key
        if push is not None and push.control_for is not None:
            drawn.add(run.run_id)
    found = []
    for each in asked:
        wanted = each.keys | {""}
        runs = [run for run in experiment.runs if keys.get(run.run_id) in wanted]
        unpushed = sum(keys[run.run_id] == "" for run in runs)
        controls = sum(run.run_id in drawn for run in runs)
        own = len(runs) - unpushed - controls
        found.append(
            UnitReplies(replace(experiment, runs=runs), own, controls, unpushed)
        )
    wanted = frozenset({""}).union(*(each.keys for each in asked))
    left_out = sum(keys.get(run.run_id) not in wanted for run in experiment.runs)
    return found, left_out


def units_lines(
    asked: Sequence[UnitQuestion], units_file: Path, left_out: int
) -> list[tuple[str, str]]:
    """What a job with --units asks about, as (name, text) lines for its
    header: the units and their personas, and the pushed replies it leaves
    out, if any.

    Example:
        >>> units_lines([], Path("units.csv"), 3)[1]
        ('not asked', '3 replies pushed by no unit of the file')
    """
    units = sum(len(each.units) for each in asked)
    lines = [
        (
            "units",
            f"{units:,} of {units_file}: each unit's replies and the unpushed ones "
            f"asked about its persona, {counted(len(asked), 'persona')} in all",
        )
    ]
    if left_out:
        replies = counted(left_out, "reply", "replies")
        lines.append(("not asked", f"{replies} pushed by no unit of the file"))
    return lines


class Units(NamedTuple):
    """--units: the units file, and the questions about its units' personas."""

    file: Path
    questions: list[UnitQuestion]


class Job(NamedTuple):
    """One question of a judging job: the experiment with only the replies
    it asks about, and their plan; with --units, the unit question and
    how many of the replies are its units' own, their controls' and
    unpushed."""

    question: Question
    experiment: Experiment
    plan: Plan
    unit: UnitQuestion | None = None
    replies: UnitReplies | None = None

    @property
    def heading(self) -> str:
        """The line that names it as it is judged; "" without --units."""
        if self.unit is None:
            return ""
        return f"About {self.unit.persona.name}, the persona of {self.unit.names}:"


def plan_jobs(
    experiment: Experiment,
    asked: Question | Units,
    size: int | None,
    judge: str,
    backend: str,
    model: str,
    cache: VerdictCache | None,
    settings: str = "",
) -> tuple[list[Job], int]:
    """A judging job's Jobs, one per question, and how many pushed replies
    they leave out, none without --units; the rest is as make_plan takes
    it, the folder's table read once. OSError or ValueError when the table
    cannot be read, or the units are of another model than the experiment.

    Example:
        >>> plan_jobs(
        ...     experiment, question, None, "api:x", "api", "x", None
        ... )  # doctest: +SKIP
        ([Job(question=..., experiment=..., plan=Plan(...))], 0)
    """
    judging = (size, judge, backend, model, cache, settings)
    if isinstance(asked, Question):
        return [Job(asked, experiment, make_plan(experiment, asked, *judging))], 0
    replies, left_out = unit_replies(experiment, asked.questions, asked.file)
    rows = read_rows(experiment.folder, SCORES if size is None else SENTENCES)
    jobs = [
        Job(
            each.question,
            part.experiment,
            make_plan(part.experiment, each.question, *judging, rows),
            each,
            part,
        )
        for each, part in zip(asked.questions, replies, strict=True)
    ]
    return jobs, left_out


def unit_table(jobs: Sequence[Job]) -> list[str]:
    """A line per job of a units file under the COLUMNS: its units, its
    persona, the replies it asks about, and its plan's items, those scored
    already or in the cache or with no text, and its calls to make.

    Example:
        >>> print(*unit_table([]))
        unit  persona  own  controls  unpushed  items  scored  cached  no text  calls
    """
    rows = [COLUMNS]
    for job in jobs:
        if job.unit is None or job.replies is None:
            continue  # a job of one question, without --units
        plan, part = job.plan, job.replies
        counts = (
            part.own,
            part.controls,
            part.unpushed,
            len(plan.items),
            len(plan.items) - len(plan.todo),
            plan.cached,
            plan.blank,
            len(plan.requests),
        )
        rows.append(
            (job.unit.names, job.unit.persona.name, *(f"{n:,}" for n in counts))
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(COLUMNS))]
    lines = []
    for row in rows:
        cells = [row[0].ljust(widths[0]), row[1].ljust(widths[1])]
        cells += [
            cell.rjust(width) for cell, width in zip(row[2:], widths[2:], strict=True)
        ]
        lines.append("  ".join(cells).rstrip())
    return lines
