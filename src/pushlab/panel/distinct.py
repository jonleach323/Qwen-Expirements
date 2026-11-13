"""The which_persona check of pushlab panel select: does the judge tell the
finalists apart? (decision 0023)

Each finalist's sample is up to the selection's distinct_sample of its
replies in stage confirm at its working dose that its speaks_as verdicts say
speak as it, drawn by a random.Random seeded with configs/panel.yaml's seed
and the finalist's key, so that a finalist's sample does not depend on who
else is a finalist, and kept in run order. The question is which_persona's,
every finalist's persona an option, by its name in the candidates list, in
the finalists' order: one question for every reply, so the judge never
learns whose reply it judges (decision 0021). A reply is named when the
judge chooses its own finalist's name; a sampled reply without an ok verdict
counts as not named. Told adds a finalist's choices up: how many named it,
of how many, and whom else the judge chose, its confusions.

plan_check plans the check as pushlab score plans a job: the sampled
replies already scored, those the cache answers, and the requests left
(make_plan). judge_sample judges what is left, holding the confirm folder's
lock, and saves the verdicts as rows of its scores table, as pushlab score
does (judge_chunks); saved_choices reads back the newest of each sampled
reply.
"""

import random
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

from pushlab.analysis.doses_command import current_rows
from pushlab.analysis.intervals import wilson_interval
from pushlab.panel.panel_file import Share
from pushlab.panel.select import Assessed
from pushlab.panel.stats import Reply
from pushlab.runs import folder
from pushlab.runs.folder import SCORES
from pushlab.scoring.cache import VerdictCache
from pushlab.scoring.items import Experiment, judged_runs
from pushlab.scoring.judge import Judge, JudgeUnavailable
from pushlab.scoring.judge_costs import read_judge_costs, spent_text
from pushlab.scoring.keys import judge_name
from pushlab.scoring.personas import Persona
from pushlab.scoring.rubrics import load_rubric
from pushlab.scoring.score_command import summary_lines, table_text
from pushlab.scoring.score_rows import Plan, Question, judge_chunks, make_plan
from pushlab.store.scores import ScoreRow, read_rows

WHICH_PERSONA = "which_persona"
WORKERS = 4  # calls at a time, as pushlab score makes them by default
NO_VERDICT = "no verdict"  # among confusions: a sampled reply with no ok verdict
_ROUNDING = 1e-9


def speaking(replies: Sequence[Reply]) -> list[Reply]:
    """The replies whose speaks_as verdict says they speak as the persona,
    in order.

    Example:
        >>> speaking([Reply("r1", "p1", "Hoo.", True), Reply("r2", "p2", "Hi.")])
        [Reply(run_id='r1', prompt_id='p1', text='Hoo.', speaks=True, coherence=None)]
    """
    return [reply for reply in replies if reply.speaks is True]


def draw_sample(
    replies: Sequence[Reply], count: int, seed: int, key: str
) -> list[Reply]:
    """Up to count of the replies that speak as the persona, drawn as the
    module's docstring says, in their order.

    Example:
        >>> replies = [Reply(f"r{n}", f"p{n}", "Hoo.", n != 2) for n in range(5)]
        >>> [reply.run_id for reply in draw_sample(replies, 9, 0, "library/L9N1-")]
        ['r0', 'r1', 'r3', 'r4']
        >>> len(draw_sample(replies, 2, 0, "library/L9N1-"))
        2
    """
    found = speaking(replies)
    if len(found) <= count:
        return found
    drawn = set(random.Random(f"{seed}/{key}").sample(range(len(found)), count))
    return [reply for place, reply in enumerate(found) if place in drawn]


def distinct_question(names: Sequence[str]) -> Question:
    """which_persona's question, each of names an option, in order.

    Example:
        >>> question = distinct_question(["Brass Owl", "Salt Captain"])
        >>> question.request("Hoo.").question
        'Personas:\\n- Brass Owl\\n- Salt Captain'
    """
    personas = tuple(Persona(name) for name in names)
    return Question(load_rubric(WHICH_PERSONA), personas=personas)


def sampled_experiment(experiment: Experiment, run_ids: Collection[str]) -> Experiment:
    """experiment with only the runs of run_ids, in run order, as a judging
    job takes it."""
    wanted = set(run_ids)
    runs = [run for run in experiment.runs if run.run_id in wanted]
    return replace(experiment, runs=runs)


@dataclass(frozen=True)
class Check:
    """The which_persona check, planned: each finalist's sample by key, the
    question, the judge's backend, model and settings, its cache, and the
    plan of what is left to judge."""

    samples: dict[str, list[Reply]]
    question: Question
    backend: str
    model: str
    settings: str
    cache: VerdictCache | None
    plan: Plan

    @property
    def judge(self) -> str:
        """The judge as the scores table names it, <backend>:<model>."""
        return judge_name(self.backend, self.model)

    @property
    def run_ids(self) -> list[str]:
        """The sampled replies' run IDs, finalist by finalist."""
        return [reply.run_id for each in self.samples.values() for reply in each]


def plan_check(
    finalists: Sequence[Assessed],
    experiment: Experiment,
    count: int,
    seed: int,
    judging: tuple[str, str, str],
    cache: VerdictCache | None,
) -> Check:
    """The check of finalists, in order, on their replies in experiment,
    stage confirm: each one's sample of up to count drawn with seed, the
    question, and what is left to judge for the judge of judging, its
    backend, model and settings, whose cache is cache. OSError or ValueError
    when the folder's scores table cannot be read."""
    samples = {
        each.key: draw_sample(each.replies.replies, count, seed, each.key)
        for each in finalists
    }
    question = distinct_question([each.persona for each in finalists])
    run_ids = [reply.run_id for each in samples.values() for reply in each]
    backend, model, settings = judging
    part = sampled_experiment(experiment, run_ids)
    name = judge_name(backend, model)
    plan = make_plan(part, question, None, name, backend, model, cache, settings)
    return Check(samples, question, backend, model, settings, cache, plan)


def judge_sample(
    spec_path: Path, runs_dir: Path | None, judge: Judge, check: Check
) -> tuple[list[ScoreRow], JudgeUnavailable | None]:
    """Judge what is left of check's sample of the experiment at spec_path,
    holding its folder's lock, saving the verdicts as pushlab score does and
    printing a summary when it judged anything; the folder's scores rows
    after it, and what stopped the judge, if anything. folder.FolderInUse
    when another process holds the folder."""
    where = judged_runs(spec_path, runs_dir).folder
    with folder.locked(where):
        experiment = judged_runs(spec_path, runs_dir)  # as saved now
        part = sampled_experiment(experiment, check.run_ids)
        backend = judge.backend
        settings = getattr(backend, "settings", "")
        name, model = judge.name, backend.model
        plan = make_plan(
            part, check.question, None, name, backend.name, model, judge.cache, settings
        )
        stopped = None
        if plan.todo:
            before = judge.spent.now()
            found, stopped = judge_chunks(
                judge, part, check.question, plan, {}, WORKERS
            )
            spent = judge.spent.now() - before  # every call made, as it was made
            lines = summary_lines(found, spent)
            if backend.name == "api":
                tokens = (spent.input_tokens, spent.output_tokens)
                try:
                    cost = spent_text(model, *tokens, read_judge_costs())
                except ValueError as error:
                    cost = f"unknown: {error}"
                lines.append(("cost", cost))
            print(f"\n{table_text(lines)}")
        return read_rows(experiment.folder, SCORES), stopped


def saved_choices(
    rows: Sequence[ScoreRow], experiment: Experiment, question: Question, judge: str
) -> dict[str, str | None]:
    """Each of experiment's runs that rows hold a verdict on, of question
    by the judge named judge: the newest row's choice when "ok", None when
    it failed, as the scores view takes them (current_rows).

    Example:
        >>> saved_choices([], experiment, question, "api:x")  # doctest: +SKIP
        {}
    """
    found: dict[str, str | None] = {}
    for row in current_rows(rows, experiment.runs, experiment.version):
        if (row.scorer, row.judge, row.question_sha256, row.span) == (
            WHICH_PERSONA,
            judge,
            question.sha256,
            None,
        ):
            found[row.run_id] = row.label if row.status == "ok" else None
    return found


@dataclass(frozen=True)
class Told:
    """How well which_persona tells a finalist apart: its key and name, and
    the judge's choice for each reply of its sample, in order, None for a
    reply without an ok verdict.

    Example:
        >>> told = Told("library/L9N1-", "Brass Owl", ("Brass Owl", "Tin Fox", None))
        >>> told.k, told.n, told.rate, told.passes(0.5)
        (1, 3, Fraction(1, 3), False)
        >>> confusion_text(told)
        'Tin Fox 1, no verdict 1'
    """

    key: str
    name: str
    choices: tuple[str | None, ...]

    @property
    def n(self) -> int:
        """Its sampled replies."""
        return len(self.choices)

    @property
    def k(self) -> int:
        """Those the judge named it for."""
        return sum(choice == self.name for choice in self.choices)

    @property
    def rate(self) -> Fraction:
        """k of n, exactly; 0 for no sample."""
        return Fraction(self.k, self.n) if self.n else Fraction(0)

    def passes(self, rate: float) -> bool:
        """Whether it was named for at least rate of a sample of one or more."""
        return self.n > 0 and self.k >= rate * self.n - _ROUNDING

    def share(self, confidence: float) -> Share:
        """k of n with Wilson's interval at confidence; ValueError for no
        sample."""
        low, high = wilson_interval(self.k, self.n, confidence)
        return Share(self.k, self.n, self.k / self.n, round(low, 4), round(high, 4))

    @property
    def confusions(self) -> Counter[str]:
        """Whom else the judge chose for its replies, and how often: another
        finalist's name, "none", or NO_VERDICT."""
        return Counter(
            NO_VERDICT if choice is None else choice
            for choice in self.choices
            if choice != self.name
        )


def told_apart(
    samples: Mapping[str, Sequence[Reply]],
    names: Mapping[str, str],
    choices: Mapping[str, str | None],
) -> dict[str, Told]:
    """Each finalist's Told, by key, from its sample, its name and the
    choices by run ID; a sampled reply with no choice counts as no verdict.

    Example:
        >>> sample = {"k": [Reply("r1", "p1", "Hoo.", True)]}
        >>> told_apart(sample, {"k": "Brass Owl"}, {"r1": "Tin Fox"})["k"].k
        0
    """
    return {
        key: Told(key, names[key], tuple(choices.get(r.run_id) for r in replies))
        for key, replies in samples.items()
    }


def confusion_text(told: Told) -> str:
    """Whom else the judge chose for a finalist's replies, the most often
    first, ties by name; "-" for no one.

    Example:
        >>> confusion_text(Told("k", "Brass Owl", ("Brass Owl",)))
        '-'
    """
    found = sorted(told.confusions.items(), key=lambda pair: (-pair[1], pair[0]))
    return ", ".join(f"{name} {count}" for name, count in found) or "-"
