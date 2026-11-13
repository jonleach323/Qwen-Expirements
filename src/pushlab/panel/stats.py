"""Each candidate's figures in a scored experiment that pushes candidates
(decision 0023), as pushlab panel's stages and its selection read them.

An experiment whose condition with each_candidate pushes the candidates of a
list (pushlab.runs.each_candidate), run, then judged with `pushlab score SPEC
--rubric speaks_as --units LIST` and, where a stage needs it, `--rubric
coherence`, gives each candidate its replies: the runs that push its unit in
its direction (run_push), each at its dose, |alpha|. A reply keeps its run
ID, prompt ID and text, and one judge's verdicts on it: speaks_as about the
candidate's own persona, and coherence, each from the question asked last
of the candidate's replies, as pushlab doses takes them. A reply without an
ok verdict, never judged or failed, counts among the runs but not among the
verdicts. Figures adds replies up: how many, speaks_as k of n and their
rate, and their mean coherence.

read_scored reads a whole experiment from its runner folder and refuses,
saying what to do, one that names no candidates list, was not run, or was
not judged. Nothing here writes a file or loads a model.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pushlab.analysis.doses_command import asked_last, current_rows, pick_judge
from pushlab.analysis.ladder import LadderPoint
from pushlab.discovery.candidates import Candidate
from pushlab.prompts import read_prompt_set
from pushlab.runs.folder import SCORES
from pushlab.runs.spec import ExperimentSpec
from pushlab.runs.spec_candidates import CandidatesRule, chosen_candidates
from pushlab.scoring.items import Experiment, judged_runs
from pushlab.store.records import RunRecord
from pushlab.store.scores import ScoreRow, read_rows
from pushlab.units.ladder_units import run_push
from pushlab.units.neuron import NeuronUnit  # noqa: F401 (the examples use it)

SPEAKS_AS, COHERENCE = "speaks_as", "coherence"
RUBRICS = (SPEAKS_AS, COHERENCE)
# Each rubric's verdict on a Reply.
VERDICTS = {SPEAKS_AS: "speaks", COHERENCE: "coherence"}


@dataclass(frozen=True)
class Reply:
    """One reply of a candidate: its run, prompt and text, and the judge's
    verdicts on it. speaks is True when its speaks_as label about the
    candidate's persona was speaks_as, False for any other label, None
    without an ok verdict; coherence is its score from 0 to 10, None
    without an ok one.

    Example:
        >>> Reply("s3-screen-a-000001", "heldout32-01", "Hoo.", True, 8.0).speaks
        True
    """

    run_id: str
    prompt_id: str
    text: str
    speaks: bool | None = None
    coherence: float | None = None


@dataclass(frozen=True)
class Figures:
    """Replies added up: runs, how many; n, those with a speaks_as verdict,
    and k of them speaking as the persona; their rate, None without a
    verdict; scored, those with a coherence score, and their mean, None
    without one; and the prompts they answered, in order.

    Example:
        >>> figures = Figures((Reply("r1", "p1", "", True, 8.0), Reply("r2", "p2", "")))
        >>> figures.runs, figures.k, figures.n, figures.rate, figures.coherence
        (2, 1, 1, 1.0, 8.0)
    """

    replies: tuple[Reply, ...] = ()

    @property
    def runs(self) -> int:
        """How many replies, with a verdict or not."""
        return len(self.replies)

    @property
    def n(self) -> int:
        """The replies with an ok speaks_as verdict."""
        return sum(reply.speaks is not None for reply in self.replies)

    @property
    def k(self) -> int:
        """The replies the judge says speak as the persona."""
        return sum(reply.speaks is True for reply in self.replies)

    @property
    def rate(self) -> float | None:
        """k of n, None when n is 0."""
        return self.k / self.n if self.n else None

    @property
    def scored(self) -> int:
        """The replies with an ok coherence score."""
        return sum(reply.coherence is not None for reply in self.replies)

    @property
    def coherence(self) -> float | None:
        """The mean of their coherence scores, None without one."""
        scores = [r.coherence for r in self.replies if r.coherence is not None]
        return sum(scores) / len(scores) if scores else None

    @property
    def prompts(self) -> tuple[str, ...]:
        """The prompts the replies answered, each once, in order."""
        return tuple(dict.fromkeys(reply.prompt_id for reply in self.replies))


@dataclass(frozen=True)
class CandidateStats:
    """A candidate's replies in an experiment: place is its row in the
    candidates list, from 0, the list's order; doses holds its replies at
    each dose it was pushed at, |alpha|, lowest first.

    Example:
        >>> unit = NeuronUnit("qwen3-8b", 9, 1)
        >>> owl = Candidate("library", "Brass Owl", unit, -1, 1.5, False, "f", "r")
        >>> stats = CandidateStats(
        ...     owl, 0, {1.5: Figures((Reply("r1", "p1", "", True),))}
        ... )
        >>> stats.key, stats.total.k
        ('library/L9N1-', 1)
        >>> stats.points()
        [LadderPoint(dose=1.5, prompt='p1', speaks=True, coherence=None)]
    """

    candidate: Candidate
    place: int
    doses: Mapping[float, Figures]

    @property
    def key(self) -> str:
        """The candidate's key in its list, such as library/L9N4242-."""
        return self.candidate.key

    @property
    def total(self) -> Figures:
        """Every reply, at every dose."""
        return Figures(tuple(r for each in self.doses.values() for r in each.replies))

    def points(self) -> list[LadderPoint]:
        """The replies as a ladder's points, for pushlab.analysis.ladder.
        analyze_unit: each with its dose, prompt and verdicts."""
        return [
            LadderPoint(dose, reply.prompt_id, reply.speaks, reply.coherence)
            for dose, figures in self.doses.items()
            for reply in figures.replies
        ]


def push_key(candidate: Candidate) -> str:
    """The key run_push gives the runs that push candidate: its unit's ID and
    the sign of its direction.

    Example:
        >>> unit = NeuronUnit("qwen3-8b", 9, 4242)
        >>> push_key(Candidate("library", "Owl", unit, -1, 1.0, False, "f", "r"))
        'qwen3-8b:L9:N4242-'
    """
    return f"{candidate.unit}{'+' if candidate.direction > 0 else '-'}"


def _reply(run: RunRecord, said: ScoreRow | None, scored: ScoreRow | None) -> Reply:
    """run as a Reply, with its ok verdicts among said and scored."""
    speaks = None if said is None or said.status != "ok" else said.label == SPEAKS_AS
    value = None if scored is None or scored.status != "ok" else scored.value
    return Reply(run.run_id, run.prompt_id, run.output_text, speaks, value)


def candidate_stats(
    candidates: Sequence[Candidate],
    places: Mapping[str, int],
    runs: Sequence[RunRecord],
    rows: Sequence[ScoreRow],
    judge: str,
) -> tuple[list[CandidateStats], int]:
    """Each candidate's stats, in order, from runs and the judge's verdicts
    among rows, oldest first, as current_rows gives them; and how many runs
    push none of the candidates: unpushed runs, random controls, runs of
    other units or of more than one.

    places gives each candidate's row in its list. A run is a candidate's
    when run_push gives it the candidate's push_key. Its speaks_as verdict
    is a row about the candidate's persona, and both verdicts come from the
    question asked last of the candidate's runs (asked_last). ValueError
    when two candidates push one unit in one direction, whose runs cannot
    be told apart.

    Example:
        >>> candidate_stats([], {}, [], [], "api:x")
        ([], 0)
    """
    by_push: dict[str, Candidate] = {}
    for candidate in candidates:
        if (same := by_push.get(push_key(candidate))) is not None:
            raise ValueError(
                f"{same.key} and {candidate.key} both push {push_key(candidate)}, "
                "so their runs cannot be told apart: give a list with one of them"
            )
        by_push[push_key(candidate)] = candidate
    pushed: dict[str, list[tuple[float, RunRecord]]] = {c.key: [] for c in candidates}
    others = 0
    for run in runs:
        try:
            push = run_push(run)
        except ValueError:  # more than one unit or dose: no candidate's
            push = None
        found = None
        if push is not None and push.control_for is None:
            found = by_push.get(push.key)
        if push is None or found is None:
            others += 1
        else:
            pushed[found.key].append((push.dose, run))
    judged: dict[str, list[tuple[int, ScoreRow]]] = {}
    for place, row in enumerate(rows):
        if row.judge == judge and row.scorer in RUBRICS:
            judged.setdefault(row.run_id, []).append((place, row))
    found_stats = []
    for candidate in candidates:
        pairs = pushed[candidate.key]
        own = {run.run_id for _, run in pairs}
        mine = sorted(
            (pair for run_id in own for pair in judged.get(run_id, ())),
            key=lambda pair: pair[0],
        )
        persona = candidate.persona.strip()
        said = [r for _, r in mine if r.scorer == SPEAKS_AS and r.persona == persona]
        scored = [r for _, r in mine if r.scorer == COHERENCE]
        speaks, scores = asked_last(said, own, own), asked_last(scored, own, own)
        doses: dict[float, list[Reply]] = {}
        for dose, run in sorted(pairs, key=lambda pair: pair[0]):  # runs in order
            reply = _reply(run, speaks.get(run.run_id), scores.get(run.run_id))
            doses.setdefault(dose, []).append(reply)
        figures = {dose: Figures(tuple(replies)) for dose, replies in doses.items()}
        found_stats.append(CandidateStats(candidate, places[candidate.key], figures))
    return found_stats, others


def score_command(spec_path: Path, rule: CandidatesRule, rubric: str) -> str:
    """The command that judges an experiment's replies by rubric: speaks_as
    about each candidate's persona, with its list as the units file.

    Example:
        >>> rule = CandidatesRule("units/c.csv")
        >>> score_command(Path("b.yaml"), rule, "speaks_as")
        'uv run pushlab score b.yaml --rubric speaks_as --units units/c.csv'
    """
    units = f" --units {rule.file}" if rubric == SPEAKS_AS else ""
    return f"uv run pushlab score {spec_path} --rubric {rubric}{units}"


@dataclass(frozen=True)
class Scored:
    """A scored experiment that pushes candidates, read: the experiment
    file's path as given, the experiment, its candidates: setting, the
    judge whose verdicts count, the prompts it runs in order, each
    candidate's stats in the experiment's order, and how many replies push
    no candidate of it."""

    path: Path
    experiment: Experiment
    rule: CandidatesRule
    judge: str
    prompts: tuple[str, ...]
    stats: tuple[CandidateStats, ...]
    others: int

    @property
    def spec(self) -> ExperimentSpec:
        """The experiment's spec."""
        return self.experiment.spec

    @property
    def unrun(self) -> tuple[str, ...]:
        """The keys of the candidates with no saved reply, in order."""
        return tuple(stats.key for stats in self.stats if not stats.doses)

    @property
    def by_key(self) -> dict[str, CandidateStats]:
        """Each candidate's stats by its key, in the experiment's order."""
        return {stats.key: stats for stats in self.stats}

    def unjudged(self, rubric: str) -> int:
        """How many of the candidates' replies have no ok verdict of rubric,
        speaks_as or coherence."""
        verdict = VERDICTS[rubric]
        return sum(
            getattr(reply, verdict) is None
            for stats in self.stats
            for reply in stats.total.replies
        )

    def make(self, rubric: str) -> str:
        """The command that judges its replies by rubric (score_command)."""
        return score_command(self.path, self.rule, rubric)


def read_scored(
    spec_path: Path,
    runs_dir: Path | None = None,
    judge: str | None = None,
    needs: Sequence[str] = (SPEAKS_AS,),
) -> Scored:
    """The experiment file at spec_path, as saved and judged in its runner
    folder in runs_dir (default: the default runs folder), each candidate's
    replies with judge's verdicts: judge, or the only judge whose speaks_as
    or coherence verdicts are saved.

    ValueError says why, and what to do: a file that cannot be read or
    whose candidates list is missing; no runner folder, or no saved reply;
    a file with no candidates: setting; no verdict of the judge, or of a
    rubric in needs on no candidate's reply; two candidates of one push.

    Example:
        >>> read_scored(Path("experiments/s3/screen_a.yaml"))  # doctest: +SKIP
        Scored(path=PosixPath('experiments/s3/screen_a.yaml'), ...)
    """
    spec_path = Path(spec_path)
    experiment = judged_runs(spec_path, runs_dir)
    spec = experiment.spec
    if (rule := spec.candidates) is None:
        raise ValueError(
            f"{spec_path} pushes no candidates of a list: a stage reads an "
            "experiment with a candidates: setting and a condition with "
            "each_candidate, as experiments/s3/screen_a.yaml has"
        )
    keys = list(dict.fromkeys(c.candidate for c in spec.conditions if c.candidate))
    # Every row of the list, in its order, which ranks candidates that tie.
    every = chosen_candidates(CandidatesRule(rule.file), spec.model)
    listed = [candidate for candidate, _ in every]
    by_key = {candidate.key: candidate for candidate in listed}
    if missing := [key for key in keys if key not in by_key]:  # changed meanwhile
        raise ValueError(f"{rule.file} has no {missing[0]}, which {spec_path} pushes")
    if not experiment.runs:
        raise ValueError(
            f"no reply of {spec.name} is saved in {experiment.folder} yet: `uv run "
            f"pushlab run {spec_path}` runs it on the pod"
        )
    saved = read_rows(experiment.folder, SCORES)
    rows = current_rows(saved, experiment.runs, experiment.version)
    make = score_command(spec_path, rule, SPEAKS_AS)
    chosen = pick_judge(rows, judge, f"`{make}` makes them")
    places = {candidate.key: place for place, candidate in enumerate(listed)}
    found, others = candidate_stats(
        [by_key[key] for key in keys], places, experiment.runs, rows, chosen
    )
    if not any(stats.doses for stats in found):
        raise ValueError(
            f"none of the {len(experiment.runs):,} replies saved for {spec.name} "
            f"pushes one of its candidates: was {rule.file} changed since it ran?"
        )
    prompts = spec.prompt_ids or read_prompt_set(spec.prompts_path).prompt_ids
    scored = Scored(
        spec_path, experiment, rule, chosen, tuple(prompts), tuple(found), others
    )
    replies = sum(stats.total.runs for stats in scored.stats)
    for rubric in needs:
        if scored.unjudged(rubric) == replies:
            raise ValueError(
                f"no reply of a candidate of {spec.name} has an ok {rubric} verdict "
                f"of {chosen}: `{scored.make(rubric)}` makes them"
            )
    return scored
