"""`pushlab verify`: an experiment's runner folder checked against its file.

It expands the experiment file and reads the current version of its folder,
<runs dir>/<name>: every mark the index has of the version, and the runs
and steps its latest marks count (pushlab.runs.shards). It checks that:
1. the version was made from this file, its prompt set and units/known.csv
   (if not, nothing else is compared with the file's plan);
2. every run the file plans is saved, ok or failed, and no other run;
3. each run is marked once in the version, or again only after marks that
   said it failed (a retry; a new version's marks of the runs it keeps count
   too); no shard holds a run twice, and no steps file a step of a run twice;
4. each saved record is the run the plan means (versions.same_run), has
   the status its latest mark gives it, and was made with the model
   download experiment.json names;
5. a run with a reply has steps 0 to n - 1 in order, one per reply token,
   with its reply's tokens and a known region, and a failed run has none;
6. each step pushes what its run's schedule pushes on the pass that chose
   that token (the prompt pass for step 0, the pass that read reply token
   j - 1 for step j; decision 0008), in the schedule's order, each amount
   of its dose's sign, 0 at dose 0, and the dose times one number per unit,
   its dose-1 amount, on every step of every run, up to 16-bit rounding;
7. manifest.json, if any, is a valid experiment manifest; it repeats
   experiment.json's experiment, version, experiment file, hashes and model
   download, and has the revision that download names, experiment.json's
   number of runs and the index's counts by status. It is not checked while
   a runner is working on the folder, as a session rewrites it when it ends.
A counted shard may hold other runs, such as runs of a batch that a later
attempt ran again; only the runs it counts are read from it. It loads no
model and writes nothing.
"""

import sys
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pyarrow as pa

from pushlab.runs import folder, plan
from pushlab.runs.expand import Expansion, expand
from pushlab.runs.spec import read_spec
from pushlab.runs.verify_steps import Bound, check_steps, count, listed
from pushlab.runs.versions import SAME_RUN, same_run
from pushlab.store.manifest import check_experiment_manifest
from pushlab.store.records import MANIFEST_FILE, RunRecord, read_runs_file
from pushlab.store.steps import StepRecord, read_steps_file

MOST_SHOWN = 50  # problems printed; a last line counts the rest
UNREADABLE = (OSError, ValueError, TypeError, pa.ArrowException)
# What the manifest repeats from experiment.json, field by field.
REPEATED = ("experiment", "version", "spec_file", *plan.IDENTITY_KEYS)


def _refuse(message: str) -> int:
    print(f"pushlab verify: {message}", file=sys.stderr)
    return 1


def _times(n: int) -> str:
    return "twice" if n == 2 else f"{n} times"


def _are(runs: Sequence[str], what: str) -> str:
    return f"{count(len(runs), what)} {'is' if len(runs) == 1 else 'are'}"


def _statuses(counts: object) -> str:
    if not isinstance(counts, dict):
        return repr(counts)
    ok, failed, pending = (counts.get(key) for key in ("ok", "failed", "pending"))
    return f"{ok} ok, {failed} failed and {pending} pending"


def shown_problems(problems: Sequence[str], most: int = MOST_SHOWN) -> list[str]:
    """The lines verify prints: the first most problems, then a count of the rest.

    Example:
        >>> shown_problems(["a", "b", "c"], most=2)
        ['a', 'b', 'and 1 more problem']
    """
    lines = list(problems[:most])
    if len(problems) > most:
        lines.append(f"and {count(len(problems) - most, 'more problem')}")
    return lines


@dataclass
class Verdict:
    """What verify found in a folder's version, and the runs and steps read."""

    experiment: str
    version: int
    runs: int = 0
    failed: int = 0
    steps: int = 0
    problems: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """The line verify ends with.

        Example:
            >>> Verdict("x", 1, runs=10, failed=2, steps=32).summary()
            'x, version 1: 10 runs (2 failed) and 32 steps checked; no problem found.'
        """
        failed = f" ({self.failed} failed)" if self.failed else ""
        found = count(len(self.problems), "problem") if self.problems else "no problem"
        return (
            f"{self.experiment}, version {self.version}: {count(self.runs, 'run')}"
            f"{failed} and {count(self.steps, 'step')} checked; {found} found."
        )


class _Check:
    """One verify of a folder's version: what it read, and what it found."""

    def __init__(self, path: Path, identity: dict, expansion: Expansion, busy: bool):
        self.path, self.identity, self.version = path, identity, identity["version"]
        self.busy = busy  # a runner works on the folder
        index = folder.read_index(path)
        self.marks = [entry for entry in index if entry["version"] == self.version]
        self.states = folder.latest(self.marks, self.version)
        runs = identity.get("runs")  # the version's number of runs
        self.total = runs if type(runs) is int else len(expansion.runs)
        self.verdict = Verdict(identity["experiment"], self.version, len(self.states))
        self.verdict.failed = sum(s["status"] == "failed" for s in self.states.values())
        self.say = self.verdict.problems.append
        self.scales: dict[str, Bound] = {}  # each unit's, over every run
        self.planned = self._plan(expansion)

    def run(self) -> Verdict:
        self._marked_again()
        by_shard: dict[str, list[str]] = {}  # in the order of their first runs
        for run_id in sorted(self.states):
            by_shard.setdefault(self.states[run_id]["shard"], []).append(run_id)
        for shard, run_ids in by_shard.items():
            self._shard(shard, run_ids)
        for unit, bound in self.scales.items():
            if where := bound.disagreement("in"):
                self.say(f"{unit}'s amount per dose {where}")
        self._manifest()
        return self.verdict

    def _plan(self, expansion: Expansion) -> dict[str, RunRecord] | None:
        """Checks 1 and 2; the planned records, or None when not comparable."""
        keys = ("spec_sha256", "prompts_sha256", "known_sha256")
        current = {key: getattr(expansion, key) for key in keys}
        if changed := plan.differences(self.identity, current):
            self.say(
                f"version {self.version} was made from another {' and '.join(changed)}"
                ", so its runs are not compared with what this file plans"
            )
            return None
        planned = {run.record.run_id: run.record for run in expansion.runs}
        if missing := [run_id for run_id in planned if run_id not in self.states]:
            if self.busy:
                how = "a runner is still working on the folder, and pushlab status"
                how += " says how far it has got"
            else:
                how = "pushlab status says how far the run has got"
            unsaved = f"{_are(missing, 'planned run')} not saved yet"
            self.say(f"{unsaved}: {listed(missing)}; {how}")
        if extra := sorted(set(self.states) - set(planned)):
            self.say(f"{_are(extra, 'saved run')} not planned: {listed(extra)}")
        return planned

    def _marked_again(self) -> None:
        """Check 3 on the index: a run is marked again only after it failed.

        Each run's marks are its statuses in the version's index lines, a
        kept run's mark in a new version among them. Runs whose marks are
        the same, shard by shard, are one problem.
        """
        marks: dict[str, list[tuple[str, str]]] = {}  # oldest first
        for entry in self.marks:
            for run_id, status in entry["runs"].items():
                marks.setdefault(run_id, []).append((str(status), entry["shard"]))
        again: dict[tuple[tuple[str, str], ...], list[str]] = {}
        for run_id in sorted(marks):
            if any(status != "failed" for status, _ in marks[run_id][:-1]):
                again.setdefault(tuple(marks[run_id]), []).append(run_id)
        for found, run_ids in again.items():
            where = ", then ".join(f"{status} in shards/{at}" for status, at in found)
            self.say(
                f"the index marks {listed(run_ids)} {_times(len(found))}: {where}; "
                "only a failed run is marked again"
            )

    def _read(self, place: str, shard: str, read: Callable) -> Any:
        """The rows of place/shard, or None once a problem says why not."""
        try:
            return read(self.path / place / shard)
        except FileNotFoundError:
            self.say(f"{place}/{shard} is missing, though the index counts its runs")
        except UNREADABLE as error:
            self.say(f"cannot read {place}/{shard}: {error}")
        return None

    def _shard(self, shard: str, run_ids: list[str]) -> None:
        """Check 3, and the checks of each run the index counts in a shard."""
        if (records := self._read(folder.SHARDS, shard, read_runs_file)) is None:
            return
        held = {record.run_id: record for record in records}
        for run_id, n in Counter(r.run_id for r in records).items():
            if n > 1:
                self.say(f"shards/{shard} holds {run_id} {_times(n)}")
        if lacking := [run_id for run_id in run_ids if run_id not in held]:
            self.say(
                f"shards/{shard} lacks {listed(lacking)}, which the index puts there"
            )
        steps = self._read(folder.STEPS, shard, read_steps_file)
        by_run: dict[str, list[StepRecord]] = {}
        for step in steps or []:
            by_run.setdefault(step.run_id, []).append(step)
        for run_id, found in by_run.items():
            counts = Counter(step.step for step in found)
            if twice := [number for number, n in counts.items() if n > 1]:
                times = "more than once" if twice[1:] else _times(counts[twice[0]])
                self.say(
                    f"steps/{shard} holds {listed(twice, 'step')} of {run_id} {times}"
                )
        for run_id in (run_id for run_id in run_ids if run_id in held):
            found = None if steps is None else by_run.get(run_id, [])
            self._run(held[run_id], shard, found)

    def _run(self, record: RunRecord, shard: str, steps: list | None) -> None:
        """Checks 4, 5 and 6 of one saved run; steps is None when unreadable."""
        run_id = record.run_id
        if planned := (self.planned or {}).get(run_id):
            planned = replace(planned, model=record.model)  # checked just below
        if planned and not same_run(record, planned):
            names = [k for k in SAME_RUN if getattr(record, k) != getattr(planned, k)]
            differ = "differs" if len(names) == 1 else "differ"
            self.say(f"{run_id} is not the planned run: its {listed(names)} {differ}")
        if (marked := self.states[run_id]["status"]) != record.status:
            self.say(
                f"{run_id} is {record.status} in shards/{shard}, but the index "
                f"marks it {marked}"
            )
        model = self.identity.get("model")  # the one download a version's runs use
        if isinstance(model, str) and record.model != model:
            self.say(
                f"{run_id}'s model is {record.model}, but "
                f"{folder.IDENTITY_FILE}'s is {model}"
            )
        if steps is None:
            return
        self.verdict.steps += len(steps)
        if record.status != "failed":
            check_steps(record, steps, self.say, self.scales)
        elif steps:
            self.say(f"{run_id} failed but steps/{shard} has {len(steps)} of its steps")

    def _manifest(self) -> None:
        """Check 7, unless a runner is working on the folder."""
        if self.busy:
            return
        try:
            manifest = folder.read_json(self.path / MANIFEST_FILE)
        except ValueError as error:
            self.say(str(error))
            return
        if manifest is None:
            return
        if not isinstance(manifest, dict):
            self.say(f"{MANIFEST_FILE} is not a JSON object")
            return
        try:
            check_experiment_manifest(manifest)
        except ValueError as error:
            self.say(f"{MANIFEST_FILE} is not a valid experiment manifest: {error}")
        source = folder.IDENTITY_FILE
        for key in REPEATED:
            if manifest.get(key) != self.identity.get(key):
                self.say(f"{MANIFEST_FILE}'s {key} is not {source}'s")
        model = self.identity.get("model")  # such as qwen3.5-4b@851bf6e8...
        if isinstance(model, str) and "@" in model:
            revision = model.rpartition("@")[2]
        else:
            revision = None
        if revision and manifest.get("model_revision") != revision:
            self.say(
                f"{MANIFEST_FILE}'s model_revision is not the revision in "
                f"{source}'s model"
            )
        if (run_count := manifest.get("run_count")) != self.total:
            self.say(
                f"{MANIFEST_FILE}'s run_count is {run_count!r}, but {source}'s "
                f"version has {count(self.total, 'run')}"
            )
        counts = Counter(str(state["status"]) for state in self.states.values())
        pending = self.total - len(self.states)
        index = {"ok": counts["ok"], "failed": counts["failed"], "pending": pending}
        if (saved := manifest.get("runs_by_status")) != index:
            self.say(
                f"{MANIFEST_FILE} counts {_statuses(saved)} runs, but the index "
                f"{_statuses(index)}"
            )


def run_verify(spec_path: Path, runs_dir: Path | None = None) -> int:
    """Check the runner folder of the experiment file at spec_path in runs_dir
    (default: folder.default_runs_dir()), print each problem and a summary,
    and return 1 if there are any, else 0. It refuses with 1, the reason on
    stderr, a file it cannot read or expand, a runs dir that does not exist,
    a runs dir or folder the runner would refuse, and a folder with nothing
    saved.

    Example:
        >>> run_verify(Path("experiments/p0/s2_pulse_lengths.yaml"))  # doctest: +SKIP
        0
    """
    try:
        spec = read_spec(spec_path)
    except ValueError as error:
        return _refuse(str(error))
    try:
        expansion = expand(spec)
    except ValueError as error:
        return _refuse(f"{spec_path}: {error}")
    runs_dir = folder.default_runs_dir() if runs_dir is None else runs_dir
    path = runs_dir / spec.name
    try:
        if problem := plan.runs_dir_problem(runs_dir):
            return _refuse(problem)
        if not runs_dir.exists():
            return _refuse(
                f"{runs_dir} does not exist; --runs-dir must be the folder that "
                "holds the experiments' folders"
            )
        if problem := plan.folder_problem(path, spec.name):
            return _refuse(problem)
        if (identity := folder.read_identity(path)) is None:
            return _refuse(f"nothing is saved for {spec.name} in {path} yet")
        busy = folder.lock_held(path)
        verdict = _Check(path, identity, expansion, busy).run()
    except (OSError, ValueError) as error:
        return _refuse(str(error))
    for line in shown_problems(verdict.problems):
        print(line)
    if busy:
        print("note: a runner is working on the folder; manifest.json is not checked")
    print(verdict.summary())
    return 1 if verdict.problems else 0
