"""What a judge is given from an experiment: its replies, or windows of them.

judged_runs reads an experiment file and its runner folder: the saved runs
of the folder's current version that have a reply, "ok" or "collapsed",
never a failed run, which has none. reply_items gives one Item per run, its
whole reply; window_items one per window of `size` sentences
(pushlab.scoring.sentences). An item keeps its run so that its verdict can
be stored against it, but a judge is only ever given its text (decision
0021): never the run's condition, schedule, prompt or ID.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pushlab.analysis.report_folder import runner_folder
from pushlab.runs import shards
from pushlab.runs.folder import default_runs_dir
from pushlab.runs.spec import ExperimentSpec, read_spec
from pushlab.scoring.sentences import Window, sentence_windows
from pushlab.store.records import RunRecord

# The run statuses that have a reply to judge.
JUDGED_STATUSES = ("ok", "collapsed")


@dataclass(frozen=True)
class Item:
    """A text to judge: a run's whole reply, or one window of its sentences."""

    run: RunRecord
    text: str
    window: Window | None = None


@dataclass(frozen=True)
class Experiment:
    """An experiment file, its runner folder and experiment.json, and the
    saved runs of its version that have a reply, in run order."""

    spec: ExperimentSpec
    folder: Path
    identity: dict[str, Any]
    runs: list[RunRecord]
    failed: int  # saved runs left out, as failed

    @property
    def version(self) -> int:
        """The runner folder's current version."""
        return int(self.identity["version"])

    def condition(self, run: RunRecord) -> str:
        """The run's condition label, from experiment.json."""
        return str(self.identity["conditions"].get(run.run_id, ""))


def judged_runs(spec_path: Path, runs_dir: Path | None = None) -> Experiment:
    """The experiment at spec_path, as saved in its runner folder in runs_dir
    (default: folder.default_runs_dir()).

    ValueError says why there is nothing to judge: an experiment file that
    cannot be read, a runner folder that is missing or not this
    experiment's (pushlab.analysis.report_folder.runner_folder), or saved
    runs that cannot be read.

    Example:
        >>> spec = Path("experiments/p0/s2_pulse_lengths.yaml")
        >>> judged_runs(spec).version  # doctest: +SKIP
        1
    """
    spec = read_spec(spec_path)
    runs_dir = default_runs_dir() if runs_dir is None else Path(runs_dir)
    path, identity = runner_folder(spec, runs_dir)
    try:
        saved = shards.read_results(path)
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read the runs saved in {path}: {error}") from None
    runs = [run for run in saved if run.status in JUDGED_STATUSES]
    return Experiment(spec, path, identity, runs, len(saved) - len(runs))


def reply_items(runs: Sequence[RunRecord]) -> list[Item]:
    """One item per run: its whole reply.

    Example:
        >>> reply_items([])
        []
    """
    return [Item(run, run.output_text) for run in runs]


def window_items(runs: Sequence[RunRecord], size: int = 1) -> list[Item]:
    """One item per window of `size` sentences of each run's reply, in order;
    a reply with no sentence, a blank one, gives none.

    Example:
        >>> window_items([], 2)
        []
    """
    return [
        Item(run, window.text, window)
        for run in runs
        for window in sentence_windows(run.output_text, size)
    ]
