"""What `pushlab report SPEC` reads from an experiment's runner folder.

runner_folder finds the folder and its experiment.json. saved_runs reads the
saved runs of the conditions to show, and their steps, one batch at a time:
a batch whose steps file is missing, as one saved before steps were kept, or
cannot be read only costs its replies their markers, each with a note that
says why. changed_notes says when the folder's version was made from another
experiment file or other known units than those given now. Nothing here
loads a model or writes a file.
"""

from pathlib import Path
from typing import Any

import pyarrow as pa

from pushlab.runs import folder, shards
from pushlab.runs.spec import ExperimentSpec
from pushlab.store.records import RunRecord
from pushlab.store.steps import StepRecord

NO_STEPS = "saved before steps were kept, so it has no markers"
# What reading a saved file can raise besides FileNotFoundError, as verify has.
UNREADABLE = (OSError, ValueError, TypeError, pa.ArrowException)


def runner_folder(spec: ExperimentSpec, runs_dir: Path) -> tuple[Path, dict[str, Any]]:
    """The experiment's runner folder, runs_dir/<name>, and its experiment.json.

    ValueError says why there is none to report on: the folder is missing,
    is not a runner folder, is another experiment's, or does not name its
    runs' conditions.

    Example:
        >>> runner_folder(spec, Path("/workspace/runs"))[0]  # doctest: +SKIP
        PosixPath('/workspace/runs/s2-pulse-lengths')
    """
    path = runs_dir / spec.name
    if not path.exists():
        raise ValueError(
            f"{path} does not exist: no run of {spec.name} is saved there yet; "
            "pushlab run makes it"
        )
    if not path.is_dir():
        raise ValueError(f"{path} is not a folder")
    identity = folder.read_identity(path)
    if identity is None:
        raise ValueError(
            f"{path} is not a runner folder: it has no {folder.IDENTITY_FILE}"
        )
    if identity["experiment"] != spec.name:
        raise ValueError(
            f"{path} belongs to the experiment {identity['experiment']!r}, "
            f"not {spec.name!r}"
        )
    labels = identity.get("conditions")
    if (
        not isinstance(labels, dict)
        or not labels
        or not all(isinstance(label, str) for label in labels.values())
    ):
        raise ValueError(
            f"{path / folder.IDENTITY_FILE} does not name each run's condition"
        )
    return path, identity


def _why(error: Exception, file: Path, shown: str) -> str:
    """What error says about file, with file's path shortened to shown."""
    why = str(error) or type(error).__name__
    for name in (str(file.absolute()), str(file)):
        why = why.replace(name, shown)
    why = why.rstrip(".")
    return why if shown in why else f"{shown}: {why}"


def saved_runs(
    path: Path, identity: dict[str, Any], chosen: list[str]
) -> tuple[list[RunRecord], dict[str, list[StepRecord]], dict[str, str]]:
    """The saved runs of the chosen conditions, in run order; their steps, by
    run ID; and a note for each run whose batch's steps cannot be read.

    path is the runner folder and identity its experiment.json. Each batch
    that holds one of the runs is read on its own: a missing steps file
    gives its runs the note NO_STEPS, and one that cannot be read the note
    "its steps in steps/<name> cannot be read, so it has no markers;
    pushlab verify says why". ValueError
    says, in words, when the index or a batch's runs shard cannot be read.

    Example:
        >>> saved_runs(Path("/no/such/folder"), {"version": 1, "conditions": {}}, [])
        ([], {}, {})
    """
    labels, cannot = identity["conditions"], f"cannot read the runs saved in {path}"
    try:
        states = folder.latest(folder.read_index(path), identity["version"])
    except (OSError, ValueError) as error:
        raise ValueError(f"{cannot}: {error}") from None
    by_shard: dict[str, dict[str, str]] = {}
    for run_id, state in sorted(states.items()):
        if labels.get(run_id) in chosen:
            by_shard.setdefault(state["shard"], {})[run_id] = state["shard"]
    runs: list[RunRecord] = []
    steps: dict[str, list[StepRecord]] = {}
    notes: dict[str, str] = {}
    for shard, wanted in by_shard.items():
        shown = f"{folder.SHARDS}/{shard}"
        try:
            runs += shards.read_records(path, wanted).values()
        except FileNotFoundError:
            why = f"{shown} is missing, though the index counts its runs"
            raise ValueError(f"{cannot}: {why}") from None
        except UNREADABLE as error:
            raise ValueError(f"{cannot}: {_why(error, path / shown, shown)}") from None
        try:
            steps |= shards.read_steps(path, wanted)
        except FileNotFoundError:
            notes |= dict.fromkeys(wanted, NO_STEPS)
        except UNREADABLE:  # pushlab verify says why, once for the batch
            note = (
                f"its steps in {folder.STEPS}/{shard} cannot be read, so it has no "
                "markers; pushlab verify says why"
            )
            notes |= dict.fromkeys(wanted, note)
    return sorted(runs, key=lambda run: run.run_id), steps, notes


def changed_notes(spec: ExperimentSpec, identity: dict[str, Any]) -> list[str]:
    """Notes for a report: whether the folder's version, whose experiment.json
    is identity, was made from another experiment file or other known units
    than those spec uses now. The prompt set has a note of its own.

    The known units are compared, as pushlab status compares them, through
    the experiment's expansion, and only when both the version and the file
    now draw controls; a file that no longer expands is noted only as changed.

    Example:
        >>> from pushlab import PROJECT_ROOT
        >>> from pushlab.runs.spec import read_spec
        >>> spec = read_spec(PROJECT_ROOT / "experiments/p0/s2_pulse_lengths.yaml")
        >>> changed_notes(spec, {"version": 1, "spec_sha256": spec.sha256})
        []
        >>> changed_notes(spec, {"version": 2, "spec_sha256": "0" * 64})[0][:53]
        'The experiment file changed since version 2 was made:'
        >>> identity = {"version": 1, "spec_sha256": spec.sha256, "known_sha256": "0"}
        >>> changed_notes(spec, identity)[0][:48]
        'units/known.csv changed since version 1 was made'
    """
    version, notes = identity["version"], []
    if spec.sha256 != identity.get("spec_sha256"):
        notes.append(
            f"The experiment file changed since version {version} was made: the "
            f"replies below are version {version}'s, and a label may mean other "
            "pushes in the file now."
        )
    if identity.get("known_sha256") is not None:  # the version drew controls
        from pushlab.runs.expand import expand

        try:
            known = expand(spec).known_sha256
        except ValueError:  # pushlab run would say what is wrong with it
            known = None
        if known not in (None, identity["known_sha256"]):
            notes.append(
                f"units/known.csv changed since version {version} was made: the "
                "controls below avoided the known units it listed then."
            )
    return notes
