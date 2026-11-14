"""An experiment's manifest: what its runs were made with, and how they ended.

`pushlab run` rewrites manifest.json in an experiment's runner folder after
every session (pushlab.runs.manifest). It holds what every manifest does
(pushlab.store.records.check_manifest: the code commit, the lockfile's
hash, the model's ID and revision, the GPU, CUDA, torch and transformers,
the start and end times and the number of runs), and beyond that which
experiment and version it is, the hashes of the experiment file and of the
calibration, the runs by status, and one entry per session.
"""

import re
from collections.abc import Mapping
from typing import Any

from pushlab.store.records import EXPERIMENT_NAME, check_manifest

EXPERIMENT_MANIFEST_FIELDS = (
    "experiment",
    "version",
    "spec_sha256",
    "calibration_sha256",
    "runs_by_status",
    "sessions",
)
RUN_STATES = ("ok", "failed", "pending")
SHA256 = re.compile(r"[0-9a-f]{64}")
# calibration_sha256 of runs whose doses take each prompt's own residual
# sizes, the per_prompt calibration, which has no file to hash.
PER_PROMPT = "per_prompt"


def _count(value: object) -> bool:
    return type(value) is int and value >= 0


def check_experiment_manifest(info: Mapping[str, Any]) -> str:
    """Raise ValueError unless info is a valid experiment manifest; return it
    as JSON text.

    Beyond check_manifest's fields, info must have every field in
    EXPERIMENT_MANIFEST_FIELDS: the experiment's name and version (from 1),
    the experiment file's SHA-256, the calibration file's (PER_PROMPT when
    each prompt's own sizes are measured instead, None when nothing is
    pushed), the runs ok, failed and pending, which add up to run_count, and
    a list of sessions, each with an ID and its start and end times.

    Example:
        >>> check_experiment_manifest({"run_count": 1})  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: the manifest lacks code_commit, lockfile_sha256, ...
    """
    text = check_manifest(info)
    if missing := [name for name in EXPERIMENT_MANIFEST_FIELDS if name not in info]:
        raise ValueError(f"the experiment manifest lacks {', '.join(missing)}")
    name = info["experiment"]
    if not (isinstance(name, str) and EXPERIMENT_NAME.fullmatch(name)):
        raise ValueError(f"experiment must be an experiment's name, not {name!r}")
    version = info["version"]
    if type(version) is not int or version < 1:
        raise ValueError(f"version must be a whole number from 1, not {version!r}")
    spec, calibration = info["spec_sha256"], info["calibration_sha256"]
    if not (isinstance(spec, str) and SHA256.fullmatch(spec)):
        raise ValueError(f"spec_sha256 must be a SHA-256 in hex, not {spec!r}")
    if calibration not in (None, PER_PROMPT) and not (
        isinstance(calibration, str) and SHA256.fullmatch(calibration)
    ):
        raise ValueError(
            f"calibration_sha256 must be a SHA-256 in hex, {PER_PROMPT} when each "
            "prompt's own sizes are measured, or None when nothing is pushed, not "
            f"{calibration!r}"
        )
    counts = info["runs_by_status"]
    if not (
        isinstance(counts, dict)
        and sorted(counts) == sorted(RUN_STATES)
        and all(map(_count, counts.values()))
    ):
        raise ValueError(
            f"runs_by_status must count the runs {', '.join(RUN_STATES)}, not "
            f"{counts!r}"
        )
    if sum(counts.values()) != info["run_count"]:
        raise ValueError(
            f"runs_by_status adds up to {sum(counts.values())} runs, not run_count "
            f"{info['run_count']}"
        )
    sessions = info["sessions"]
    if not (
        isinstance(sessions, list)
        and sessions
        and all(
            isinstance(s, dict) and {"id", "started_at", "finished_at"} <= set(s)
            for s in sessions
        )
    ):
        raise ValueError(
            "sessions must list each session, with its id, started_at and finished_at"
        )
    return text
