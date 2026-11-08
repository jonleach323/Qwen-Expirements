"""The runner's manifest.json: one entry per session, and the code each ran.

After every session the runner rewrites the experiment's manifest
(pushlab.store.manifest), with this session last in its sessions. When it
runs no session, as when a new version keeps every run or a session was
killed after its last batch, restamp_manifest brings the version, hashes
and counts up to date. Before a session runs, code_changes says whether
the sessions that made the version's runs so far, kept ones included, ran
other code or libraries.
"""

from collections import Counter
from pathlib import Path
from typing import Any

from pushlab.runs import folder, plan
from pushlab.runs.spec import ExperimentSpec
from pushlab.store.manifest import check_experiment_manifest
from pushlab.store.records import MANIFEST_FILE, write_manifest

# What a manifest's session records about the code and libraries it ran.
SESSION_ENVIRONMENT = ("code_commit", "code_dirty", "lockfile_sha256", "gpu")
SESSION_ENVIRONMENT += ("cuda", "driver", "torch", "transformers", "packages")
# What must be the same for a version's runs to be made by one code.
CODE_KEYS = ("code_commit", "lockfile_sha256", "torch", "transformers")


def earlier_sessions(path: Path, session_id: str) -> list[dict[str, Any]]:
    """The sessions the folder's manifest lists, other than session_id.

    Example:
        >>> earlier_sessions(Path("/no/such/folder"), "a")
        []
    """
    earlier = folder.read_json(path / MANIFEST_FILE) or {}
    return [s for s in earlier.get("sessions", []) if s.get("id") != session_id]


def code_changes(
    path: Path, version: int, session_id: str, environment: dict[str, Any]
) -> list[str]:
    """The CODE_KEYS in which the sessions that made the version's runs so
    far, kept ones included, differ from environment.

    A kept run's mark names the session that made it (pushlab.runs.versions).

    Example:
        >>> code_changes(Path("/no/such/folder"), 1, "a", {})
        []
    """
    makers = {
        entry.get("session")
        for entry in folder.read_index(path)
        if entry.get("version") == version
    }
    changed: list[str] = []
    for session in earlier_sessions(path, session_id):
        if session.get("id") in makers:
            for key in CODE_KEYS:
                if session.get(key) != environment.get(key) and key not in changed:
                    changed.append(key)
    return changed


def write_experiment_manifest(
    path: Path,
    spec: ExperimentSpec,
    identity: dict[str, Any],
    loaded: Any,
    session: dict[str, Any],
    states: dict[str, str],
    total: int,
) -> dict[str, Any]:
    """Check and rewrite the folder's manifest.json; return what it holds.

    session is this session's entry: its environment (SESSION_ENVIRONMENT and
    the rest of what environment_info gives), its id, times, state, version,
    batches, runs and GPU-hours, and its device. states are the version's
    saved runs' statuses by run ID, of total runs.

    Example:
        >>> write_experiment_manifest(
        ...     path, spec, identity, loaded, session, states, 50
        ... )["runs_by_status"]  # doctest: +SKIP
        {'ok': 50, 'failed': 0, 'pending': 0}
    """
    environment = session["environment"]
    sessions = earlier_sessions(path, session["id"])
    sessions.append(
        {key: environment.get(key) for key in SESSION_ENVIRONMENT}
        | {key: value for key, value in session.items() if key != "environment"}
    )
    counts = Counter(states.values())
    recorded = ("spec_file", *plan.IDENTITY_KEYS)
    info = environment | {key: identity.get(key) for key in recorded}
    info |= {
        "experiment": spec.name,
        "version": identity["version"],
        "model_id": loaded.spec.hf_id,
        "model_revision": loaded.revision,
        "device": session.get("device"),
        "dtype": spec.decoding["dtype"],
        "started_at": min(s["started_at"] for s in sessions),
        "finished_at": sessions[-1]["finished_at"],
        "run_count": total,
        "runs_by_status": {
            "ok": counts["ok"],
            "failed": counts["failed"],
            "pending": total - len(states),
        },
        "sessions": sessions,
    }
    check_experiment_manifest(info)
    write_manifest(path, info, overwrite=True)
    return info


def restamp_manifest(
    path: Path, identity: dict[str, Any], states: dict[str, str], total: int
) -> bool:
    """Bring manifest.json's version, hashes and counts up to experiment.json
    and the index, keeping its sessions; True if it had to be rewritten.

    It writes only when something changed, so a finished experiment run
    again changes no file, and does nothing without a manifest.

    Example:
        >>> restamp_manifest(Path("/no/such/folder"), {}, {}, 0)
        False
    """
    info = folder.read_json(path / MANIFEST_FILE)
    if info is None:
        return False
    counts = Counter(states.values())
    fresh = dict(info) | {key: identity.get(key) for key in plan.IDENTITY_KEYS}
    fresh |= {
        "version": identity["version"],
        "spec_file": identity.get("spec_file"),
        "model_revision": str(identity["model"]).rpartition("@")[2],
        "run_count": total,
        "runs_by_status": {
            "ok": counts["ok"],
            "failed": counts["failed"],
            "pending": total - len(states),
        },
    }
    if fresh == info:
        return False
    check_experiment_manifest(fresh)
    write_manifest(path, fresh, overwrite=True)
    return True
