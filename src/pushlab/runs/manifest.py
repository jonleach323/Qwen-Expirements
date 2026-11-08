"""The runner's manifest.json: one entry per session, and the code each ran.

After every session the runner rewrites the experiment's manifest
(pushlab.store.manifest) with record_session, this session last in its
sessions. When it runs no session, as when a new version keeps every run or
a session was killed after its last batch, restamp_manifest brings what the
manifest repeats from experiment.json and the index up to date. Before a
session runs, code_changes says whether the sessions that made the
version's runs so far, kept ones included, ran other code or libraries.

A manifest.json that is not valid JSON, or not a JSON object, lists no
earlier sessions: restamp_manifest leaves it as it is and says why, and the
next session rewrites it whole.
"""

from collections import Counter
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from pushlab.runs import folder, plan
from pushlab.store.manifest import check_experiment_manifest
from pushlab.store.records import MANIFEST_FILE, write_manifest

# What a manifest's session records about the code and libraries it ran.
SESSION_ENVIRONMENT = ("code_commit", "code_dirty", "lockfile_sha256", "gpu")
SESSION_ENVIRONMENT += ("cuda", "driver", "torch", "transformers", "packages")
# What must be the same for a version's runs to be made by one code.
CODE_KEYS = ("code_commit", "lockfile_sha256", "torch", "transformers")
# What an earlier session's entry must give, as text, to be kept.
SESSION_KEYS = ("id", "started_at", "finished_at")
# What restamp_manifest says of a manifest it cannot bring up to date.
LEFT_ALONE = "it is left as it is, and the next session of pushlab run rewrites it"


@dataclass(frozen=True)
class SessionEntry:
    """One session of pushlab run, as the manifest records it.

    environment is environment_info's: the code, machine and libraries the
    session ran with. Its entry in the manifest's sessions (entry()) keeps
    SESSION_ENVIRONMENT of them and the rest of these fields; the manifest's
    top level takes all of the last session's environment, and its device.
    state is how the session ended, such as "finished" or "stopped at
    --limit".

    Example:
        >>> session = SessionEntry(
        ...     id="5f0c9a",
        ...     environment={"code_commit": "e374dbf", "python": "3.11.9"},
        ...     device="cuda",
        ...     version=1,
        ...     started_at="2026-11-08T10:00:00+00:00",
        ...     finished_at="2026-11-08T10:30:00+00:00",
        ...     state="finished",
        ...     batches=5,
        ...     runs=50,
        ...     gpu_hours=0.5,
        ... )
        >>> entry = session.entry()
        >>> entry["code_commit"], entry["runs"], "python" in entry
        ('e374dbf', 50, False)
    """

    id: str
    environment: dict[str, Any]
    device: str
    version: int
    started_at: str
    finished_at: str
    state: str
    batches: int
    runs: int
    gpu_hours: float

    def entry(self) -> dict[str, Any]:
        """The session's entry in the manifest's sessions (see the example)."""
        own = {f.name: getattr(self, f.name) for f in fields(self)}
        del own["environment"]
        return {key: self.environment.get(key) for key in SESSION_ENVIRONMENT} | own


def _read(path: Path) -> dict[str, Any] | None:
    """The folder's manifest.json, or None when it has none.

    ValueError says when it is not valid JSON or not a JSON object.
    """
    file = path / MANIFEST_FILE
    info = folder.read_json(file)  # ValueError when not valid JSON
    if info is None and not file.exists():
        return None
    if not isinstance(info, dict):
        raise ValueError(f"{file} is not a JSON object")
    return info


def earlier_sessions(path: Path, session_id: str) -> list[dict[str, Any]]:
    """The sessions the folder's manifest lists, other than session_id.

    An entry without an id, a start and an end, each as text, is left out,
    and a manifest that is not valid JSON or not a JSON object lists none:
    the session then rewrites it whole.

    Example:
        >>> earlier_sessions(Path("/no/such/folder"), "a")
        []
    """
    try:
        info = _read(path) or {}
    except ValueError:  # the runner's restamp_manifest has warned of it
        return []
    sessions = info.get("sessions")
    return [
        session
        for session in (sessions if isinstance(sessions, list) else [])
        if isinstance(session, dict)
        and all(isinstance(session.get(key), str) for key in SESSION_KEYS)
        and session["id"] != session_id
    ]


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


def _stamp(
    identity: dict[str, Any], states: dict[str, str], total: int
) -> dict[str, Any]:
    """What a manifest repeats from experiment.json (identity) and the index
    (states, of total runs): the experiment, its version and file, the
    hashes, the model's download and revision, the dtype and the runs by
    status."""
    counts = Counter(states.values())
    stamp = {key: identity.get(key) for key in plan.IDENTITY_KEYS} | {
        "experiment": identity["experiment"],
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
    spec = identity.get("spec")
    decoding = spec.get("decoding") if isinstance(spec, dict) else None
    if isinstance(decoding, dict) and "dtype" in decoding:
        stamp["dtype"] = decoding["dtype"]
    return stamp


def record_session(
    path: Path,
    identity: dict[str, Any],
    session: SessionEntry,
    *,
    model_id: str,
    states: dict[str, str],
    total: int,
) -> None:
    """Check and rewrite the folder's manifest.json, with session last in
    its sessions, in place of its own earlier entry if it has one.

    identity is experiment.json, model_id the Hugging Face ID of the model
    the session loaded, and states the version's saved runs' statuses by
    run ID, of total runs. The top level holds what experiment.json and the
    index say, the last session's environment, model ID and device, the
    first session's start and the last one's end.

    Example:
        >>> record_session(
        ...     path,
        ...     identity,
        ...     session,
        ...     model_id="Qwen/Qwen3.5-4B",
        ...     states=states,
        ...     total=50,
        ... )  # doctest: +SKIP
    """
    sessions = earlier_sessions(path, session.id)
    sessions.append(session.entry())
    info = dict(session.environment) | _stamp(identity, states, total)
    info |= {
        "model_id": model_id,
        "device": session.device,
        "started_at": min(s["started_at"] for s in sessions),
        "finished_at": sessions[-1]["finished_at"],
        "sessions": sessions,
    }
    check_experiment_manifest(info)
    write_manifest(path, info, overwrite=True)


def restamp_manifest(
    path: Path, identity: dict[str, Any], states: dict[str, str], total: int
) -> bool:
    """Bring what manifest.json repeats from experiment.json and the index up
    to date, keeping the rest; True if it had to be rewritten.

    What it brings up to date is the experiment, its version and file, the
    hashes, the model's download and revision, the dtype and the runs by
    status. The rest stays as the sessions left it: the last session's code,
    machine and libraries (code_commit, gpu, torch and the like), model ID
    and device, the first session's start, the last one's end, and the
    sessions. So until a session of a new version runs, those are an earlier
    version's.

    It writes only when something changed, so a finished experiment run
    again changes no file, and does nothing without a manifest. ValueError
    says when manifest.json is not valid JSON, not a JSON object or cannot
    be brought up to date, and leaves it as it is: the next session of
    pushlab run rewrites it whole.

    Example:
        >>> restamp_manifest(Path("/no/such/folder"), {}, {}, 0)
        False
    """
    try:
        info = _read(path)
    except ValueError as error:
        raise ValueError(f"{error}; {LEFT_ALONE}") from None
    if info is None:
        return False
    fresh = info | _stamp(identity, states, total)
    if fresh == info:
        return False
    try:
        check_experiment_manifest(fresh)
    except ValueError as error:
        file = path / MANIFEST_FILE
        raise ValueError(
            f"{file} cannot be brought up to date: {error}; {LEFT_ALONE}"
        ) from None
    write_manifest(path, fresh, overwrite=True)
    return True
