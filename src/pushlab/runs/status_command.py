"""`pushlab status`: how far an experiment has got, and how long the rest will take.

It reads the experiment file for the experiment's name, then its runner
folder: experiment.json, the index and the heartbeat. It prints the runs
saved ok, failed and pending, the batches saved, the speed measured over
them, the GPU-hours used of max_gpu_hours and the time left at that speed
(or, before any batch, at the speed estimates use). It loads no model,
takes no lock and writes nothing, so it works while a run is going.
"""

import socket
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from pushlab.runs import folder
from pushlab.runs.estimate import duration, exact, read_costs, speed_for
from pushlab.runs.spec import read_spec

# The most failed runs it names; the count says how many there are.
FAILED_SHOWN = 5


def _refuse(message: str) -> int:
    print(f"pushlab status: {message}", file=sys.stderr)
    return 1


def _runner(path: Path, beat: dict[str, Any] | None) -> str:
    """Whether a runner works on the folder, from its heartbeat."""
    if beat is None:
        return "not running: no session has started"
    if beat.get("state") != "running":
        ended = f"{beat['updated_at']} ({beat['state']})"
        return f"not running; its last session ended at {ended}"
    pid, host = beat["pid"], beat.get("host")
    if host != socket.gethostname():
        return f"running on {host}, pid {pid}, as of {beat['updated_at']}"
    if folder.pid_alive(pid):
        return f"running, pid {pid}; the last batch was saved at {beat['updated_at']}"
    return (
        f"stopped without finishing: pid {pid} is gone; run the same command to go on"
    )


def status_lines(spec_path: Path, runs_dir: Path) -> list[tuple[str, str]]:
    """What `pushlab status` prints, as (name, value) lines.

    Example:
        >>> status_lines(
        ...     Path("experiments/p0/s2_pulse_lengths.yaml"), RUNS
        ... )  # doctest: +SKIP
        [('experiment', 's2-pulse-lengths, version 1, in /workspace/runs/...'), ...]
    """
    from pushlab.runs.plan import folder_problem

    spec = read_spec(spec_path)
    path = runs_dir / spec.name
    if problem := folder_problem(path, spec.name):
        raise ValueError(problem)
    identity = folder.read_identity(path)
    costs = read_costs()
    speed = speed_for(spec.model, spec.decoding["dtype"], runs_dir, costs)
    if identity is None:
        from pushlab.runs.expand import expand  # for the runs it would make

        total = len(expand(spec).runs)
        where = "does not exist yet" if not path.exists() else "holds no runs yet"
        lines = [
            ("experiment", spec.name),
            ("runner", f"not started: {path} {where}"),
            ("runs", f"{total}: 0 ok, 0 failed, {total} pending"),
        ]
        if speed is not None:
            seconds = total * spec.max_new_tokens / speed.reply_tokens_per_second
            lines.append(
                ("time left", f"at most {duration(seconds)}, at {speed.text()}")
            )
        return lines
    version, total = identity["version"], identity["runs"]
    entries = [e for e in folder.read_index(path) if e.get("version") == version]
    states = folder.latest(entries, version)
    counts = Counter(state["status"] for state in states.values())
    pending = total - len(states)
    ran = [e for e in entries if "kept_from" not in e]
    kept = len(entries) - len(ran)
    batches = f"{len(ran)} saved" + (
        f", and {kept} kept from earlier versions" if kept else ""
    )
    lines = [
        ("experiment", f"{spec.name}, version {version}, in {path}"),
        ("runner", _runner(path, folder.read_json(path / folder.HEARTBEAT_FILE))),
        (
            "runs",
            f"{total}: {counts['ok']} ok, {counts['failed']} failed, {pending} pending",
        ),
        ("batches", batches),
    ]
    good = [e for e in ran if not e.get("error")]
    tokens = sum(e.get("reply_tokens", 0) for e in good)
    seconds = sum(e.get("generate_seconds", 0.0) for e in good)
    used = folder.used_seconds(entries, version) / 3600
    if tokens and seconds:
        rate = tokens / seconds
        lines.append(
            (
                "speed",
                f"{rate:.3g} reply tokens a second, measured over {len(good)} batches",
            )
        )
    cap = exact(identity.get("max_gpu_hours", spec.max_gpu_hours))
    lines.append(("gpu-hours", f"{used:.2f} used of max_gpu_hours {cap}"))
    if not pending:
        lines.append(("time left", "none: every run is saved"))
    elif tokens and seconds:
        runs_done = sum(len(e["runs"]) for e in good)
        left = pending * tokens / runs_done / (tokens / seconds)
        lines.append(
            ("time left", f"about {duration(left)} for {pending} runs at that speed")
        )
    elif speed is not None:
        left = pending * identity["max_new_tokens"] / speed.reply_tokens_per_second
        lines.append(("time left", f"at most {duration(left)}, at {speed.text()}"))
    failed = sorted(run_id for run_id, s in states.items() if s["status"] == "failed")
    for run_id in failed[:FAILED_SHOWN]:
        lines.append(("failed", f"{run_id}: {states[run_id]['error']}"))
    if len(failed) > FAILED_SHOWN:
        lines.append(("failed", f"and {len(failed) - FAILED_SHOWN} more"))
    if identity["spec_sha256"] != spec.sha256:
        lines.append(
            (
                "note",
                f"the experiment file changed since version {version}: pushlab run "
                "will refuse it without --overwrite-version",
            )
        )
    return lines


def run_status(spec_path: Path, runs_dir: Path | None = None) -> int:
    """Print how far the experiment at spec_path has got; return the exit status.

    The status is 0, or 1 with the reason on stderr when the experiment file
    or its folder cannot be read.

    Example:
        >>> run_status(Path("experiments/p0/s2_pulse_lengths.yaml"))  # doctest: +SKIP
        0
    """
    runs_dir = folder.default_runs_dir() if runs_dir is None else runs_dir
    try:
        lines = status_lines(spec_path, runs_dir)
    except ValueError as error:
        return _refuse(str(error))
    width = max(len(name) for name, _ in lines) + 2
    for name, value in lines:
        print(f"{name:<{width}}{value}")
    return 0
