"""One session of `pushlab run`: the model loaded once, then batch after batch.

After each batch the session saves its shard and marks it in the index
(pushlab.runs.folder), then rewrites the heartbeat and progress files, so
pushlab status always has a fresh account. It stops after the batch that
brings the version's GPU time to max_gpu_hours, after the batches --limit
allows, or at Ctrl-C, which loses only the batch that was running; then it
saves the speed it measured and the manifest, with this session in it.
"""

import os
import socket
import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pushlab.runs import folder, plan
from pushlab.runs.batch import run_batch, session_inputs
from pushlab.runs.estimate import exact
from pushlab.runs.expand import Expansion
from pushlab.runs.plan import Batch
from pushlab.runs.spec import ExperimentSpec
from pushlab.store.records import MANIFEST_FILE, RunRecord

EXIT_FAILED, EXIT_INTERRUPTED = 2, 130
GO_ON = (
    "To go on, raise max_gpu_hours in the experiment file and run it with "
    "--overwrite-version, which keeps the saved runs."
)


def _refuse(message: str) -> int:
    print(f"pushlab run: {message}", file=sys.stderr)
    return 1


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def runs_text(count: int) -> str:
    """A number of runs in words.

    Example:
        >>> runs_text(1), runs_text(1200)
        ('1 run', '1,200 runs')
    """
    return f"{count:,} run" + ("" if count == 1 else "s")


@dataclass
class Tally:
    """What one session did."""

    runs: int = 0
    ok: int = 0
    failed: int = 0
    batches: int = 0
    seconds: float = 0.0
    generate_seconds: float = 0.0  # of batches that ran ok, for the speed
    reply_tokens: int = 0


class Session:
    """One `pushlab run` that runs batches: the model, the files, the counts."""

    def __init__(
        self,
        spec: ExperimentSpec,
        expansion: Expansion,
        path: Path,
        identity: dict[str, Any],
        states: dict[str, dict[str, Any]],
        used: float,
        left: int,
        clock: Callable[[], float],
    ) -> None:
        self.spec, self.expansion, self.path = spec, expansion, path
        self.clock = clock  # the session's wall clock, in seconds
        self.identity, self.version = identity, identity["version"]
        self.states = {run_id: state["status"] for run_id, state in states.items()}
        self.used, self.left = used, left
        self.tally = Tally()
        self.started_at = _now()
        self.state = "running"
        self.batch: str | None = None

    def _beat(self) -> None:
        """Rewrite the heartbeat and the progress file."""
        counts = Counter(self.states.values())
        total = len(self.expansion.runs)
        rate = None
        if self.tally.generate_seconds:
            rate = self.tally.reply_tokens / self.tally.generate_seconds
        hours_left = None
        if rate and self.tally.ok:
            tokens_per_run = self.tally.reply_tokens / self.tally.ok
            hours_left = self.left * tokens_per_run / rate / 3600
        now = _now()
        folder.write_json(
            self.path / folder.HEARTBEAT_FILE,
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "state": self.state,
                "batch": self.batch,
                "session_started_at": self.started_at,
                "updated_at": now,
            },
        )
        folder.write_json(
            self.path / folder.PROGRESS_FILE,
            {
                "experiment": self.spec.name,
                "version": self.version,
                "state": self.state,
                "runs": total,
                "ok": counts["ok"],
                "failed": counts["failed"],
                "pending": total - len(self.states),
                "session_runs": self.tally.runs,
                "session_batches": self.tally.batches,
                "gpu_hours": (self.used + self.tally.seconds) / 3600,
                "max_gpu_hours": self.spec.max_gpu_hours,
                "reply_tokens_per_second": rate,
                "hours_left": hours_left,
                "updated_at": now,
            },
        )

    def run(self, chosen: list[Batch], limit: int | None) -> int:
        """Load the model, run the chosen batches, and say how it ended."""
        import torch

        from pushlab.engine import generate as generate_module  # looked up per call
        from pushlab.models import load
        from pushlab.store.environment import environment_info

        spec = self.spec
        last = self.clock()
        self._beat()
        saved = len(self.states)
        print(
            f"{spec.name}, version {self.version}: {runs_text(self.left)} to run, "
            f"{saved} of {len(self.expansion.runs)} saved."
        )
        device = load.default_device()
        dtype_name = spec.decoding["dtype"]
        where = "GPU" if device == "cuda" else "CPU"
        print(f"Loading {spec.model} in {dtype_name} on the {where}...")
        loaded = load.load_model(spec.model, getattr(torch, dtype_name), device=device)
        problem = session_inputs(spec, loaded.revision, self.identity, chosen)
        if isinstance(problem, str):
            self.state = "stopped: the inputs changed"
            self._beat()
            return _refuse(problem)
        norms, texts = problem
        environment = environment_info(device)
        self._manifest(loaded, device, environment)
        ending = "finished"
        try:
            current = chosen[0].key
            for number, batch in enumerate(chosen):
                current = batch.key
                records, error, generated, tokens = run_batch(
                    batch, spec, loaded, norms, texts, generate_module.generate
                )
                name = folder.next_shard_name(self.path, self.version, batch.key)
                folder.write_shard(self.path, name, records)
                now = self.clock()
                seconds, last = now - last, now
                folder.append_index(
                    self.path,
                    self._entry(
                        name, batch, records, error, seconds, generated, tokens
                    ),
                )
                self._count(batch, records, error, seconds, generated, tokens)
                if (self.used + self.tally.seconds) >= spec.max_gpu_hours * 3600:
                    if self.left:
                        ending = "stopped at max_gpu_hours"
                        break
                if number == len(chosen) - 1 and self.left:
                    ending = "stopped at --limit"
        except KeyboardInterrupt:
            ending = "stopped by Ctrl-C"
        except OSError as error:  # the disk, not a run: stop, and keep what is saved
            ending = f"stopped: cannot save {current}: {error}"
        return self._finish(ending, limit, loaded, device, environment)

    def _entry(
        self,
        name: str,
        batch: Batch,
        records: list[RunRecord],
        error: str | None,
        seconds: float,
        generated: float,
        tokens: int,
    ) -> dict[str, Any]:
        return {
            "version": self.version,
            "shard": name,
            "batch": batch.key,
            "runs": {record.run_id: record.status for record in records},
            "error": error,
            "seconds": seconds,
            "generate_seconds": generated,
            "reply_tokens": tokens,
            "saved_at": _now(),
        }

    def _count(
        self,
        batch: Batch,
        records: list[RunRecord],
        error: str | None,
        seconds: float,
        generated: float,
        tokens: int,
    ) -> None:
        """Count a saved batch, print its line and rewrite the progress files."""
        tally = self.tally
        ok = sum(record.status == "ok" for record in records)
        tally.runs += len(records)
        tally.ok += ok
        tally.failed += len(records) - ok
        tally.batches += 1
        tally.seconds += seconds
        if error is None:
            tally.generate_seconds += generated
            tally.reply_tokens += tokens
        self.states |= {record.run_id: record.status for record in records}
        self.left -= len(records)
        self.batch = batch.key
        saved = len(self.states)
        total = len(self.expansion.runs)
        line = (
            f"{batch.key}: {ok} ok, {len(records) - ok} failed, {tokens:,} reply "
            f"tokens in {generated:.1f} s; {saved:,} of {total:,} runs saved"
        )
        print(line + (f" ({error})" if error else ""), flush=True)
        self._beat()

    def _manifest(self, loaded: Any, device: str, environment: dict[str, Any]) -> None:
        """Rewrite manifest.json, with this session last in its sessions."""
        from pushlab.store.records import check_manifest, write_manifest

        earlier = folder.read_json(self.path / MANIFEST_FILE) or {}
        sessions = [
            s for s in earlier.get("sessions", []) if s["started_at"] != self.started_at
        ]
        counts = Counter(self.states.values())
        sessions.append(
            {
                "started_at": self.started_at,
                "finished_at": _now(),
                "state": self.state,
                "version": self.version,
                "code_commit": environment["code_commit"],
                "code_dirty": environment["code_dirty"],
                "gpu": environment["gpu"],
                "batches": self.tally.batches,
                "runs": self.tally.runs,
                "gpu_hours": self.tally.seconds / 3600,
            }
        )
        info = environment | {
            k: self.identity.get(k) for k in ("spec_file", *plan.IDENTITY_KEYS)
        }
        info |= {
            "experiment": self.spec.name,
            "version": self.version,
            "model_id": loaded.spec.hf_id,
            "model_revision": loaded.revision,
            "device": device,
            "dtype": self.spec.decoding["dtype"],
            "started_at": sessions[0]["started_at"],
            "finished_at": sessions[-1]["finished_at"],
            "run_count": len(self.expansion.runs),
            "runs_by_status": {
                "ok": counts["ok"],
                "failed": counts["failed"],
                "pending": len(self.expansion.runs) - len(self.states),
            },
            "sessions": sessions,
        }
        check_manifest(info)
        write_manifest(self.path, info, overwrite=True)

    def _finish(
        self,
        ending: str,
        limit: int | None,
        loaded: Any,
        device: str,
        environment: dict[str, Any],
    ) -> int:
        """Save the speed and the files, and say how the session ended."""
        spec, tally = self.spec, self.tally
        self.state = ending
        if tally.generate_seconds and tally.reply_tokens:
            folder.save_speed(
                self.path.parent,
                f"{spec.model}/{spec.decoding['dtype']}",
                {
                    "reply_tokens_per_second": tally.reply_tokens
                    / tally.generate_seconds,
                    "reply_tokens": tally.reply_tokens,
                    "seconds": tally.generate_seconds,
                    "batch_size": spec.batch_size,
                    "gpu": environment["gpu"],
                    "experiment": spec.name,
                    "measured_at": _now(),
                },
            )
        self._beat()
        self._manifest(loaded, device, environment)
        counts = Counter(self.states.values())
        total = len(self.expansion.runs)
        if ending == "stopped at max_gpu_hours":
            hours = (self.used + tally.seconds) / 3600
            print(
                f"Stopped at max_gpu_hours {exact(spec.max_gpu_hours)}: {hours:.2f} "
                f"GPU-hours are used; {runs_text(self.left)} left. {GO_ON}"
            )
        elif ending == "stopped at --limit":
            print(
                f"Stopped at --limit {limit}: {runs_text(tally.runs)} this time; "
                f"{runs_text(self.left)} left."
            )
        elif ending == "stopped by Ctrl-C":
            print(
                "Stopped by Ctrl-C: the batch that was running is not saved. Run "
                "the same command to go on."
            )
            return EXIT_INTERRUPTED
        elif ending.startswith("stopped:"):
            return _refuse(ending.removeprefix("stopped: "))
        elif len(self.states) == total:
            print(
                f"Done: all {total} runs are saved, {counts['ok']} ok and "
                f"{counts['failed']} failed, in {self.path}."
            )
        else:
            print(f"Done: {runs_text(total - len(self.states))} are still to run.")
        if counts["failed"]:
            them = "it" if counts["failed"] == 1 else "them"
            failed = (
                "1 run failed"
                if counts["failed"] == 1
                else f"{counts['failed']} runs failed"
            )
            print(f"{failed}; rerun {them} with --retry-failed.")
        return EXIT_FAILED if tally.failed else 0
