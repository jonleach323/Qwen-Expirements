"""One session of `pushlab run`: the model loaded once, then call after call.

After each call the session saves its shard and marks it in the index
(pushlab.runs.folder), then rewrites the heartbeat and progress files, so
pushlab status always has a fresh account. It stops after the call that
brings the version's GPU time to max_gpu_hours, after the calls --limit
allows, or at Ctrl-C, which loses only the call that was running; then it
saves the speed it measured and the manifest, with this session in it.
Each session has an ID, which its index lines and its manifest entry
carry, with the code and library versions it ran.
"""

import os
import socket
import sys
import time
import traceback
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pushlab.runs import estimate, folder, manifest, shards
from pushlab.runs.batch import Inputs, Outcome, run_call, session_inputs
from pushlab.runs.estimate import exact
from pushlab.runs.expand import Expansion
from pushlab.runs.plan import Call
from pushlab.runs.spec import ExperimentSpec

EXIT_FAILED, EXIT_STOPPED, EXIT_INTERRUPTED = 2, 3, 130
GO_ON = (
    "To go on, raise max_gpu_hours in the experiment file through a pull "
    "request, then run it with --overwrite-version, which keeps the saved runs."
)


def refuse(message: str) -> int:
    """Print why `pushlab run` refuses, and return its exit status, 1.

    Example:
        >>> refuse("no such file")  # doctest: +SKIP
        1
    """
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
    calls: int = 0
    seconds: float = 0.0
    ok_calls: int = 0  # calls that ran ok, and their rows, for the speed
    rows: int = 0
    generate_seconds: float = 0.0
    reply_tokens: int = 0
    load_seconds: float | None = None


class Session:
    """One `pushlab run` that runs calls: the model, the files, the counts."""

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
        self.id = uuid.uuid4().hex
        self.started_at = _now()
        self.state = "running"
        self.call: str | None = None
        self.environment: dict[str, Any] | None = None
        self.loaded: Any = None
        self.device = "cpu"

    def _rate(self) -> float | None:
        tally = self.tally
        return (
            tally.reply_tokens / tally.generate_seconds
            if tally.generate_seconds
            else None
        )

    def _beat(self) -> None:
        """Rewrite the heartbeat and the progress file."""
        counts = Counter(self.states.values())
        total = len(self.expansion.runs)
        rate, hours_left = self._rate(), None
        if rate and self.tally.rows:
            per_row = self.tally.reply_tokens / self.tally.rows
            hours_left = self.left * per_row / rate / 3600
        now = _now()
        folder.write_json(
            self.path / folder.HEARTBEAT_FILE,
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "session": self.id,
                "state": self.state,
                "batch": self.call,
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
                "session_batches": self.tally.calls,
                "gpu_hours": (self.used + self.tally.seconds) / 3600,
                "max_gpu_hours": self.spec.max_gpu_hours,
                "reply_tokens_per_second": rate,
                "hours_left": hours_left,
                "updated_at": now,
            },
        )

    def run(self, chosen: list[Call], limit: int | None) -> int:
        """Load the model, run the chosen calls, and say how it ended.

        Returns the exit status (see pushlab.runs.runner).

        Example:
            >>> Session(
            ...     spec, expansion, path, identity, states, 0.0, 50, time.monotonic
            ... ).run(chosen, None)  # doctest: +SKIP
            0
        """
        import torch

        from pushlab.engine import generate as generate_module  # looked up per call
        from pushlab.models import load
        from pushlab.store.environment import environment_info

        spec = self.spec
        last = self.clock()
        self._beat()
        print(
            f"{spec.name}, version {self.version}: {runs_text(self.left)} to run, "
            f"{len(self.states)} of {len(self.expansion.runs)} saved."
        )
        ending, step = "finished", f"load {spec.model}"  # step: for an error
        try:
            device = self.device = load.default_device()
            dtype = spec.decoding["dtype"]
            where = "GPU" if device == "cuda" else "CPU"
            print(f"Loading {spec.model} in {dtype} on the {where}...")
            loading = time.perf_counter()
            self.loaded = load.load_model(
                spec.model, getattr(torch, dtype), device=device
            )
            self.tally.load_seconds = time.perf_counter() - loading
            step = "read the calibration and the prompts"
            inputs = session_inputs(spec, self.loaded.revision, self.identity, chosen)
            step = "write the manifest"
            self.environment = environment_info(device)
            self._warn_if_code_changed()
            self._manifest()
            for number, call in enumerate(chosen):
                step = f"save {call.key}"
                name = shards.next_shard_name(self.path, self.version, call.key)
                outcome = self._run_call(call, name, inputs, generate_module.generate)
                shards.write_shard(self.path, name, outcome.records)
                shards.write_steps(self.path, name, outcome.steps)
                now = self.clock()
                seconds, last = now - last, now
                folder.append_index(
                    self.path, self._entry(name, call, outcome, seconds)
                )
                self._count(call, outcome, seconds)
                if (
                    self.left
                    and self.used + self.tally.seconds >= spec.max_gpu_hours * 3600
                ):
                    ending = "stopped at max_gpu_hours"
                    break
                if number == len(chosen) - 1 and self.left:
                    ending = "stopped at --limit"
        except KeyboardInterrupt:
            ending = "stopped by Ctrl-C"
        except ValueError as error:  # the inputs changed while the model loaded
            ending = f"stopped: {error}"
        except OSError as error:  # the disk or the download, not a run
            ending = f"stopped: cannot {step}: {error}"
        except Exception as error:  # anything else: say where, keep what is saved
            traceback.print_exc()
            ending = f"stopped: cannot {step}: {type(error).__name__}: {error}"
        return self._finish(ending, limit)

    def _run_call(
        self, call: Call, name: str, inputs: Inputs, generate: Any
    ) -> Outcome:
        stem = name.removesuffix(".parquet")
        return run_call(call, stem, self.spec, self.loaded, inputs, generate)

    def _entry(self, name: str, call: Call, outcome: Outcome, seconds: float) -> dict:
        return {
            "version": self.version,
            "session": self.id,
            "shard": name,
            "batch": call.key,
            "runs": {record.run_id: record.status for record in outcome.records},
            "rows": len(call.runs),
            "error": outcome.error,
            "seconds": seconds,
            "generate_seconds": outcome.seconds,
            "reply_tokens": outcome.reply_tokens,
            "saved_at": _now(),
        }

    def _count(self, call: Call, outcome: Outcome, seconds: float) -> None:
        """Count a saved call, print its line and rewrite the progress files."""
        tally, records = self.tally, outcome.records
        ok = sum(record.status == "ok" for record in records)
        tally.runs += len(records)
        tally.ok += ok
        tally.failed += len(records) - ok
        tally.calls += 1
        tally.seconds += seconds
        if outcome.error is None:
            tally.ok_calls += 1
            tally.rows += len(call.runs)
            tally.generate_seconds += outcome.seconds
            tally.reply_tokens += outcome.reply_tokens
        self.states |= {record.run_id: record.status for record in records}
        self.left -= len(records)
        self.call = call.key
        line = (
            f"{call.key}: {ok} ok, {len(records) - ok} failed, "
            f"{outcome.reply_tokens:,} reply tokens in {outcome.seconds:.1f} s; "
            f"{len(self.states):,} of {len(self.expansion.runs):,} runs saved"
        )
        print(line + (f" ({outcome.error})" if outcome.error else ""), flush=True)
        self._beat()

    def _warn_if_code_changed(self) -> None:
        """Say when this version's runs so far, kept ones included, ran other
        code or libraries."""
        assert self.environment is not None
        changed = manifest.code_changes(
            self.path, self.version, self.id, self.environment
        )
        if changed:
            print(
                f"pushlab run: warning: version {self.version}'s runs so far ran "
                f"with another {', '.join(changed)}; this session's runs are made "
                "with the code and libraries as they are now",
                file=sys.stderr,
            )

    def _manifest(self) -> None:
        """Rewrite manifest.json, with this session last in its sessions."""
        if self.environment is None or self.loaded is None:
            return  # stopped before the model loaded: nothing ran to record
        session = {
            "environment": self.environment,
            "id": self.id,
            "started_at": self.started_at,
            "finished_at": _now(),
            "state": self.state,
            "version": self.version,
            "device": self.device,
            "batches": self.tally.calls,
            "runs": self.tally.runs,
            "gpu_hours": self.tally.seconds / 3600,
        }
        manifest.write_experiment_manifest(
            self.path,
            self.spec,
            self.identity,
            self.loaded,
            session,
            self.states,
            len(self.expansion.runs),
        )

    def _save_speed(self) -> None:
        tally, spec = self.tally, self.spec
        rate = self._rate()
        if not rate or self.environment is None:
            return
        estimate.save_speed(
            self.path.parent,
            f"{spec.model}/{spec.decoding['dtype']}",
            {
                "reply_tokens_per_second": rate,
                "reply_tokens": tally.reply_tokens,
                "seconds": tally.generate_seconds,
                "load_seconds": tally.load_seconds,
                "rows_per_batch": tally.rows / tally.ok_calls,
                "batch_size": spec.batch_size,
                "gpu": self.environment["gpu"],
                "experiment": spec.name,
                "measured_at": _now(),
            },
        )

    def _finish(self, ending: str, limit: int | None) -> int:
        """Save the speed and the files, and say how the session ended."""
        spec, tally = self.spec, self.tally
        self.state = ending
        for step in (self._save_speed, self._beat, self._manifest):
            try:
                step()
            except Exception as error:  # say so, and keep what is saved
                print(f"pushlab run: warning: {error}", file=sys.stderr)
        counts = Counter(self.states.values())
        total = len(self.expansion.runs)
        failed = counts["failed"]
        if ending == "stopped by Ctrl-C":
            print(
                "Stopped by Ctrl-C: the batch that was running is not saved. Run "
                "the same command to go on."
            )
            return EXIT_INTERRUPTED
        if ending.startswith("stopped:"):
            return refuse(ending.removeprefix("stopped: "))
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
        elif len(self.states) == total:
            print(
                f"Done: all {total} runs are saved, {counts['ok']} ok and "
                f"{failed} failed, in {self.path}."
            )
        else:
            rest = total - len(self.states)  # only after --retry-failed
            print(
                f"Done: {runs_text(rest)} {'is' if rest == 1 else 'are'} still to "
                "run; the same command without --retry-failed runs "
                f"{'it' if rest == 1 else 'them'}."
            )
        if failed and ending != "stopped at max_gpu_hours":
            what = (
                "1 run failed; rerun it"
                if failed == 1
                else f"{failed} runs failed; rerun them"
            )
            print(f"{what} with --retry-failed.")
        if tally.failed:
            return EXIT_FAILED
        return EXIT_STOPPED if ending.startswith("stopped at") else 0
