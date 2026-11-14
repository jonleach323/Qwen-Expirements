"""The judge's cache: every verdict that parsed, by its key.

One JSON Lines file per judge, <cache dir>/<backend>-<model>.jsonl, each
line a verdict: its key (pushlab.scoring.keys.request_key, a SHA-256 of the
key's version, the backend, the model, the backend's settings, the rubric
and its version, and the whole request, the text included); its label,
value, reason and raw answer, each of which a line must have; and the
rubric, its version and when it was found. Lines are only ever added. A
line a crash cut short, or any line that is not such a verdict, is skipped
when the file is read, so the verdict is asked for again; and the judge
reads a cached raw answer again by its rubric before it uses it
(Judge.verdict). The cache holds the judge's reasons, which may quote
replies: it lives on the pod's volume, never in git.
"""

import json
import math
import os
import re
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# The pod's volume; PUSHLAB_CACHE_DIR names another.
CACHE_ROOT = Path("/workspace/cache")
KEY = re.compile(r"[0-9a-f]{64}")


def default_cache_dir() -> Path:
    """The folder of the judge's cache files: $PUSHLAB_CACHE_DIR/judge, else
    /workspace/cache/judge.

    Example:
        >>> default_cache_dir()  # doctest: +SKIP
        PosixPath('/workspace/cache/judge')
    """
    return Path(os.environ.get("PUSHLAB_CACHE_DIR") or CACHE_ROOT) / "judge"


def cache_file_name(judge: str) -> str:
    """The file a judge's verdicts go to, named after it.

    Example:
        >>> cache_file_name("local:ibm-granite/granite-4.1-8b")
        'local-ibm-granite_granite-4.1-8b.jsonl'
    """
    backend, _, model = judge.partition(":")
    return re.sub(r"[^A-Za-z0-9._-]", "_", f"{backend}-{model}") + ".jsonl"


def _verdict(line: bytes) -> dict[str, Any] | None:
    """The verdict on a line of the file, or None for one that is not."""
    try:
        entry = json.loads(line)
    except (ValueError, RecursionError):  # not JSON, or nested too deep
        return None
    if not (
        isinstance(entry, dict)
        and {"key", "label", "value", "reason", "raw"} <= set(entry)
        and isinstance(entry["key"], str)
        and KEY.fullmatch(entry["key"])
        and isinstance(entry["reason"], str)
        and isinstance(entry["raw"], str)
        and (entry["label"] is None or isinstance(entry["label"], str))
        and (
            entry["value"] is None
            or type(entry["value"]) is int  # any size: never turned into a float
            or (type(entry["value"]) is float and math.isfinite(entry["value"]))
        )
        and (entry["label"] is not None or entry["value"] is not None)
    ):
        return None
    return entry


class VerdictCache:
    """A judge's verdicts, read from its file and added to it.

    Safe to share between threads. skipped counts the lines read that were
    not verdicts, such as one a crash cut short.

    Example:
        >>> cache = VerdictCache(default_cache_dir(), "api:m")  # doctest: +SKIP
        >>> cache.get("0" * 64) is None  # doctest: +SKIP
        True
    """

    def __init__(self, folder: Path, judge: str) -> None:
        self.path = Path(folder) / cache_file_name(judge)
        self.skipped = 0
        self._lock = threading.Lock()
        self._entries: dict[str, dict[str, Any]] = {}
        self._torn = False  # whether the file ends without a line feed
        try:
            content = self.path.read_bytes()
        except FileNotFoundError:
            return
        self._torn = bool(content) and not content.endswith(b"\n")
        for line in content.split(b"\n"):
            if not line.strip():
                continue
            entry = _verdict(line)
            if entry is None:
                self.skipped += 1
            else:
                self._entries[entry["key"]] = entry

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, key: object) -> bool:
        return key in self._entries

    def get(self, key: str) -> dict[str, Any] | None:
        """The verdict cached under key, or None."""
        return self._entries.get(key)

    def put(self, key: str, verdict: dict[str, Any]) -> None:
        """Cache a verdict that parsed: add its line to the file and keep it.

        verdict holds its label, value, reason and raw answer, and may hold
        more, such as the rubric; the time it was found is added.
        """
        found = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        entry = {"key": key} | verdict | {"found_at": found}
        if _verdict(json.dumps(entry).encode()) is None:
            raise ValueError(f"not a verdict to cache: {entry!r}")
        line = json.dumps(entry, ensure_ascii=False, sort_keys=True, allow_nan=False)
        data = (line + "\n").encode("utf-8")
        with self._lock:
            if self._torn:  # end a line a crash cut short before adding one
                data = b"\n" + data
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                while data:
                    data = data[os.write(handle, data) :]
            finally:
                os.close(handle)
            self._torn = False
            self._entries[key] = entry
