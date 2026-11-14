"""The golden fingerprints file: each case's fingerprint and the environment
that made it, and how a new run compares with it (decision 0023).

`pushlab golden record` writes it on the pod, outside the repository, as
JSON (pushlab.runs.golden_command): the format's version, the model, its
revision and data type, when it was recorded and from which cases file,
the environment (pushlab.store.environment.environment_info, with the
fixed calibration's SHA-256), and by case ID each case as plain data with
its reply's fingerprint (pushlab.runs.golden). The owner brings it in as
tests/golden/<model>.json through a pull request, and
tests/gpu/test_golden_qwen3_8b.py checks the model against it:
recording_problem says when the cases changed since, changed_cases which
replies changed, and environment_changes what of the environment differs.
"""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from pushlab.runs.golden import FIRST_TOKENS, Fingerprint
from pushlab.runs.golden_cases import GOLDEN_DIR, GoldenCases, _mapping
from pushlab.store.environment import environment_info
from pushlab.store.folders import shown_path

RECORDING_FORMAT = 1
SHA256 = re.compile(r"[0-9a-f]{64}")
AGAIN = (
    "record the fingerprints again with `pushlab golden record` and bring the "
    'new file in through a pull request (docs/runbooks/pod.md, "Golden tests")'
)
_ABSENT = object()  # a setting one environment records and the other lacks


@dataclass(frozen=True)
class RecordedCase:
    """One case as it was recorded: its plain data (GoldenCase.as_data) and
    its reply's fingerprint."""

    case: dict[str, Any]
    fingerprint: Fingerprint


@dataclass(frozen=True)
class Recording:
    """A whole fingerprints file, as the module's docstring describes it;
    cases are by case ID, in the cases file's order."""

    model: str
    revision: str
    dtype: str
    recorded_at: str
    cases_file: str
    environment: dict[str, Any]
    cases: dict[str, RecordedCase]


def fingerprints_path(model: str) -> Path:
    """Where the repository keeps a model's golden fingerprints.

    Example:
        >>> fingerprints_path("qwen3-8b").name
        'qwen3-8b.json'
    """
    return GOLDEN_DIR / f"{model}.json"


def norms_sha256(norms: Sequence[float] | None) -> str | None:
    """The SHA-256 of a calibration's sizes as compact JSON; None for none.

    Example:
        >>> norms_sha256([7.0, 10.0])[:16], norms_sha256(None)
        ('359952069b873b2c', None)
    """
    if norms is None:
        return None
    text = json.dumps([float(norm) for norm in norms], separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def golden_environment(
    device: torch.device | str, fixed: Sequence[float] | None
) -> dict[str, Any]:
    """What a recording keeps of the code and the machine: environment_info's
    record, and fixed_calibration_sha256, the fixed calibration's sizes'
    SHA-256 (None when no case takes them).

    Example:
        >>> environment = golden_environment("cpu", [7.0, 10.0])
        >>> environment["gpu"], environment["fixed_calibration_sha256"][:16]
        (None, '359952069b873b2c')
    """
    return {
        **environment_info(device),
        "fixed_calibration_sha256": norms_sha256(fixed),
    }


def make_recording(
    cases: GoldenCases,
    cases_file: Path,
    revision: str,
    environment: Mapping[str, Any],
    found: Mapping[str, Fingerprint],
    recorded_at: datetime | None = None,
) -> Recording:
    """The recording of a run of every case: found holds each reply's
    fingerprint by case ID (pushlab.runs.golden.run_cases).

    Example:
        >>> found = run_cases(
        ...     loaded.model, loaded.tokenizer, cases, fixed
        ... )  # doctest: +SKIP
        >>> make_recording(
        ...     cases, path, revision, environment, found
        ... ).dtype  # doctest: +SKIP
        'bfloat16'
    """
    moment = recorded_at or datetime.now(UTC)
    return Recording(
        model=cases.model,
        revision=revision,
        dtype=cases.dtype,
        recorded_at=moment.isoformat(timespec="seconds"),
        cases_file=shown_path(Path(cases_file)),
        environment=dict(environment),
        cases={
            case.case_id: RecordedCase(case.as_data(), found[case.case_id])
            for case in cases.cases
        },
    )


def recording_mapping(recording: Recording) -> dict[str, Any]:
    """The recording as plain data, as the file holds it.

    Example:
        >>> hi = RecordedCase({"prompt": "Hi"}, Fingerprint("0" * 64, 1, (5,), "Hey"))
        >>> recording = Recording("tiny", "abc", "float32", "", "", {}, {"hi": hi})
        >>> recording_mapping(recording)["cases"]["hi"]["fingerprint"]["first_tokens"]
        [5]
    """
    return {
        "version": RECORDING_FORMAT,
        "model": recording.model,
        "revision": recording.revision,
        "dtype": recording.dtype,
        "recorded_at": recording.recorded_at,
        "cases_file": recording.cases_file,
        "environment": recording.environment,
        "cases": {
            case_id: {
                "case": recorded.case,
                "fingerprint": asdict(recorded.fingerprint)
                | {"first_tokens": list(recorded.fingerprint.first_tokens)},
            }
            for case_id, recorded in recording.cases.items()
        },
    }


def _text(where: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be text, not {value!r}")
    return value


def _fingerprint(where: str, value: object) -> Fingerprint:
    keys = ("sha256", "token_count", "first_tokens", "text_start")
    data = _mapping(where, value, keys)
    sha256, count, first = data["sha256"], data["token_count"], data["first_tokens"]
    if not isinstance(sha256, str) or not SHA256.fullmatch(sha256):
        raise ValueError(f"{where}.sha256 must be a SHA-256 in lowercase hex")
    if type(count) is not int or count < 1:
        raise ValueError(f"{where}.token_count must be a whole number from 1")
    if (
        not isinstance(first, list)
        or len(first) != min(count, FIRST_TOKENS)
        or not all(type(token) is int and token >= 0 for token in first)
    ):
        raise ValueError(
            f"{where}.first_tokens must list the reply's first token IDs, "
            f"{min(count, FIRST_TOKENS)} of them"
        )
    if not isinstance(data["text_start"], str):
        raise ValueError(f"{where}.text_start must be text")
    return Fingerprint(sha256, count, tuple(first), data["text_start"])


def recording_from_mapping(
    data: object, where: str = "the golden fingerprints"
) -> Recording:
    """A recording from plain data, every field checked; ValueError names
    the first problem.

    Example:
        >>> hi = RecordedCase({"prompt": "Hi"}, Fingerprint("0" * 64, 1, (5,), "Hey"))
        >>> recording = Recording("tiny", "abc", "float32", "now", "c", {}, {"hi": hi})
        >>> recording_from_mapping(recording_mapping(recording)) == recording
        True
        >>> recording_from_mapping({"version": 2})  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: the golden fingerprints lacks model, revision, ...
    """
    keys = ("version", "model", "revision", "dtype", "recorded_at", "cases_file")
    data = _mapping(where, data, (*keys, "environment", "cases"))
    if type(data["version"]) is not int or data["version"] != RECORDING_FORMAT:
        raise ValueError(f"{where}: version must be {RECORDING_FORMAT}")
    texts = {key: _text(f"{where}.{key}", data[key]) for key in keys[1:]}
    if not isinstance(data["environment"], dict):
        raise ValueError(f"{where}.environment must be a mapping")
    if not isinstance(data["cases"], dict) or not data["cases"]:
        raise ValueError(f"{where}.cases must map each case's ID to its record")
    cases = {}
    for case_id, value in data["cases"].items():
        here = f"{where}.cases.{case_id}"
        item = _mapping(here, value, ("case", "fingerprint"))
        if not isinstance(item["case"], dict):
            raise ValueError(f"{here}.case must be a mapping")
        found = _fingerprint(f"{here}.fingerprint", item["fingerprint"])
        cases[case_id] = RecordedCase(item["case"], found)
    return Recording(**texts, environment=data["environment"], cases=cases)


def recording_text(recording: Recording) -> str:
    """The recording as the file's JSON text, checked to read back as it is.

    Example:
        >>> recording_text(make_recording(...))  # doctest: +SKIP
        '{\\n  "version": 1,\\n  "model": "qwen3-8b",\\n ...'
    """
    data = recording_mapping(recording)
    text = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if recording_from_mapping(json.loads(text)) != recording:  # a bug, never yours
        raise ValueError("the fingerprints do not read back as written")
    return text


def read_recording(path: Path) -> Recording:
    """The fingerprints file at path, checked; ValueError says what is wrong.

    Example:
        >>> read_recording(fingerprints_path("qwen3-8b")).model  # doctest: +SKIP
        'qwen3-8b'
    """
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        why = error.strerror or error
        raise ValueError(f"cannot read the golden fingerprints {path}: {why}") from None
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from None
    return recording_from_mapping(data, str(path))


def not_recorded(path: Path, model: str) -> str | None:
    """Why the golden test cannot run: its fingerprints file is missing, and
    how to make it; None once the file is there.

    Example:
        >>> why = not_recorded(Path("/no/such/qwen3-8b.json"), "qwen3-8b")
        >>> why.split(": ")[0], "pushlab golden record --model qwen3-8b" in why
        ('/no/such/qwen3-8b.json is not recorded yet', True)
    """
    if Path(path).exists():
        return None
    return (
        f"{shown_path(Path(path))} is not recorded yet: on the pod, run `uv run "
        f"pushlab golden record --model {model} --out /workspace/golden/{model}.json`"
        f", then bring the file in as tests/golden/{model}.json through a pull "
        'request (docs/runbooks/pod.md, "Golden tests")'
    )


def recording_problem(recording: Recording, cases: GoldenCases) -> str | None:
    """Why the recording cannot be checked against the cases: it is of
    another model or data type, or cases were added, removed or changed
    since it was made. None when every case is as it was recorded."""
    if (recording.model, recording.dtype) != (cases.model, cases.dtype):
        return (
            f"the fingerprints are of {recording.model} in {recording.dtype}, but "
            f"the cases are of {cases.model} in {cases.dtype}: {AGAIN}"
        )
    now = {case.case_id: case.as_data() for case in cases.cases}
    parts = []
    if added := [case_id for case_id in now if case_id not in recording.cases]:
        parts.append(f"not recorded: {', '.join(added)}")
    if gone := [case_id for case_id in recording.cases if case_id not in now]:
        parts.append(f"no longer in the cases: {', '.join(gone)}")
    changed = [
        case_id
        for case_id, data in now.items()
        if case_id in recording.cases and recording.cases[case_id].case != data
    ]
    if changed:
        parts.append(f"changed since: {', '.join(changed)}")
    if not parts:
        return None
    return f"the cases differ from the recorded ones ({'; '.join(parts)}): {AGAIN}"


def _described(found: Fingerprint) -> str:
    tokens = "1 token" if found.token_count == 1 else f"{found.token_count} tokens"
    return f"{tokens} starting {list(found.first_tokens)}, {found.text_start!r}"


def changed_cases(
    recording: Recording, cases: GoldenCases, found: Mapping[str, Fingerprint]
) -> list[str]:
    """Each case whose reply's fingerprint differs from the recorded one, in
    words and in the cases' order; [] when none does. A case the recording
    lacks is recording_problem's."""
    lines = []
    for case in cases.cases:
        recorded = recording.cases.get(case.case_id)
        if recorded is not None and recorded.fingerprint != found[case.case_id]:
            lines.append(
                f"{case.case_id}: recorded {_described(recorded.fingerprint)}; now "
                f"{_described(found[case.case_id])}"
            )
    return lines


def _flat(data: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Nested settings as one mapping, such as packages.torch."""
    flat: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, Mapping):
            flat |= _flat(value, f"{prefix}{key}.")
        else:
            flat[f"{prefix}{key}"] = value
    return flat


def environment_changes(
    recording: Recording, revision: str, environment: Mapping[str, Any]
) -> list[str]:
    """What differs between the environment recorded and this one, with the
    model's revision: one line per setting, in words.

    Example:
        >>> then = Recording("m", "abc", "bfloat16", "", "", {"torch": "2.9.0"}, {})
        >>> environment_changes(then, "abc", {"torch": "2.9.1"})
        ['torch: 2.9.0 when recorded, 2.9.1 now']
    """
    then = {"revision": recording.revision, **_flat(recording.environment)}
    now = {"revision": revision, **_flat(environment)}
    lines = []
    for key in dict.fromkeys([*then, *now]):
        old, new = then.get(key, _ABSENT), now.get(key, _ABSENT)
        if old != new:
            shown = ["absent" if v is _ABSENT else str(v) for v in (old, new)]
            lines.append(f"{key}: {shown[0]} when recorded, {shown[1]} now")
    return lines


def failure_text(changed: Sequence[str], differences: Sequence[str], total: int) -> str:
    """The golden test's failure: each changed case, then what of the
    environment differs, then what to do.

    Example:
        >>> changed = ["kettle-greedy: recorded ...; now ..."]
        >>> text = failure_text(changed, ["torch: 2.9.0 when recorded, 2.9.1 now"], 18)
        >>> print("\\n".join(text.splitlines()[:4]))
        1 of the 18 golden replies changed since they were recorded:
        - kettle-greedy: recorded ...; now ...
        What differs from the environment they were recorded in:
        - torch: 2.9.0 when recorded, 2.9.1 now
    """
    lines = [
        f"{len(changed)} of the {total} golden replies changed since they were "
        "recorded:",
        *(f"- {line}" for line in changed),
    ]
    if differences:
        lines.append("What differs from the environment they were recorded in:")
        lines += [f"- {line}" for line in differences]
    else:
        lines.append(
            "Nothing recorded about the environment differs: the same code, "
            "libraries and GPU gave other replies, so something is not "
            "deterministic."
        )
    lines.append(
        "If the change is meant, such as a new torch, record the fingerprints "
        "again and bring them in through a pull request, with a decision note "
        '(docs/runbooks/pod.md, "Golden tests").'
    )
    return "\n".join(lines)
