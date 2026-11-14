"""The golden fingerprints file (decision 0023): a recording reads back as
written and refuses what is broken, and a new run is checked against it,
naming the cases that changed since, each reply that changed, and what of
the environment differs. The recordings here hold made-up replies; a real
one, made on the tiny model, is test_golden_command.py's.
"""

import importlib.util
import json
import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import pytest

from pushlab.engine.generate import Generation
from pushlab.runs.golden import fingerprint
from pushlab.runs.golden_cases import GoldenCases, cases_from_mapping
from pushlab.runs.golden_fingerprints import (
    Recording,
    changed_cases,
    environment_changes,
    failure_text,
    make_recording,
    not_recorded,
    read_recording,
    recording_problem,
    recording_text,
)

# The made-up cases, loaded by path, as test modules import nothing from
# one another.
_SPEC = importlib.util.spec_from_file_location(
    "golden_helpers", Path(__file__).with_name("golden_helpers.py")
)
helpers = importlib.util.module_from_spec(_SPEC)  # type: ignore[arg-type]
_SPEC.loader.exec_module(helpers)  # type: ignore[union-attr]
set_key, drop_key = helpers.set_key, helpers.drop_key


def _recording(cases: GoldenCases) -> Recording:
    """A recording of made-up replies: case i's is tokens i, 7 and 9."""
    found = {
        case.case_id: fingerprint(
            Generation(f"reply {i}", (i, 7, 9), "end_of_turn", ())
        )
        for i, case in enumerate(cases.cases)
    }
    environment = {"torch": "2.9.0", "packages": {"numpy": "2.3.0"}, "gpu": None}
    moment = datetime(2026, 11, 13, 21, 0, tzinfo=UTC)
    where = Path("/no/such/cases.yaml")
    return make_recording(cases, where, "abc", environment, found, moment)


@pytest.fixture
def cases() -> GoldenCases:
    return cases_from_mapping(helpers.golden_data())


def test_a_recording_reads_back_as_it_was_written(
    cases: GoldenCases, tmp_path: Path
) -> None:
    recording = _recording(cases)
    path = tmp_path / "golden.json"
    path.write_text(recording_text(recording), encoding="utf-8")
    assert read_recording(path) == recording
    data = json.loads(path.read_text(encoding="utf-8"))
    assert list(data) == [
        "version",
        "model",
        "revision",
        "dtype",
        "recorded_at",
        "cases_file",
        "environment",
        "cases",
    ]
    assert (data["version"], data["model"], data["revision"], data["dtype"]) == (
        1,
        "tiny",
        "abc",
        "float32",
    )
    assert data["recorded_at"] == "2026-11-13T21:00:00+00:00"
    assert data["cases_file"] == "/no/such/cases.yaml"
    assert list(data["cases"]) == helpers.IDS
    greedy = data["cases"]["hi-greedy"]
    assert greedy["case"] == cases.cases[0].as_data()
    assert greedy["fingerprint"]["first_tokens"] == [0, 7, 9]
    assert greedy["fingerprint"]["text_start"] == "reply 0"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (set_key("version", 2), "version must be 1"),
        (set_key("revision", ""), "revision must be text"),
        (set_key("cases.hi-greedy.fingerprint.sha256", "ab"), "must be a SHA-256"),
        (set_key("cases.hi-greedy.fingerprint.token_count", 0), "a whole number"),
        (set_key("cases.hi-greedy.fingerprint.first_tokens", [0]), "first_tokens"),
        (drop_key("cases.hi-greedy.case"), "lacks case"),
        (set_key("environment", []), "environment must be a mapping"),
        (set_key("cases", {}), "cases must map each case's ID to its record"),
    ],
)
def test_a_broken_recording_is_refused(
    cases: GoldenCases,
    tmp_path: Path,
    change: Callable[[dict], None],
    message: str,
) -> None:
    data = json.loads(recording_text(_recording(cases)))
    change(data)
    path = tmp_path / "golden.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(message)):
        read_recording(path)


def test_a_recording_that_is_not_json_or_missing_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "golden.json"
    with pytest.raises(ValueError, match="cannot read the golden fingerprints"):
        read_recording(path)
    path.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="golden.json is not valid JSON"):
        read_recording(path)


def test_cases_changed_since_the_recording_are_named(cases: GoldenCases) -> None:
    recording = _recording(cases)
    assert recording_problem(recording, cases) is None
    data = helpers.golden_data()
    data["dtype"] = "bfloat16"
    problem = recording_problem(recording, cases_from_mapping(data))
    assert problem.startswith(
        "the fingerprints are of tiny in float32, but the cases are of tiny in "
        "bfloat16: record the fingerprints again"
    )
    data = helpers.golden_data()
    data["cases"][1]["push"]["alpha"] = 0.5  # changed
    data["cases"][2]["id"] = "hi-neuron-own"  # added, and hi-neuron-per-prompt gone
    problem = recording_problem(recording, cases_from_mapping(data))
    assert problem.startswith(
        "the cases differ from the recorded ones (not recorded: hi-neuron-own; no "
        "longer in the cases: hi-neuron-per-prompt; changed since: hi-neuron-fixed)"
        ": record the fingerprints again with `pushlab golden record`"
    )


def test_a_changed_reply_is_caught_with_what_changed(cases: GoldenCases) -> None:
    recording = _recording(cases)
    found = {case_id: item.fingerprint for case_id, item in recording.cases.items()}
    assert changed_cases(recording, cases, found) == []
    other = fingerprint(Generation("other", (5,), "end_of_turn", ()))
    (line,) = changed_cases(recording, cases, found | {"hi-set-per-prompt": other})
    assert line == (
        "hi-set-per-prompt: recorded 3 tokens starting [3, 7, 9], 'reply 3'; now "
        "1 token starting [5], 'other'"
    )


def test_what_differs_in_the_environment_is_named_setting_by_setting(
    cases: GoldenCases,
) -> None:
    recording = _recording(cases)
    assert environment_changes(recording, "abc", recording.environment) == []
    now = {"torch": "2.9.1", "packages": {"numpy": "2.3.0", "triton": "3.5"}}
    assert environment_changes(recording, "def", now) == [
        "revision: abc when recorded, def now",
        "torch: 2.9.0 when recorded, 2.9.1 now",
        "gpu: None when recorded, absent now",
        "packages.triton: absent when recorded, 3.5 now",
    ]


def test_the_failure_lists_each_change_then_the_environment() -> None:
    changed = ["hi-greedy: recorded ...; now ..."]
    text = failure_text(changed, ["torch: 2.9.0 when recorded, 2.9.1 now"], 6)
    assert text.splitlines()[:4] == [
        "1 of the 6 golden replies changed since they were recorded:",
        "- hi-greedy: recorded ...; now ...",
        "What differs from the environment they were recorded in:",
        "- torch: 2.9.0 when recorded, 2.9.1 now",
    ]
    assert "with a decision note" in text
    assert "something is not deterministic" in failure_text(changed, [], 6)


def test_a_missing_recording_says_how_to_make_it(tmp_path: Path) -> None:
    why = not_recorded(tmp_path / "qwen3-8b.json", "qwen3-8b")
    assert "is not recorded yet" in why
    assert "uv run pushlab golden record --model qwen3-8b --out " in why
    assert "as tests/golden/qwen3-8b.json through a pull request" in why
    (tmp_path / "qwen3-8b.json").write_text("{}", encoding="utf-8")
    assert not_recorded(tmp_path / "qwen3-8b.json", "qwen3-8b") is None
