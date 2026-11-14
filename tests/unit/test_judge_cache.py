"""The judge's cache (pushlab.scoring.cache): verdicts by key, one append-only
JSON Lines file per judge."""

import doctest
import json
from pathlib import Path

import pytest

from pushlab.scoring import cache as cache_module
from pushlab.scoring.cache import VerdictCache, cache_file_name, default_cache_dir

KEY = "a" * 64
OTHER = "b" * 64
RAW = '{"reason": "Hoots.", "label": "speaks_as"}'
VERDICT = {"label": "speaks_as", "value": None, "reason": "Hoots.", "raw": RAW}


def test_a_verdict_put_is_found_now_and_by_a_later_reader(tmp_path: Path) -> None:
    cache = VerdictCache(tmp_path, "api:claude-haiku-5-5")
    assert cache.get(KEY) is None and len(cache) == 0
    cache.put(KEY, VERDICT | {"rubric": "speaks_as", "version": 1})
    assert cache.get(KEY)["label"] == "speaks_as" and KEY in cache
    again = VerdictCache(tmp_path, "api:claude-haiku-5-5")
    entry = again.get(KEY)
    assert entry is not None and entry["reason"] == "Hoots."
    assert entry["found_at"].endswith("Z") and again.skipped == 0
    assert cache.path == tmp_path / "api-claude-haiku-5-5.jsonl"


def test_each_judge_has_its_own_file(tmp_path: Path) -> None:
    VerdictCache(tmp_path, "api:claude-haiku-5-5").put(KEY, VERDICT)
    assert VerdictCache(tmp_path, "local:granite-4.1-8b").get(KEY) is None
    assert cache_file_name("local:ibm-granite/granite-4.1-8b") == (
        "local-ibm-granite_granite-4.1-8b.jsonl"
    )


def test_a_torn_line_is_skipped_and_the_next_verdict_starts_a_line(
    tmp_path: Path,
) -> None:
    cache = VerdictCache(tmp_path, "api:m")
    cache.put(KEY, VERDICT)
    with cache.path.open("a", encoding="utf-8") as file:
        file.write('{"key": "' + OTHER + '", "lab')  # a crash mid-line
    reread = VerdictCache(tmp_path, "api:m")
    assert reread.get(OTHER) is None and reread.skipped == 1
    seven = '{"reason": "Fine.", "score": 7}'
    reread.put(OTHER, {"label": None, "value": 7.0, "reason": "Fine.", "raw": seven})
    lines = cache.path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[-1])["key"] == OTHER  # on a line of its own
    final = VerdictCache(tmp_path, "api:m")
    assert final.get(OTHER)["value"] == 7.0 and final.get(KEY) is not None


@pytest.mark.parametrize(
    "line",
    [
        "not json",
        "[1, 2]",
        '{"key": "short", "label": "x", "value": null, "reason": ""}',
        '{"key": "' + OTHER + '", "label": null, "value": null, "reason": ""}',
        '{"key": "' + OTHER + '", "label": 3, "value": null, "reason": ""}',
        # Each of label, value and raw must be there, and a value no bool.
        '{"key": "' + OTHER + '", "value": 7, "reason": "", "raw": "{}"}',
        '{"key": "' + OTHER + '", "label": "x", "reason": "", "raw": "{}"}',
        '{"key": "' + OTHER + '", "label": "x", "value": null, "reason": ""}',
        '{"key": "' + OTHER + '", "label": null, "value": true, "reason": "", '
        '"raw": "{}"}',
        # Not JSON that Python can read: nested too deep.
        "[" * 100_000 + "]" * 100_000,
    ],
)
def test_a_line_that_is_no_verdict_is_skipped(tmp_path: Path, line: str) -> None:
    path = tmp_path / cache_file_name("api:m")
    path.write_text(line + "\n", encoding="utf-8")
    cache = VerdictCache(tmp_path, "api:m")
    assert len(cache) == 0 and cache.skipped == 1


def test_a_verdict_with_neither_label_nor_value_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not a verdict to cache"):
        VerdictCache(tmp_path, "api:m").put(
            KEY, {"label": None, "value": None, "reason": "", "raw": "{}"}
        )


def test_the_cache_lives_under_the_cache_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PUSHLAB_CACHE_DIR", raising=False)
    assert default_cache_dir() == Path("/workspace/cache/judge")
    monkeypatch.setenv("PUSHLAB_CACHE_DIR", "/tmp/somewhere")
    assert default_cache_dir() == Path("/tmp/somewhere/judge")


def test_docstring_examples_run() -> None:
    result = doctest.testmod(cache_module)
    assert result.attempted > 0
    assert result.failed == 0


def test_a_value_too_big_for_a_float_is_read_not_a_crash(tmp_path: Path) -> None:
    # A whole number of 309 digits passes a float's range; it is kept as it
    # is, and the judge reads the raw answer again before using it anyway.
    big = "9" * 309
    path = tmp_path / cache_file_name("api:m")
    line = f'{{"key": "{OTHER}", "label": null, "value": {big}, "reason": "", '
    path.write_text(line + '"raw": "{}"}\n', encoding="utf-8")
    cache = VerdictCache(tmp_path, "api:m")
    assert cache.skipped == 0 and cache.get(OTHER)["value"] == int(big)
