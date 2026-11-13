"""The names and keys a judge's verdicts are kept under (pushlab.scoring.keys);
test_judge.py checks what the key holds."""

import doctest

from pushlab.scoring import keys
from pushlab.scoring.keys import KEY_VERSION, judge_name, request_key, text_sha256
from pushlab.scoring.rubrics import load_rubric


def test_the_key_is_of_its_version_and_the_whole_request() -> None:
    request = load_rubric("coherence").request("Hi.")
    assert KEY_VERSION == 2
    key = request_key("api", "claude-haiku-5-5", request, "effort=low")
    assert len(key) == 64 and int(key, 16) >= 0
    assert key != request_key("api", "claude-haiku-5-5", request)
    assert judge_name("local", "granite-4.1-8b") == "local:granite-4.1-8b"
    assert text_sha256("Hi.") != text_sha256("Hi")


def test_docstring_examples_run() -> None:
    result = doctest.testmod(keys)
    assert result.attempted > 0
    assert result.failed == 0
