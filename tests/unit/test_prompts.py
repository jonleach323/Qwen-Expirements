"""Prompt sets: the JSON Lines files in prompts/, read by pushlab.prompts."""

import doctest
import hashlib
from pathlib import Path

import pytest

from pushlab import PROJECT_ROOT, prompts
from pushlab.prompts import PROMPTS_DIR, read_prompt_set

GOOD = '{"prompt_id": "a", "version": 1, "prompt": "Hi"}'


def test_a_prompt_set_keeps_its_prompts_in_order_with_the_files_hash(
    tmp_path: Path,
) -> None:
    path = tmp_path / "prompts.jsonl"
    path.write_text(GOOD + '\n{"prompt_id": "b", "version": 1, "prompt": "Bye"}\n')
    prompt_set = read_prompt_set(path)
    assert prompt_set.version == 1
    assert prompt_set.prompt_ids == ("a", "b")
    assert prompt_set.prompts == ("Hi", "Bye")
    assert prompt_set.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_prompt_sets_live_in_the_repositorys_prompts_folder() -> None:
    assert PROMPTS_DIR == PROJECT_ROOT / "prompts"


@pytest.mark.parametrize(
    ("lines", "problem"),
    [
        ([], "no prompts"),
        (["not json"], "line 1 is not JSON"),
        ([GOOD, ""], "line 2 is not JSON"),  # a blank line
        (['["a", 1, "Hi"]'], "exactly"),
        (['{"prompt_id": "a", "version": 1}'], "exactly"),
        (['{"prompt_id": "a", "version": 1, "prompt": "Hi", "note": "x"}'], "exactly"),
        ([GOOD, GOOD], "used twice"),
        ([GOOD, '{"prompt_id": "b", "version": 2, "prompt": "Yo"}'], "same version"),
        (['{"prompt_id": "a", "version": 0, "prompt": "Hi"}'], "version"),
        (['{"prompt_id": "a", "version": true, "prompt": "Hi"}'], "version"),
        (['{"prompt_id": "a", "version": "1", "prompt": "Hi"}'], "version"),
        (['{"prompt_id": "a", "version": 1, "prompt": " "}'], "prompt must be"),
        (['{"prompt_id": "", "version": 1, "prompt": "Hi"}'], "prompt_id must be"),
    ],
)
def test_a_broken_prompt_set_is_refused(
    tmp_path: Path, lines: list[str], problem: str
) -> None:
    path = tmp_path / "prompts.jsonl"
    path.write_text("".join(line + "\n" for line in lines))
    with pytest.raises(ValueError, match=problem):
        read_prompt_set(path)


def test_a_prompt_set_that_is_not_utf8_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "prompts.jsonl"
    path.write_bytes(b"\xff\xfe\n")
    with pytest.raises(ValueError, match="UTF-8"):
        read_prompt_set(path)


def test_docstring_examples_run() -> None:
    result = doctest.testmod(prompts)
    assert result.attempted > 0
    assert result.failed == 0
