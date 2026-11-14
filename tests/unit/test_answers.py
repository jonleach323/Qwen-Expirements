"""A prompt set's answers file, read beside it (pushlab.scoring.answers).

test_capability.py checks the grader that uses the answers, and
test_capability_set.py the capability set's own answers file.
"""

import doctest
import json
from pathlib import Path

import pytest

from pushlab.prompts import PromptSet
from pushlab.scoring import answers as answers_module
from pushlab.scoring.answers import answers_path, read_answers
from pushlab.scoring.capability import Answer


def test_the_answers_file_sits_beside_its_prompt_set() -> None:
    assert answers_path(Path("prompts/capability100.jsonl")) == Path(
        "prompts/capability100-answers.jsonl"
    )


SET = PromptSet(1, ("q-1", "q-2"), ("What is 17 × 23?", "Australia's capital?"), "0")


def _record(prompt_id: str = "q-1", **changes: object) -> dict:
    record = {
        "prompt_id": prompt_id,
        "version": 1,
        "kind": "arithmetic",
        "type": "number",
        "answer": "391",
        "also": [],
        "units": [],
        "check": "17 * 23",
    }
    record.update(changes)
    return record


CANBERRA_RECORD = _record(
    "q-2", kind="fact", type="text", answer="Canberra", check="Its capital."
)


def _write(tmp_path: Path, records: list) -> Path:
    lines = [json.dumps(record, ensure_ascii=False) for record in records]
    path = tmp_path / "set-answers.jsonl"
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


def test_an_answers_file_gives_each_prompt_its_answer_in_the_sets_order(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, [CANBERRA_RECORD, _record(also=["391.0"], units=["cm"])])
    answers = read_answers(path, SET)
    assert list(answers) == ["q-1", "q-2"]
    # Each answer holds the numbers its question gives, for the grader.
    assert answers["q-1"] == Answer(
        "q-1",
        "arithmetic",
        "number",
        "391",
        ("391.0",),
        ("cm",),
        "17 * 23",
        ("17", "23"),
    )
    assert answers["q-2"].answer == "Canberra"


@pytest.mark.parametrize(
    ("records", "problem"),
    [
        (["not json"], r"line 1 is not JSON"),
        ([CANBERRA_RECORD, ""], r"line 2 is not JSON"),
        ([["q-1", 1]], r"line 1 needs exactly prompt_id, version, kind, type"),
        ([{**_record(), "note": "x"}], r"line 1 needs exactly"),
        (
            [{k: v for k, v in _record().items() if k != "check"}],
            r"line 1 needs exactly",
        ),
        ([_record("q-9")], r"line 1: 'q-9' is not a prompt of the set"),
        ([_record(["q-1"])], r"line 1: \['q-1'\] is not a prompt of the set"),
        ([_record(), _record()], r"line 2 answers q-1 again, after line 1"),
        ([_record(version=2)], r"line 1: version must be the prompt set's, 1, not 2"),
        ([_record(version=True)], r"line 1: version must be"),
        ([_record(kind="maths")], r"line 1: kind must be arithmetic, fact or"),
        ([_record(type="word")], r"line 1: type must be number or text, not 'word'"),
        ([_record(answer="")], r"line 1: answer must be non-empty text"),
        ([_record(answer=391)], r"line 1: answer must be non-empty text, not 391"),
        ([_record(answer="**")], r"line 1: answer '\*\*' is empty once normalized"),
        ([_record(answer="391 cm")], r"line 1: a number answer must be a number alone"),
        ([_record(answer="three")], r"line 1: a number .* not 'three'"),
        ([_record(also="391")], r"line 1: also must be a list of spellings"),
        ([_record(also=["390"])], r"line 1: also spelling '390' is not the number"),
        ([_record(also=[""])], r"line 1: also must hold non-empty spellings"),
        ([_record(units=[3])], r"line 1: units must hold non-empty spellings"),
        ([_record(check=" ")], r"line 1: check must be non-empty text"),
        ([{**CANBERRA_RECORD, "units": ["km"]}], r"line 1: a text answer has no"),
    ],
)
def test_a_broken_answers_file_is_refused_naming_the_line(
    tmp_path: Path, records: list, problem: str
) -> None:
    path = tmp_path / "set-answers.jsonl"
    lines = [r if isinstance(r, str) else json.dumps(r) for r in records]
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    with pytest.raises(ValueError, match=rf"set-answers\.jsonl, {problem}"):
        read_answers(path, SET)


def test_an_answers_file_must_answer_every_prompt(tmp_path: Path) -> None:
    path = _write(tmp_path, [_record()])
    with pytest.raises(ValueError, match=r"set-answers\.jsonl has no answer for q-2$"):
        read_answers(path, SET)
    many = PromptSet(1, tuple(f"p-{n}" for n in range(5)), ("Q?",) * 5, "0")
    with pytest.raises(ValueError, match=r"no answer for p-0, p-1, p-2 and 2 more$"):
        read_answers(_write(tmp_path, []), many)


def test_an_unreadable_answers_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot read the answers file"):
        read_answers(tmp_path / "none-answers.jsonl", SET)
    path = tmp_path / "set-answers.jsonl"
    path.write_bytes(b"\xff\xfe\n")
    with pytest.raises(ValueError, match="is not UTF-8 text"):
        read_answers(path, SET)


def test_windows_line_endings_are_accepted(tmp_path: Path) -> None:
    lines = [json.dumps(_record()), json.dumps(CANBERRA_RECORD)]
    path = tmp_path / "set-answers.jsonl"
    path.write_bytes("\r\n".join([*lines, ""]).encode())
    assert list(read_answers(path, SET)) == ["q-1", "q-2"]


def test_docstring_examples_run() -> None:
    result = doctest.testmod(answers_module)
    assert result.attempted > 0
    assert result.failed == 0
