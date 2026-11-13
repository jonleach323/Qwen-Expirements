"""A prompt set's answers file: each prompt's exact answer, kept beside the
set (decision 0022).

A prompt set's lines hold exactly prompt_id, version and prompt, so a
model never sees an answer: a set's answers live beside it, in
<set>-answers.jsonl, a line per prompt, as the capability set's do
(prompts/capability100-answers.jsonl):

    {"prompt_id": "capability100-001", "version": 1, "kind": "arithmetic",
     "type": "number", "answer": "123", "also": [], "units": [],
     "check": "48 + 75"}

kind is arithmetic, fact or reasoning, and type number or text. also lists
other accepted spellings; units, the spellings of a number's unit; check,
how the answer was checked: the expression that gives it, or one line on
why it is right. read_answers gives each Answer the numbers its question
gives too, which the grader passes over (pushlab.scoring.capability).
"""

import json
from dataclasses import replace
from pathlib import Path

from pushlab.prompts import PromptSet
from pushlab.scoring.capability import Answer, given_numbers

# An answers file's keys, in the order its lines give them.
ANSWER_KEYS = (
    "prompt_id",
    "version",
    "kind",
    "type",
    "answer",
    "also",
    "units",
    "check",
)


def answers_path(prompt_set_path: Path) -> Path:
    """The answers file of a prompt set: <set>-answers.jsonl beside it.

    Example:
        >>> answers_path(Path("prompts/capability100.jsonl")).name
        'capability100-answers.jsonl'
    """
    path = Path(prompt_set_path)
    return path.with_name(f"{path.stem}-answers.jsonl")


def read_answers(path: Path, prompt_set: PromptSet) -> dict[str, Answer]:
    """Each prompt's answer, from an answers file, in the prompt set's order.

    Every line needs exactly the keys of ANSWER_KEYS, the prompt set's
    version, a prompt of the set that no line before it answered, and an
    answer Answer accepts; every prompt of the set needs its line.
    ValueError names the line of each problem, or the prompts unanswered.

    Example:
        >>> from pushlab.prompts import PROMPTS_DIR, read_prompt_set
        >>> path = PROMPTS_DIR / "capability100.jsonl"
        >>> answers = read_answers(answers_path(path), read_prompt_set(path))
        >>> len(answers), answers["capability100-001"].kind
        (100, 'arithmetic')
    """
    path = Path(path)
    try:
        text = path.read_bytes().decode("utf-8")
    except OSError as error:
        raise ValueError(f"cannot read the answers file {path}: {error}") from None
    except UnicodeDecodeError as error:
        raise ValueError(f"{path} is not UTF-8 text: {error}") from None
    # Only a line feed ends a line, as in a prompt set (pushlab.prompts).
    lines = [line.removesuffix("\r") for line in text.split("\n")]
    if lines[-1] == "":
        lines.pop()  # the file's final line feed
    answers: dict[str, Answer] = {}
    lines_of: dict[str, int] = {}
    questions = dict(zip(prompt_set.prompt_ids, prompt_set.prompts, strict=True))
    for number, line in enumerate(lines, start=1):
        where = f"{path}, line {number}"
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{where} is not JSON: {error}") from None
        if not isinstance(record, dict) or set(record) != set(ANSWER_KEYS):
            raise ValueError(f"{where} needs exactly {', '.join(ANSWER_KEYS)}")
        prompt_id = record["prompt_id"]
        if not isinstance(prompt_id, str) or prompt_id not in prompt_set.prompt_ids:
            raise ValueError(f"{where}: {prompt_id!r} is not a prompt of the set")
        if prompt_id in lines_of:
            raise ValueError(
                f"{where} answers {prompt_id} again, after line {lines_of[prompt_id]}"
            )
        version = record["version"]
        if type(version) is not int or version != prompt_set.version:
            raise ValueError(
                f"{where}: version must be the prompt set's, "
                f"{prompt_set.version}, not {version!r}"
            )
        for key in ("also", "units"):
            if not isinstance(record[key], list):
                raise ValueError(
                    f"{where}: {key} must be a list of spellings, not {record[key]!r}"
                )
        try:
            answer = Answer(
                prompt_id,
                record["kind"],
                record["type"],
                record["answer"],
                tuple(record["also"]),
                tuple(record["units"]),
                record["check"],
            )
            given = given_numbers(questions[prompt_id], answer.units)
            answers[prompt_id] = replace(answer, given=given)
        except ValueError as error:
            raise ValueError(f"{where}: {error}") from None
        lines_of[prompt_id] = number
    if missing := [key for key in prompt_set.prompt_ids if key not in answers]:
        listed = ", ".join(missing[:3])
        if len(missing) > 3:
            listed += f" and {len(missing) - 3} more"
        raise ValueError(f"{path} has no answer for {listed}")
    return {prompt_id: answers[prompt_id] for prompt_id in prompt_set.prompt_ids}
