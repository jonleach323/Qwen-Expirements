"""Prompt sets: versioned lists of prompts in the repository's prompts/ folder.

A prompt set is a JSON Lines file with one prompt per line. Each line is an
object with the prompt's ID, the set's version and the prompt itself:

    {"prompt_id": "calibration-01", "version": 1, "prompt": "How do I ...?"}
"""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from pushlab import PROJECT_ROOT

PROMPTS_DIR = PROJECT_ROOT / "prompts"
KEYS = ("prompt_id", "version", "prompt")


@dataclass(frozen=True)
class PromptSet:
    """A versioned set of prompts. sha256 is the hash of the file's exact bytes.

    Example:
        >>> PromptSet(1, ("greeting",), ("Say hi.",), "0" * 64).prompts
        ('Say hi.',)
    """

    version: int
    prompt_ids: tuple[str, ...]
    prompts: tuple[str, ...]
    sha256: str


def read_prompt_set(path: Path) -> PromptSet:
    """Read and check a prompt set file.

    Every line needs exactly prompt_id, version and prompt, every line the
    same version, and every prompt its own ID. The hash covers the exact
    bytes, so any edit to the file gives it another hash.

    Example:
        >>> prompt_set = read_prompt_set(PROMPTS_DIR / "calibration.jsonl")
        >>> prompt_set.version, len(prompt_set.prompts), prompt_set.prompt_ids[0]
        (1, 32, 'calibration-01')
    """
    content = path.read_bytes()
    try:
        lines = content.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValueError(f"not UTF-8 text: {error}") from error
    if not lines:
        raise ValueError("it holds no prompts")
    ids: list[str] = []
    prompts: list[str] = []
    versions: set[int] = set()
    for number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"line {number} is not JSON: {error}") from error
        if not isinstance(record, dict) or set(record) != set(KEYS):
            raise ValueError(f"line {number} needs exactly {', '.join(KEYS)}")
        prompt_id, version, prompt = (record[key] for key in KEYS)
        if not isinstance(prompt_id, str) or not prompt_id.strip():
            raise ValueError(f"line {number}: prompt_id must be non-empty text")
        if prompt_id in ids:
            raise ValueError(f"line {number}: prompt_id {prompt_id!r} is used twice")
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            raise ValueError(
                f"line {number}: version must be a whole number from 1, not {version!r}"
            )
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"line {number}: prompt must be non-empty text")
        ids.append(prompt_id)
        prompts.append(prompt)
        versions.add(version)
    if len(versions) > 1:
        raise ValueError(f"every line needs the same version, not {sorted(versions)}")
    sha256 = hashlib.sha256(content).hexdigest()
    return PromptSet(versions.pop(), tuple(ids), tuple(prompts), sha256)
