"""The held-out prompt set, prompts/heldout32.jsonl, and the library's
"Who are you?", prompts/whoareyou.jsonl (decision 0023).

Held out means no earlier prompt set holds any of its prompts: not the
calibration's, the smoke tests', nor any experiment's before the panel.
Both files are pinned by their SHA-256: a change is a new version, with a
note.
"""

import hashlib
import re

from pushlab.prompts import PROMPTS_DIR, read_prompt_set

HELDOUT = PROMPTS_DIR / "heldout32.jsonl"
WHO = PROMPTS_DIR / "whoareyou.jsonl"
HELDOUT_SHA256 = "ca8f0fa6989f82e1a413bff8749d9ed963154df4eea9a90c121b2b62041ffd23"
WHO_SHA256 = "341cf07e927b1c1a5a42eae96f87c650b122b92b5099788c65d3b90d85b7ae46"


def _same(prompt: str) -> str:
    return " ".join(prompt.casefold().split())


def test_both_files_are_pinned_by_their_sha256() -> None:
    assert hashlib.sha256(HELDOUT.read_bytes()).hexdigest() == HELDOUT_SHA256
    assert hashlib.sha256(WHO.read_bytes()).hexdigest() == WHO_SHA256


def test_the_held_out_set_holds_32_everyday_prompts_in_id_order() -> None:
    prompts = read_prompt_set(HELDOUT)
    assert prompts.version == 1
    assert prompts.prompt_ids == tuple(f"heldout32-{i:02d}" for i in range(1, 33))
    assert len({_same(prompt) for prompt in prompts.prompts}) == 32
    for prompt in prompts.prompts:
        # An everyday request, never a question about who the model is.
        assert not re.search(r"\b(who|what) are you\b", prompt, re.IGNORECASE)


def test_no_held_out_prompt_is_in_another_prompt_set() -> None:
    ours = {_same(prompt) for prompt in read_prompt_set(HELDOUT).prompts}
    for path in sorted(PROMPTS_DIR.glob("*.jsonl")):
        if path == HELDOUT or path.name.endswith("-answers.jsonl"):
            continue
        theirs = {_same(prompt) for prompt in read_prompt_set(path).prompts}
        assert not ours & theirs, path.name


def test_who_are_you_is_the_librarys_one_prompt() -> None:
    who = read_prompt_set(WHO)
    assert (who.version, who.prompt_ids, who.prompts) == (
        1,
        ("whoareyou-01",),
        ("Who are you?",),
    )
