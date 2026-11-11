"""Two checks of every reply an experiment report shows: thinking-mode text,
and whether the reply was cut off at max_new_tokens.

Thinking-mode text. With thinking off, Qwen's chat template ends the prompt
with an empty thinking block, `<think>\\n\\n</think>\\n\\n`
(pushlab.models.chat), so a reply is the answer alone and should never hold
a thinking tag. A reply that writes <think> or </think> has thinking-mode
text: the model has started to reason aloud, or, under a strong push, is
falling apart into its template's tokens, as occcat saw at dose 2
("<think>}{"). Either is worth seeing before reading the reply as a persona.
A tag counts in the reply's text or as the tag's token among its tokens,
for a tokenizer whose decoding would hide it.

Cut off. A reply keeps the token that ended it (pushlab.engine.generate),
so a reply was cut off when it holds max_new_tokens tokens and its last is
not one that ends a reply. One that ended its turn on its last allowed
token was not. The tokens that end a reply are the tokenizer's end token
and those the model's generation config names, as
pushlab.models.chat.stop_token_ids finds them; reply_stops reads them from
the downloaded model's folder, without its weights.

counts and count_lines give, by condition, how many replies were cut off
and which have thinking-mode text, for the top of the report.
"""

from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pushlab.store.records import RunRecord

THINKING_TAGS = ("<think>", "</think>")


@dataclass(frozen=True)
class ReplyChecks:
    """The tokenizer's facts the checks need: the IDs its vocabulary gives
    the thinking tags, and the tokens that end a reply."""

    thinking_ids: frozenset[int]
    stops: frozenset[int]


def thinking_ids(tokenizer: Any) -> frozenset[int]:
    """The IDs of the thinking tags in tokenizer's vocabulary, added tokens
    included; none for a tokenizer that has them only as several tokens.

    Example:
        >>> sorted(thinking_ids(tokenizer))  # doctest: +SKIP
        [248068, 248069]
    """
    vocabulary = tokenizer.get_vocab()
    return frozenset(vocabulary[tag] for tag in THINKING_TAGS if tag in vocabulary)


def reply_stops(tokenizer: Any, folder: Path) -> frozenset[int]:
    """The tokens that end a reply of the model downloaded in folder: the
    tokenizer's end token and those its generation config names, as
    pushlab.models.chat.stop_token_ids finds them, without loading weights.
    A folder without a generation config gives the tokenizer's end token.

    Example:
        >>> sorted(reply_stops(tokenizer, folder))  # doctest: +SKIP
        [248044, 248046]
    """
    from transformers import GenerationConfig  # here, so importing loads no torch

    ids = {tokenizer.eos_token_id}
    try:
        config = GenerationConfig.from_pretrained(folder, local_files_only=True)
    except OSError:  # no generation_config.json
        config = None
    configured = None if config is None else config.eos_token_id
    if isinstance(configured, int):
        ids.add(configured)
    elif configured is not None:
        ids.update(configured)
    ids.discard(None)
    return frozenset(ids)


def has_thinking(text: str, tokens: Sequence[int], ids: Collection[int]) -> bool:
    """Whether a reply holds thinking-mode text: a thinking tag in its text,
    or the token of one, by ids (thinking_ids), among its tokens.

    Example:
        >>> has_thinking("Sure. <think>", [1, 2], set())
        True
        >>> has_thinking("Hi", [7], {7}), has_thinking("I think so.", [1, 2], {7})
        (True, False)
    """
    return any(tag in text for tag in THINKING_TAGS) or any(t in ids for t in tokens)


def cut_off(tokens: Sequence[int], max_new_tokens: int, stops: Collection[int]) -> bool:
    """Whether a reply hit max_new_tokens: it holds that many tokens, and its
    last is not one of stops, the tokens that end a reply.

    Example:
        >>> cut_off([5, 6, 7], 3, {9}), cut_off([5, 6, 9], 3, {9})
        (True, False)
        >>> cut_off([5, 9], 3, {9})  # ended before the limit
        False
    """
    return len(tokens) >= max_new_tokens > 0 and tokens[-1] not in stops


@dataclass
class Count:
    """One condition's replies, or one control's: how many there are, how
    many were cut off, the prompts of those with thinking-mode text, and the
    runs that failed, which have no reply."""

    replies: int = 0
    cut_off: int = 0
    thinking: list[str] = field(default_factory=list)
    failed: int = 0


def counts(
    shown: Iterable[tuple[str, RunRecord]], checks: ReplyChecks
) -> dict[str, Count]:
    """Each group's Count, in the order the groups first show, from shown,
    pairs of a group's name and a run, such as ("e-1", run) and ("e-1,
    control 1", its control).

    Example:
        >>> found = counts(pairs, checks)  # doctest: +SKIP
        >>> found["e-1"].cut_off, found["e-1"].thinking  # doctest: +SKIP
        (30, ['neutral32-07'])
    """
    found: dict[str, Count] = {}
    for group, run in shown:
        count = found.setdefault(group, Count())
        if run.status == "failed":
            count.failed += 1
            continue
        count.replies += 1
        tokens = run.output_tokens
        limit = run.decoding.get("max_new_tokens")
        if isinstance(limit, int) and cut_off(tokens, limit, checks.stops):
            count.cut_off += 1
        if has_thinking(run.output_text, tokens, checks.thinking_ids):
            count.thinking.append(run.prompt_id)
    return found


def _listed(items: Sequence[str]) -> str:
    return items[0] if len(items) == 1 else f"{', '.join(items[:-1])} and {items[-1]}"


def count_lines(found: dict[str, Count], limits: Collection[int]) -> list[str]:
    """count's lines for a report, before Markdown escaping: a heading
    line, then one line per group. limits are the runs' max_new_tokens.

    Example:
        >>> lines = count_lines({"e-1": Count(32, 30, ["neutral32-07"])}, {200})
        >>> for line in lines:  # doctest: +NORMALIZE_WHITESPACE
        ...     print(line)
        Replies cut off at the 200-token limit, and replies with thinking-mode
        text (a <think> or </think> tag, which a reply should never hold with
        thinking off), by condition:
        e-1: 30 of 32 cut off; 1 with thinking-mode text: neutral32-07
    """
    limit = (
        f"the {next(iter(limits))}-token limit" if len(limits) == 1 else "their limit"
    )
    lines = [
        f"Replies cut off at {limit}, and replies with thinking-mode text (a "
        "<think> or </think> tag, which a reply should never hold with thinking "
        "off), by condition:"
    ]
    for group, count in found.items():
        line = f"{group}: {count.cut_off} of {count.replies} cut off; "
        if count.thinking:
            line += f"{len(count.thinking)} with thinking-mode text: "
            line += _listed(count.thinking)
        else:
            line += "none with thinking-mode text"
        if count.failed:
            line += f"; {count.failed} failed, with no reply"
        lines.append(line)
    return lines
