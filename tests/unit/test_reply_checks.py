"""pushlab.analysis.reply_checks: thinking-mode text and replies cut off at
max_new_tokens, and how `pushlab report SPEC` flags and counts them.

The report test runs no push and a pulse with one control on two prompts,
16 reply tokens, on tiny-vl. A stand-in for generate() plants a thinking
tag in one reply and ends another with the end-of-turn token on its last
allowed token, so the report has one of each to find.
"""

import dataclasses
import doctest
import html
import re
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from transformers import PreTrainedTokenizerBase

from pushlab.analysis import experiment_report, reply_checks
from pushlab.analysis.reply_checks import (
    Count,
    ReplyChecks,
    count_lines,
    counts,
    cut_off,
    has_thinking,
    reply_stops,
    thinking_ids,
)
from pushlab.cli import main
from pushlab.engine import generate as generate_module
from pushlab.models.chat import stop_token_ids
from pushlab.models.load import load_model
from pushlab.models.registry import ModelSpec
from pushlab.runs.folder import read_identity
from pushlab.runs.shards import read_results
from pushlab.store.records import RunRecord

NAME = "tiny-pulses"
CONDITIONS = """\
conditions:
  - {label: none, segments: []}
  - label: pulse
    shape: pulse
    params: {unit: tiny-vl:L2:N100, alpha: 1.0, start: 0, length: 4}
    controls: 1
    control_seed: 0
"""
FLAG = experiment_report.THINKING + " "


def _record(
    run_id: str, tokens: list[int], text: str = "", **fields: object
) -> RunRecord:
    values = {
        "run_id": run_id,
        "experiment": "x",
        "model": "tiny@main",
        "prompt_id": "p-1",
        "schedule": [],
        "decoding": {"max_new_tokens": 3},
        "seed": 0,
        "batch_key": "b000001:0",
        "control_of": "",
        "output_text": text,
        "output_tokens": tokens,
        "status": "ok",
    } | fields
    return RunRecord(**values)  # type: ignore[arg-type]


def test_thinking_ids_are_the_tags_tokens(
    tiny_tokenizer: PreTrainedTokenizerBase,
) -> None:
    ids = {tiny_tokenizer.convert_tokens_to_ids(tag) for tag in ("<think>", "</think>")}
    assert thinking_ids(tiny_tokenizer) == ids and len(ids) == 2

    class Plain:  # a tokenizer whose vocabulary has no thinking tag
        def get_vocab(self) -> dict[str, int]:
            return {"a": 0, "<": 1}

    assert thinking_ids(Plain()) == frozenset()


def test_a_thinking_tag_counts_in_the_text_or_as_its_token() -> None:
    assert has_thinking("Fine. <think>", [1, 2], set())
    assert has_thinking("so </think> then", [1], set())
    assert has_thinking("Fine.", [1, 7, 2], {7})  # hidden from the text
    assert not has_thinking("I think so. <thinking>", [1, 2], {7})


@pytest.mark.parametrize(
    ("tokens", "limit", "cut"),
    [
        ([5, 6, 7], 3, True),  # the limit, without an ending token
        ([5, 6, 9], 3, False),  # ended its turn on its last allowed token
        ([5, 9], 3, False),  # ended before the limit
        ([5, 6], 3, False),  # shorter, as a reply saved without its ending
        ([], 3, False),
    ],
)
def test_a_reply_is_cut_off_at_the_limit_without_an_ending_token(
    tokens: list[int], limit: int, cut: bool
) -> None:
    assert cut_off(tokens, limit, {9, 10}) is cut


# tiny-vl ends a chat turn and its text with different tokens, as Qwen3.5-4B
# does; tiny-lm's config names its tokenizer's end token.
@pytest.mark.parametrize(("name", "how_many"), [("tiny-vl", 2), ("tiny-lm", 1)])
def test_the_stops_read_from_the_folder_are_generate_s(
    tiny_downloads: dict[str, ModelSpec], tmp_path: Path, name: str, how_many: int
) -> None:
    loaded = load_model(name, torch.float32, device="cpu")
    found = reply_stops(loaded.tokenizer, loaded.folder)
    assert found == frozenset(stop_token_ids(loaded.model, loaded.tokenizer))
    assert len(found) == how_many
    # Without a generation config, the tokenizer's end token alone.
    assert reply_stops(loaded.tokenizer, tmp_path) == {loaded.tokenizer.eos_token_id}


def test_counts_keep_each_control_apart_and_count_failed_runs() -> None:
    checks = ReplyChecks(frozenset({7}), frozenset({9}))
    own = _record("x-000001", [1, 2, 3])
    control = _record("x-000002", [1, 7, 9], control_of="x-000001")
    other = _record("x-000003", [1, 9], "a <think>", prompt_id="p-2")
    failed = _record("x-000004", [], status="failed", prompt_id="p-2")
    shown = [("a", own), ("a, control 1", control), ("a", other), ("a", failed)]
    found = counts(shown, checks)
    assert list(found) == ["a", "a, control 1"]
    assert found["a"] == Count(replies=2, cut_off=1, thinking=["p-2"], failed=1)
    assert found["a, control 1"] == Count(replies=1, cut_off=0, thinking=["p-1"])


def test_count_lines_say_each_group_s_counts_in_words() -> None:
    found = {
        "none": Count(2, 2),
        "pulse": Count(3, 1, ["p-1", "p-2", "p-3"], failed=1),
        "pulse, control 1": Count(4, 0, ["p-2"]),
    }
    heading, *lines = count_lines(found, {16})
    assert heading.startswith("Replies cut off at the 16-token limit, and replies")
    assert "<think> or </think>" in heading
    assert lines == [
        "none: 2 of 2 cut off; none with thinking-mode text",
        "pulse: 1 of 3 cut off; 3 with thinking-mode text: p-1, p-2 and p-3; "
        "1 failed, with no reply",
        "pulse, control 1: 0 of 4 cut off; 1 with thinking-mode text: p-2",
    ]
    assert count_lines(found, {16, 32})[0].startswith("Replies cut off at their limit")


@pytest.fixture
def planted(
    monkeypatch: pytest.MonkeyPatch, tiny_tokenizer: PreTrainedTokenizerBase
) -> None:
    """generate() as it is, but the unpushed reply to "Say hi." gets <think>
    as its fourth token, and the one to "Count to three." ends its turn with
    its last allowed token."""
    real = generate_module.generate
    think = tiny_tokenizer.convert_tokens_to_ids("<think>")

    def planting(model, tokenizer, prompts, *args, schedules=None, **kwargs):  # type: ignore[no-untyped-def]
        replies = real(model, tokenizer, prompts, *args, schedules=schedules, **kwargs)
        changed = []
        for prompt, schedule, reply in zip(prompts, schedules, replies, strict=True):
            tokens = list(reply.token_ids)
            if not schedule.segments and prompt == "Say hi.":
                tokens[3] = think
            elif not schedule.segments and prompt == "Count to three.":
                tokens[-1] = tokenizer.eos_token_id
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            changed.append(
                dataclasses.replace(reply, token_ids=tuple(tokens), text=text)
            )
        return changed

    monkeypatch.setattr(generate_module, "generate", planting)


def _plain(markdown: str) -> str:
    """Report text as plain text again: no backslash escapes or entities."""
    return html.unescape(re.sub(r"\\(.)", r"\1", markdown))


def test_the_report_flags_thinking_text_and_counts_cut_off_replies(
    tiny_spec: Callable[..., Path],
    two_prompts: Path,
    tiny_tokenizer: PreTrainedTokenizerBase,
    planted: None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    spec = tiny_spec(
        CONDITIONS, prompts=str(two_prompts), max_new_tokens="16", batch_size="8"
    )
    runs, out = tmp_path / "runs", tmp_path / "report.md"
    assert main(["run", str(spec), "--runs-dir", str(runs)]) == 0
    argv = ["report", str(spec), "--runs-dir", str(runs), "--out", str(out)]
    assert main(argv) == 0
    markdown = out.read_text(encoding="utf-8")
    head, *sections = markdown.split("\n## ")
    # What the report should say, worked out here from the saved runs.
    tags = {
        tiny_tokenizer.convert_tokens_to_ids(tag) for tag in ("<think>", "</think>")
    }
    stops = {tiny_tokenizer.eos_token_id, tiny_tokenizer.pad_token_id}
    conditions = read_identity(runs / NAME)["conditions"]
    expected: dict[str, list[RunRecord]] = {}
    for record in read_results(runs / NAME):
        group = conditions[record.run_id] + (", control 1" if record.control_of else "")
        expected.setdefault(group, []).append(record)
    lines = []
    for group, records in expected.items():
        cut = sum(
            len(r.output_tokens) == 16 and r.output_tokens[-1] not in stops
            for r in records
        )
        flagged = [
            r.prompt_id
            for r in records
            if tags & set(r.output_tokens)
            or "<think>" in r.output_text
            or "</think>" in r.output_text
        ]
        thinking = "none with thinking-mode text"
        if flagged:  # two prompts: one or both
            thinking = f"{len(flagged)} with thinking-mode text: "
            thinking += " and ".join(flagged)
        lines.append(f"- {group}: {cut} of {len(records)} cut off; {thinking}")
    assert [_plain(line) for line in head.split("\n") if line.startswith("- ")] == lines
    assert "Replies cut off at the 16-token limit" in head
    # The planted replies: one flagged, one not cut off though 16 tokens long.
    say_hi = next(s for s in sections if s.startswith("p-1\n"))
    (none,) = [line for line in say_hi.split("\n") if line.startswith("**none**: ")]
    assert none.removeprefix("**none**: ").startswith(FLAG)
    count = next(s for s in sections if s.startswith("p-2\n"))
    (ended,) = [line for line in count.split("\n") if line.startswith("**none**: ")]
    assert "cut off" not in ended and "16-token limit" not in ended
    assert "- none: 1 of 2 cut off;" in _plain(head)  # p-1's, not p-2's


def test_docstring_examples_run() -> None:
    assert doctest.testmod(reply_checks).failed == 0
