"""`pushlab report SPEC`: an experiment's replies, with markers where its pushes
switch (pushlab.analysis.experiment_report).

tiny-vl runs four conditions on two prompts, greedy, 16 reply tokens: no
push, a pulse of A on reply tokens 2 to 4 with one random control, a handoff
from A to B at reply token 4, and A and B taking turns every 2 reply tokens.
A push on reply token k first changes token k + 1 (decision 0008), so each
marker goes before the token of the step after the pass that switched.
"""

import doctest
import html
import json
import os
import re
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
from transformers import PreTrainedTokenizerBase

from pushlab.analysis import experiment_report
from pushlab.cli import main
from pushlab.runs import shards
from pushlab.runs.folder import read_identity
from pushlab.runs.shards import read_results
from pushlab.store.records import RunRecord, make_run_id, write_runs

NAME = "tiny-pulses"
A, B = "tiny-vl:L2:N100", "tiny-vl:L1:N3"
CONDITIONS = f"""\
conditions:
  - {{label: none, segments: []}}
  - label: pulse
    shape: pulse
    params: {{unit: {A}, alpha: 1.0, start: 2, length: 3}}
    controls: 1
    control_seed: 0
  - label: handoff
    shape: handoff
    params: {{a: {A}, b: {B}, alpha_a: 1.0, alpha_b: -1.0, switch_at: 4}}
  - label: alternate
    shape: alternate
    params: {{a: {A}, b: {B}, alpha_a: 1.0, alpha_b: -1.0, period: 2}}
"""
# Each condition's markers in a 16-token reply, by step. The pulse pushes
# reply tokens 2 to 4: steps 3 to 5. The handoff pushes A on the prompt pass
# (step 0) and on reply tokens 0 to 3, then B from reply token 4 (step 5).
# The alternation pushes A on reply tokens 0 and 1 (steps 1 and 2), B on 2
# and 3, A on 4 and 5, and so on: 8 turns in steps 1 to 15.
EXPECTED = {
    "none": [],
    "pulse": [(3, "A on"), (6, "A off")],
    "handoff": [(0, "A on"), (5, "A off, B on")],
    "alternate": [(1, "A/B x8")],
}
# A pulse with two controls, and a push that starts after the reply ends.
SAMPLED = f"""\
conditions:
  - label: pulse
    shape: pulse
    params: {{unit: {A}, alpha: 1.0, start: 0, length: 2}}
    controls: 2
    control_seed: 0
  - label: late
    shape: onset
    params: {{unit: {B}, alpha: 1.0, start: 10}}
"""
MARKER = re.compile(r"\[\*\*(.+?)\*\*\]")
LIMIT = " *(cut off at the 16-token limit)*"


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


@pytest.fixture
def spec(tiny_spec: Callable[..., Path], two_prompts: Path) -> Path:
    """The four conditions on the two prompts, 16 reply tokens, batches of 8."""
    return tiny_spec(
        CONDITIONS, prompts=str(two_prompts), max_new_tokens="16", batch_size="8"
    )


def _run(spec: Path, runs: Path, *options: str) -> int:
    return main(["run", str(spec), "--runs-dir", str(runs), *options])


def _report(spec: Path, runs: Path, out: Path, *options: str) -> int:
    argv = ["report", str(spec), "--runs-dir", str(runs), "--out", str(out)]
    return main([*argv, *options])


def _plain(markdown: str) -> str:
    """Report text as plain text again: <br> a line feed, no escapes."""
    return html.unescape(re.sub(r"\\(.)", r"\1", markdown.replace("<br>", "\n")))


def _breaks(text: str) -> str:
    """text with each line break, of any kind, a line feed, as the report has."""
    return "\n".join(re.split(r"\r\n|[\n\r\v\f\x1c\x1d\x1e\x85\u2028\u2029]", text))


def _replies(markdown: str) -> dict[tuple[str, str], str]:
    """Each reply's Markdown in the report, by its prompt ID and plain label."""
    replies = {}
    for section in markdown.split("\n## ")[1:]:
        prompt_id, *lines = section.split("\n")
        for line in lines:
            if line.startswith("**") and "**: " in line:
                label, reply = line[2:].split("**: ", 1)
                replies[(_plain(prompt_id), _plain(label))] = reply
    return replies


def _label(record: object, conditions: dict[str, str]) -> str:
    """The label the report gives a run: a control's names its stand-in."""
    label = conditions[record.run_id]  # type: ignore[attr-defined]
    if record.control_of:  # type: ignore[attr-defined]
        (segment,) = record.schedule  # type: ignore[attr-defined]
        assert segment["control_for"] == A
        label += f", control 1 (A is {segment['unit']})"
    return label


def test_markers_sit_before_the_tokens_of_their_steps(
    spec: Path,
    tiny_tokenizer: PreTrainedTokenizerBase,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs, out = tmp_path / "runs", tmp_path / "reports" / "markers.md"
    assert _run(spec, runs) == 0
    capsys.readouterr()
    assert _report(spec, runs, out) == 0
    assert capsys.readouterr().out == f"Saved the report on 10 replies: {out}\n"
    markdown = out.read_text(encoding="utf-8")
    lines = markdown.splitlines()
    assert "A = tiny-vl:L2:N100 · B = tiny-vl:L1:N3" in lines  # the legend
    assert not any(line.startswith("|") for line in lines)  # no wide table
    conditions = read_identity(runs / NAME)["conditions"]
    records = read_results(runs / NAME)
    replies = _replies(markdown)
    assert len(records) == len(replies) == 10
    for record in records:
        tokens, text = record.output_tokens, record.output_text
        assert len(tokens) == 16  # each reply runs to the limit, so says so
        reply = replies[(record.prompt_id, _label(record, conditions))]
        reply = reply.removeprefix(f"{experiment_report.THINKING} ")  # a flag
        assert reply.endswith(LIMIT)
        pieces = MARKER.split(reply.removesuffix(LIMIT))  # text, marker, text...
        found = pieces[1::2]
        expected = EXPECTED[conditions[record.run_id]]
        assert found == [marker for _, marker in expected]
        assert all(len(f"[{marker}]") <= 20 for marker in found)
        assert _plain("".join(pieces[::2])) == _breaks(text)
        for i, (step, _) in enumerate(expected):
            before = tiny_tokenizer.decode(tokens[:step], skip_special_tokens=True)
            place = len(os.path.commonprefix([text, before]))
            assert _plain("".join(pieces[: 2 * i + 1 : 2])) == _breaks(text[:place])


def test_each_prompt_shows_its_conditions_in_order_with_the_controls_after(
    spec: Path, tmp_path: Path
) -> None:
    runs, out = tmp_path / "runs", tmp_path / "report.md"
    assert _run(spec, runs) == 0
    assert _report(spec, runs, out) == 0
    markdown = out.read_text(encoding="utf-8")
    assert "Say hi." in markdown and "Count to three." in markdown
    conditions = read_identity(runs / NAME)["conditions"]
    records = read_results(runs / NAME)
    sections = markdown.split("\n## ")[1:]
    assert [section.split("\n")[0] for section in sections] == ["p-1", "p-2"]
    for section in sections:
        prompt_id = section.split("\n")[0]
        replies = _replies("\n## " + section)
        control = next(r for r in records if r.prompt_id == prompt_id and r.control_of)
        assert [label for _, label in replies] == [
            "none",
            "pulse",
            _label(control, conditions),
            "handoff",
            "alternate",
        ]
        # The control pushes a random neuron, which stands in for A: its
        # markers are A's, and the legend gives it no letter of its own.
        assert control.schedule[0]["unit"] not in (A, B)
        reply = replies[(prompt_id, _label(control, conditions))]
        assert MARKER.findall(reply) == ["A on", "A off"]
        assert control.schedule[0]["unit"] not in markdown.split("\n## ")[0]


def test_conditions_choose_the_replies_shown_and_their_order(
    spec: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    runs, out = tmp_path / "runs", tmp_path / "report.md"
    assert _run(spec, runs) == 0
    capsys.readouterr()
    assert _report(spec, runs, out, "--conditions", "handoff", "none") == 0
    assert capsys.readouterr().out == f"Saved the report on 4 replies: {out}\n"
    replies = _replies(out.read_text(encoding="utf-8"))
    assert list(replies) == [
        ("p-1", "handoff"),
        ("p-1", "none"),
        ("p-2", "handoff"),
        ("p-2", "none"),
    ]


def test_labels_name_each_seed_and_number_each_runs_controls(
    tiny_spec: Callable[..., Path], two_prompts: Path, tmp_path: Path
) -> None:
    spec = tiny_spec(
        SAMPLED,
        prompts=str(two_prompts),
        decoding="{strategy: sample, temperature: 0.7, dtype: float32}",
        seeds="[0, 1]",
        batch_size="8",
    )
    runs, out = tmp_path / "runs", tmp_path / "report.md"
    assert _run(spec, runs) == 0
    assert _report(spec, runs, out) == 0
    replies = _replies(out.read_text(encoding="utf-8"))
    records = read_results(runs / NAME)
    first = next(r for r in records if not r.schedule[0].get("control_for"))
    units = [r.schedule[0]["unit"] for r in records if r.control_of == first.run_id]
    pulses = [
        [f"pulse, seed {seed}"]
        + [
            f"pulse, seed {seed}, control {k} (A is {u})"
            for k, u in enumerate(units, 1)
        ]
        for seed in (0, 1)
    ]
    expected = [*pulses[0], *pulses[1], "late, seed 0", "late, seed 1"]
    for prompt_id in ("p-1", "p-2"):
        assert [label for p, label in replies if p == prompt_id] == expected
    for (_, label), reply in replies.items():
        late = label.startswith("late")
        assert ("*(the reply ended before any push)*" in reply) == late
        assert bool(MARKER.search(reply)) != late


def test_unknown_conditions_and_odd_outs_are_refused_with_a_message(
    tiny_spec: Callable[..., Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec, runs, out = tiny_spec(CONDITIONS), tmp_path / "runs", tmp_path / "r.md"
    assert _report(spec, runs, out) == 1  # nothing run yet
    err = capsys.readouterr().err
    assert err.startswith("pushlab report: ") and "does not exist" in err
    assert _run(spec, runs, "--limit", "2") == 3
    capsys.readouterr()
    assert _report(spec, runs, out, "--conditions", "pulse", "pulsed") == 1
    err = capsys.readouterr().err
    assert err.startswith("pushlab report: ")
    assert '"pulsed"' in err
    assert '"none", "pulse", "handoff" and "alternate"' in err
    assert not out.exists()
    assert _report(spec, runs, tmp_path) == 1
    assert "must be a file" in capsys.readouterr().err
    out.write_text("my notes")
    assert _report(spec, runs, out) == 1
    assert "--overwrite" in capsys.readouterr().err
    assert out.read_text() == "my notes"
    assert _report(spec, runs, out, "--overwrite") == 0
    assert out.read_text() != "my notes"
    identity = runs / NAME / "experiment.json"
    before = identity.read_bytes()
    assert _report(spec, runs, identity, "--overwrite") == 1
    assert "--out must be outside" in capsys.readouterr().err
    assert identity.read_bytes() == before


def test_a_folder_that_is_not_the_experiments_is_refused(
    tiny_spec: Callable[..., Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec, out = tiny_spec(CONDITIONS), tmp_path / "r.md"
    other = tmp_path / "other" / NAME
    other.mkdir(parents=True)
    (other / "experiment.json").write_text(
        json.dumps({"experiment": "x", "version": 1})
    )
    assert _report(spec, tmp_path / "other", out) == 1
    assert "belongs to the experiment 'x'" in capsys.readouterr().err
    stray = tmp_path / "stray" / NAME
    stray.mkdir(parents=True)
    (stray / "notes.txt").write_text("mine")
    assert _report(spec, tmp_path / "stray", out) == 1
    assert "is not a runner folder" in capsys.readouterr().err
    assert main(["report", str(tmp_path / "missing.yaml"), "--out", str(out)]) == 1
    assert "cannot read it" in capsys.readouterr().err
    assert not out.exists()


def test_a_tokenizer_that_is_not_downloaded_is_refused(
    tiny_spec: Callable[..., Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    spec, runs, out = tiny_spec(CONDITIONS), tmp_path / "runs", tmp_path / "r.md"
    assert _run(spec, runs, "--limit", "2") == 3
    monkeypatch.setenv("PUSHLAB_MODELS_DIR", str(tmp_path / "no-models"))
    capsys.readouterr()
    assert _report(spec, runs, out) == 1
    err = capsys.readouterr().err
    assert err.startswith("pushlab report: ") and "is not downloaded" in err
    assert not out.exists()


def test_a_failed_run_is_listed_as_failed_without_its_error(
    spec: Path,
    tmp_path: Path,
    failing_generate: Callable[..., None],
    capsys: pytest.CaptureFixture[str],
) -> None:
    runs, out = tmp_path / "runs", tmp_path / "report.md"
    failing_generate("Say hi.", RuntimeError("boom <b>"))  # p-1's batch
    assert _run(spec, runs) == 2
    capsys.readouterr()
    assert _report(spec, runs, out) == 0
    assert capsys.readouterr().out == f"Saved the report on 5 replies: {out}\n"
    markdown = out.read_text(encoding="utf-8")
    replies = _replies(markdown)
    failed = [reply for (prompt_id, _), reply in replies.items() if prompt_id == "p-1"]
    assert len(failed) == 5
    assert all(reply.startswith("*failed") for reply in failed)
    assert "boom" not in markdown


def test_runs_not_saved_yet_are_counted_and_changed_prompts_noted(
    spec: Path, two_prompts: Path, tmp_path: Path
) -> None:
    runs, out = tmp_path / "runs", tmp_path / "report.md"
    assert _run(spec, runs, "--limit", "5") == 3  # p-1's five runs
    assert _report(spec, runs, out) == 0
    markdown = out.read_text(encoding="utf-8")
    assert {prompt_id for prompt_id, _ in _replies(markdown)} == {"p-1"}
    assert "Note: Not saved yet: 5 runs of these conditions;" in markdown
    assert "changed since" not in markdown
    two_prompts.write_text(two_prompts.read_text().replace("Say hi.", "Say hey."))
    assert _report(spec, runs, out, "--overwrite") == 0
    markdown = out.read_text(encoding="utf-8")
    assert f"{two_prompts.name} changed since version 1" in _plain(markdown)
    assert "Say hey." in markdown


def test_a_reply_whose_steps_do_not_match_its_tokens_has_no_markers(
    spec: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runs, out = tmp_path / "runs", tmp_path / "report.md"
    assert _run(spec, runs, "--limit", "5") == 3
    real = shards.read_steps

    def cut(path: Path, wanted: dict[str, str]) -> dict:  # each last step lost
        return {run_id: steps[:-1] for run_id, steps in real(path, wanted).items()}

    monkeypatch.setattr(shards, "read_steps", cut)
    assert _report(spec, runs, out) == 0
    replies = _replies(out.read_text(encoding="utf-8"))
    assert len(replies) == 5
    for reply in replies.values():
        assert not MARKER.search(reply)
        assert "steps do not match its tokens" in reply


def test_the_s1_runs_mode_still_reports_run_folders(
    spec: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    folder, out = tmp_path / "s1", tmp_path / "side.md"
    decoding = {"strategy": "greedy", "max_new_tokens": 8, "thinking": False}
    record = RunRecord(
        make_run_id("steer", 1),
        "steer",
        "tiny-vl@main",
        "p1",
        [],
        decoding | {"dtype": "float32"},
        0,
        "b000001:0",
        "",
        "hi",
        [1],
        "ok",
    )
    write_runs([record], folder)
    assert main(["report", "--runs", str(folder), "--out", str(out)]) == 0
    assert "| **p1** | hi |" in out.read_text()
    argv = ["report", str(spec), "--runs", str(folder), "--out", str(out)]
    assert main([*argv, "--overwrite"]) == 1
    assert "not both" in capsys.readouterr().err


def test_docstring_examples_run() -> None:
    result = doctest.testmod(experiment_report)
    assert result.failed == 0
