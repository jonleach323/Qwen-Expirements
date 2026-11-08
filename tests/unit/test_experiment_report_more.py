"""More of `pushlab report SPEC` (pushlab.analysis.experiment_report and
report_folder): batches whose steps or runs are missing or cannot be read,
the notes on what changed since the version was made and on the tokenizer,
the line on what markers mean, and the guards on condition labels, letters
and token IDs.

tiny-vl runs four conditions on two prompts, greedy, 4 reply tokens, one
batch per prompt (p-1's six runs, then p-2's): no push; a push of A on every
reply token, with one random control; a handoff from A to B at reply token
2, with one random control; and a push of C from reply token 10, after the
reply has ended, so that no step shows C.
"""

import doctest
import html
import re
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from pushlab.analysis import report_folder
from pushlab.cli import main
from pushlab.models import load
from pushlab.runs import shards
from pushlab.runs.folder import read_identity, read_index, write_identity
from pushlab.runs.shards import read_results
from pushlab.units import controls

NAME = "tiny-pulses"
A, B, C = "tiny-vl:L2:N100", "tiny-vl:L1:N3", "tiny-vl:L0:N5"
CONDITIONS = f"""\
conditions:
  - {{label: none, segments: []}}
  - label: pulse
    segments: [{{unit: {A}, alpha: 1.0}}]
    controls: 1
    control_seed: 0
  - label: handoff
    shape: handoff
    params: {{a: {A}, b: {B}, alpha_a: 1.0, alpha_b: -1.0, switch_at: 2}}
    controls: 1
    control_seed: 0
  - label: late
    shape: onset
    params: {{unit: {C}, alpha: 1.0, start: 10}}
"""
FIRST = "v1-b000001-1.parquet"  # p-1's batch
LEGEND = f"A = {A} · B = {B} · C = {C}"
MARKER = re.compile(r"\[\*\*(.+?)\*\*\]")
NO_STEPS = " *(saved before steps were kept, so it has no markers)*"


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _run_all(tiny_spec: Callable[..., Path], prompts: Path, runs: Path) -> Path:
    """Run the four conditions on the two prompts to the end; the file."""
    spec = tiny_spec(CONDITIONS, prompts=str(prompts), batch_size="6")
    assert main(["run", str(spec), "--runs-dir", str(runs)]) == 0
    return spec


@pytest.fixture
def ran(
    tiny_spec: Callable[..., Path], two_prompts: Path, tmp_path: Path
) -> tuple[Path, Path]:
    """The experiment, run to its end: its file and the runs dir."""
    runs = tmp_path / "runs"
    return _run_all(tiny_spec, two_prompts, runs), runs


def _report(spec: Path, runs: Path, out: Path, *options: str) -> int:
    argv = ["report", str(spec), "--runs-dir", str(runs), "--out", str(out)]
    return main([*argv, *options])


def _plain(markdown: str) -> str:
    """Report text as plain text again: <br> a line feed, no escapes."""
    return html.unescape(re.sub(r"\\(.)", r"\1", markdown.replace("<br>", "\n")))


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


def _notes(markdown: str) -> list[str]:
    """The report's notes, in order, as plain text."""
    return [_plain(line) for line in markdown.splitlines() if line.startswith("Note: ")]


def _pushed(label: str) -> bool:
    """Whether a reply of the label has markers when its steps are whole."""
    return not label.startswith(("none", "late"))


def test_a_batch_without_steps_shows_its_replies_without_markers(
    ran: tuple[Path, Path], tmp_path: Path
) -> None:
    # As a batch saved before steps were kept: the rest keep their markers.
    spec, runs = ran
    (runs / NAME / "steps" / FIRST).unlink()
    out = tmp_path / "r.md"
    assert _report(spec, runs, out) == 0
    replies = _replies(out.read_text(encoding="utf-8"))
    assert len(replies) == 12
    for (prompt_id, label), reply in replies.items():
        if prompt_id == "p-1":
            assert NO_STEPS in reply and not MARKER.search(reply)
        else:
            assert NO_STEPS not in reply
            assert bool(MARKER.search(reply)) == _pushed(label)
    # A folder saved before steps were kept at all.
    shutil.rmtree(runs / NAME / "steps")
    assert _report(spec, runs, out, "--overwrite") == 0
    replies = _replies(out.read_text(encoding="utf-8"))
    assert len(replies) == 12
    assert all(NO_STEPS in reply for reply in replies.values())


@pytest.mark.parametrize("broken", ["not-parquet", "other-columns"])
def test_a_steps_file_that_cannot_be_read_gives_its_replies_a_note(
    ran: tuple[Path, Path], tmp_path: Path, broken: str
) -> None:
    spec, runs = ran
    path = runs / NAME / "steps" / FIRST
    if broken == "not-parquet":
        path.write_bytes(b"not a Parquet file")
    else:
        pq.write_table(pa.table({"run_id": ["x"], "step_no": [0]}), path)
    out = tmp_path / "r.md"
    assert _report(spec, runs, out) == 0
    markdown = out.read_text(encoding="utf-8")
    body = _plain(markdown.split("\n## ", 1)[1])
    assert str(path.parent) not in body  # the folder's own path is not repeated
    for (prompt_id, label), reply in _replies(markdown).items():
        note = re.search(r" \*\((its steps cannot be read: .+?)\)\*", reply)
        assert (note is not None) == (prompt_id == "p-1")
        assert bool(MARKER.search(reply)) == (prompt_id == "p-2" and _pushed(label))
        if note is None:
            continue
        assert f"steps/{FIRST}" in note[1]
        assert note[1].endswith(", so it has no markers")
        if broken == "other-columns":  # escaped for Markdown, the _ too
            assert (
                "is not a steps table: its columns are run\\_id, step\\_no" in note[1]
            )


def test_a_missing_or_broken_runs_shard_is_refused_in_words(
    ran: tuple[Path, Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec, runs = ran
    shard, out = runs / NAME / "shards" / FIRST, tmp_path / "r.md"
    shard.unlink()
    capsys.readouterr()
    assert _report(spec, runs, out) == 1
    assert capsys.readouterr().err == (
        f"pushlab report: cannot read the runs saved in {runs / NAME}: "
        f"shards/{FIRST} is missing, though the index counts its runs\n"
    )
    shard.write_bytes(b"not a Parquet file")
    assert _report(spec, runs, out) == 1
    err = capsys.readouterr().err
    assert err.startswith(
        f"pushlab report: cannot read the runs saved in {runs / NAME}: "
    )
    assert f"shards/{FIRST}" in err and str(shard) not in err
    assert not out.exists()


def test_only_the_batches_of_the_chosen_conditions_are_read(
    tiny_spec: Callable[..., Path], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # One run per batch: the broken batches of none do not stop a report on pulse.
    spec, runs, out = tiny_spec(batch_size="1"), tmp_path / "runs", tmp_path / "r.md"
    assert main(["run", str(spec), "--runs-dir", str(runs)]) == 0
    labels = read_identity(runs / NAME)["conditions"]
    for entry in read_index(runs / NAME):
        if [labels[run_id] for run_id in entry["runs"]] == ["none"]:
            (runs / NAME / "shards" / entry["shard"]).write_bytes(b"broken")
            (runs / NAME / "steps" / entry["shard"]).unlink()
    assert _report(spec, runs, out, "--conditions", "pulse") == 0
    replies = _replies(out.read_text(encoding="utf-8"))
    assert len(replies) == 5
    assert all(
        MARKER.search(reply) and NO_STEPS not in reply for reply in replies.values()
    )
    capsys.readouterr()
    assert _report(spec, runs, out, "--overwrite") == 1  # every condition
    assert "cannot read the runs saved in" in capsys.readouterr().err


def test_the_meaning_line_explains_each_kind_of_marker(
    ran: tuple[Path, Path], tmp_path: Path
) -> None:
    spec, runs = ran
    out = tmp_path / "r.md"
    assert _report(spec, runs, out) == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    # C has a letter though no step shows it: its schedule names it.
    meaning = lines[lines.index(LEGEND) + 2]
    assert meaning.startswith("A marker sits before the first token chosen with ")
    assert "x8 is 8 turns: " in meaning
    for form, words in [
        ("A/B x8", "A, then B, 4 times"),
        ("A on/off x8", "A on, then off, 4 times"),
        ("A off/on x8", "A off, then on, 4 times"),
    ]:
        assert f"[**{form}**] is {words}" in meaning


def test_a_controls_label_names_its_stand_ins_in_letter_order(
    ran: tuple[Path, Path], tmp_path: Path
) -> None:
    spec, runs = ran
    out = tmp_path / "r.md"
    assert _report(spec, runs, out) == 0
    replies = _replies(out.read_text(encoding="utf-8"))
    control = next(r for r in read_results(runs / NAME) if r.run_id == f"{NAME}-000005")
    stand_in = {segment["control_for"]: segment["unit"] for segment in control.schedule}
    assert stand_in[B] < stand_in[A]  # so units in text order would read B first
    label = f"handoff, control 1 (A is {stand_in[A]}, B is {stand_in[B]})"
    assert MARKER.findall(replies[("p-1", label)]) == ["A on", "A off, B on"]


def test_a_note_says_when_the_experiment_file_changed_since_the_version(
    ran: tuple[Path, Path], tmp_path: Path
) -> None:
    spec, runs = ran
    out = tmp_path / "r.md"
    assert _report(spec, runs, out) == 0
    assert _notes(out.read_text(encoding="utf-8")) == []
    # The same label, now another dose.
    spec.write_text(spec.read_text().replace("alpha: 1.0}]", "alpha: 2.0}]"))
    assert _report(spec, runs, out, "--overwrite") == 0
    assert _notes(out.read_text(encoding="utf-8")) == [
        "Note: The experiment file changed since version 1 was made: the replies "
        "below are version 1's, and a label may mean other pushes in the file now."
    ]


def test_a_note_says_when_the_known_units_changed_since_the_version(
    tiny_spec: Callable[..., Path],
    two_prompts: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    known = tmp_path / "known.csv"
    known.write_bytes(controls.KNOWN_UNITS_FILE.read_bytes())
    monkeypatch.setattr(controls, "KNOWN_UNITS_FILE", known)
    runs, out = tmp_path / "runs", tmp_path / "r.md"
    spec = _run_all(tiny_spec, two_prompts, runs)
    assert _report(spec, runs, out) == 0
    assert _notes(out.read_text(encoding="utf-8")) == []
    known.write_bytes(known.read_bytes() + b"\n")  # the same units, another file
    assert _report(spec, runs, out, "--overwrite") == 0
    assert _notes(out.read_text(encoding="utf-8")) == [
        "Note: units/known.csv changed since version 1 was made: the controls below "
        "avoided the known units it listed then."
    ]


def test_a_note_says_when_the_tokenizer_is_another_downloads(
    ran: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, runs = ran
    out = tmp_path / "r.md"
    assert _report(spec, runs, out) == 0
    assert _notes(out.read_text(encoding="utf-8")) == []
    model = read_identity(runs / NAME)["model"]
    monkeypatch.setattr(load, "downloaded_revision", lambda *args, **kwargs: "f" * 40)
    assert _report(spec, runs, out, "--overwrite") == 0
    assert _notes(out.read_text(encoding="utf-8")) == [
        f"Note: The runs were made with {model}, and the markers placed with the "
        f"tokenizer of tiny-vl@{'f' * 40}."
    ]


@pytest.mark.parametrize(
    "labels", [None, {}, {f"{NAME}-000001": 3}], ids=["gone", "empty", "not-text"]
)
def test_a_folder_that_does_not_name_each_runs_condition_is_refused(
    ran: tuple[Path, Path],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    labels: object,
) -> None:
    spec, runs = ran
    identity = read_identity(runs / NAME)
    if labels is None:
        del identity["conditions"]  # as a folder saved before labels were kept
    else:
        identity["conditions"] = labels
    write_identity(runs / NAME, identity)
    out = tmp_path / "r.md"
    capsys.readouterr()
    assert _report(spec, runs, out) == 1
    assert capsys.readouterr().err == (
        f"pushlab report: {runs / NAME / 'experiment.json'} does not name each "
        "run's condition\n"
    )
    assert not out.exists()


def test_a_reply_whose_steps_name_other_tokens_has_no_markers(
    ran: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec, runs = ran
    real, changed = shards.read_steps, f"{NAME}-000004"  # p-1's handoff

    def other_token(folder: Path, wanted: dict[str, str]) -> dict:
        found = real(folder, wanted)
        if changed in found:
            first, *rest = found[changed]
            found[changed] = [replace(first, token_id=first.token_id + 1), *rest]
        return found

    monkeypatch.setattr(shards, "read_steps", other_token)
    out = tmp_path / "r.md"
    assert _report(spec, runs, out) == 0
    replies = _replies(out.read_text(encoding="utf-8"))
    assert not MARKER.search(replies[("p-1", "handoff")])
    assert "*(its steps do not match its tokens," in replies[("p-1", "handoff")]
    assert MARKER.findall(replies[("p-2", "handoff")]) == ["A on", "A off, B on"]


def test_docstring_examples_run() -> None:
    result = doctest.testmod(report_folder)
    assert result.attempted > 0
    assert result.failed == 0
