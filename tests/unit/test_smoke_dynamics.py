"""experiments/s2/smoke_dynamics.yaml, the first dynamics experiment.

It expands to 31 conditions, its 16 and 15 matched random controls, times
its 32 neutral prompts: 992 runs, every one passing validation. Each
condition pushes what its label says on every pass: the prompt pass (None)
and reply tokens 0 to 95 (with 96 reply tokens the last pass reads token
94; 95 is checked too). The file's order puts runs that must match exactly
in one batch, and the runbook's check of those pairs works on a run.
"""

import ast
import contextlib
import io
import re
import textwrap
import warnings
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import pytest
import torch

from pushlab import PROJECT_ROOT
from pushlab.cli import main
from pushlab.models.registry import REGISTRY
from pushlab.prompts import PROMPTS_DIR, read_prompt_set
from pushlab.runs.batch import record_schedule
from pushlab.runs.expand import Expansion, expand
from pushlab.runs.spec import read_spec
from pushlab.schedules.model import Schedule
from pushlab.schedules.validation import validate
from pushlab.units.controls import known_units

SPEC = PROJECT_ROOT / "experiments" / "s2" / "smoke_dynamics.yaml"
RUNBOOK = PROJECT_ROOT / "docs" / "runbooks" / "experiments.md"
E, W = "qwen3.5-4b:L14:N8949", "qwen3.5-4b:L13:N7956"
# In the file's order: the first 8 fill each prompt's first batch, the rest
# its second (decision 0017).
LABELS = [
    "baseline",
    "onset-e-10",
    "onset-e-30",
    "ramp-e-0-32",
    "pulse-e-4",
    "pulse-e-16",
    "pulse-e-64",
    "alternate-e-w-32",
    "constant-e",
    "constant-w",
    "handoff-e-w-10",
    "handoff-e-w-30",
    "handoff-w-e-10",
    "handoff-w-e-30",
    "alternate-e-w-2",
    "alternate-e-w-8",
]
PROMPT_IDS = [f"neutral32-{n:02d}" for n in range(1, 33)]
PASSES = [None, *range(96)]  # the prompt pass, then reply tokens 0 to 95
# The runbook's pairs, as decision 0017's table lists them: pushed alike, in
# one batch, so their replies share every token up to the last one named.
PAIRS = [
    ("onset-e-10", "baseline", 10),
    ("onset-e-30", "baseline", 30),
    ("ramp-e-0-32", "baseline", 1),
    ("pulse-e-64", "baseline", 0),
    ("pulse-e-4", "pulse-e-64", 4),
    ("pulse-e-16", "pulse-e-64", 16),
    ("alternate-e-w-32", "pulse-e-64", 32),
    ("handoff-e-w-10", "constant-e", 10),
    ("handoff-e-w-30", "constant-e", 30),
    ("handoff-w-e-10", "constant-w", 10),
    ("handoff-w-e-30", "constant-w", 30),
    ("alternate-e-w-2", "alternate-e-w-8", 2),
]
ALIKE = ": runs 5 of 5, controls 5 of 5"  # the check's line for a pair alike


@pytest.fixture(scope="module")
def expansion() -> Expansion:
    return expand(read_spec(SPEC))


def test_it_expands_to_31_conditions_times_32_prompts(expansion: Expansion) -> None:
    assert len(expansion.runs) == 992
    assert expansion.warnings == ()
    groups = Counter((run.condition, run.control) for run in expansion.runs)
    assert len(groups) == 31
    assert set(groups.values()) == {32}  # every prompt in every group
    assert list(dict.fromkeys(run.condition for run in expansion.runs)) == LABELS
    controlled = {run.condition for run in expansion.runs if run.control}
    assert controlled == set(LABELS) - {"baseline"}
    assert {run.control for run in expansion.runs} == {0, 1}  # one control each
    assert {run.record.prompt_id for run in expansion.runs} == set(PROMPT_IDS)


def test_each_prompt_fills_two_batches_in_the_planned_layout(
    expansion: Expansion,
) -> None:
    batches: dict[tuple[str, str], list[tuple[str, int]]] = {}
    for run in expansion.runs:
        key = (run.record.prompt_id, run.record.batch_key.split(":")[0])
        batches.setdefault(key, []).append((run.condition, run.control))
    first = [("baseline", 0)] + [(label, k) for label in LABELS[1:8] for k in (0, 1)]
    second = [(label, k) for label in LABELS[8:] for k in (0, 1)]
    assert len(first) == 15 and len(second) == 16
    assert Counter(tuple(rows) for rows in batches.values()) == {
        tuple(first): 32,
        tuple(second): 32,
    }


def test_its_settings_are_the_ones_planned() -> None:
    spec = read_spec(SPEC)
    assert spec.name == "s2-smoke-dynamics"
    assert spec.model == "qwen3.5-4b"
    assert spec.prompts_path == PROMPTS_DIR / "neutral32.jsonl"
    assert spec.decoding == {
        "strategy": "greedy",
        "temperature": None,
        "top_p": 1.0,
        "dtype": "bfloat16",
    }
    assert spec.seeds == (0,)
    assert spec.max_new_tokens == 96
    assert spec.batch_size == 16
    assert spec.group_by_length is True
    assert spec.max_gpu_hours == 1


def test_the_file_s_hash_is_pinned() -> None:
    # It covers what no other test here looks at, such as allow_over_ceiling
    # or a setting that changes no push. Update it when the file changes on
    # purpose.
    assert read_spec(SPEC).sha256 == (
        "9147f31bbcd5090dfd07e978e5545d009ce19cf03b3606e50a7a2bf9eaaf2cad"
    )


def test_every_schedule_passes_validation_without_a_warning(
    expansion: Expansion,
) -> None:
    model = REGISTRY["qwen3.5-4b"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a warning would fail the test
        for run in expansion.runs:
            assert validate(record_schedule(run.record), model, 96) == []


def _planned(label: str, token: int | None) -> dict[str, float]:
    """What a condition's own run pushes on the pass of token, by unit."""
    reply = token is not None
    if label == "baseline":
        return {}
    if label.startswith("constant-"):
        unit, dose = (E, 1.0) if label.endswith("-e") else (W, -1.0)
        return {unit: dose}
    if label.startswith("pulse-e-"):
        length = int(label.rsplit("-", 1)[1])
        return {E: 1.0} if reply and token < length else {}
    if label.startswith("onset-e-"):
        start = int(label.rsplit("-", 1)[1])
        return {E: 1.0} if reply and token >= start else {}
    if label.startswith("handoff-"):
        first, second = (E, W) if label.startswith("handoff-e-w") else (W, E)
        switch = int(label.rsplit("-", 1)[1])
        unit = first if not reply or token < switch else second
        return {unit: 1.0 if unit == E else -1.0}
    if label.startswith("alternate-e-w-"):
        period = int(label.rsplit("-", 1)[1])
        if not reply:
            return {}
        return {E: 1.0} if (token // period) % 2 == 0 else {W: -1.0}
    assert label == "ramp-e-0-32"
    return {E: token / 31} if reply and token < 32 else {}


@pytest.mark.parametrize("label", LABELS)
def test_each_condition_pushes_what_its_label_says(
    expansion: Expansion, label: str
) -> None:
    own = next(r for r in expansion.runs if r.condition == label and not r.control)
    schedule = record_schedule(own.record)
    for token in PASSES:
        doses = {str(unit): dose for unit, dose in schedule.doses_at(token).items()}
        want = _planned(label, token)
        assert doses.keys() == want.keys(), (label, token)
        for unit, dose in want.items():
            assert doses[unit] == pytest.approx(dose, abs=1e-12), (label, token)


def test_each_control_mirrors_its_run_on_two_fixed_random_neurons(
    expansion: Expansion,
) -> None:
    own = {
        (run.condition, run.record.prompt_id): run
        for run in expansion.runs
        if not run.control
    }
    stand_ins: dict[str, set[str]] = {E: set(), W: set()}
    avoid = {str(unit) for unit in known_units()} | {E, W}
    for run in (run for run in expansion.runs if run.control):
        mirrored = own[(run.condition, run.record.prompt_id)]
        assert run.record.control_of == mirrored.record.run_id
        theirs, mine = record_schedule(mirrored.record), record_schedule(run.record)
        assert len(theirs.segments) == len(mine.segments)
        for real, control, stored in zip(
            theirs.segments, mine.segments, run.record.schedule, strict=True
        ):
            assert stored["control_for"] == str(real.unit)
            assert control.unit.layer == real.unit.layer  # the same layer
            assert str(control.unit) not in avoid  # not E, W or a known unit
            # The same doses on the same passes: only the neuron differs.
            assert (control.alpha_start, control.alpha_end) == (
                real.alpha_start,
                real.alpha_end,
            )
            assert (control.start, control.end, control.region) == (
                real.start,
                real.end,
                real.region,
            )
            assert (control.on_for, control.off_for) == (real.on_for, real.off_for)
            stand_ins[str(real.unit)].add(str(control.unit))
    # One control_seed: every condition's control uses the same two neurons,
    # which decision 0017 names.
    assert stand_ins == {E: {"qwen3.5-4b:L14:N1929"}, W: {"qwen3.5-4b:L13:N3302"}}


def _runbook_section() -> str:
    """The runbook's section on this experiment."""
    book = RUNBOOK.read_text(encoding="utf-8")
    section = book[book.index("## Run S2's smoke test of dynamics") :]
    return section[: section.index("\n## ")]


def _runbook_check() -> str:
    """The Python of the runbook's step that checks the exact pairs."""
    pattern = r"uv run python -I - <<'EOF'\n(.*?)\n *EOF\n"
    (code,) = re.findall(pattern, _runbook_section(), flags=re.DOTALL)
    return textwrap.dedent(code)


def _runbook_pairs() -> list[tuple[str, str, int]]:
    """The runbook check's PAIRS: (condition, condition, last shared token)."""
    found = re.findall(r"^PAIRS = (\[.*?^\])", _runbook_check(), re.DOTALL | re.M)
    return ast.literal_eval(found[0])


def _pushed(schedule: Schedule, token: int | None) -> dict[str, float]:
    """The doses on a pass by unit, without doses of 0, which add nothing."""
    return {str(u): d for u, d in schedule.doses_at(token).items() if d != 0}


def test_the_runbook_s_pairs_are_pushed_alike_in_one_batch(
    expansion: Expansion,
) -> None:
    pairs = _runbook_pairs()
    assert pairs == PAIRS
    assert {label for pair in pairs for label in pair[:2]} == set(LABELS)
    runs = {(r.condition, r.control, r.record.prompt_id): r for r in expansion.runs}
    for a, b, last in pairs:
        for control, prompt_id in ((k, p) for k in (0, 1) for p in PROMPT_IDS):
            mine = runs[a, control, prompt_id]
            theirs = runs.get((b, control, prompt_id)) or runs[b, 0, prompt_id]
            batch = {r.record.batch_key.split(":")[0] for r in (mine, theirs)}
            assert len(batch) == 1, (a, b, prompt_id)  # one batch layout
            # Alike up to the pass that reads token last, which chooses
            # token last + 1, so the replies share tokens 0 to last.
            schedules = record_schedule(mine.record), record_schedule(theirs.record)
            for token in (None, *range(last)):
                assert _pushed(schedules[0], token) == _pushed(schedules[1], token)
            assert _pushed(schedules[0], last) != _pushed(schedules[1], last)


@pytest.mark.slow
def test_the_runbook_s_check_finds_every_pair_alike_on_a_tiny_model(
    tiny_spec: Callable[..., Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The file's conditions on tiny-vl, run on the CPU, then the runbook's
    check on its folder: every pair alike on all 5 smoke prompts, and most
    pairs apart once each names one token more, so the check can fail."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    text = SPEC.read_text(encoding="utf-8")
    conditions = text[text.index("conditions:") :]
    for unit, tiny in ((E, "tiny-vl:L2:N100"), (W, "tiny-vl:L1:N3")):
        conditions = conditions.replace(unit, tiny)
    spec = tiny_spec(
        conditions, name="s2-smoke-dynamics", max_new_tokens="96", batch_size="16"
    )
    runs = tmp_path / "runs"
    assert main(["run", str(spec), "--runs-dir", str(runs)]) == 0
    code = _runbook_check()
    assert code.count("connect()") == 1
    code = code.replace("connect()", f"connect(Path({str(runs)!r}))")
    lines = _printed(code)
    assert len(lines) == len(PAIRS)
    assert all(line.endswith(ALIKE) for line in lines)
    # Each pair one token later: pushed apart on the pass before it.
    pattern, later = r"^( +\(.*, )(\d+)\),$", lambda m: f"{m[1]}{int(m[2]) + 1}),"
    code, raised = re.subn(pattern, later, code, flags=re.M)
    assert raised == len(PAIRS)
    apart = [line for line in _printed(code) if not line.endswith(ALIKE)]
    assert len(apart) >= len(PAIRS) // 2, apart


def _printed(code: str) -> list[str]:
    """The lines a script prints when run here."""
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        exec(compile(code, str(RUNBOOK), "exec"), {"Path": Path})
    return printed.getvalue().splitlines()


def test_each_runbook_report_holds_conditions_of_one_batch(
    expansion: Expansion,
) -> None:
    reports = re.findall(r"--conditions (.*?) --out \S+/(\S+)", _runbook_section())
    assert [name for _, name in reports] == [
        "s2-smoke-onsets.md",
        "s2-smoke-pulses.md",
        "s2-smoke-e-first.md",
        "s2-smoke-w-first.md",
    ]
    assert {label for shown, _ in reports for label in shown.split()} == set(LABELS)
    for shown, name in reports:
        for prompt_id in PROMPT_IDS:
            batches = {
                run.record.batch_key.split(":")[0]
                for run in expansion.runs
                if run.record.prompt_id == prompt_id and run.condition in shown.split()
            }
            assert len(batches) == 1, (name, prompt_id)


def _bare(prompt: str) -> str:
    """A prompt's letters and digits, in lower case: what makes it the same."""
    return re.sub(r"[^a-z0-9]", "", prompt.casefold())


def test_the_prompt_set_is_32_new_prompts() -> None:
    neutral = read_prompt_set(PROMPTS_DIR / "neutral32.jsonl")
    assert neutral.version == 1
    assert list(neutral.prompt_ids) == PROMPT_IDS
    # Pinned: the runner refuses a changed set for a folder it has started.
    assert neutral.sha256 == (
        "5ca28963eaee920963c19db7a9b360fbd74eaa42a4dfdc37306087a32b844f45"
    )
    bare = {_bare(prompt) for prompt in neutral.prompts}
    assert len(bare) == 32
    for name in ("calibration.jsonl", "smoke.jsonl"):
        others = {_bare(p) for p in read_prompt_set(PROMPTS_DIR / name).prompts}
        assert not bare & others, name
