"""experiments/s2/evil_diagnosis.yaml, the evil neuron's diagnosis.

It expands to 7 conditions and one matched random control, for e-1, on the
32 neutral prompts: 256 runs, each prompt's 8 in one batch. Each condition
pushes what decision 0018 says on every pass: the prompt pass (None) and
reply tokens 0 to 199 (the last pass reads token 198; 199 is checked too).
Only e-1-5, at 1.5, is above the ceiling of 1.4, and the file allows it on
purpose.
"""

from collections import Counter

import pytest

from pushlab import PROJECT_ROOT
from pushlab.models.registry import REGISTRY
from pushlab.prompts import PROMPTS_DIR
from pushlab.runs.batch import record_schedule
from pushlab.runs.expand import Expansion, expand
from pushlab.runs.spec import read_spec
from pushlab.schedules.validation import validate
from pushlab.units.controls import known_units

SPEC = PROJECT_ROOT / "experiments" / "s2" / "evil_diagnosis.yaml"
E, W = "qwen3.5-4b:L14:N8949", "qwen3.5-4b:L13:N7956"
# occcat's RESULTS.md: "L16 N8927 +1 -> 12.5/100w", and "-1 stays polite".
L16 = "qwen3.5-4b:L16:N8927"
LABELS = [
    "baseline",
    "e-1",
    "e-1-25",
    "e-1-5",
    "e-1-reply-only",
    "e-w-recipe",
    "l16-n8927",
]
PROMPT_IDS = [f"neutral32-{n:02d}" for n in range(1, 33)]
PASSES = [None, *range(200)]  # the prompt pass, then reply tokens 0 to 199
# What each condition's own run pushes on the prompt pass and on every reply
# token, by unit.
PUSHED = {
    "baseline": ({}, {}),
    "e-1": ({E: 1.0}, {E: 1.0}),
    "e-1-25": ({E: 1.25}, {E: 1.25}),
    "e-1-5": ({E: 1.5}, {E: 1.5}),
    "e-1-reply-only": ({}, {E: 1.0}),
    "e-w-recipe": ({E: 1.0, W: -1.0}, {E: 1.0, W: -1.0}),
    "l16-n8927": ({L16: 1.0}, {L16: 1.0}),
}


@pytest.fixture(scope="module")
def expansion() -> Expansion:
    return expand(read_spec(SPEC))


def test_it_expands_to_7_conditions_and_a_control_times_32_prompts(
    expansion: Expansion,
) -> None:
    assert len(expansion.runs) == 256
    assert expansion.warnings == ()
    groups = Counter((run.condition, run.control) for run in expansion.runs)
    assert groups == {(label, 0): 32 for label in LABELS} | {("e-1", 1): 32}
    assert list(dict.fromkeys(run.condition for run in expansion.runs)) == LABELS
    assert {run.record.prompt_id for run in expansion.runs} == set(PROMPT_IDS)


def test_each_prompt_s_8_runs_share_one_batch(expansion: Expansion) -> None:
    batches: dict[str, list[tuple[str, int]]] = {}
    prompts: dict[str, set[str]] = {}
    for run in expansion.runs:
        batch = run.record.batch_key.split(":")[0]
        batches.setdefault(batch, []).append((run.condition, run.control))
        prompts.setdefault(batch, set()).add(run.record.prompt_id)
    layout = [("baseline", 0), ("e-1", 0), ("e-1", 1)] + [(c, 0) for c in LABELS[2:]]
    assert len(batches) == 32
    assert all(len(ids) == 1 for ids in prompts.values())  # one prompt each
    assert all(rows == layout for rows in batches.values())


def test_its_settings_are_the_ones_planned() -> None:
    spec = read_spec(SPEC)
    assert spec.name == "s2-evil-diagnosis"
    assert spec.model == "qwen3.5-4b"
    assert spec.prompts_path == PROMPTS_DIR / "neutral32.jsonl"
    assert spec.decoding == {
        "strategy": "greedy",
        "temperature": None,
        "top_p": 1.0,
        "dtype": "bfloat16",
    }
    assert spec.seeds == (0,)
    assert spec.max_new_tokens == 200
    assert spec.batch_size == 8
    assert spec.group_by_length is True
    assert spec.max_gpu_hours == 1


def test_the_file_s_hash_is_pinned() -> None:
    # It covers what no other test here looks at, such as a setting that
    # changes no push. Update it when the file changes on purpose.
    assert read_spec(SPEC).sha256 == (
        "783c5df814ef56762c4af3de691c290e0769ee06c037b997246cc2b68707867e"
    )


def test_only_the_dose_of_1_5_goes_above_the_ceiling_and_on_purpose(
    expansion: Expansion,
) -> None:
    spec, model = read_spec(SPEC), REGISTRY["qwen3.5-4b"]
    allowed = {
        label: schedule.allow_over_ceiling
        for condition in spec.conditions
        for label, schedule in condition.points
    }
    assert allowed == {label: label == "e-1-5" for label in LABELS}
    for run in expansion.runs:
        schedule = record_schedule(run.record)  # a run keeps no allow_over_ceiling
        if run.condition == "e-1-5":
            with pytest.raises(ValueError, match="above the ceiling of 1.4"):
                validate(schedule, model, 200)
            assert validate(schedule, model, 200, ceiling=None) == []
        else:
            assert validate(schedule, model, 200) == []


@pytest.mark.parametrize("label", LABELS)
def test_each_condition_pushes_what_its_label_says(
    expansion: Expansion, label: str
) -> None:
    own = next(r for r in expansion.runs if r.condition == label and not r.control)
    schedule = record_schedule(own.record)
    on_prompt, on_reply = PUSHED[label]
    for token in PASSES:
        doses = {str(unit): dose for unit, dose in schedule.doses_at(token).items()}
        assert doses == (on_prompt if token is None else on_reply), (label, token)


def test_e_1_s_control_mirrors_it_on_one_random_neuron_of_layer_14(
    expansion: Expansion,
) -> None:
    own = {
        r.record.prompt_id: r
        for r in expansion.runs
        if r.condition == "e-1" and not r.control
    }
    controls = [run for run in expansion.runs if run.control]
    assert {run.condition for run in controls} == {"e-1"}
    avoid = {str(unit) for unit in known_units()} | {E, W, L16}
    stand_ins = set()
    for run in controls:
        mirrored = own[run.record.prompt_id]
        assert mirrored.control == 0
        assert run.record.control_of == mirrored.record.run_id
        theirs, mine = record_schedule(mirrored.record), record_schedule(run.record)
        for real, control, stored in zip(
            theirs.segments, mine.segments, run.record.schedule, strict=True
        ):
            assert stored["control_for"] == E == str(real.unit)
            assert control.unit.layer == 14 and str(control.unit) not in avoid
            assert (control.alpha_start, control.alpha_end) == (1.0, 1.0)
            assert (control.start, control.end, control.region) == (
                real.start,
                real.end,
                real.region,
            )
            stand_ins.add(str(control.unit))
    # Control seed 0's first draw for E, as in S1 and the smoke test.
    assert stand_ins == {"qwen3.5-4b:L14:N1929"}
