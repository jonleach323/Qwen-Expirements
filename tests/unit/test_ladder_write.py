"""A dose ladder's experiment file (pushlab.runs.ladder): its settings and
text, and the file it writes, read back and expanded; and the doses, seeds
and name `pushlab ladder` reads from its options (pushlab.runs.ladder_command).

The units are tiny-vl's, which the tiny_downloads fixture registers, on the
five smoke prompts; their personas and sources are invented.
test_ladder_command covers `pushlab ladder` itself.
"""

import doctest
from dataclasses import replace
from pathlib import Path

import pytest

from pushlab.runs import ladder as ladder_module
from pushlab.runs.ladder import Ladder, default_hours, unit_labels, write_ladder
from pushlab.runs.ladder_command import experiment_name, parse_doses, parse_seeds
from pushlab.runs.spec import read_spec, spec_from_mapping
from pushlab.schedules.yaml_text import load_yaml
from pushlab.store.records import MAX_SEED
from pushlab.units.ladder_units import LadderUnit, run_push
from pushlab.units.neuron import NeuronUnit

DOSES = (0.25, 0.5, 0.75, 1.0, 1.25)
SMOKE = "prompts/smoke.jsonl"


@pytest.fixture
def units(tiny_downloads: object) -> tuple[LadderUnit, ...]:
    """Two invented units of tiny-vl: one pushed up, one pushed down."""
    return (
        LadderUnit(NeuronUnit.parse("tiny-vl:L1:N3"), 1, "Brass Owl", "", "invented"),
        LadderUnit(NeuronUnit.parse("tiny-vl:L2:N100"), -1, "Salt Captain"),
    )


def _ladder(units: tuple[LadderUnit, ...], **settings: object) -> Ladder:
    """The two units at DOSES on the smoke prompts, settings replaced."""
    ladder = Ladder("tiny-ladder", units, DOSES, SMOKE, units_file="units.csv")
    return replace(ladder, **settings)


def _comments(text: str) -> str:
    """The file's comments as one line of words, wherever they wrap."""
    lines = [line.strip() for line in text.splitlines()]
    comments = [line.lstrip("#") for line in lines if line.startswith("#")]
    return " ".join(" ".join(comments).split())


def _segments(run: object) -> list[tuple[str, float, str, object]]:
    return [
        (s["unit"], s["alpha"], s["region"], s["control_for"])
        for s in run.record.schedule  # type: ignore[attr-defined]
    ]


def test_a_ladder_of_two_units_runs_each_at_every_dose_on_every_prompt(
    units: tuple[LadderUnit, ...], tmp_path: Path
) -> None:
    out = tmp_path / "tiny-ladder.yaml"
    spec, expansion = write_ladder(_ladder(units), out)
    assert [condition.label for condition in spec.conditions] == ["u01", "u02"]
    assert len(expansion.runs) == 2 * 5 * 5 == 50
    counts = expansion.counts()
    assert list(counts) == [
        f"{label} alpha={sign * dose!r}"
        for label, sign in (("u01", 1), ("u02", -1))
        for dose in DOSES
    ]
    assert set(counts.values()) == {(5, 0)}
    for run in expansion.runs:
        label, alpha = run.condition.split(" alpha=")
        unit = units[["u01", "u02"].index(label)]
        assert float(alpha) == unit.direction * abs(float(alpha))
        # A constant push of its unit at its signed dose, the prompt pass too.
        assert _segments(run) == [
            (str(unit.unit), float(alpha), "prompt", None),
            (str(unit.unit), float(alpha), "reply", None),
        ]
        assert run.record.schedule[1]["end"] is None
        push = run_push(run.record)
        assert push is not None and push.key == unit.key
        assert abs(float(alpha)) in DOSES
    assert spec.decoding["strategy"] == "greedy" and spec.seeds == (0,)
    assert (spec.max_new_tokens, spec.batch_size) == (160, 16)


def test_the_baseline_adds_a_run_per_prompt_and_controls_double_the_pushed_runs(
    units: tuple[LadderUnit, ...], tmp_path: Path
) -> None:
    _, plain = write_ladder(_ladder(units, baseline=True), tmp_path / "a.yaml")
    assert len(plain.runs) == 50 + 5
    assert plain.counts()["baseline"] == (5, 0)
    baseline = [run for run in plain.runs if run.condition == "baseline"]
    assert len(baseline) == 5 and not any(run.record.schedule for run in baseline)
    ladder = _ladder(units, controls=1, control_seed=3)
    spec, drawn = write_ladder(ladder, tmp_path / "b.yaml")
    assert len(drawn.runs) == 2 * 50
    assert {(c.controls, c.control_seed) for c in spec.conditions} == {(1, 3)}
    for run in drawn.runs:
        push = run_push(run.record)
        assert push is not None
        if run.control:  # a random neuron of the unit's layer, at its dose
            own = NeuronUnit.parse(push.control_for or "")
            assert NeuronUnit.parse(push.unit).layer == own.layer
            assert push.unit != push.control_for
        assert push.key in {unit.key for unit in units}
    both = _ladder(units, baseline=True, controls=2, control_seed=0)
    _, everything = write_ladder(both, tmp_path / "c.yaml")
    assert len(everything.runs) == 50 * 3 + 5
    assert ladder.run_count(5) == 100 and both.run_count(5) == 155


def test_doses_above_the_ceiling_are_allowed_on_purpose(
    units: tuple[LadderUnit, ...], tmp_path: Path
) -> None:
    high = _ladder(units, doses=(1.0, 1.5))
    assert high.over_ceiling
    spec, expansion = write_ladder(high, tmp_path / "high.yaml")
    assert len(expansion.runs) == 2 * 2 * 5
    for condition in spec.conditions:
        assert all(schedule.allow_over_ceiling for _, schedule in condition.points)
    text = (tmp_path / "high.yaml").read_text(encoding="utf-8")
    assert "allow_over_ceiling: true" in text
    assert "above the ceiling of 1.4 are allowed on purpose" in _comments(text)
    # 1.4 is the ceiling itself, not above it.
    edge = _ladder(units, doses=(1.0, 1.4))
    assert not edge.over_ceiling
    spec, _ = write_ladder(edge, tmp_path / "edge.yaml")
    for condition in spec.conditions:
        assert not any(schedule.allow_over_ceiling for _, schedule in condition.points)
    assert "allow_over_ceiling" not in (tmp_path / "edge.yaml").read_text()


def test_the_written_file_reads_back_to_the_same_spec(
    units: tuple[LadderUnit, ...], tmp_path: Path
) -> None:
    for settings in (
        {},
        {"baseline": True, "controls": 1, "control_seed": 0},
        {"seeds": (0, 1, 2), "prompt_pass": False, "doses": (0.5, 1.5)},
        {"name": "2026", "prompts": str(tmp_path / "my prompts: #1.jsonl")},
    ):
        ladder = _ladder(units, **settings)
        text = ladder.text()
        assert load_yaml(text) == ladder.mapping()
        if settings.get("name") == "2026":
            continue  # its prompt set does not exist, so it is not written
        out = tmp_path / "ladder.yaml"
        spec, _ = write_ladder(ladder, out, overwrite=True)
        assert read_spec(out) == spec == spec_from_mapping(ladder.mapping())
        assert out.read_text(encoding="utf-8") == text


def test_the_files_comments_name_each_unit_and_say_what_to_do(
    units: tuple[LadderUnit, ...],
) -> None:
    text = _ladder(units, max_gpu_hours=2.5, hours_note="Given here.").text()
    lines = text.splitlines()
    assert lines[0].startswith("# A dose ladder written by pushlab ladder")
    owl = "#   u01: tiny-vl:L1:N3, direction +1, persona Brass Owl, source invented"
    assert owl in lines
    assert "#   u02: tiny-vl:L2:N100, direction -1, persona Salt Captain" in lines
    assert lines[lines.index("max_gpu_hours: 2.5") - 1] == "# Given here."
    comments = _comments(text)
    assert "units.csv" in comments and "pushlab run --dry-run" in comments
    assert "docs/runbooks/ladder.md" in comments
    assert "Greedy decoding draws no random numbers" in comments
    assert "Neuron Persona Library" not in comments
    library = replace(units[0], source="library")
    comments = _comments(_ladder((library, units[1])).text())
    assert "holds the Neuron Persona Library's data" in comments
    assert "out of git (decision 0020)" in comments
    # A line break in a persona's name stays in its comment.
    odd = replace(units[0], persona="Brass\nOwl\tmax_new_tokens: 1")
    ladder = _ladder((odd, units[1]))
    assert load_yaml(ladder.text()) == ladder.mapping()
    assert "persona Brass Owl max_new_tokens: 1" in ladder.text()


def test_several_seeds_sample_at_temperature_1_and_no_prompt_pass_leaves_it(
    units: tuple[LadderUnit, ...], tmp_path: Path
) -> None:
    ladder = _ladder(units, seeds=(0, 1), prompt_pass=False)
    assert ladder.decoding() == {"strategy": "sample", "temperature": 1.0}
    spec, expansion = write_ladder(ladder, tmp_path / "s.yaml")
    assert spec.decoding["temperature"] == 1.0 and spec.seeds == (0, 1)
    assert len(expansion.runs) == 2 * 5 * 5 * 2
    for run in expansion.runs:
        ((_, alpha, region, _),) = _segments(run)
        assert region == "reply" and abs(alpha) in DOSES
    text = ladder.text()
    assert "include_prompt: false" in text and "temperature: 1.0" in text


def test_a_ladder_that_does_not_expand_is_never_written(
    units: tuple[LadderUnit, ...], tmp_path: Path
) -> None:
    deep = LadderUnit(NeuronUnit.parse("tiny-vl:L9:N3"), 1)  # tiny-vl has 4 layers
    folder = tmp_path / "experiments"
    folder.mkdir()
    out = folder / "deep.yaml"
    with pytest.raises(ValueError, match="layer 9 is out of range"):
        write_ladder(_ladder((deep,)), out)
    assert list(folder.iterdir()) == []
    out.write_text("kept\n", encoding="utf-8")
    with pytest.raises(ValueError, match="layer 9"):
        write_ladder(_ladder((deep,)), out, overwrite=True)
    assert out.read_text() == "kept\n" and list(folder.iterdir()) == [out]
    with pytest.raises(FileExistsError):
        write_ladder(_ladder(units), out)
    assert out.read_text() == "kept\n"


def test_a_ladder_refuses_no_unit_two_models_and_doses_out_of_order(
    units: tuple[LadderUnit, ...],
) -> None:
    other = LadderUnit(NeuronUnit.parse("qwen3-8b:L1:N1"), 1)
    for settings, problem in [
        ({"units": ()}, "at least one unit"),
        ({"units": (units[0], other)}, "all be of one model"),
        ({"doses": ()}, "a list of doses"),
        ({"doses": (0.5, 0.5)}, "must increase, each once"),
        ({"doses": (1.0, 0.5)}, "must increase, each once"),
        ({"doses": (0.0, 1.0)}, "above 0, not 0.0"),
        ({"doses": (True,)}, "above 0, not True"),
        ({"doses": (float("inf"),)}, "above 0, not inf"),
    ]:
        with pytest.raises(ValueError, match=problem):
            replace(_ladder(units), **settings)
    ladder = replace(_ladder(units), units=list(units), doses=[1, 2])
    assert (ladder.doses, ladder.units) == ((1.0, 2.0), units)


@pytest.mark.parametrize(
    ("text", "doses"),
    [
        ("0.25,0.5,0.75,1.0,1.25", DOSES),
        (" 0.5 , 1 ", (0.5, 1.0)),
        ("1e-1,.5,2", (0.1, 0.5, 2.0)),
    ],
)
def test_doses_are_read_from_a_comma_list(text: str, doses: tuple[float, ...]) -> None:
    assert parse_doses(text) == doses


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("", "give doses such as 0.25,0.5,0.75,1.0,1.25"),
        ("0.5,,1", "'' is not a dose"),
        ("half", "'half' is not a dose"),
        ("nan", "'nan' is not a dose"),
        ("1e999", "too large"),
        ("0", "above 0"),
        ("-0.5", "above 0"),
        ("0.5,0.50", "0.50 is the dose 0.5 again"),
        ("0.5,0.25", "increasing order: 0.25 comes after 0.5"),
    ],
)
def test_bad_doses_are_refused(text: str, problem: str) -> None:
    with pytest.raises(ValueError, match="--doses") as error:
        parse_doses(text)
    assert problem in str(error.value)


def test_seeds_are_read_from_a_comma_list() -> None:
    assert parse_seeds("0") == (0,)
    assert parse_seeds("3, 1,2") == (3, 1, 2)
    for text, problem in [
        ("", "whole numbers"),
        ("-1", "whole numbers"),
        ("1.5", "whole numbers"),
        ("1,1", "seed 1 is given twice"),
        (str(MAX_SEED + 1), f"at most {MAX_SEED}"),
    ]:
        with pytest.raises(ValueError, match="--seeds") as error:
            parse_seeds(text)
        assert problem in str(error.value)


def test_the_name_is_given_or_made_from_the_out_files_stem() -> None:
    made = [
        experiment_name(None, Path(out))
        for out in (
            "/workspace/experiments/panel-1-ladder.yaml",
            "Panel_1  Ladder.v2.yaml",
            "--Évian--.yaml",
        )
    ]
    assert made == ["panel-1-ladder", "panel-1-ladder-v2", "vian"]
    assert experiment_name("owl-ladder", Path("x.yaml")) == "owl-ladder"
    with pytest.raises(ValueError, match="cannot make an experiment name from ___"):
        experiment_name(None, Path("___.yaml"))
    with pytest.raises(ValueError, match="--name must be lowercase letters"):
        experiment_name("Owl", Path("x.yaml"))


def test_labels_are_zero_padded_to_the_counts_width() -> None:
    assert unit_labels(1) == ("u01",)
    assert unit_labels(10)[-2:] == ("u09", "u10")
    labels = unit_labels(100)
    assert (labels[0], labels[-1], len(set(labels))) == ("u001", "u100", 100)


def test_the_default_gpu_hours_are_rounded_up_to_a_quarter_hour() -> None:
    # 50 runs of 160 tokens at 22 tokens a second: 0.101 hours, times 1.5.
    assert default_hours(50, 160, 22.0) == 0.25
    assert default_hours(1_000, 160, 22.0) == 3.25  # 2.02 x 1.5 = 3.03
    assert default_hours(3_000, 160, 40.0) == 5.0  # exactly 3.33 x 1.5
    assert default_hours(0, 160, 22.0) == 0.25


def test_docstring_examples_run() -> None:
    result = doctest.testmod(ladder_module)
    assert result.attempted > 0
    assert result.failed == 0
