"""The conditions each_candidate refuses to make, and why (pushlab.runs.
each_candidate, decision 0023): each refusal names the condition by its
place and label. The lists are made up (tests/unit/candidate_lists.py).
"""

import importlib.util
from collections.abc import Callable
from pathlib import Path

import pytest

from pushlab.discovery.candidates import write_candidates
from pushlab.runs.conditions import condition_from_mapping
from pushlab.runs.spec import ExperimentSpec, spec_from_yaml

_SPEC = importlib.util.spec_from_file_location(
    "candidate_lists", Path(__file__).with_name("candidate_lists.py")
)
lists = importlib.util.module_from_spec(_SPEC)  # type: ignore[arg-type]
_SPEC.loader.exec_module(lists)  # type: ignore[union-attr]

SPEC, EACH = lists.EXPERIMENT, lists.EACH


@pytest.fixture
def write_list(tmp_path: Path) -> Callable[[], Path]:
    """Write 3 made-up neurons and 2 sets as a list; return its path."""

    def write() -> Path:
        candidates = [*lists.library_neurons(3, [9, 14]), *lists.mayner_sets(2)]
        return write_candidates(tmp_path / "candidates.csv", candidates)

    return write


@pytest.fixture
def spec_of(write_list: Callable[[], Path]) -> Callable[..., ExperimentSpec]:
    """The spec of conditions (EACH by default) on the list, with rule after
    its file in candidates:."""

    def make(conditions: str = EACH, rule: str = "") -> ExperimentSpec:
        text = SPEC.format(file=write_list(), rule=rule, conditions=conditions)
        return spec_from_yaml(text)

    return make


REFUSALS = {
    "two": (
        EACH + "  - {label: again, each_candidate: {}}\n",
        "",
        "conditions[1] (again): an experiment takes one condition with "
        "each_candidate, and conditions[0] (screen) is one",
    ),
    "schedule": (
        "  - {label: c, each_candidate: {}, shape: constant}\n",
        "",
        "conditions[0] (c): each_candidate makes each candidate's schedule, a "
        "constant push, so it takes no shape",
    ),
    "grid-and-segments": (
        "  - {label: c, each_candidate: {}, segments: [], grid: {alpha: [1]}}\n",
        "",
        "so it takes no segments or grid",
    ),
    "ceiling-outside": (
        "  - {label: c, each_candidate: {}, allow_over_ceiling: true}\n",
        "",
        "conditions[0] (c): allow_over_ceiling goes inside each_candidate, as "
        "each_candidate: {allow_over_ceiling: true}",
    ),
    "unknown-key": (
        "  - {label: c, each_candidate: {}, colour: red}\n",
        "",
        "conditions[0] (c): unknown key 'colour'; it takes label, each_candidate, "
        "controls, control_seed",
    ),
    "not-a-mapping": (
        "  - {label: c, each_candidate: [1.0]}\n",
        "",
        "conditions[0] (c): each_candidate must be a mapping, such as {doses: "
        "[1.0]}, or {} for its defaults, not [1.0]",
    ),
    "inside-unknown-key": (
        "  - {label: c, each_candidate: {dose: 1.0}}\n",
        "",
        "conditions[0] (c): each_candidate: unknown key 'dose'; it takes doses, "
        "include_prompt, allow_over_ceiling",
    ),
    "doses": (
        "  - {label: c, each_candidate: {doses: [0]}}\n",
        "",
        "conditions[0] (c): each_candidate: a dose must be a number above 0",
    ),
    "include-prompt": (
        "  - {label: c, each_candidate: {include_prompt: 'no'}}\n",
        "",
        "conditions[0] (c): each_candidate: include_prompt must be true or false, "
        "not 'no'",
    ),
    "controls": (
        "  - {label: c, each_candidate: {}, controls: 1}\n",
        "",
        "conditions[0] (c): 1 control needs a control_seed",
    ),
    "label": (
        "  - {label: C, each_candidate: {}}\n",
        "",
        "conditions[0]: label must be lowercase letters and digits",
    ),
    "clash-before": (
        "  - {label: c-0002, segments: []}\n  - {label: c, each_candidate: {}}\n",
        "",
        "conditions[1] (c): it labels a candidate c-0002, which is conditions[0]'s "
        "label; give one of them another",
    ),
    "clash-after": (
        "  - {label: c, each_candidate: {}}\n  - {label: c-0002, segments: []}\n",
        "",
        "conditions[0] (c): it labels a candidate c-0002, which is conditions[1]'s",
    ),
    "list": (
        EACH,
        ", source: zoo",
        "candidates: source must be library or mayner, not 'zoo'",
    ),
}


@pytest.mark.parametrize(
    "conditions, rule, problem", REFUSALS.values(), ids=REFUSALS.keys()
)
def test_refusals_name_the_condition_and_the_problem(
    spec_of: Callable[..., ExperimentSpec], conditions: str, rule: str, problem: str
) -> None:
    with pytest.raises(ValueError) as error:
        spec_of(conditions, rule)
    assert problem in str(error.value)


def test_each_candidate_needs_candidates_and_candidates_need_it(
    spec_of: Callable[..., ExperimentSpec], write_list: Callable[..., Path]
) -> None:
    text = SPEC.format(file=write_list(), rule="", conditions=EACH)
    without = text.replace(f"candidates: {{file: {write_list()}}}\n", "")
    with pytest.raises(ValueError) as error:
        spec_from_yaml(without)
    assert str(error.value) == (
        "conditions[0] (screen): each_candidate pushes the candidates of the "
        "experiment's candidates: setting, which it lacks, such as candidates: "
        "{file: units/candidates_qwen3-8b.csv, source: library}"
    )
    with pytest.raises(ValueError) as error:
        spec_of("  - {label: none, segments: []}\n")
    assert str(error.value) == (
        "candidates: names candidates to push, but no condition has "
        "each_candidate: to push them"
    )


def test_a_condition_read_alone_refuses_each_candidate() -> None:
    with pytest.raises(ValueError) as error:
        condition_from_mapping({"label": "c", "each_candidate": {}}, 4)
    assert str(error.value).startswith(
        "conditions[4] (c): each_candidate makes a condition per candidate of the "
        "experiment's candidates list, so it is read with the whole experiment"
    )
