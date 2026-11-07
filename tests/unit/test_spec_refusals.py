"""Experiment specs: the files and settings spec_from_yaml refuses, and why."""

import pytest

from pushlab.runs.spec import spec_from_yaml

# The spec of test_spec.py, which each refusal changes in one place.
SPEC = """\
name: pulse-lengths
model: qwen3.5-4b
prompts: prompts/smoke.jsonl
decoding: {strategy: sample, temperature: 0.7, top_p: 0.9}
seeds: [0, 1]
max_new_tokens: 64
batch_size: 4
max_gpu_hours: 0.5
conditions:
  - label: none
    segments: []
  - label: pulse
    shape: pulse
    params: {unit: qwen3.5-4b:L14:N8949, alpha: 1.0, start: 0}
    grid: {length: [4, 16, 64]}
    controls: 2
    control_seed: 7
"""
SAMPLE = "{strategy: sample, temperature: 0.7, top_p: 0.9}"


def _with(old: str, new: str) -> str:
    assert old in SPEC, old
    return SPEC.replace(old, new)


REFUSALS = {
    "unknown-key": (
        _with("batch_size: 4", "batch_size: 4\ncolour: red"),
        "unknown key 'colour'",
    ),
    "missing": (_with("max_gpu_hours: 0.5\n", ""), "needs max_gpu_hours"),
    "name": (
        _with("name: pulse-lengths", "name: Pulse_Lengths"),
        "name must be lowercase",
    ),
    "name-number": (
        _with("name: pulse-lengths", "name: 2026"),
        "name must be text, such as pulse-lengths, but YAML read 2026; put it in",
    ),
    "model-for-tests": (
        _with("model: qwen3.5-4b", "model: tiny"),
        "model 'tiny' is the test suite's model; an experiment can use qwen3-8b, ",
    ),
    "prompts": (_with("prompts: prompts/smoke.jsonl", "prompts: 3"), "prompts must be"),
    "decoding-not-a-mapping": (_with(SAMPLE, "greedy"), "decoding must be a mapping"),
    "decoding-key": (
        _with("top_p: 0.9", "top_p: 0.9, beams: 2"),
        "unknown key 'beams'",
    ),
    "strategy": (_with("strategy: sample", "strategy: beam"), "strategy must be"),
    "no-temperature": (_with("temperature: 0.7, ", ""), "sampling needs a temperature"),
    "top-p": (_with("top_p: 0.9", "top_p: 1.5"), "top_p must be above 0 and at most 1"),
    "dtype": (_with("top_p: 0.9", "top_p: 0.9, dtype: int8"), "dtype must be"),
    "greedy-temperature": (
        _with(SAMPLE, "{strategy: greedy, temperature: 0.7}"),
        "greedy decoding takes no temperature; use strategy: sample",
    ),
    "greedy-top-p": (
        _with(SAMPLE, "{strategy: greedy, top_p: 0.9}"),
        "greedy decoding takes no top_p",
    ),
    "greedy-seeds": (
        _with(SAMPLE, "{strategy: greedy}"),
        "greedy decoding takes one seed",
    ),
    "seeds": (_with("[0, 1]", "[]"), "seeds must be a list of whole numbers"),
    "seed-twice": (_with("[0, 1]", "[1, 1]"), "seed 1 is given twice"),
    "seed-float": (
        _with("[0, 1]", "[0, 1.5]"),
        "seeds must be a list of whole numbers",
    ),
    "tokens": (
        _with("max_new_tokens: 64", "max_new_tokens: 0"),
        "max_new_tokens must be",
    ),
    "batch": (_with("batch_size: 4", "batch_size: 0"), "batch_size must be"),
    "hours": (_with("max_gpu_hours: 0.5", "max_gpu_hours: 0"), "max_gpu_hours must be"),
    "hours-text": (
        _with("max_gpu_hours: 0.5", "max_gpu_hours: '1'"),
        "max_gpu_hours must be",
    ),
    "hours-huge": (
        _with("max_gpu_hours: 0.5", "max_gpu_hours: " + "9" * 400),
        "max_gpu_hours must be a number above 0",
    ),
    "conditions": (
        _with("conditions:", "conditions: []\nx:").split("\nx:")[0],
        "at least one condition",
    ),
    "condition-not-a-mapping": (
        _with("  - label: none\n    segments: []\n", "  - none\n"),
        r"conditions\[0\] must be a mapping with a label and a schedule, not 'none'",
    ),
    "condition-key": (
        _with("controls: 2", "controls: 2\n    colour: red"),
        r"conditions\[1\] \(pulse\): unknown key 'colour'",
    ),
    "label": (
        _with("- label: none", "- label: No Push"),
        r"conditions\[0\]: label must be",
    ),
    "label-number": (
        _with("- label: none", "- label: 16"),
        r"conditions\[0\]: label must be text, .* YAML read 16; put it in quotes",
    ),
    "no-label": (
        _with("- label: none\n    segments: []", "- segments: []"),
        r"conditions\[0\] needs a label",
    ),
    "label-twice": (
        _with("- label: pulse", "- label: none"),
        "the label 'none' is used twice",
    ),
    "grid-segments": (
        _with("    segments: []", "    segments: []\n    grid: {start: [1]}"),
        r"\(none\): a grid goes with a shape and its params, not with segments",
    ),
    "grid-alone": (
        _with("    segments: []", "    grid: {start: [1]}"),
        r"\(none\): a grid goes with a shape and its params$",
    ),
    "grid-not-a-mapping": (
        _with("{length: [4, 16, 64]}", "[4, 16, 64]"),
        r"\(pulse\): grid must be a mapping of params to lists of values",
    ),
    "grid-of-nothing": (
        _with("{length: [4, 16, 64]}", "{}"),
        r"\(pulse\): grid must be a mapping of params to lists of values",
    ),
    "grid-params": (
        _with(
            "params: {unit: qwen3.5-4b:L14:N8949, alpha: 1.0, start: 0}", "params: 3"
        ),
        r"\(pulse\): shape pulse: params must be a mapping, not 3",
    ),
    "grid-and-params": (
        _with("{length: [4, 16, 64]}", "{start: [0, 2]}"),
        r"\(pulse\): start is in both params and grid",
    ),
    "grid-empty": (
        _with("[4, 16, 64]", "[]"),
        r"\(pulse\): grid length must be a list of values",
    ),
    "grid-twice": (
        _with("[4, 16, 64]", "[4, 16, 4]"),
        r"\(pulse\): grid length repeats 4",
    ),
    "grid-param": (
        _with("grid: {length: [4, 16, 64]}", "grid: {colour: [1]}"),
        "unknown param 'colour'",
    ),
    "controls": (_with("controls: 2", "controls: -1"), r"\(pulse\): controls must be"),
    "control-seed": (
        _with("    control_seed: 7\n", ""),
        r"\(pulse\): 2 controls need a control_seed",
    ),
    "control-seed-one": (
        _with("    controls: 2\n    control_seed: 7\n", "    controls: 1\n"),
        r"\(pulse\): 1 control needs a control_seed",
    ),
    "control-seed-alone": (
        _with("    segments: []", "    segments: []\n    control_seed: 3"),
        r"\(none\): control_seed goes with controls",
    ),
    "control-seed-range": (
        _with("control_seed: 7", "control_seed: -1"),
        r"\(pulse\): control_seed must be a whole number from 0",
    ),
    "controls-of-no-push": (
        _with(
            "    segments: []", "    segments: []\n    controls: 1\n    control_seed: 0"
        ),
        r"\(none\): it pushes no unit, so it cannot have controls",
    ),
    "shape": (
        _with("shape: pulse", "shape: wave"),
        r"conditions\[1\] \(pulse\): unknown shape 'wave'",
    ),
    "not-yaml": ("name: [", "is not valid YAML"),
    "not-a-mapping": ("- 1", "an experiment must be a mapping"),
}


@pytest.mark.parametrize("text, problem", REFUSALS.values(), ids=REFUSALS.keys())
def test_refusals_name_the_problem(text: str, problem: str) -> None:
    with pytest.raises(ValueError, match=problem):
        spec_from_yaml(text)
