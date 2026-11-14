"""Golden cases (decision 0023): the cases file's checks and refusals, a
reply's fingerprint, and each case run alone with the sizes its calibration
names. The fingerprints file is test_golden_fingerprints.py's, and `pushlab
golden record` test_golden_command.py's.

The made-up cases come from tests/unit/golden_helpers.py: neurons of the
tiny models, and a set of three across their layers 0 and 1. None comes
from the Neuron Persona Library.
"""

import copy
import doctest
import hashlib
import importlib.util
import re
from collections import Counter
from pathlib import Path

import pytest
import torch

from pushlab.engine import generate as generate_module
from pushlab.engine.decoding import Decoding
from pushlab.engine.generate import Generation, generate
from pushlab.models.load import load_model
from pushlab.prompts import PROMPTS_DIR, read_prompt_set
from pushlab.runs import golden, golden_cases, golden_command, golden_fingerprints
from pushlab.runs.golden import fingerprint, fixed_norms, run_case, run_cases
from pushlab.runs.golden_cases import (
    GoldenCase,
    GoldenCases,
    cases_from_mapping,
    cases_path,
    read_cases,
)
from pushlab.schedules.shapes import constant
from pushlab.schedules.yaml_text import dump_yaml
from pushlab.units import calibrate
from pushlab.units.calibrate import load_calibration
from pushlab.units.calibration_mode import per_prompt_norms
from pushlab.units.neuron import NeuronUnit
from pushlab.units.neuron_set import NeuronSet

# The made-up cases, loaded by path, as test modules import nothing from
# one another.
_SPEC = importlib.util.spec_from_file_location(
    "golden_helpers", Path(__file__).with_name("golden_helpers.py")
)
helpers = importlib.util.module_from_spec(_SPEC)  # type: ignore[arg-type]
_SPEC.loader.exec_module(helpers)  # type: ignore[union-attr]
set_key, drop_key = helpers.set_key, helpers.drop_key
SPARE = {"entries": [[0, 1, 1.0]], "source": "made up"}


def _kind(case: GoldenCase) -> tuple:
    """How a case runs: its decoding, and its push's kind of unit, dose and
    calibration."""
    push, decoding = case.push, case.decoding
    pushed = (None, None, None)
    if push is not None:
        pushed = (type(push.unit).__name__, push.alpha, push.calibration)
    return (decoding.strategy, decoding.temperature, decoding.seed, *pushed)


def test_the_qwen3_8b_cases_run_three_everyday_prompts_six_ways() -> None:
    cases = read_cases(cases_path("qwen3-8b"))
    assert (cases.model, cases.dtype, cases.max_new_tokens) == (
        "qwen3-8b",
        "bfloat16",
        32,
    )
    six = Counter(
        [
            ("greedy", None, None, None, None, None),
            ("greedy", None, None, "NeuronUnit", 1.0, "fixed"),
            ("greedy", None, None, "NeuronUnit", 1.0, "per_prompt"),
            ("greedy", None, None, "NeuronSet", 1.0, "per_prompt"),
            ("sample", 0.7, 1, None, None, None),
            ("sample", 0.7, 2, None, None, None),
        ]
    )
    prompts = list(dict.fromkeys(case.prompt for case in cases.cases))
    assert len(prompts) == 3
    for prompt in prompts:
        kinds = Counter(_kind(case) for case in cases.cases if case.prompt == prompt)
        assert kinds == six, prompt
    # One made-up neuron, and one made-up set of three neurons in two layers,
    # which needs a size of its own: the file names them, the code never.
    units = {case.push.unit for case in cases.cases if case.push is not None}
    (neuron,) = [unit for unit in units if isinstance(unit, NeuronUnit)]
    (trio,) = [unit for unit in units if isinstance(unit, NeuronSet)]
    assert neuron.model_name == trio.model_name == "qwen3-8b"
    assert (len(trio.entries), len(trio.layers)) == (3, 2) and trio.size is not None


def test_no_golden_prompt_comes_from_a_prompt_set() -> None:
    def same(prompt: str) -> str:
        return " ".join(prompt.casefold().split())

    ours = {same(case.prompt) for case in read_cases(cases_path("qwen3-8b")).cases}
    for path in sorted(PROMPTS_DIR.glob("*.jsonl")):
        if not path.name.endswith("-answers.jsonl"):
            theirs = {same(prompt) for prompt in read_prompt_set(path).prompts}
            assert not ours & theirs, path.name


def test_made_up_cases_read_as_written() -> None:
    cases = cases_from_mapping(helpers.golden_data())
    assert [case.case_id for case in cases.cases] == helpers.IDS
    assert cases.torch_dtype == torch.float32 and cases.needs_fixed_calibration
    trio = cases.cases[3].push.unit
    assert isinstance(trio, NeuronSet)
    assert (str(trio), trio.size) == ("tiny:set:trio", 0.2)
    assert cases.cases[3].as_data()["push"] == {  # the set's definition beside its ID
        "unit": "tiny:set:trio",
        "set": {
            "entries": [[0, 5, 1.0], [1, 9, -0.5], [1, 17, 0.75]],
            "size": 0.2,
            "source": "made up for the tests",
        },
        "alpha": 1.0,
        "calibration": "per_prompt",
    }
    assert cases.cases[4].decoding == Decoding(
        strategy="sample", max_new_tokens=5, temperature=0.7, seed=1
    )


REFUSALS = {
    "a key missing": (drop_key("dtype"), "lacks dtype"),
    "a key unknown": (set_key("colour", "red"), "has colour, which it does not take"),
    "another version": (set_key("version", 2), "version must be 1"),
    "a model not registered": (set_key("model", "gpt-2"), "'gpt-2' is not in the"),
    "float16": (set_key("dtype", "float16"), "dtype must be bfloat16 or float32"),
    "no reply": (set_key("max_new_tokens", 0), "max_new_tokens must be a whole"),
    "a prompt with no text": (set_key("prompts.hi", " "), "must be the prompt's text"),
    "a prompt no case asks": (set_key("prompts.bye", "Bye."), "no case asks bye"),
    "an unknown prompt": (
        set_key("cases.0.prompt", "bye"),
        "one of the file's prompts",
    ),
    "a case ID twice": (set_key("cases.1.id", "hi-greedy"), "is cases[0]'s already"),
    "an ID unlike a name": (set_key("cases.0.id", "Hi greedy"), "lowercase letters"),
    "no cases": (set_key("cases", []), "a list of at least one case"),
    "greedy with a seed": (set_key("cases.0.decoding.seed", 1), "takes no seed"),
    "sampled with no seed": (drop_key("cases.4.decoding.seed"), "needs a seed"),
    "top_k": (set_key("cases.4.decoding.top_k", 40), "has top_k, which it does not"),
    "another model's neuron": (
        set_key("cases.1.push.unit", "qwen3-8b:L1:N5"),
        "is a neuron of qwen3-8b, but the cases are tiny's",
    ),
    "a layer out of range": (
        set_key("cases.1.push.unit", "tiny:L2:N5"),
        "layer 2 is out of range",
    ),
    "a set the file lacks": (
        set_key("cases.3.push.unit", "tiny:set:other"),
        "is neither a set the file's sets define nor a neuron",
    ),
    "a dose above the ceiling": (
        set_key("cases.3.push.alpha", -1.5),
        "the combined push reaches 1.5",
    ),
    "a dose as text": (set_key("cases.1.push.alpha", "1"), "must be a finite number"),
    "a calibration unknown": (
        set_key("cases.1.push.calibration", "auto"),
        "calibration must be fixed or per_prompt",
    ),
    "no calibration": (drop_key("cases.1.push.calibration"), "lacks calibration"),
    "a set no case pushes": (
        set_key("sets.tiny:set:spare", SPARE),
        "no case pushes tiny:set:spare",
    ),
    "another model's set": (
        set_key("sets.qwen3-8b:set:spare", SPARE),
        "is a set of qwen3-8b's neurons",
    ),
    "a set ID unlike one": (set_key("sets.spare", SPARE), "'spare' is not a set's ID"),
    "a set across layers with no size": (
        drop_key("sets.tiny:set:trio.size"),
        "it needs a size",
    ),
}


@pytest.mark.parametrize("mistake", list(REFUSALS))
def test_each_mistake_in_a_cases_file_is_refused_by_name(mistake: str) -> None:
    change, message = REFUSALS[mistake]
    data = copy.deepcopy(helpers.golden_data())
    change(data)
    with pytest.raises(ValueError, match=re.escape(message)):
        cases_from_mapping(data)


def test_reading_a_cases_file_names_it_in_what_it_refuses(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot read the golden cases .*none.yaml"):
        read_cases(tmp_path / "none.yaml")
    twice = tmp_path / "twice.yaml"
    twice.write_text("version: 1\nversion: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"(?s)twice\.yaml: .*duplicate key 'version'"):
        read_cases(twice)
    data = helpers.golden_data()
    data["dtype"] = "float16"
    wrong = tmp_path / "wrong.yaml"
    wrong.write_text(dump_yaml(data), encoding="utf-8")
    with pytest.raises(ValueError, match=r"wrong\.yaml: dtype must be"):
        read_cases(wrong)


def test_a_fingerprint_is_the_sha256_of_the_token_ids_as_compact_json() -> None:
    reply = Generation("x" * 100, tuple(range(10)), "max_new_tokens", (9,))
    found = fingerprint(reply)
    assert found.sha256 == hashlib.sha256(b"[0,1,2,3,4,5,6,7,8,9]").hexdigest()
    assert (found.token_count, found.first_tokens) == (10, tuple(range(8)))
    assert found.text_start == "x" * golden.TEXT_START
    # Any token changes it, the last too; the text alone never does.
    other = Generation("x" * 100, (*range(9), 10), "max_new_tokens", (9,))
    assert fingerprint(other).sha256 != found.sha256
    same = Generation("y", tuple(range(10)), "end_of_turn", (9,))
    assert fingerprint(same).sha256 == found.sha256


@pytest.fixture
def tiny_cases(tiny_vl_calibration: Path) -> GoldenCases:
    """The made-up cases for tiny-vl, downloaded and calibrated."""
    return cases_from_mapping(helpers.golden_data("tiny-vl"))


def test_each_case_runs_alone_in_one_generate_call(
    tiny_cases: GoldenCases, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = load_model("tiny-vl", torch.float32, device="cpu")
    calls = []
    real = generate_module.generate

    def spy(model, tokenizer, prompts, *args, schedules, **kwargs):  # type: ignore[no-untyped-def]
        calls.append((list(prompts), [schedule.name for schedule in schedules]))
        return real(model, tokenizer, prompts, *args, schedules=schedules, **kwargs)

    monkeypatch.setattr(generate_module, "generate", spy)
    fixed = fixed_norms(tiny_cases, loaded.revision)
    seen = []
    found = run_cases(
        loaded.model,
        loaded.tokenizer,
        tiny_cases,
        fixed,
        lambda case, reply: seen.append((case.case_id, reply)),
    )
    assert calls == [(["Say hi."], [case_id]) for case_id in helpers.IDS]
    assert list(found) == helpers.IDS and seen == list(found.items())
    assert all(1 <= reply.token_count <= 5 for reply in found.values())


def test_a_case_pushes_its_unit_with_the_sizes_its_calibration_names(
    tiny_cases: GoldenCases,
) -> None:
    loaded = load_model("tiny-vl", torch.float32, device="cpu")
    model, tokenizer = loaded.model, loaded.tokenizer
    fixed = fixed_norms(tiny_cases, loaded.revision)
    assert fixed == load_calibration("tiny-vl", loaded.revision).norms
    own = per_prompt_norms(model, tokenizer, ["Say hi."])
    greedy = Decoding(strategy="greedy", max_new_tokens=5)
    by_id = {case.case_id: case for case in tiny_cases.cases}

    def alone(unit: NeuronUnit | NeuronSet, norms: object) -> Generation:
        schedules = [constant(unit, 1.0)]
        (reply,) = generate(
            model, tokenizer, ["Say hi."], [], greedy, norms, schedules=schedules
        )
        return reply

    neuron = NeuronUnit.parse("tiny-vl:L1:N5")
    replies = {
        case_id: run_case(model, tokenizer, by_id[case_id], fixed)
        for case_id in ("hi-neuron-fixed", "hi-neuron-per-prompt")
    }
    assert replies["hi-neuron-fixed"] == alone(neuron, fixed)
    assert replies["hi-neuron-per-prompt"] == alone(neuron, own)
    # The prompt's own sizes are not the fixed ones, so the amounts differ.
    assert replies["hi-neuron-fixed"].amounts != replies["hi-neuron-per-prompt"].amounts
    # A set with a size pushes the same under either calibration (decision 0019).
    trio = by_id["hi-set-per-prompt"].push.unit
    assert run_case(model, tokenizer, by_id["hi-set-per-prompt"], fixed) == alone(
        trio, fixed
    )


def test_a_fixed_case_needs_the_fixed_calibration(
    tiny_downloads: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(calibrate, "CALIBRATION_DIR", tmp_path / "calibration")
    cases = cases_from_mapping(helpers.golden_data("tiny-vl"))
    loaded = load_model("tiny-vl", torch.float32, device="cpu")
    with pytest.raises(FileNotFoundError, match="pushlab calibrate --model tiny-vl"):
        fixed_norms(cases, loaded.revision)
    with pytest.raises(ValueError, match="hi-neuron-fixed: the fixed calibration"):
        run_case(loaded.model, loaded.tokenizer, cases.cases[1], None)
    data = helpers.golden_data("tiny-vl")
    del data["cases"][1]  # per_prompt alone needs no fixed calibration
    assert fixed_norms(cases_from_mapping(data), loaded.revision) is None


@pytest.mark.parametrize(
    "module", [golden, golden_cases, golden_fingerprints, golden_command]
)
def test_docstring_examples_run(module: object) -> None:
    result = doctest.testmod(module)  # type: ignore[arg-type]
    assert result.attempted > 0
    assert result.failed == 0
