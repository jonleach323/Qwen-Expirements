"""Model families (pushlab.scoring.family): a judge of the tested model's
family is refused."""

import doctest

import pytest

from pushlab.scoring import family as family_module
from pushlab.scoring.family import check_family, family, same_family


@pytest.mark.parametrize(
    ("judge_model", "model", "same"),
    [
        ("granite-4.1-8b", "qwen3-8b", False),
        ("ibm-granite/granite-4.1-8b", "qwen3.5-4b", False),
        ("claude-haiku-5-5", "qwen3-8b", False),
        ("Qwen/Qwen2.5-7B-Instruct", "qwen3-8b", True),
        ("qwen3-8b", "qwen3-8b@b968826d", True),
        ("deepseek-r1-distill-qwen-7b", "qwen3-8b", True),
        ("dolphin-2.9", "phi-4", False),  # "phi" inside a word is no family
        ("fake-judge", "tiny-vl", False),
        ("tiny-judge", "tiny-vl", True),
    ],
)
def test_a_judge_of_the_tested_models_family_is_refused(
    judge_model: str, model: str, same: bool
) -> None:
    assert same_family(judge_model, model) is same
    if same:
        with pytest.raises(ValueError, match="choose a judge of another family"):
            check_family(judge_model, [model])
    else:
        check_family(judge_model, [model])


def test_a_models_family_is_the_first_word_of_its_name() -> None:
    assert family("meta-llama/Llama-3.1-8B") == "llama"
    assert family("google/gemma-3-12b-it") == "gemma"
    assert family("qwen3-8b@b968826d") == "qwen"


def test_docstring_examples_run() -> None:
    result = doctest.testmod(family_module)
    assert result.attempted > 0
    assert result.failed == 0
