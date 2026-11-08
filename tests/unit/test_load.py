"""Loading a registered model from its downloaded folder: pushlab.models.load."""

import dataclasses
import doctest
import json
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM

from pushlab.models import load
from pushlab.models.discovery import ModelInfo
from pushlab.models.download import MARKER
from pushlab.models.load import (
    default_device,
    default_dtype,
    default_models_dir,
    downloaded_revision,
    load_model,
    loading_class,
)
from pushlab.models.registry import REGISTRY, ModelSpec


def _commit(folder: Path) -> str:
    return json.loads((folder / MARKER).read_text())["commit"]


def test_qwen3_5_loads_with_the_model_cards_class() -> None:
    # Qwen3.5-4B's model card loads it with AutoModelForMultimodalLM; pushlab
    # then gives it text only.
    assert loading_class(REGISTRY["qwen3.5-4b"]) is AutoModelForMultimodalLM
    assert loading_class(REGISTRY["qwen3-8b"]) is AutoModelForCausalLM


def test_loads_the_vision_language_model_with_its_tokenizer(
    tiny_downloads: dict[str, ModelSpec], tiny_models_dir: Path
) -> None:
    loaded = load_model("tiny-vl", torch.float32, device="cpu")
    assert type(loaded.model).__name__ == "Qwen3_5ForConditionalGeneration"
    assert loaded.model.dtype == torch.float32
    assert loaded.model.device.type == "cpu"
    assert not loaded.model.training
    assert loaded.tokenizer.eos_token == "<|im_end|>"
    assert loaded.spec == tiny_downloads["tiny-vl"]
    assert loaded.info == ModelInfo(layers=4, hidden_size=64, mlp_size=128)
    assert loaded.folder == tiny_models_dir / "tiny-vl"
    assert loaded.revision == _commit(loaded.folder)


def test_loads_a_plain_causal_model_with_the_causal_class(
    tiny_downloads: dict[str, ModelSpec],
) -> None:
    loaded = load_model("tiny-lm", torch.float32, device="cpu")
    assert type(loaded.model).__name__ == "Qwen3ForCausalLM"
    assert loaded.info == ModelInfo(layers=2, hidden_size=64, mlp_size=128)


def test_loads_in_the_requested_dtype(tiny_downloads: dict[str, ModelSpec]) -> None:
    loaded = load_model("tiny-vl", torch.bfloat16, device="cpu")
    assert loaded.model.dtype == torch.bfloat16


def test_models_folder_can_be_given_directly(
    tiny_downloads: dict[str, ModelSpec],
    tiny_models_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PUSHLAB_MODELS_DIR")
    loaded = load_model(
        "tiny-lm", torch.float32, device="cpu", models_dir=tiny_models_dir
    )
    assert loaded.folder == tiny_models_dir / "tiny-lm"


def test_bfloat16_on_a_gpu_and_float32_on_a_cpu() -> None:
    assert default_dtype("cuda") == torch.bfloat16
    assert default_dtype(torch.device("cuda", 0)) == torch.bfloat16
    assert default_dtype("cpu") == torch.float32


def test_the_gpu_is_used_when_there_is_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert default_device() == "cuda"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert default_device() == "cpu"


def test_models_live_on_the_pods_volume_unless_told_otherwise(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PUSHLAB_MODELS_DIR", raising=False)
    assert default_models_dir() == Path("/workspace/models")
    monkeypatch.setenv("PUSHLAB_MODELS_DIR", "/data/models")
    assert default_models_dir() == Path("/data/models")


def test_a_model_that_is_not_downloaded_is_refused_with_a_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUSHLAB_MODELS_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="setup_pod.sh"):
        load_model("qwen3.5-4b", torch.bfloat16, device="cpu")


def test_the_downloaded_revision_is_read_without_loading_the_model(
    tiny_downloads: dict[str, ModelSpec],
    tiny_models_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_loading(*args: object, **kwargs: object) -> None:
        raise AssertionError("downloaded_revision must not load the model")

    with monkeypatch.context() as patch:
        patch.setattr(AutoModelForMultimodalLM, "from_pretrained", no_loading)
        assert downloaded_revision("tiny-vl") == _commit(tiny_models_dir / "tiny-vl")
    assert (
        downloaded_revision("tiny-vl")
        == load_model("tiny-vl", torch.float32, device="cpu").revision
    )


def test_a_revision_needs_a_finished_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUSHLAB_MODELS_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError, match="setup_pod.sh"):
        downloaded_revision("qwen3.5-4b")
    with pytest.raises(ValueError, match="not in the model registry"):
        downloaded_revision("qwen9-1t")


def test_a_download_of_another_revision_is_refused(
    tiny_downloads: dict[str, ModelSpec], monkeypatch: pytest.MonkeyPatch
) -> None:
    newer = dataclasses.replace(tiny_downloads["tiny-vl"], revision="v2")
    monkeypatch.setitem(REGISTRY, "tiny-vl", newer)
    with pytest.raises(FileNotFoundError, match="tiny-vl"):
        load_model("tiny-vl", torch.float32, device="cpu")


def test_a_model_of_the_wrong_shape_is_refused(
    tiny_downloads: dict[str, ModelSpec], monkeypatch: pytest.MonkeyPatch
) -> None:
    wrong = dataclasses.replace(tiny_downloads["tiny-vl"], mlp_size=256)
    monkeypatch.setitem(REGISTRY, "tiny-vl", wrong)
    with pytest.raises(ValueError, match="registry"):
        load_model("tiny-vl", torch.float32, device="cpu")


@pytest.mark.parametrize(
    ("name", "problem"),
    [("tiny", "built by the tests"), ("qwen9-1t", "not in the model registry")],
)
def test_the_tiny_test_model_and_unknown_names_are_refused(
    name: str, problem: str
) -> None:
    with pytest.raises(ValueError, match=problem):
        load_model(name, torch.float32, device="cpu")


def test_docstring_examples_run() -> None:
    result = doctest.testmod(load)
    assert result.attempted > 0
    assert result.failed == 0


def test_the_tokenizer_loads_alone(tiny_downloads: dict[str, ModelSpec]) -> None:
    from pushlab.models.load import load_tokenizer

    alone = load_tokenizer("tiny-vl")
    loaded = load_model("tiny-vl", torch.float32, device="cpu")
    text = "Say hi."
    assert alone(text)["input_ids"] == loaded.tokenizer(text)["input_ids"]
    with pytest.raises(FileNotFoundError, match="is not downloaded"):
        load_tokenizer("qwen3.5-4b")
