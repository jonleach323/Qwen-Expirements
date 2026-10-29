"""Shared test setup: seeds, GPU-test skipping, and the tiny test models."""

import json
import os
import random
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import torch

# Tests never download from Hugging Face; an accidental download fails fast.
# Set before transformers is imported: huggingface_hub reads it once, then.
os.environ["HF_HUB_OFFLINE"] = "1"

from tokenizers import Tokenizer, decoders, models, pre_tokenizers  # noqa: E402
from transformers import (  # noqa: E402
    AutoTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from pushlab.models.download import MARKER  # noqa: E402
from pushlab.models.registry import REGISTRY, ModelSpec  # noqa: E402

pytest_plugins = ["pytester"]

TEST_SEED = 0


def _gpu_requested(markexpr: str) -> bool:
    """True if gpu appears in a -m expression under an even number of `not`s."""
    negated = [False]  # one entry per open parenthesis
    pending_not = False
    for token in re.findall(r"[()]|[^\s()]+", markexpr):
        if token == "not":
            pending_not = not pending_not
        elif token == "(":
            negated.append(negated[-1] != pending_not)
            pending_not = False
        elif token == ")":
            negated.pop()
        else:
            if token == "gpu" and negated[-1] == pending_not:
                return True
            pending_not = False
    return False


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip gpu-marked tests unless the run was started with -m gpu."""
    if _gpu_requested(config.getoption("markexpr")):
        return
    skip_gpu = pytest.mark.skip(reason="needs a GPU; run with -m gpu")
    for item in items:
        if item.get_closest_marker("gpu") is not None:
            item.add_marker(skip_gpu)


def _seed_everything() -> None:
    random.seed(TEST_SEED)
    np.random.seed(TEST_SEED)
    torch.manual_seed(TEST_SEED)  # also seeds every GPU


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Seed before any fixture is set up, so module-scoped ones are fixed too."""
    _seed_everything()


@pytest.fixture(autouse=True)
def fixed_seeds() -> int:
    """Seed Python, NumPy and torch again right before each test runs."""
    _seed_everything()
    return TEST_SEED


def _build_tiny_model() -> Qwen3ForCausalLM:
    config = Qwen3Config(
        num_hidden_layers=2,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        # Qwen3 sets head_dim apart from the hidden size. Here the attention
        # width (4 x 32 = 128) differs from the hidden size (64), as it does
        # in Qwen3.5-4B (16 x 256 vs 2560). The class default would be 128.
        head_dim=32,
        vocab_size=512,
        dtype="float32",
    )
    # Seed only inside a forked random state, so the weights are always the
    # same and the test's own random numbers are left untouched.
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(TEST_SEED)
        model = Qwen3ForCausalLM(config)
    return model.to(torch.float32).eval()


@pytest.fixture
def make_tiny_model() -> Callable[[], Qwen3ForCausalLM]:
    """A function that builds a fresh tiny model, identical on every call."""
    return _build_tiny_model


@pytest.fixture
def tiny_model(make_tiny_model: Callable[[], Qwen3ForCausalLM]) -> Qwen3ForCausalLM:
    """A 2-layer Qwen3-style model with seeded random weights, float32, eval mode."""
    return make_tiny_model()


# A Qwen-style chat template cut down to what pushlab uses: user and assistant
# turns, and the empty thinking block that turns thinking off. The real
# Qwen3.5 template ends its prompt the same way.
TINY_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] }}"
    "{{ '<|im_end|>\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}"
    "{% if enable_thinking is defined and enable_thinking is false %}"
    "{{ '<think>\n\n</think>\n\n' }}"
    "{% else %}"
    "{{ '<think>\n' }}"
    "{% endif %}"
    "{% endif %}"
)

# Saved tiny checkpoints, registered under these names by the tiny_downloads
# fixture. Unlike "tiny", they load through pushlab's real loading code.
TINY_DOWNLOADS = {
    spec.name: spec
    for spec in [
        ModelSpec(
            name="tiny-vl",
            hf_id="local:tiny-qwen3.5",
            revision="main",
            layers=4,
            hidden_size=64,
            mlp_size=128,
            kind="vision_language",
        ),
        ModelSpec(
            name="tiny-lm",
            hf_id="local:tiny-qwen3",
            revision="main",
            layers=2,
            hidden_size=64,
            mlp_size=128,
            kind="causal_lm",
        ),
    ]
}
TINY_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _build_tiny_tokenizer() -> PreTrainedTokenizerFast:
    """A byte-level tokenizer like Qwen's, with no merges: one token per byte."""
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {character: index for index, character in enumerate(alphabet)}
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="<|im_end|>",
        pad_token="<|endoftext|>",
        extra_special_tokens=["<|im_start|>"],
        chat_template=TINY_CHAT_TEMPLATE,
    )
    tokenizer.add_tokens(["<think>", "</think>"])
    return tokenizer


def _build_tiny_vl_model() -> Qwen3_5ForConditionalGeneration:
    tokenizer = _build_tiny_tokenizer()
    text_config = dict(
        # Three linear-attention layers, then one full-attention layer.
        num_hidden_layers=4,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        vocab_size=len(tokenizer),
        # Like Qwen3.5-4B, whose config ends text at <|endoftext|> while its
        # chat turns end at <|im_end|>.
        eos_token_id=tokenizer.convert_tokens_to_ids("<|endoftext|>"),
    )
    # The vision tower is never used, but the default one is 1.8 GB.
    vision_config = dict(
        depth=1,
        hidden_size=16,
        intermediate_size=32,
        num_heads=2,
        out_hidden_size=64,
        num_position_embeddings=16,
    )
    config = Qwen3_5Config(text_config=text_config, vision_config=vision_config)
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(TEST_SEED)
        model = Qwen3_5ForConditionalGeneration(config)
    return model.to(torch.float32).eval()


@pytest.fixture
def tiny_vl_model() -> Qwen3_5ForConditionalGeneration:
    """A 4-layer Qwen3.5-style model in its vision-language wrapper, float32, eval."""
    return _build_tiny_vl_model()


@pytest.fixture(scope="session")
def tiny_models_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A models folder like the pod's, with finished downloads of the tiny models.

    tiny-vl is a Qwen3.5-style checkpoint and tiny-lm a Qwen3 one. Both come
    with the tiny tokenizer and a download record, as if setup_pod.sh had
    downloaded them.
    """
    models_dir = tmp_path_factory.mktemp("models")
    builders = {"tiny-vl": _build_tiny_vl_model, "tiny-lm": _build_tiny_model}
    for name, build in builders.items():
        spec = TINY_DOWNLOADS[name]
        folder = models_dir / name
        build().save_pretrained(folder)
        _build_tiny_tokenizer().save_pretrained(folder)
        record = {"hf_id": spec.hf_id, "revision": spec.revision, "commit": TINY_COMMIT}
        (folder / MARKER).write_text(json.dumps(record))
    return models_dir


@pytest.fixture
def tiny_downloads(
    monkeypatch: pytest.MonkeyPatch, tiny_models_dir: Path
) -> dict[str, ModelSpec]:
    """Register tiny-vl and tiny-lm, and point PUSHLAB_MODELS_DIR at them.

    Returns the registered specs by name. Each also records TINY_COMMIT as
    the commit it was downloaded at.
    """
    for name, spec in TINY_DOWNLOADS.items():
        monkeypatch.setitem(REGISTRY, name, spec)
    monkeypatch.setenv("PUSHLAB_MODELS_DIR", str(tiny_models_dir))
    return TINY_DOWNLOADS


@pytest.fixture
def tiny_tokenizer(tiny_models_dir: Path) -> PreTrainedTokenizerBase:
    """The tiny models' tokenizer, loaded back from disk as pushlab loads it."""
    return AutoTokenizer.from_pretrained(tiny_models_dir / "tiny-vl")
