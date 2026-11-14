"""Shared test setup: seeds, GPU-test skipping, the tiny test models, and
Hugging Face's generate() as the reference for greedy replies."""

import importlib
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
# Tests never call the Anthropic API either, now that its SDK is installed:
# no key, and the SDK pointed at a closed local port, so a test that makes a
# real client by mistake fails at once instead of making a paid call. A test
# that needs a key sets a made-up one, with a stand-in for the SDK.
os.environ.pop("ANTHROPIC_API_KEY", None)
os.environ["ANTHROPIC_BASE_URL"] = "http://127.0.0.1:9"

from tokenizers import Tokenizer, decoders, models, pre_tokenizers  # noqa: E402
from transformers import (  # noqa: E402
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    Qwen3_5Config,
    Qwen3_5ForConditionalGeneration,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from pushlab.models.chat import chat_prompt_ids, stop_token_ids  # noqa: E402
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


# The functions Qwen3.5's linear-attention layers call. transformers points
# each at a fast GPU kernel when causal-conv1d or flash-linear-attention is
# installed, as on the pod, and those kernels refuse CPU tensors.
QWEN3_5_KERNEL_FUNCTIONS = (
    "causal_conv1d_fn",
    "causal_conv1d_update",
    "torch_chunk_gated_delta_rule",
    "torch_recurrent_gated_delta_rule",
)
_PYTORCH_SWITCH = pytest.StashKey[pytest.MonkeyPatch]()


def _run_qwen3_5_on_pytorch(patch: pytest.MonkeyPatch) -> None:
    """Point Qwen3.5's linear-attention functions at transformers' PyTorch code."""
    module = importlib.import_module("transformers.models.qwen3_5.modeling_qwen3_5")
    for name in QWEN3_5_KERNEL_FUNCTIONS:
        function = getattr(module, name)
        # functools.wraps keeps the PyTorch function as __wrapped__. A test
        # session started inside a test finds it already switched.
        patch.setattr(module, name, getattr(function, "__wrapped__", function))


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Seed before any fixture is set up, so module-scoped ones are fixed too.

    For the same reason, every test not marked gpu runs Qwen3.5 on
    transformers' PyTorch code from here on. Then a Qwen3.5 model in any CPU
    test, or in any fixture, runs on the CPU even where the GPU kernels are
    installed, so the unit tests pass on the pod too. GPU tests keep the
    fast kernels.
    """
    _seed_everything()
    if item.get_closest_marker("gpu") is None:
        patch = pytest.MonkeyPatch()
        _run_qwen3_5_on_pytorch(patch)
        item.stash[_PYTORCH_SWITCH] = patch


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item: pytest.Item) -> None:
    """Undo the switch to PyTorch code once the test's fixtures are gone."""
    if (patch := item.stash.get(_PYTORCH_SWITCH, None)) is not None:
        patch.undo()


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
def make_tiny_vl_model() -> Callable[[], Qwen3_5ForConditionalGeneration]:
    """A function that builds a fresh tiny-vl model, identical on every call."""
    return _build_tiny_vl_model


@pytest.fixture
def tiny_vl_model(
    make_tiny_vl_model: Callable[[], Qwen3_5ForConditionalGeneration],
) -> Qwen3_5ForConditionalGeneration:
    """A 4-layer Qwen3.5-style model in its vision-language wrapper, float32, eval."""
    return make_tiny_vl_model()


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


# tiny-vl's own residual sizes, as calibrate_norms measures them on
# prompts/calibration.jsonl, so a test dose pushes as hard as a real one:
# larger sizes would swamp the model, and every dose would give one reply.
TINY_VL_NORMS = (0.1636, 0.1649, 0.1653, 0.1698)


@pytest.fixture
def tiny_vl_calibration(
    tiny_downloads: dict[str, ModelSpec],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """tiny-vl downloaded and calibrated with TINY_VL_NORMS; the calibration file.

    The file goes in a fresh folder that stands in for calibration/.
    """
    from pushlab.units import calibrate

    monkeypatch.setattr(calibrate, "CALIBRATION_DIR", tmp_path / "calibration")
    calibration = calibrate.Calibration(
        "tiny-vl", TINY_COMMIT, TINY_VL_NORMS, "0" * 64, {"note": "made up"}
    )
    return calibrate.save_calibration(calibration)


@pytest.fixture
def tiny_tokenizer(tiny_models_dir: Path) -> PreTrainedTokenizerBase:
    """The tiny models' tokenizer, loaded back from disk as pushlab loads it."""
    return AutoTokenizer.from_pretrained(tiny_models_dir / "tiny-vl")


def _hugging_face_greedy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompts: list[str],
    max_new_tokens: int,
) -> tuple[list[list[int]], list[torch.Tensor]]:
    rows = [chat_prompt_ids(tokenizer, prompt, thinking=False) for prompt in prompts]
    width = max(len(row) for row in rows)
    pad = tokenizer.pad_token_id
    stops = stop_token_ids(model, tokenizer)
    with torch.no_grad():
        output = model.generate(
            input_ids=torch.tensor([[pad] * (width - len(row)) + row for row in rows]),
            attention_mask=torch.tensor(
                [[0] * (width - len(row)) + [1] * len(row) for row in rows]
            ),
            do_sample=False,
            max_new_tokens=max_new_tokens,
            eos_token_id=stops,
            pad_token_id=pad,
            output_logits=True,
            return_dict_in_generate=True,
        )
    replies = []
    for tokens in output.sequences[:, width:].tolist():
        end = next((i + 1 for i, t in enumerate(tokens) if t in stops), len(tokens))
        replies.append(tokens[:end])  # up to its first stop token, which it keeps
    return replies, [step.float() for step in output.logits]


@pytest.fixture
def hf_greedy() -> Callable[..., tuple[list[list[int]], list[torch.Tensor]]]:
    """Hugging Face's own generate(), the reference for greedy replies.

    Called with a model, a tokenizer, prompts and max_new_tokens, it renders
    each prompt in the chat template with thinking off, as pushlab does,
    pads them on the left into one batch with an attention mask, and returns
    each reply's tokens up to and including its first stop token, and each
    step's scores for the next token in float32: (rows, vocabulary). A row's
    scores after its reply has ended belong to no reply.
    """
    return _hugging_face_greedy
