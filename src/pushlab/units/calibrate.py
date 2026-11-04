"""Calibration: each layer's median residual size, the scale of a dose.

A dose of 1 pushes a neuron as far as the typical residual stream at its
layer (see units.dose). calibrate_norms measures that typical size once per
model: it runs a fixed set of short prompts through the chat template with
thinking off, takes the length of the residual stream leaving every decoder
layer at each token, leaves out each prompt's first token, and keeps each
layer's median. `pushlab calibrate` saves the result as
calibration/<model>-<revision>.json.
"""

import json
import math
import re
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from pushlab import PROJECT_ROOT
from pushlab.models.chat import chat_prompt_ids
from pushlab.models.discovery import find_decoder_layers
from pushlab.models.registry import REGISTRY
from pushlab.prompts import PROMPTS_DIR, read_prompt_set
from pushlab.store.records import _replace_atomically

PROMPTS_FILE = PROMPTS_DIR / "calibration.jsonl"
CALIBRATION_DIR = PROJECT_ROOT / "calibration"

# Stored with every calibration, so the file says what its numbers are.
MEASURE = (
    "median length of each decoder layer's output, taken in float32, with each "
    "prompt's first token left out; prompts in the chat template, thinking off"
)

# A commit or a branch name, safe to put in a file name.
REVISION = re.compile(r"[0-9A-Za-z][0-9A-Za-z._-]*")
SHA256 = re.compile(r"[0-9a-f]{64}")
# The calibration's own fields; the rest of a file is its info.
FIELDS = ("model", "revision", "norms", "prompt_set_sha256")


def _push_hooks(model: nn.Module) -> list[str]:
    """The modules with forward hooks that transformers did not add itself.

    transformers leaves hooks of its own on the decoder layers once a forward
    pass has asked for hidden states; they only record. Any other hook may
    change the residual stream, as a push does.
    """
    found = []
    for name, module in model.named_modules():
        hooks = [*module._forward_pre_hooks.values(), *module._forward_hooks.values()]
        if any(
            not str(getattr(h, "__module__", "")).startswith("transformers.")
            for h in hooks
        ):
            found.append(name or type(module).__name__)
    return found


def layer_outputs(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompts: Sequence[str]
) -> list[list[torch.Tensor]]:
    """The residual stream leaving every decoder layer, for each prompt.

    Each prompt runs alone, in the chat template with thinking off, so there
    is no padding. outputs[p][L] is the output of decoder layer L (counting
    from 0) for prompt p: one row per token, on the model's device and in
    its data type. That is never the embeddings, a layer's input or the
    final norm's output. A model with a push or another hook active is
    refused, and every hook added here is removed again, even after an error.

    Example:
        >>> outputs = layer_outputs(model, tokenizer, ["Hi"])  # doctest: +SKIP
        >>> len(outputs[0]), outputs[0][14].shape  # doctest: +SKIP
        (32, torch.Size([13, 2560]))
    """
    if not prompts:
        raise ValueError("calibration needs at least one prompt")
    if hooked := _push_hooks(model):
        raise ValueError(
            "calibration measures the model with no pushes active, but these "
            f"modules have forward hooks: {', '.join(hooked)}"
        )
    blocks = find_decoder_layers(model)
    device = next(model.parameters()).device
    current: list[torch.Tensor | None] = [None] * len(blocks)

    def recorder(layer: int) -> Callable[[nn.Module, object, object], None]:
        def record(module: nn.Module, args: object, output: object) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            current[layer] = hidden[0].detach()  # the batch's only row

        return record

    handles = [
        block.register_forward_hook(recorder(i)) for i, block in enumerate(blocks)
    ]
    outputs = []
    try:
        for prompt in prompts:
            ids = chat_prompt_ids(tokenizer, prompt, thinking=False)
            if len(ids) < 2:
                raise ValueError(
                    f"{prompt!r} has only {len(ids)} token in the template"
                )
            current[:] = [None] * len(blocks)
            with torch.no_grad():
                model(
                    input_ids=torch.tensor([ids], device=device),
                    use_cache=False,
                    logits_to_keep=1,
                )
            if any(output is None for output in current):
                raise RuntimeError("a decoder layer did not run")
            outputs.append(list(current))
    finally:
        for handle in handles:
            handle.remove()
    return outputs


def median_norms(outputs: Sequence[Sequence[torch.Tensor]]) -> list[float]:
    """Each layer's median token length, each prompt's first token left out.

    outputs[p][L] is prompt p's output of layer L, one row per token, as
    layer_outputs gives it. A token's length is its row's L2 norm, taken in
    float32 whatever the outputs' type. Each layer pools the lengths of all
    prompts; with an even count, the median is the mean of the middle two.

    Example:
        >>> rows = [[9.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 3.0], [4.0, 0.0]]
        >>> median_norms([[torch.tensor(rows)]])  # 9 left out: 1, 2, 3, 4
        [2.5]
    """
    counts = {len(per_prompt) for per_prompt in outputs}
    if len(counts) != 1:
        raise ValueError("every prompt needs the same number of layers")
    medians = []
    for layer in range(counts.pop()):
        lengths = [
            torch.linalg.vector_norm(per_prompt[layer][1:].float(), dim=-1)
            for per_prompt in outputs
        ]
        medians.append(statistics.median(torch.cat(lengths).tolist()))
    return medians


def calibrate_norms(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, prompts: Sequence[str]
) -> list[float]:
    """Each layer's median residual size, from layer 0: what a dose of 1 means.

    The prompts go through the chat template with thinking off, one at a
    time and with no push active. Layer L's size is the median length of
    decoder layer L's output over every token but each prompt's first.

    Example:
        >>> calibrate_norms(model, tokenizer, ["Hi", "Bye"])  # doctest: +SKIP
        [1.43, 2.07, ...]
    """
    return median_norms(layer_outputs(model, tokenizer, prompts))


@dataclass(frozen=True)
class Calibration:
    """One model revision's median residual size per layer, and its record.

    norms[L] is layer L's size. info holds how, when and with which library
    versions it was measured.

    Example:
        >>> calibration = Calibration("tiny", "main", (1.5, 2.5), "0" * 64)
        >>> calibration.norms[1]
        2.5
    """

    model: str
    revision: str
    norms: tuple[float, ...]
    prompt_set_sha256: str
    info: dict[str, Any] = field(default_factory=dict)


def calibration_path(model: str, revision: str, folder: Path | None = None) -> Path:
    """The file for one model revision: folder/<model>-<revision>.json.

    folder is the repository's calibration/ folder unless given.

    Example:
        >>> calibration_path("qwen3.5-4b", "851bf6e", Path("calibration"))
        PosixPath('calibration/qwen3.5-4b-851bf6e.json')
    """
    if not isinstance(revision, str) or not REVISION.fullmatch(revision):
        raise ValueError(f"revision must be a commit or branch name, not {revision!r}")
    return (folder or CALIBRATION_DIR) / f"{model}-{revision}.json"


def _check(calibration: Calibration) -> None:
    """Raise ValueError unless calibration can be saved and used for doses."""
    spec = REGISTRY.get(calibration.model)
    if spec is None:
        raise ValueError(f"{calibration.model!r} is not in the model registry")
    calibration_path(calibration.model, calibration.revision)  # checks the revision
    norms = calibration.norms
    if not isinstance(norms, tuple) or len(norms) != spec.layers:
        raise ValueError(
            f"norms needs one per layer: {spec.layers} for {spec.name}, not {norms!r}"
        )
    for layer, norm in enumerate(norms):
        number = isinstance(norm, int | float) and not isinstance(norm, bool)
        if not (number and math.isfinite(norm) and norm > 0):
            raise ValueError(
                f"norms[{layer}] must be a finite number above 0, not {norm!r}"
            )
    sha256 = calibration.prompt_set_sha256
    if not (isinstance(sha256, str) and SHA256.fullmatch(sha256)):
        raise ValueError("prompt_set_sha256 must be a SHA-256 in lowercase hex")
    if clash := sorted(set(FIELDS) & set(calibration.info)):
        raise ValueError(f"info cannot hold the calibration's own fields: {clash}")


def save_calibration(
    calibration: Calibration, folder: Path | None = None, *, overwrite: bool = False
) -> Path:
    """Check calibration and save it as JSON; return the file's path.

    An existing file for the same model revision raises FileExistsError
    unless overwrite is True. The file is written in one step, so it is
    never left half written.

    Example:
        >>> save_calibration(calibration)  # doctest: +SKIP
        PosixPath('.../calibration/qwen3.5-4b-851bf6e....json')
    """
    _check(calibration)
    record = {
        "model": calibration.model,
        "revision": calibration.revision,
        "norms": list(calibration.norms),
        "prompt_set_sha256": calibration.prompt_set_sha256,
        **calibration.info,
    }
    try:
        text = json.dumps(record, indent=2, allow_nan=False) + "\n"
    except (TypeError, ValueError) as error:
        raise ValueError(f"info must be plain JSON: {error}") from error
    path = calibration_path(calibration.model, calibration.revision, folder)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{path} already exists; pass overwrite=True to replace it"
        )
    _replace_atomically(path, lambda temporary: temporary.write_text(text))
    return path


def load_calibration(
    model: str, revision: str, folder: Path | None = None
) -> Calibration:
    """Read and check the calibration of model at revision.

    Example:
        >>> load_calibration("qwen3.5-4b", "851bf6e...").norms[14]  # doctest: +SKIP
        10.4...
    """
    path = calibration_path(model, revision, folder)
    try:
        record = json.loads(path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{model} at {revision} has no calibration in {path.parent}; on the "
            f"GPU pod, run `uv run pushlab calibrate --model {model}`"
        ) from None
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(record, dict):
        raise ValueError(f"{path} must hold a JSON object")
    if missing := sorted(set(FIELDS) - set(record)):
        raise ValueError(f"{path} lacks {', '.join(missing)}")
    if (record["model"], record["revision"]) != (model, revision):
        raise ValueError(
            f"{path} holds a calibration of {record['model']} at "
            f"{record['revision']}, not {model} at {revision}"
        )
    norms = record["norms"]
    if not isinstance(norms, list):
        raise ValueError(f"{path}: norms must be a list, not {norms!r}")
    info = {key: value for key, value in record.items() if key not in FIELDS}
    calibration = Calibration(
        model, revision, tuple(norms), record["prompt_set_sha256"], info
    )
    _check(calibration)
    return replace(calibration, norms=tuple(float(norm) for norm in norms))


def _refuse(message: str) -> int:
    print(f"pushlab calibrate: {message}", file=sys.stderr)
    return 1


def _shown(path: Path) -> str:
    """path relative to the repository when it is inside it."""
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def run_calibrate(model_name: str, overwrite: bool = False) -> int:
    """Measure model_name's residual sizes and save them; return the exit status.

    Returns 0 once the file is written, and 1 with the reason on stderr when
    it refuses: a broken prompt set before the model loads, and a model
    revision that already has a calibration unless overwrite is True.

    Example:
        >>> run_calibrate("qwen3.5-4b")  # doctest: +SKIP
        0
    """
    try:
        prompt_set = read_prompt_set(PROMPTS_FILE)
    except (OSError, ValueError) as error:
        return _refuse(f"cannot use {PROMPTS_FILE}: {error}")

    from pushlab.models.load import default_device, default_dtype, load_model
    from pushlab.store.environment import environment_info

    started = datetime.now(UTC)
    device = default_device()
    environment = environment_info(device)  # read before the checkout can change
    dtype = default_dtype(device)
    dtype_name = str(dtype).removeprefix("torch.")
    where = "GPU" if device == "cuda" else "CPU"
    print(f"Loading {model_name} in {dtype_name} on the {where}...")
    try:
        loaded = load_model(model_name, dtype, device=device)
    except FileNotFoundError as error:
        return _refuse(str(error))
    path = calibration_path(model_name, loaded.revision)
    if path.exists() and not overwrite:
        return _refuse(f"{path} already exists; pass --overwrite to measure again")

    clock = time.perf_counter()
    norms = calibrate_norms(loaded.model, loaded.tokenizer, prompt_set.prompts)
    seconds = time.perf_counter() - clock
    tokens = sum(
        len(chat_prompt_ids(loaded.tokenizer, prompt, thinking=False)) - 1
        for prompt in prompt_set.prompts
    )
    info = {
        "model_id": loaded.spec.hf_id,
        "measure": MEASURE,
        "prompt_set": _shown(PROMPTS_FILE),
        "prompt_set_version": prompt_set.version,
        "prompt_count": len(prompt_set.prompts),
        "token_count": tokens,
        "dtype": dtype_name,
        "device": device,
        "measured_at": started.isoformat(timespec="seconds"),
        **environment,
    }
    calibration = Calibration(
        model_name, loaded.revision, tuple(norms), prompt_set.sha256, info
    )
    try:
        path = save_calibration(calibration, overwrite=overwrite)
    except FileExistsError:  # written meanwhile
        return _refuse(f"{path} already exists; pass --overwrite to measure again")

    print(
        f"Measured {len(prompt_set.prompts)} prompts, {tokens} tokens after each "
        f"prompt's first, in {seconds:.1f} s."
    )
    print("Median length of the residual stream leaving each layer:")
    for start in range(0, len(norms), 4):
        row = range(start, min(start + 4, len(norms)))
        print("  " + "   ".join(f"L{layer:<3} {norms[layer]:9.3f}" for layer in row))
    print(f"Saved {path}")
    return 0
