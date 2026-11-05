"""`pushlab calibrate`: measure a downloaded model's residual sizes and save them.

The measuring and the file belong to pushlab.units.calibrate. This is the
command around them: it loads the model, refuses what it must, and prints
one size per layer.
"""

import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pushlab import PROJECT_ROOT
from pushlab.models.chat import chat_prompt_ids
from pushlab.prompts import read_prompt_set
from pushlab.units import calibrate


def summary_lines(norms: Sequence[float]) -> list[str]:
    """The sizes in rows of four layers, as `pushlab calibrate` prints them.

    Example:
        >>> print("\\n".join(summary_lines([2.534, 3.078, 3.326, 4.679, 5.584])))
          L0       2.534   L1       3.078   L2       3.326   L3       4.679
          L4       5.584
    """
    starts = range(0, len(norms), 4)
    rows = [range(start, min(start + 4, len(norms))) for start in starts]
    return [
        "  " + "   ".join(f"L{layer:<3} {norms[layer]:9.3f}" for layer in row)
        for row in rows
    ]


def _refuse(message: str) -> int:
    print(f"pushlab calibrate: {message}", file=sys.stderr)
    return 1


def _shown(path: Path) -> str:
    """path relative to the repository when it is inside it."""
    if path.is_relative_to(PROJECT_ROOT):
        return path.relative_to(PROJECT_ROOT).as_posix()
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
    prompts_file = calibrate.PROMPTS_FILE
    try:
        prompt_set = read_prompt_set(prompts_file)
    except (OSError, ValueError) as error:
        return _refuse(f"cannot use {prompts_file}: {error}")

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
    path = calibrate.calibration_path(model_name, loaded.revision)
    if path.exists() and not overwrite:
        return _refuse(f"{path} already exists; pass --overwrite to measure again")

    clock = time.perf_counter()
    norms = calibrate.calibrate_norms(
        loaded.model, loaded.tokenizer, prompt_set.prompts
    )
    seconds = time.perf_counter() - clock
    tokens = sum(
        len(chat_prompt_ids(loaded.tokenizer, prompt, thinking=False)) - 1
        for prompt in prompt_set.prompts
    )
    info = {
        "model_id": loaded.spec.hf_id,
        "measure": calibrate.MEASURE,
        "prompt_set": _shown(prompts_file),
        "prompt_set_version": prompt_set.version,
        "prompt_count": len(prompt_set.prompts),
        "token_count": tokens,
        "dtype": dtype_name,
        "device": device,
        "measured_at": started.isoformat(timespec="seconds"),
        **environment,
    }
    calibration = calibrate.Calibration(
        model_name, loaded.revision, tuple(norms), prompt_set.sha256, info
    )
    try:
        path = calibrate.save_calibration(calibration, overwrite=overwrite)
    except FileExistsError:  # written meanwhile
        return _refuse(f"{path} already exists; pass --overwrite to measure again")

    print(
        f"Measured {len(prompt_set.prompts)} prompts, {tokens} tokens after each "
        f"prompt's first, in {seconds:.1f} s."
    )
    print("Median length of the residual stream leaving each layer:")
    print("\n".join(summary_lines(norms)))
    print(f"Saved {path}")
    return 0
