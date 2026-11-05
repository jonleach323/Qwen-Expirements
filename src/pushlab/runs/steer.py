"""`pushlab steer`: replies to a prompt set with chosen neurons pushed, saved as runs.

One call is one condition: the same pushes on every prompt of a prompt set,
or none. It writes OUT/runs.parquet, one run per prompt in the set's order,
and OUT/manifest.json, which also records the prompt set's and the
calibration file's hashes. Each run's schedule lists its pushes: unit,
alpha, start, end, include_prompt, the amount added, and control_for.

With --control random, each unit is replaced by a random neuron of its
layer, at the same dose, drawn from --control-seed (pushlab.units.controls).
Controls are numbered steer-control-000001 and on, and each names in
control_of the run it mirrors: the steer run of the same prompt.

Every check that needs no model runs before the model loads.
"""

import math
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pushlab.analysis.report import describe_push
from pushlab.models.discovery import ModelInfo
from pushlab.models.registry import REGISTRY
from pushlab.prompts import read_prompt_set
from pushlab.store.folders import already_there, folder_problem, holds_a_run, shown_path
from pushlab.store.records import (
    RunRecord,
    check_manifest,
    make_run_id,
    write_manifest,
    write_runs,
)
from pushlab.units.neuron import NeuronUnit

EXPERIMENT = "steer"
CONTROL_EXPERIMENT = "steer-control"


def _refuse(message: str) -> int:
    print(f"pushlab steer: {message}", file=sys.stderr)
    return 1


def _units(
    model_name: str, units: Sequence[str], alphas: Sequence[float]
) -> list[tuple[NeuronUnit, float]] | str:
    """The units and their doses, or why they are refused."""
    if len(units) != len(alphas):
        return (
            f"give one --alpha for each --unit, in the same order; got "
            f"{len(units)} --unit and {len(alphas)} --alpha"
        )
    spec = REGISTRY[model_name]
    pairs = []
    for text, alpha in zip(units, alphas, strict=True):
        try:
            unit = NeuronUnit.parse(text)
        except ValueError as error:
            return f"--unit {text!r}: {error}"
        if unit.model_name != model_name:
            return (
                f"--unit {text} is a neuron of {unit.model_name}, not of {model_name}"
            )
        try:
            unit.validate(ModelInfo(spec.layers, spec.hidden_size, spec.mlp_size))
        except ValueError as error:
            return f"--unit {text}: {error}"
        if not math.isfinite(alpha):
            return f"--alpha must be a finite number, not {alpha} (for {text})"
        if unit in [seen for seen, _ in pairs]:
            return f"--unit {text} is given twice; give it once, with the doses added"
        pairs.append((unit, alpha))
    return pairs


def _control_problem(
    units: Sequence[object], control: str | None, control_seed: int | None
) -> str | None:
    if control is None:
        if control_seed is not None:
            return "--control-seed only goes with --control random"
        return None
    if control_seed is None:
        return "--control random needs --control-seed, the seed it draws from"
    if not units:
        return "--control random replaces the units given with --unit; give one"
    return None


def run_steer(
    *,
    model_name: str,
    prompts_file: Path,
    out: Path,
    units: Sequence[str] = (),
    alphas: Sequence[float] = (),
    start: int = 0,
    end: int | None = None,
    include_prompt: bool = True,
    control: str | None = None,
    control_seed: int | None = None,
    max_new_tokens: int = 200,
    overwrite: bool = False,
) -> int:
    """Reply to every prompt in prompts_file with the pushes, and save the runs.

    Returns the exit status: 0 once both files are written, 1 with the
    reason on stderr when it refuses, before loading the model if it can.

    Example:
        >>> run_steer(  # doctest: +SKIP
        ...     model_name="qwen3.5-4b",
        ...     prompts_file=Path("prompts/smoke.jsonl"),
        ...     out=Path("/workspace/runs/evil-dose1"),
        ...     units=["qwen3.5-4b:L14:N8949"],
        ...     alphas=[1.0],
        ... )
        0
    """
    if problem := folder_problem(out, overwrite):
        return _refuse(problem)
    pairs = _units(model_name, units, alphas)
    if isinstance(pairs, str):
        return _refuse(pairs)
    if problem := _control_problem(pairs, control, control_seed):
        return _refuse(problem)
    if end is not None and end <= start:
        return _refuse(f"--end ({end}) must come after --start ({start})")
    try:
        prompt_set = read_prompt_set(prompts_file)
    except (OSError, ValueError) as error:
        return _refuse(f"cannot use {prompts_file}: {error}")

    # Imported here, so that --help answers without first loading torch.
    from pushlab.engine.decoding import Decoding
    from pushlab.engine.generate import generate
    from pushlab.engine.push import Push
    from pushlab.models import load
    from pushlab.store.environment import environment_info, file_sha256
    from pushlab.units import calibrate, controls
    from pushlab.units.dose import amount

    try:
        revision = load.downloaded_revision(model_name)
        calibration = (
            calibrate.load_calibration(model_name, revision) if pairs else None
        )
    except (FileNotFoundError, ValueError) as error:
        return _refuse(str(error))  # names the command that makes a calibration
    calibration_file = calibrate.calibration_path(model_name, revision)
    # Hashed as soon as it is read, so the hash is of the sizes in use.
    calibration_sha256 = file_sha256(calibration_file) if pairs else None
    used = [unit for unit, _ in pairs]
    if control is not None:
        try:
            avoid = controls.known_units()
            used = controls.random_controls(used, control_seed, avoid)
        except (OSError, ValueError) as error:
            return _refuse(f"cannot draw the random controls: {error}")
    pushes = [
        Push(
            unit=unit, alpha=alpha, start=start, end=end, include_prompt=include_prompt
        )
        for unit, (_, alpha) in zip(used, pairs, strict=True)
    ]

    started = datetime.now(UTC)
    clock = time.monotonic()  # never steps back, unlike the system clock
    device = load.default_device()
    environment = environment_info(device)  # read before the checkout can change
    dtype = load.default_dtype(device)
    dtype_name = str(dtype).removeprefix("torch.")
    where = "GPU" if device == "cuda" else "CPU"
    print(f"Loading {model_name} in {dtype_name} on the {where}...")
    loaded = load.load_model(model_name, dtype, device=device)
    norms = list(calibration.norms) if calibration else []
    schedule = [
        {
            "unit": str(push.unit),
            "alpha": float(push.alpha),
            "start": start,
            "end": end,
            "include_prompt": include_prompt,
            "amount": amount(push.unit, push.alpha, loaded.model, norms),
            "control_for": str(unit) if control is not None else None,
        }
        for push, (unit, _) in zip(pushes, pairs, strict=True)
    ]
    for segment in schedule:
        print(f"Pushing {describe_push(segment)}.")
    greedy = Decoding(strategy="greedy", max_new_tokens=max_new_tokens)
    clock_generation = time.perf_counter()
    replies = generate(
        loaded.model, loaded.tokenizer, prompt_set.prompts, pushes, greedy, norms
    )
    seconds = time.perf_counter() - clock_generation
    print(f"Generated {len(replies)} replies in {seconds:.1f} s:")
    for prompt_id, reply in zip(prompt_set.prompt_ids, replies, strict=True):
        ending = "ended its turn" if reply.stop_reason == "end_of_turn" else "cut off"
        preview = " ".join(reply.text.split())[:60]
        print(f"  {prompt_id}: {len(reply.token_ids)} tokens, {ending}: {preview}")

    experiment = EXPERIMENT if control is None else CONTROL_EXPERIMENT
    decoding = {
        "strategy": "greedy",
        "max_new_tokens": max_new_tokens,
        "thinking": False,
        "dtype": dtype_name,
    }
    records = [
        RunRecord(
            run_id=make_run_id(experiment, row + 1),
            experiment=experiment,
            model=f"{model_name}@{loaded.revision}",
            prompt_id=prompt_id,
            schedule=schedule,
            decoding=decoding,
            seed=control_seed if control is not None else 0,
            batch_key=f"b000001:{row}",  # batch 1, this row
            control_of=make_run_id(EXPERIMENT, row + 1) if control is not None else "",
            output_text=reply.text,
            output_tokens=list(reply.token_ids),
            status="ok",
        )
        for row, (prompt_id, reply) in enumerate(
            zip(prompt_set.prompt_ids, replies, strict=True)
        )
    ]
    finished = started + timedelta(seconds=time.monotonic() - clock)
    info = environment | {
        "command": "pushlab steer",
        "experiment": experiment,
        "model": model_name,
        "model_id": loaded.spec.hf_id,
        "model_revision": loaded.revision,
        "device": device,
        "dtype": dtype_name,
        "prompt_set": shown_path(prompts_file),
        "prompt_set_version": prompt_set.version,
        "prompt_set_sha256": prompt_set.sha256,
        "prompts": dict(zip(prompt_set.prompt_ids, prompt_set.prompts, strict=True)),
        "schedule": schedule,
        "control": None
        if control is None
        else {
            "kind": control,
            "seed": control_seed,
            "neurons": {
                str(unit): str(push.unit)
                for push, (unit, _) in zip(pushes, pairs, strict=True)
            },
        },
        "calibration_file": shown_path(calibration_file) if pairs else None,
        "calibration_sha256": calibration_sha256,
        "decoding": decoding,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "run_count": len(records),
        "runs_by_status": {"ok": len(records)},
    }
    check_manifest(info)  # before writing, so a bad manifest leaves no run file
    try:
        runs_path = write_runs(records, out, overwrite=overwrite)
        manifest_path = write_manifest(out, info, overwrite=overwrite)
    except FileExistsError:
        if holds_a_run(out):  # written meanwhile
            return _refuse(already_there(out))
        raise
    print(f"Saved {len(records)} runs and their manifest:")
    print(f"  {runs_path}\n  {manifest_path}")
    return 0
