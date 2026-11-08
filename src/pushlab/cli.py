"""The pushlab command. `pushlab --help` lists its subcommands.

`pushlab baseline` generates one unsteered reply and saves it in the
project's run-record format: OUT/runs.parquet with one run, and
OUT/manifest.json saying what produced it. Its experiment file,
experiments/p0/s0_baseline.yaml, names the experiment, seed and prompt.

`pushlab calibrate` measures each layer's median residual size, the scale
of a dose, and saves it in calibration/ (pushlab.units.calibrate).
`pushlab amount` prints what a dose adds to a neuron
(pushlab.units.amount_command), `pushlab steer` saves pushed replies to a
prompt set (pushlab.runs.steer), and `pushlab report` lays run folders'
replies side by side (pushlab.analysis.report). `pushlab expand` lists the
runs an experiment file means without running them
(pushlab.runs.expand_command), `pushlab run` runs them batch by batch
(pushlab.runs.runner), `pushlab status` says how far a run has got
(pushlab.runs.status_command), and `pushlab verify` checks its saved runs
and steps (pushlab.runs.verify). `pushlab report` also shows an
experiment's replies with markers where its pushes switch
(pushlab.analysis.experiment_report). The options of every subcommand are
in pushlab.cli_parser.
"""

import hashlib
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml

from pushlab import PROJECT_ROOT
from pushlab.cli_parser import build_parser
from pushlab.store.folders import already_there, file_in_the_way
from pushlab.store.records import (
    EXPERIMENT_NAME,
    MANIFEST_FILE,
    MAX_SEED,
    RUNS_FILE,
    RunRecord,
    check_manifest,
    make_run_id,
    write_manifest,
    write_runs,
)

BASELINE_EXPERIMENT = PROJECT_ROOT / "experiments" / "p0" / "s0_baseline.yaml"


@dataclass(frozen=True)
class BaselineExperiment:
    """The baseline's settings from its experiment file.

    sha256 is the hash of the exact bytes the settings were read from.
    """

    experiment: str
    seed: int
    prompt_id: str
    prompt: str
    sha256: str


def read_baseline_experiment(path: Path) -> BaselineExperiment:
    """Read and check the baseline's experiment file.

    Example:
        >>> read_baseline_experiment(BASELINE_EXPERIMENT).experiment
        's0-baseline'
    """
    content = path.read_bytes()
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as error:
        raise ValueError(f"not valid YAML: {error}") from error
    if not isinstance(data, dict):
        raise ValueError("expected the keys experiment, seed, prompt_id and prompt")
    name = data.get("experiment")
    if not isinstance(name, str) or not EXPERIMENT_NAME.fullmatch(name):
        raise ValueError(
            "experiment must be lowercase letters and digits joined by hyphens, "
            f"such as s0-baseline, not {name!r}"
        )
    seed = data.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed <= MAX_SEED:
        raise ValueError(
            f"seed must be a whole number from 0 to {MAX_SEED}, not {seed!r}"
        )
    for key in ("prompt_id", "prompt"):
        value = data.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be non-empty text, not {value!r}")
    sha256 = hashlib.sha256(content).hexdigest()
    return BaselineExperiment(name, seed, data["prompt_id"], data["prompt"], sha256)


def custom_prompt_id(prompt: str) -> str:
    """The prompt ID for a prompt given with --prompt, made from its text.

    The same text always gets the same ID, so runs on it can be grouped.

    Example:
        >>> custom_prompt_id("Say hi.")
        'custom-e276e57b8ac9'
    """
    return "custom-" + hashlib.sha256(prompt.encode()).hexdigest()[:12]


def _now() -> datetime:
    return datetime.now(UTC)


def _refuse(message: str) -> int:
    print(f"pushlab baseline: {message}", file=sys.stderr)
    return 1


def run_baseline(
    model_name: str,
    out: Path,
    prompt: str | None = None,
    max_new_tokens: int = 200,
    overwrite: bool = False,
) -> int:
    """Generate one unsteered reply with model_name and save it in out.

    Returns the exit status: 0 once both files are written, 1 with the
    reason on stderr when it refuses, before loading the model if it can.

    Example:
        >>> run_baseline("qwen3.5-4b", Path("outputs/s0"))  # doctest: +SKIP
        0
    """
    if (in_the_way := file_in_the_way(out)) is not None:
        what = "a file" if in_the_way.exists() else "a link to nothing"
        return _refuse(f"--out must be a folder, but {in_the_way} is {what}")
    existing = [out / name for name in (RUNS_FILE, MANIFEST_FILE)]
    if not overwrite and any(path.exists() for path in existing):
        return _refuse(already_there(out))
    if prompt is not None and not prompt.strip():
        return _refuse("--prompt is empty")
    try:
        experiment = read_baseline_experiment(BASELINE_EXPERIMENT)
    except (OSError, ValueError) as error:
        return _refuse(f"cannot use {BASELINE_EXPERIMENT}: {error}")
    prompt_id = experiment.prompt_id if prompt is None else custom_prompt_id(prompt)
    prompt = experiment.prompt if prompt is None else prompt

    # Imported here, so that --help answers without first loading torch.
    import torch

    from pushlab.engine.decoding import Decoding
    from pushlab.engine.generate import generate
    from pushlab.models.load import default_device, default_dtype, load_model
    from pushlab.store.environment import environment_info

    started = _now()
    clock = time.monotonic()  # never steps back, unlike the system clock
    device = default_device()
    # Read now, so the manifest describes the code and files that start the
    # run even if the checkout changes while it runs.
    environment = environment_info(device)
    dtype = default_dtype(device)
    dtype_name = str(dtype).removeprefix("torch.")
    where = "GPU" if device == "cuda" else "CPU"
    print(f"Loading {model_name} in {dtype_name} on the {where}...")
    try:
        loaded = load_model(model_name, dtype, device=device)
    except FileNotFoundError as error:
        return _refuse(str(error))

    greedy = Decoding(strategy="greedy", max_new_tokens=max_new_tokens)
    torch.manual_seed(experiment.seed)
    start = time.perf_counter()
    (reply,) = generate(loaded.model, loaded.tokenizer, [prompt], [], greedy)
    seconds = time.perf_counter() - start
    tokens, text = list(reply.token_ids), reply.text
    print(f"Generated {len(tokens)} tokens in {seconds:.1f} s:\n\n{text}\n")

    record = RunRecord(
        run_id=make_run_id(experiment.experiment, 1),
        experiment=experiment.experiment,
        model=f"{model_name}@{loaded.revision}",
        prompt_id=prompt_id,
        schedule=[],
        decoding={
            "strategy": "greedy",
            "max_new_tokens": max_new_tokens,
            "thinking": False,
            "dtype": dtype_name,
        },
        seed=experiment.seed,
        batch_key="b000001:0",  # batch 1, row 0
        control_of="",
        output_text=text,
        output_tokens=tokens,
        status="ok",
    )
    finished = started + timedelta(seconds=time.monotonic() - clock)
    info = environment | {
        "experiment": experiment.experiment,
        "experiment_file": BASELINE_EXPERIMENT.relative_to(PROJECT_ROOT).as_posix(),
        "experiment_sha256": experiment.sha256,
        "model": model_name,
        "model_id": loaded.spec.hf_id,
        "model_revision": loaded.revision,
        "device": device,
        "dtype": dtype_name,
        "prompt_id": prompt_id,
        "prompt": prompt,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "run_count": 1,
        "runs_by_status": {"ok": 1},
    }
    check_manifest(info)  # before writing, so a bad manifest leaves no run file
    try:
        runs_path = write_runs([record], out, overwrite=overwrite)
        manifest_path = write_manifest(out, info, overwrite=overwrite)
    except FileExistsError:
        if any(path.exists() for path in existing):  # written meanwhile
            return _refuse(already_there(out))
        raise
    print(f"Saved run {record.run_id} and its manifest:")
    print(f"  {runs_path}\n  {manifest_path}")
    return 0


def _report(args: Any) -> int:
    """pushlab report: on an experiment's runner folder, or on run folders."""
    if args.spec is not None and args.runs:
        print(
            "pushlab report: give an experiment file or --runs, not both",
            file=sys.stderr,
        )
        return 1
    if args.spec is not None:
        from pushlab.analysis.experiment_report import run_experiment_report

        return run_experiment_report(
            args.spec, args.conditions, args.out, args.runs_dir, args.overwrite
        )
    if not args.runs:
        if args.conditions:  # --conditions takes every word after it
            print(
                "pushlab report: error: give the experiment file before "
                "--conditions, such as: pushlab report SPEC --conditions none "
                "--out FILE (or run folders with --runs DIR)",
                file=sys.stderr,
            )
        else:
            print(
                "pushlab report: error: give an experiment file, or run folders "
                "with --runs DIR",
                file=sys.stderr,
            )
        raise SystemExit(2)
    if args.conditions or args.runs_dir is not None:
        print(
            "pushlab report: --conditions and --runs-dir go with an experiment "
            "file, not with --runs",
            file=sys.stderr,
        )
        return 1
    from pushlab.analysis.report import run_report

    return run_report(args.runs, args.out, args.overwrite)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pushlab command with argv; return the exit status.

    Example:
        >>> main(["baseline", "--model", "qwen3.5-4b", "--out", "x"])  # doctest: +SKIP
        0
    """
    args = build_parser().parse_args(argv)
    # Each command is imported here, so that --help answers without first
    # loading torch.
    if args.command == "calibrate":
        from pushlab.units.calibrate_command import run_calibrate

        return run_calibrate(args.model, args.overwrite)
    if args.command == "amount":
        from pushlab.units.amount_command import run_amount

        return run_amount(args.unit, args.alpha)
    if args.command == "steer":
        from pushlab.runs.steer import run_steer

        return run_steer(
            model_name=args.model,
            prompts_file=args.prompts,
            out=args.out,
            units=args.unit or [],
            alphas=args.alpha or [],
            start=args.start,
            end=args.end,
            include_prompt=args.include_prompt,
            control=args.control,
            control_seed=args.control_seed,
            max_new_tokens=args.max_new_tokens,
            overwrite=args.overwrite,
        )
    if args.command == "report":
        return _report(args)
    if args.command == "verify":
        from pushlab.runs.verify import run_verify

        return run_verify(args.spec, args.runs_dir)
    if args.command == "expand":
        from pushlab.runs.expand_command import run_expand

        return run_expand(args.spec, args.summary)
    if args.command == "run":
        from pushlab.runs.runner import run_experiment

        return run_experiment(
            args.spec,
            runs_dir=args.runs_dir,
            dry_run=args.dry_run,
            limit=args.limit,
            retry_failed=args.retry_failed,
            overwrite_version=args.overwrite_version,
        )
    if args.command == "status":
        from pushlab.runs.status_command import run_status

        return run_status(args.spec, args.runs_dir)
    return run_baseline(
        args.model, args.out, args.prompt, args.max_new_tokens, args.overwrite
    )


if __name__ == "__main__":
    sys.exit(main())
