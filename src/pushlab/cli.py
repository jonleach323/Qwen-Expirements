"""The pushlab command. `pushlab --help` lists its subcommands.

`pushlab baseline` generates one unsteered reply and saves it in the
project's run-record format: OUT/runs.parquet with one run, and
OUT/manifest.json saying what produced it. Its experiment file,
experiments/p0/s0_baseline.yaml, names the experiment, seed and prompt.
"""

import argparse
import hashlib
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import yaml

from pushlab import PROJECT_ROOT
from pushlab.models.registry import REGISTRY
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


def _positive_int(text: str) -> int:
    if not text.isdigit() or int(text) < 1:
        raise argparse.ArgumentTypeError(f"must be a whole number from 1, not {text!r}")
    return int(text)


def build_parser() -> argparse.ArgumentParser:
    """The pushlab command's arguments, with one subparser per subcommand.

    Example:
        >>> args = ["baseline", "--model", "qwen3.5-4b", "--out", "outputs/s0"]
        >>> build_parser().parse_args(args).max_new_tokens
        200
    """
    parser = argparse.ArgumentParser(
        prog="pushlab",
        description="Research code for pushing LLM neurons on schedules.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    baseline = commands.add_parser(
        "baseline",
        help="generate one unsteered reply and save it as a run",
        description="Generate one unsteered reply, with thinking off and greedy "
        "decoding, and save it as one run in OUT/runs.parquet with "
        "OUT/manifest.json. It uses the GPU in bfloat16 if there is one, "
        "otherwise the CPU in float32. The model must already be downloaded, "
        "as scripts/setup_pod.sh does on the GPU pod.",
    )
    models = sorted(name for name, spec in REGISTRY.items() if spec.kind != "test")
    baseline.add_argument(
        "--model", required=True, choices=models, help="registry name of the model"
    )
    baseline.add_argument(
        "--out",
        required=True,
        type=Path,
        help="folder for runs.parquet and manifest.json, created if missing",
    )
    baseline.add_argument(
        "--prompt",
        help="the user's message (default: the prompt in "
        "experiments/p0/s0_baseline.yaml)",
    )
    baseline.add_argument(
        "--max-new-tokens",
        type=_positive_int,
        default=200,
        help="most tokens the reply may have (default: %(default)s)",
    )
    baseline.add_argument(
        "--overwrite",
        action="store_true",
        help="replace runs.parquet and manifest.json if OUT already has them",
    )
    return parser


def _now() -> datetime:
    return datetime.now(UTC)


def _refuse(message: str) -> int:
    print(f"pushlab baseline: {message}", file=sys.stderr)
    return 1


def _file_in_the_way(out: Path) -> Path | None:
    """The file, if any, at out or above it that stops out being a folder.

    A link to something missing counts too, since no folder can be made there.

    Example:
        >>> _file_in_the_way(Path("/no/such/folder")) is None
        True
    """
    for path in (out, *out.parents):
        if path.is_symlink() or path.exists():  # exists() alone misses a broken link
            return None if path.is_dir() else path
    return None


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
    if (in_the_way := _file_in_the_way(out)) is not None:
        what = "a file" if in_the_way.exists() else "a link to nothing"
        return _refuse(f"--out must be a folder, but {in_the_way} is {what}")
    already_there = (
        f"{out} already holds a run; choose another --out, or pass --overwrite "
        "to replace it"
    )
    existing = [out / name for name in (RUNS_FILE, MANIFEST_FILE)]
    if not overwrite and any(path.exists() for path in existing):
        return _refuse(already_there)
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

    from pushlab.engine.generate import greedy_generate
    from pushlab.models.chat import chat_prompt_ids, stop_token_ids
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

    ids = chat_prompt_ids(loaded.tokenizer, prompt, thinking=False)
    stops = stop_token_ids(loaded.model, loaded.tokenizer)
    torch.manual_seed(experiment.seed)
    start = time.perf_counter()
    tokens = greedy_generate(loaded.model, ids, max_new_tokens, stops)
    seconds = time.perf_counter() - start
    text = loaded.tokenizer.decode(tokens, skip_special_tokens=True)
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
            return _refuse(already_there)
        raise
    print(f"Saved run {record.run_id} and its manifest:")
    print(f"  {runs_path}\n  {manifest_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pushlab command with argv; return the exit status.

    Example:
        >>> main(["baseline", "--model", "qwen3.5-4b", "--out", "x"])  # doctest: +SKIP
        0
    """
    args = build_parser().parse_args(argv)
    return run_baseline(
        args.model, args.out, args.prompt, args.max_new_tokens, args.overwrite
    )


if __name__ == "__main__":
    sys.exit(main())
