"""`pushlab golden record`: the golden cases' fingerprints (decision 0023).

It reads the golden cases (pushlab.runs.golden_cases) from
tests/golden/<model>.yaml, unless --cases names another file. It loads the
model in the cases' data type, runs each case alone (pushlab.runs.golden),
and writes every reply's fingerprint with the environment that made it to
--out, as JSON (pushlab.runs.golden_fingerprints). A GPU job: the owner
runs it on the pod, writing outside the repository, which the pod never
changes, and brings the file in as tests/golden/<model>.json from a cloud
session, through a pull request (docs/runbooks/pod.md, "Golden tests").

It refuses, loading nothing, with the reason on stderr: cases that cannot
be read, or that are another model's; an --out inside the repository,
that exists, or whose folder cannot be made; cases in bfloat16 when torch
sees no GPU; a model that is not downloaded; and, when a case takes the
fixed calibration, a model revision with none.
"""

import argparse
import sys
from pathlib import Path

from pushlab import PROJECT_ROOT
from pushlab.models.load import default_device, downloaded_revision, load_model
from pushlab.runs import golden, golden_cases
from pushlab.runs.golden_fingerprints import (
    golden_environment,
    make_recording,
    recording_text,
)
from pushlab.store.folders import file_in_the_way, shown_path
from pushlab.units.calibrate import calibration_path

OUTSIDE = "/workspace/golden"  # where the pod writes fingerprints
SHOWN = 48  # characters of each reply a progress line shows


def _refuse(message: str) -> int:
    print(f"pushlab golden: {message}", file=sys.stderr)
    return 1


def inside_repository(path: Path) -> bool:
    """Whether a file written at path lands inside the repository: its folder
    with any link followed, then its own name.

    Example:
        >>> inside_repository(PROJECT_ROOT / "tests" / "golden" / "qwen3-8b.json")
        True
        >>> inside_repository(Path("/workspace/golden/qwen3-8b.json"))
        False
    """
    where = Path(path).absolute()
    return (where.parent.resolve() / where.name).is_relative_to(PROJECT_ROOT)


def out_problem(out: Path, model: str) -> str | None:
    """Why --out cannot take the fingerprints, or None when it can.

    Example:
        >>> out_problem(Path("/no/such/folder/qwen3-8b.json"), "qwen3-8b") is None
        True
        >>> print(
        ...     out_problem(PROJECT_ROOT / "g.json", "qwen3-8b")
        ... )  # doctest: +ELLIPSIS
        --out must be outside the repository, which the pod never changes: ...
    """
    if inside_repository(out):
        return (
            "--out must be outside the repository, which the pod never changes: "
            f"write to {OUTSIDE}/{model}.json, then bring the file in as "
            f"tests/golden/{model}.json from a cloud session, through a pull request"
        )
    if out.exists() or out.is_symlink():
        return f"{out} already exists; choose another --out"
    if (in_the_way := file_in_the_way(out.parent)) is not None:
        return f"--out's folder cannot be made, as {in_the_way} is a file"
    return None


def case_line(case_id: str, found: golden.Fingerprint, width: int) -> str:
    """A case's line as the record runs: its ID, its reply's length, and the
    start of its text.

    Example:
        >>> found = golden.Fingerprint("0" * 64, 32, (40,), "To descale a kettle")
        >>> case_line("kettle-greedy", found, 15)
        "  kettle-greedy     32 tokens  'To descale a kettle'"
    """
    return (
        f"  {case_id:<{width}}  {found.token_count:>3} tokens  "
        f"{found.text_start[:SHOWN]!r}"
    )


def record(model: str, cases_file: Path | None, out: Path) -> int:
    """Run every golden case of model and write the fingerprints to out; the
    exit status: 0 once written, 1 with the reason on stderr when refused.

    Example:
        >>> record(
        ...     "qwen3-8b", None, Path("/workspace/golden/qwen3-8b.json")
        ... )  # doctest: +SKIP
        0
    """
    path = golden_cases.cases_path(model) if cases_file is None else Path(cases_file)
    if cases_file is None and not path.exists():
        return _refuse(
            f"there are no golden cases for {model}: {shown_path(path)} does not "
            "exist; give --cases FILE"
        )
    try:
        cases = golden_cases.read_cases(path)
    except ValueError as error:
        return _refuse(str(error))
    if cases.model != model:
        return _refuse(
            f"the cases in {shown_path(path)} are {cases.model}'s, not {model}'s: "
            f"give --model {cases.model}, or other --cases"
        )
    if (problem := out_problem(out, model)) is not None:
        return _refuse(problem)
    device = default_device()
    if device == "cpu" and cases.dtype == "bfloat16":
        return _refuse(
            "torch sees no GPU, and these cases run in bfloat16 on the GPU, as the "
            "golden test does: record them on the pod, where `bash "
            "scripts/setup_pod.sh` sets torch up to use it"
        )
    try:
        revision = downloaded_revision(model)
        fixed = golden.fixed_norms(cases, revision)
    except (FileNotFoundError, ValueError) as error:
        return _refuse(str(error))
    environment = golden_environment(device, fixed)  # before the checkout changes
    where = "GPU" if device == "cuda" else "CPU"
    print(f"Loading {model} in {cases.dtype} on the {where}...", flush=True)
    try:
        loaded = load_model(model, cases.torch_dtype, device=device)
    except (FileNotFoundError, ValueError) as error:
        return _refuse(str(error))
    if fixed is not None:
        shown = shown_path(calibration_path(model, revision))
        print(f"The fixed calibration's sizes come from {shown}.")
    print(f"Running the {len(cases.cases)} cases of {shown_path(path)}, each alone:")
    width = max(len(case.case_id) for case in cases.cases)
    found = golden.run_cases(
        loaded.model,
        loaded.tokenizer,
        cases,
        fixed,
        lambda case, reply: print(case_line(case.case_id, reply, width), flush=True),
    )
    text = recording_text(make_recording(cases, path, revision, environment, found))
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("x", encoding="utf-8") as file:  # never over another
            file.write(text)
    except OSError as error:
        return _refuse(f"cannot write {out}: {error}")
    print(f"Wrote the {len(found)} fingerprints to {out}.")
    print(
        f"Next, bring it in as tests/golden/{model}.json from a cloud session, "
        "through a pull request; then `uv run pytest -m gpu -k golden` on the pod "
        'checks the model against it (docs/runbooks/pod.md, "Golden tests").'
    )
    return 0


def run_golden_command(args: argparse.Namespace) -> int:
    """`pushlab golden` with its parsed options (pushlab.cli_golden); the exit
    status.

    Example:
        >>> out = Path("/workspace/golden/qwen3-8b.json")
        >>> args = argparse.Namespace(
        ...     golden_command="record", model="qwen3-8b", cases=None, out=out
        ... )
        >>> run_golden_command(args)  # doctest: +SKIP
        0
    """
    if args.golden_command == "record":
        return record(args.model, args.cases, args.out)
    return _refuse(f"no command {args.golden_command!r}; the one command is record")
