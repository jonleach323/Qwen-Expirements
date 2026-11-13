"""`pushlab golden record` (decision 0023): its help, which loads nothing
heavy; its refusals, each before the model loads; and records of made-up
cases on the tiny Qwen3.5, on the CPU, where float32 is deterministic: two
records give the same fingerprints, and a model changed since is caught.
Also the GPU test's wiring: the runbook's command selects it alone, and it
skips, naming the record command, until its fingerprints are recorded.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

from pushlab import PROJECT_ROOT
from pushlab.cli import main
from pushlab.models.load import load_model
from pushlab.runs import golden_command
from pushlab.runs.golden import fixed_norms, run_cases
from pushlab.runs.golden_cases import read_cases
from pushlab.runs.golden_fingerprints import (
    changed_cases,
    environment_changes,
    failure_text,
    golden_environment,
    norms_sha256,
    read_recording,
    recording_problem,
)
from pushlab.schedules.yaml_text import dump_yaml
from pushlab.store.environment import environment_info
from pushlab.units import calibrate
from pushlab.units.calibrate import load_calibration

COMMIT = "0123456789abcdef0123456789abcdef01234567"  # tests/conftest.py's tiny-vl
GPU_TEST = PROJECT_ROOT / "tests" / "gpu" / "test_golden_qwen3_8b.py"
HELP_IMPORTS = """
import contextlib, sys
from pushlab.cli import main
with contextlib.suppress(SystemExit):
    main(["golden", "record", "--help"])
print([name for name in ("torch", "pandas", "transformers") if name in sys.modules])
"""


def _load(name: str, path: Path) -> Any:
    """A module loaded by path, as test modules import nothing from one another."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


helpers = _load("golden_helpers", Path(__file__).with_name("golden_helpers.py"))


@pytest.fixture
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU even on a machine with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _cases_file(folder: Path, **settings: object) -> Path:
    """The made-up cases for tiny-vl, with settings replaced, as a file."""
    path = folder / "tiny-vl.yaml"
    data = helpers.golden_data("tiny-vl") | settings
    path.write_text(dump_yaml(data), encoding="utf-8")
    return path


def _record(cases: Path | None, out: Path, model: str = "tiny-vl") -> int:
    more = [] if cases is None else ["--cases", str(cases)]
    return main(["golden", "record", "--model", model, *more, "--out", str(out)])


def test_the_help_lists_the_options_and_loads_nothing_heavy(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["golden", "record", "--help"])
    assert exit_info.value.code == 0
    text = " ".join(capsys.readouterr().out.split())
    for phrase in ["--model MODEL", "--cases FILE", "--out FILE", "A GPU job"]:
        assert phrase in text
    assert "(default: tests/golden/<model>.yaml)" in text
    pushlab = str(Path(sys.executable).with_name("pushlab"))
    for argv in (
        [pushlab, "golden", "record", "--help"],
        [sys.executable, "-c", HELP_IMPORTS],
    ):
        done = subprocess.run(argv, capture_output=True, text=True, cwd=tmp_path)
        assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines()[-1] == "[]"


def test_two_records_of_the_same_cases_give_the_same_fingerprints(
    cpu_only: None,
    tiny_vl_calibration: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cases = _cases_file(tmp_path)
    first, second = (tmp_path / "golden" / f"{n}.json" for n in ("one", "two"))
    assert _record(cases, first) == 0
    assert _record(cases, second) == 0
    output = capsys.readouterr().out
    assert output.count("Loading tiny-vl in float32 on the CPU...") == 2
    assert output.count("The fixed calibration's sizes come from ") == 2
    assert output.count("Wrote the 6 fingerprints to ") == 2
    for case_id in helpers.IDS:  # one progress line per case and record
        assert output.count(f"\n  {case_id} ") == 2, case_id
    one, two = (
        json.loads(path.read_text(encoding="utf-8")) for path in (first, second)
    )
    assert one["cases"] == two["cases"]
    assert list(one["cases"]) == helpers.IDS
    assert (one["model"], one["revision"], one["dtype"]) == (
        "tiny-vl",
        COMMIT,
        "float32",
    )
    norms = load_calibration("tiny-vl", COMMIT).norms
    assert one["environment"]["fixed_calibration_sha256"] == norms_sha256(norms)
    assert set(environment_info("cpu")) < set(one["environment"])
    # Each seed draws its own reply; the set's definition is kept with its case.
    seeds = [one["cases"][f"hi-sampled-seed-{s}"]["fingerprint"] for s in (1, 2)]
    assert seeds[0]["sha256"] != seeds[1]["sha256"]
    assert one["cases"]["hi-set-per-prompt"]["case"]["push"]["set"]["size"] == 0.2
    assert recording_problem(read_recording(first), read_cases(cases)) is None


def test_a_model_changed_since_the_record_is_caught(
    cpu_only: None,
    tiny_vl_calibration: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "golden.json"
    assert _record(_cases_file(tmp_path), out) == 0
    recording, cases = read_recording(out), read_cases(tmp_path / "tiny-vl.yaml")
    loaded = load_model("tiny-vl", torch.float32, device="cpu")
    fixed = fixed_norms(cases, loaded.revision)
    again = run_cases(loaded.model, loaded.tokenizer, cases, fixed)
    assert changed_cases(recording, cases, again) == []
    # The greedy reply's first token can no longer win: its score is now 0.
    first = recording.cases["hi-greedy"].fingerprint.first_tokens[0]
    with torch.no_grad():
        loaded.model.lm_head.weight[first] = 0.0
    found = run_cases(loaded.model, loaded.tokenizer, cases, fixed)
    changed = changed_cases(recording, cases, found)
    assert changed[0].startswith("hi-greedy: recorded ")
    environment = golden_environment("cpu", fixed)
    differences = environment_changes(recording, loaded.revision, environment)
    text = failure_text(changed, differences, len(cases.cases))
    assert text.startswith(f"{len(changed)} of the 6 golden replies changed since")


@pytest.fixture
def no_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if the command loads the model."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the model must not load when the command refuses")

    monkeypatch.setattr(golden_command, "load_model", refuse)


def _refused(capsys: pytest.CaptureFixture[str], code: int) -> str:
    """The reason the command gave, once it refused with exit status 1."""
    assert code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("pushlab golden: ")
    return captured.err.removeprefix("pushlab golden: ").strip()


def test_an_out_inside_the_repository_is_refused(
    cpu_only: None,
    no_loading: None,
    tiny_vl_calibration: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = PROJECT_ROOT / "tests" / "golden" / "tiny-vl.json"
    why = _refused(capsys, _record(_cases_file(tmp_path), out))
    assert why == (
        "--out must be outside the repository, which the pod never changes: write "
        "to /workspace/golden/tiny-vl.json, then bring the file in as "
        "tests/golden/tiny-vl.json from a cloud session, through a pull request"
    )
    assert not out.exists()


@pytest.mark.parametrize("where", ["a file", "a file's folder"])
def test_an_out_that_cannot_be_written_new_is_refused(
    cpu_only: None,
    no_loading: None,
    tiny_vl_calibration: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    where: str,
) -> None:
    taken = tmp_path / "taken.json"
    taken.write_text("{}", encoding="utf-8")
    out = taken if where == "a file" else taken / "golden.json"
    why = _refused(capsys, _record(_cases_file(tmp_path), out))
    if where == "a file":
        assert why == f"{taken} already exists; choose another --out"
        assert taken.read_text(encoding="utf-8") == "{}"
    else:
        assert why == f"--out's folder cannot be made, as {taken} is a file"


def test_cases_that_cannot_be_used_are_refused(
    cpu_only: None,
    no_loading: None,
    tiny_vl_calibration: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "golden.json"
    why = _refused(capsys, _record(None, out, model="tiny-lm"))
    assert why == (
        "there are no golden cases for tiny-lm: tests/golden/tiny-lm.yaml does not "
        "exist; give --cases FILE"
    )
    why = _refused(capsys, _record(tmp_path / "none.yaml", out))
    assert why.startswith(f"cannot read the golden cases {tmp_path / 'none.yaml'}")
    cases = _cases_file(tmp_path)
    why = _refused(capsys, _record(cases, out, model="tiny-lm"))
    assert why == (
        f"the cases in {cases} are tiny-vl's, not tiny-lm's: give --model tiny-vl, "
        "or other --cases"
    )
    why = _refused(capsys, _record(_cases_file(tmp_path, dtype="float16"), out))
    assert why.startswith(f"{cases}: dtype must be bfloat16 or float32")
    assert not out.exists()


def test_a_machine_that_cannot_run_the_cases_is_refused(
    cpu_only: None,
    no_loading: None,
    tiny_vl_calibration: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "golden.json"
    why = _refused(capsys, _record(_cases_file(tmp_path, dtype="bfloat16"), out))
    assert why.startswith(
        "torch sees no GPU, and these cases run in bfloat16 on the GPU, as the "
        "golden test does: record them on the pod"
    )
    cases = _cases_file(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(calibrate, "CALIBRATION_DIR", tmp_path / "no-calibration")
        why = _refused(capsys, _record(cases, out))
    assert why.startswith(f"tiny-vl at {COMMIT} has no calibration in ")
    assert why.endswith("run `uv run pushlab calibrate --model tiny-vl`")
    monkeypatch.setenv("PUSHLAB_MODELS_DIR", str(tmp_path / "no-models"))
    why = _refused(capsys, _record(cases, out))
    assert why.startswith("tiny-vl (local:tiny-qwen3.5 at main) is not downloaded")
    assert not out.exists()


def test_the_runbooks_command_selects_the_golden_gpu_test_alone(
    pytester: pytest.Pytester,
) -> None:
    # docs/runbooks/pod.md runs `uv run pytest -m gpu -k golden`. Only the
    # GPU tests' folder is collected here: test_setup_pod.py collects every
    # test in this process too, and pytester forgets the modules each run
    # imported, so a second run over tests/unit would import duckdb again,
    # which aborts the process.
    recorder = pytester.inline_run(
        "--collect-only",
        "-p",
        "no:cacheprovider",
        "-m",
        "gpu",
        "-k",
        "golden",
        f"--rootdir={PROJECT_ROOT}",
        f"--config-file={PROJECT_ROOT / 'pyproject.toml'}",
        str(PROJECT_ROOT / "tests" / "gpu"),
    )
    (call,) = recorder.getcalls("pytest_collection_finish")
    assert [item.nodeid for item in call.session.items] == [
        "tests/gpu/test_golden_qwen3_8b.py::"
        "test_every_golden_case_gives_its_recorded_reply"
    ]


def test_the_gpu_test_skips_naming_the_record_command_until_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load("golden_gpu_test", GPU_TEST)
    monkeypatch.setattr(module, "FINGERPRINTS", tmp_path / "qwen3-8b.json")
    with pytest.raises(pytest.skip.Exception, match="pushlab golden record"):
        module.test_every_golden_case_gives_its_recorded_reply()
