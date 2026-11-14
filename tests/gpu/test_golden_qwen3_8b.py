"""GPU golden test: Qwen3-8B's replies to its golden cases, against the
fingerprints recorded for them (decision 0023; development plan, "Golden
tests").

tests/golden/qwen3-8b.yaml holds the cases: three everyday prompts, each
run six ways, greedy and sampled, unpushed and with a made-up neuron or
set. tests/golden/qwen3-8b.json holds each reply's fingerprint, the SHA-256
of its token IDs, with the environment that made it. `pushlab golden
record` writes that file on the pod, outside the repository, and it comes
in through a pull request (docs/runbooks/pod.md, "Golden tests"). Until it
does, this test skips, saying how to make it.

It needs a GPU, Qwen3-8B downloaded by scripts/setup_pod.sh and its fixed
calibration (`pushlab calibrate --model qwen3-8b`). It loads the model once,
in bfloat16, and runs every case alone, as the record did. Any changed
reply means the code or the environment changed, so it fails with each
changed case and what of the environment differs from the recorded one.
Run it with `uv run pytest -m gpu -k golden`.
"""

import pytest
import torch

from pushlab.models.load import load_model
from pushlab.runs import golden, golden_cases
from pushlab.runs import golden_fingerprints as fingerprints

MODEL = "qwen3-8b"
CASES = golden_cases.cases_path(MODEL)
FINGERPRINTS = fingerprints.fingerprints_path(MODEL)


@pytest.mark.gpu
@pytest.mark.slow
def test_every_golden_case_gives_its_recorded_reply() -> None:
    if (why := fingerprints.not_recorded(FINGERPRINTS, MODEL)) is not None:
        pytest.skip(why)
    assert torch.cuda.is_available(), "torch sees no GPU; see test_cuda_is_available"
    cases = golden_cases.read_cases(CASES)
    recording = fingerprints.read_recording(FINGERPRINTS)
    if (problem := fingerprints.recording_problem(recording, cases)) is not None:
        pytest.fail(problem, pytrace=False)
    loaded = load_model(MODEL, cases.torch_dtype, device="cuda")
    try:
        fixed = golden.fixed_norms(cases, loaded.revision)
    except FileNotFoundError as error:
        pytest.skip(str(error))
    found = golden.run_cases(loaded.model, loaded.tokenizer, cases, fixed)
    changed = fingerprints.changed_cases(recording, cases, found)
    if changed:
        environment = fingerprints.golden_environment("cuda", fixed)
        differences = fingerprints.environment_changes(
            recording, loaded.revision, environment
        )
        total = len(cases.cases)
        pytest.fail(
            fingerprints.failure_text(changed, differences, total), pytrace=False
        )
