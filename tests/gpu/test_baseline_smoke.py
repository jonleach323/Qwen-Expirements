"""GPU smoke test: `pushlab baseline` on the real Qwen3.5-4B.

It needs a GPU and the model downloaded by scripts/setup_pod.sh, whose last
step runs it with the other smoke tests (`pytest -m gpu -k smoke`). It loads
the model once and writes one reply into a temporary folder.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pushlab.store.records import REQUIRED_MANIFEST_FIELDS, read_runs


@pytest.mark.gpu
@pytest.mark.slow
def test_the_baseline_command_writes_a_reply_and_a_manifest(tmp_path: Path) -> None:
    out = tmp_path / "baseline"
    command = Path(sys.executable).with_name("pushlab")
    result = subprocess.run(
        [str(command), "baseline", "--model", "qwen3.5-4b", "--out", str(out)],
        capture_output=True,
        text=True,
        timeout=1200,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    (run,) = read_runs(out)
    assert run.run_id == "s0-baseline-000001"
    assert run.model.startswith("qwen3.5-4b@")
    assert run.status == "ok"
    assert run.decoding["thinking"] is False
    assert run.decoding["dtype"] == "bfloat16"
    assert 0 < len(run.output_tokens) <= run.decoding["max_new_tokens"]
    assert run.output_text.strip()
    # Thinking is off, so the reply holds no reasoning block.
    assert "</think>" not in run.output_text

    manifest = json.loads((out / "manifest.json").read_text())
    assert set(REQUIRED_MANIFEST_FIELDS) <= set(manifest)
    assert manifest["model_id"] == "Qwen/Qwen3.5-4B"
    assert manifest["model_revision"] == run.model.split("@", 1)[1]
    assert manifest["gpu"]
    assert manifest["cuda"]
    assert manifest["run_count"] == 1
