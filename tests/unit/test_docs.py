"""Commands the docs give, checked against what they must never do."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
# The S0 reference run (docs/decisions/0005-s0-baseline.md). Its runs.parquet
# is the only copy of that reply's token IDs, so no command may write there.
REFERENCE_RUN = "/workspace/runs/s0-baseline"


@pytest.mark.parametrize("doc", ["README.md", "CLAUDE.md", "docs/runbooks/pod.md"])
def test_no_documented_baseline_run_goes_into_the_reference_folder(doc: str) -> None:
    text = (ROOT / doc).read_text()
    outs = re.findall(r'pushlab baseline\b[^\n`]*?--out\s+"?([^\s`"]+)', text)
    assert outs, f"{doc} shows no pushlab baseline command with --out"
    assert REFERENCE_RUN not in [out.rstrip("/") for out in outs]
