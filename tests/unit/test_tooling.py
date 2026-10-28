"""Tool versions that must stay in step."""

import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_pre_commit_ruff_matches_locked_ruff() -> None:
    # Otherwise the commit hook and `uv run ruff` could format code differently.
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    (hook,) = [r for r in config["repos"] if r["repo"].endswith("/ruff-pre-commit")]
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    (locked,) = [p["version"] for p in lock["package"] if p["name"] == "ruff"]
    assert hook["rev"] == f"v{locked}"
