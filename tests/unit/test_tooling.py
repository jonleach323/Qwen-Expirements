"""Tool versions that must stay in step."""

import tomllib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_pre_commit_ruff_matches_locked_ruff() -> None:
    # Otherwise the commit hook and `uv run ruff` could format code differently.
    config = yaml.safe_load((ROOT / ".pre-commit-config.yaml").read_text())
    hooks = [r for r in config["repos"] if "astral-sh/ruff-pre-commit" in r["repo"]]
    assert len(hooks) == 1, "expected one ruff-pre-commit repo in .pre-commit-config"
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    (locked,) = [p["version"] for p in lock["package"] if p["name"] == "ruff"]
    assert hooks[0]["rev"] == f"v{locked}"
