"""The GitHub Actions workflow in .github/workflows/tests.yml."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_each_push_to_main_gets_its_own_concurrency_group() -> None:
    # GitHub keeps one waiting run per concurrency group and cancels it when a
    # newer run joins, whatever cancel-in-progress says. If every push to main
    # shared one group, quick merges could cancel a merge's only check.
    workflow = yaml.safe_load((ROOT / ".github/workflows/tests.yml").read_text())
    group = workflow["concurrency"]["group"]
    assert "github.ref == 'refs/heads/main' && github.sha" in group
