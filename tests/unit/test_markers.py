"""GPU tests are skipped by default and run only when selected with -m gpu."""

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

SAMPLE_TESTS = """
import pytest


@pytest.mark.gpu
def test_gpu():
    pass


@pytest.mark.gpu
@pytest.mark.slow
def test_gpu_and_slow():
    pass


def test_cpu():
    pass
"""


@pytest.mark.gpu
def test_dummy_gpu_test_is_skipped_by_default(request: pytest.FixtureRequest) -> None:
    # Reaching this line without -m gpu means the default skip is broken.
    assert "gpu" in request.config.getoption("markexpr")


@pytest.fixture
def sample_project(pytester: pytest.Pytester) -> pytest.Pytester:
    """A throwaway project using this repository's real pytest settings."""
    pytester.makepyprojecttoml((ROOT / "pyproject.toml").read_text())
    pytester.makeconftest((ROOT / "tests" / "conftest.py").read_text())
    pytester.makepyfile(test_sample=SAMPLE_TESTS)
    return pytester


@pytest.mark.parametrize(
    ("args", "outcomes"),
    [
        ([], {"passed": 1, "skipped": 2}),
        (["-m", "gpu"], {"passed": 2, "deselected": 1}),
        (["-m", "gpu and not slow"], {"passed": 1, "deselected": 2}),
        (["-m", "slow"], {"skipped": 1, "deselected": 2}),
        (["-m", "not gpu"], {"passed": 1, "deselected": 2}),
    ],
)
def test_gpu_tests_run_only_with_m_gpu(
    sample_project: pytest.Pytester, args: list[str], outcomes: dict[str, int]
) -> None:
    result = sample_project.runpytest("test_sample.py", *args)
    result.assert_outcomes(**outcomes)


def test_skip_reason_says_how_to_run_gpu_tests(
    sample_project: pytest.Pytester,
) -> None:
    result = sample_project.runpytest("test_sample.py")
    result.stdout.fnmatch_lines(["*SKIPPED*needs a GPU; run with -m gpu*"])
