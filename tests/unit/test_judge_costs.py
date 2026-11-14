"""What judging costs (pushlab.scoring.judge_costs): configs/costs.yaml's
judges section, the dry run's estimate of a job's tokens and dollars, and
the dollars a job's tokens cost."""

import doctest
from pathlib import Path

import pytest

from pushlab import PROJECT_ROOT
from pushlab.runs import estimate
from pushlab.runs.estimate import read_costs
from pushlab.scoring import judge_costs
from pushlab.scoring.judge_costs import (
    JudgeCosts,
    Prices,
    estimate_judging,
    read_judge_costs,
    spent,
)
from pushlab.scoring.rubrics import Request
from pushlab.store.folders import shown_path

# A costs file's other keys, as configs/costs.yaml has them.
TOP = "gpu: RTX 4090\ndollars_per_hour: [0.34, 0.74]\nreply_tokens_per_second: {}\n"
JUDGES = """\
judges:
  characters_per_token: 3.5
  output_tokens_per_call: 300
  api:
    claude-haiku-5-5: {input: 0.10, output: 0.50}
  local:
    calls_per_second: 2
"""
# 7,000 requests of 120 characters each: 85 of instructions, and a user's
# message of 35, the 20-character text between <text> and </text>.
REQUEST = Request("coherence", 1, "s" * 85, "", "t" * 20, {})
REQUESTS = [REQUEST] * 7000


def _costs(tmp_path: Path, judges: str = JUDGES) -> JudgeCosts:
    path = tmp_path / "costs.yaml"
    path.write_text(TOP + judges, encoding="utf-8")
    return read_judge_costs(path)


def test_the_costs_file_has_haiku_s_prices_and_the_assumptions() -> None:
    costs = read_judge_costs()  # configs/costs.yaml itself
    assert costs.path == PROJECT_ROOT / "configs" / "costs.yaml"
    assert costs.prices == {"claude-haiku-5-5": Prices(0.10, 0.50)}
    assert (costs.characters_per_token, costs.output_tokens_per_call) == (3.5, 300)
    assert costs.calls_per_second == 2.0
    assert (costs.gpu, costs.dollars_per_hour) == ("RTX 4090", (0.34, 0.74))


def test_the_runs_estimates_still_read_the_file_with_its_judges() -> None:
    costs = read_costs()
    assert costs.gpu == "RTX 4090" and costs.dollars_per_hour == (0.34, 0.74)
    assert costs.reply_tokens_per_second == {"qwen3.5-4b": 44.0, "qwen3-8b": 22.0}


def test_the_file_is_read_where_estimate_points_when_called(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "costs.yaml"
    path.write_text(TOP + JUDGES.replace("3.5", "4"), encoding="utf-8")
    monkeypatch.setattr(estimate, "COSTS_FILE", path)
    costs = read_judge_costs()
    assert costs.path == path and costs.characters_per_token == 4.0


@pytest.mark.parametrize(
    ("judges", "problem"),
    [
        ("", "needs a judges section"),
        ("judges: [1]\n", "judges must be a mapping of characters_per_token"),
        (
            JUDGES.replace("  output_tokens_per_call: 300\n", ""),
            "judges lacks output_tokens_per_call",
        ),
        (JUDGES + "  speed: 1\n", "judges has unknown keys speed"),
        (
            JUDGES.replace("3.5", "0"),
            "judges.characters_per_token must be a number above 0, not 0",
        ),
        (
            JUDGES.replace("300", "2.5"),
            "judges.output_tokens_per_call must be a whole number from 1, not 2.5",
        ),
        (
            JUDGES.replace("0.10", "free"),
            "judges.api.claude-haiku-5-5.input must be a number above 0, not 'free'",
        ),
        (
            JUDGES.replace(", output: 0.50", ""),
            "judges.api.claude-haiku-5-5 must be a mapping of input and output",
        ),
        (
            JUDGES.replace(
                "    claude-haiku-5-5: {input: 0.10, output: 0.50}\n", ""
            ).replace("api:", "api: [x]"),
            "judges.api must map models to their prices",
        ),
        (
            JUDGES.replace("calls_per_second: 2", "calls_per_second: -1"),
            "judges.local.calls_per_second must be a number above 0, not -1",
        ),
        (
            JUDGES.replace("calls_per_second", "calls_per_minute"),
            "judges.local must be a mapping of calls_per_second",
        ),
    ],
)
def test_a_broken_judges_section_is_refused_naming_what_is_wrong(
    tmp_path: Path, judges: str, problem: str
) -> None:
    with pytest.raises(ValueError, match=problem):
        _costs(tmp_path, judges)


def test_a_broken_file_is_refused_as_the_runs_estimates_refuse_it(
    tmp_path: Path,
) -> None:
    path = tmp_path / "costs.yaml"
    path.write_text(JUDGES, encoding="utf-8")  # no gpu, prices or speeds
    with pytest.raises(ValueError, match="needs gpu, dollars_per_hour"):
        read_judge_costs(path)
    with pytest.raises(ValueError, match="cannot read"):
        read_judge_costs(tmp_path / "missing.yaml")


def test_an_api_estimate_prices_the_tokens_at_the_models_list_prices(
    tmp_path: Path,
) -> None:
    costs = _costs(tmp_path)
    assert len(REQUEST.system) + len(REQUEST.user) == 120
    lines = estimate_judging("api", "claude-haiku-5-5", REQUESTS, 7000, costs)
    # 840,000 characters are 240,000 tokens; 7,000 answers 2,100,000 tokens:
    # $0.024 and $1.05.
    assert lines == [
        ("input tokens", "about 240,000: 840,000 characters at 3.5 a token, assumed"),
        (
            "output tokens",
            "about 2,100,000: 300 a call, the judge's thinking included, assumed",
        ),
        (
            "cost",
            "about $1.07, at claude-haiku-5-5's list prices of $0.10 and $0.50 per "
            "million input and output tokens",
        ),
    ]


def test_a_local_estimate_prices_the_gpu_time_at_the_pods_prices(
    tmp_path: Path,
) -> None:
    costs = _costs(tmp_path)
    lines = dict(estimate_judging("local", "granite-4.1-8b", REQUESTS, 7000, costs))
    # 7,000 calls at 2 a second: 3,500 s, 0.97 hours, at $0.34 to $0.74 an hour.
    assert lines["gpu-hours"] == "about 0.97: 58 minutes at 2 calls a second, assumed"
    assert (
        lines["cost"]
        == "about $0.33 to $0.72, on an RTX 4090 at $0.34 to $0.74 an hour"
    )
    assert lines["input tokens"].startswith("about 240,000:")


# Calls at 2 a second, at $0.34 to $0.74 an hour: 48 cost $0.0023 to $0.0049,
# 49 $0.0023 to $0.0050, 60 $0.0028 to $0.0062, and 106 $0.0050 to $0.0109.
@pytest.mark.parametrize(
    ("calls", "cost"),
    [
        (48, "under $0.01"),
        (49, "under $0.01 to $0.01"),
        (60, "under $0.01 to $0.01"),
        (106, "about $0.01 to $0.01"),
    ],
)
def test_a_local_estimate_under_a_cent_is_never_about(
    tmp_path: Path, calls: int, cost: str
) -> None:
    lines = dict(
        estimate_judging("local", "granite-4.1-8b", [], calls, _costs(tmp_path))
    )
    assert lines["cost"] == f"{cost}, on an RTX 4090 at $0.34 to $0.74 an hour"


def test_an_api_model_without_prices_has_an_unknown_cost(tmp_path: Path) -> None:
    costs = _costs(tmp_path)
    lines = dict(estimate_judging("api", "claude-opus-5-5", REQUESTS[:2], 2, costs))
    where = shown_path(tmp_path / "costs.yaml")
    assert lines["cost"] == f"unknown: add the prices of claude-opus-5-5 to {where}"
    assert spent("claude-opus-5-5", 10, 10, costs) is None
    real = read_judge_costs()
    assert dict(estimate_judging("api", "claude-x", [], 0, real))["cost"] == (
        "unknown: add the prices of claude-x to configs/costs.yaml"
    )


def test_nothing_to_send_costs_nothing(tmp_path: Path) -> None:
    lines = dict(estimate_judging("api", "claude-haiku-5-5", [], 0, _costs(tmp_path)))
    assert lines["input tokens"].startswith("about 0:")
    assert lines["cost"].startswith("$0, at claude-haiku-5-5's list prices")


def test_small_costs_read_as_under_a_cent(tmp_path: Path) -> None:
    lines = dict(
        estimate_judging("api", "claude-haiku-5-5", REQUESTS[:7], 7, _costs(tmp_path))
    )
    assert lines["cost"].startswith("under $0.01, at claude-haiku-5-5's")


def test_spent_prices_a_jobs_tokens(tmp_path: Path) -> None:
    costs = _costs(tmp_path)
    assert spent("claude-haiku-5-5", 2_000_000, 1_000_000, costs) == pytest.approx(0.7)
    assert spent("claude-haiku-5-5", 0, 0, costs) == 0.0


def test_docstring_examples_run() -> None:
    result = doctest.testmod(judge_costs)
    assert result.attempted > 0
    assert result.failed == 0
