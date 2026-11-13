"""What judging costs: configs/costs.yaml's judges section, the estimate of a
job's tokens and dollars, and the dollars the tokens a job used cost.

The judges section (decision 0021) holds what a judge's calls are taken to
need until they are measured: characters_per_token, the characters a token
holds in a judge's prompt, and output_tokens_per_call, the tokens each
answer takes, the judge's thinking included; under api, each model's list
prices in dollars per million input and output tokens; under local,
calls_per_second, how fast a model that vLLM serves on the pod answers,
whose time costs the GPU's dollars_per_hour from the same file
(pushlab.runs.estimate). `pushlab score --dry-run` prints estimate_judging's
lines, and its summary after a job prices the tokens used with spent.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pushlab.runs import estimate
from pushlab.schedules.yaml_text import load_yaml
from pushlab.scoring.rubrics import Request
from pushlab.store.folders import shown_path

JUDGES = "judges"
JUDGES_KEYS = ("characters_per_token", "output_tokens_per_call", "api", "local")
PRICE_KEYS = ("input", "output")
LOCAL_KEYS = ("calls_per_second",)
MILLION = 1_000_000


@dataclass(frozen=True)
class Prices:
    """A model's list prices, in dollars per million input and output tokens."""

    input: float
    output: float


@dataclass(frozen=True)
class JudgeCosts:
    """The judges section of a costs file, with the GPU and its prices."""

    characters_per_token: float
    output_tokens_per_call: int
    prices: dict[str, Prices]  # the api models', by model
    calls_per_second: float  # a local judge's
    gpu: str
    dollars_per_hour: tuple[float, float]
    path: Path


def _number(value: object, where: str) -> float:
    """value, a number above 0; ValueError naming where it is otherwise."""
    if (
        not isinstance(value, int | float)
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{where} must be a number above 0, not {value!r}")
    return float(value)


def _mapping(value: object, keys: tuple[str, ...], where: str) -> dict[Any, Any]:
    """value, a mapping of exactly keys; ValueError otherwise."""
    if not isinstance(value, dict) or set(value) != set(keys):
        listed = ", ".join(keys[:-1]) + " and " if len(keys) > 1 else ""
        raise ValueError(f"{where} must be a mapping of {listed}{keys[-1]}")
    return value


def read_judge_costs(path: Path | None = None) -> JudgeCosts:
    """Read and check the judges section of configs/costs.yaml, or of the
    file path names, with the GPU and its prices from the rest of it.

    The file is pushlab.runs.estimate.COSTS_FILE, looked up when called.
    ValueError names what is wrong: the rest of the file, as read_costs
    checks it, a missing section, a key it lacks or does not know, and a
    value that is not a number above 0 (output_tokens_per_call: a whole
    number from 1).

    Example:
        >>> costs = read_judge_costs()
        >>> costs.prices["claude-haiku-5-5"], costs.calls_per_second
        (Prices(input=0.1, output=0.5), 2.0)
    """
    path = estimate.COSTS_FILE if path is None else Path(path)
    rest = estimate.read_costs(path)  # the rest of the file, checked
    data = load_yaml(path.read_text(encoding="utf-8"))
    if JUDGES not in data:
        raise ValueError(f"{path} needs a {JUDGES} section, as configs/costs.yaml has")
    judges = data[JUDGES]
    where = f"{path}: {JUDGES}"
    if not isinstance(judges, dict):
        _mapping(judges, JUDGES_KEYS, where)
    if missing := [key for key in JUDGES_KEYS if key not in judges]:
        raise ValueError(f"{where} lacks {', '.join(missing)}")
    if unknown := sorted(map(str, set(judges) - set(JUDGES_KEYS))):
        raise ValueError(f"{where} has unknown keys {', '.join(unknown)}")
    per_token = _number(judges["characters_per_token"], f"{where}.characters_per_token")
    per_call = judges["output_tokens_per_call"]
    if type(per_call) is not int or per_call < 1:
        raise ValueError(
            f"{where}.output_tokens_per_call must be a whole number from 1, "
            f"not {per_call!r}"
        )
    if not isinstance(judges["api"], dict):
        raise ValueError(f"{where}.api must map models to their prices")
    prices = {}
    for model, listed in judges["api"].items():
        listed = _mapping(listed, PRICE_KEYS, f"{where}.api.{model}")
        prices[str(model)] = Prices(
            *(_number(listed[key], f"{where}.api.{model}.{key}") for key in PRICE_KEYS)
        )
    local = _mapping(judges["local"], LOCAL_KEYS, f"{where}.local")
    speed = _number(local["calls_per_second"], f"{where}.local.calls_per_second")
    return JudgeCosts(
        per_token, per_call, prices, speed, rest.gpu, rest.dollars_per_hour, path
    )


def dollars(amount: float) -> str:
    """An amount of dollars in words: none, under a cent, or to the cent.

    Example:
        >>> dollars(0), dollars(0.0012), dollars(1234.567)
        ('$0', 'under $0.01', '$1,234.57')
    """
    if amount == 0:
        return "$0"
    if amount < 0.005:
        return "under $0.01"
    return f"${amount:,.2f}"


def _about(amount: float) -> str:
    """An estimate in dollars: about so many, none, or under a cent."""
    return dollars(amount) if amount < 0.005 else f"about {dollars(amount)}"


def _price(price: float) -> str:
    """A price in dollars, to the cent when that is exact.

    Example:
        >>> _price(0.1), _price(0.025)
        ('$0.10', '$0.025')
    """
    return f"${price:.2f}" if round(price, 2) == price else f"${estimate.exact(price)}"


def prices_text(model: str, prices: Prices) -> str:
    """A model's list prices in words.

    Example:
        >>> prices_text("m", Prices(0.1, 0.5))
        "m's list prices of $0.10 and $0.50 per million input and output tokens"
    """
    return (
        f"{model}'s list prices of {_price(prices.input)} and "
        f"{_price(prices.output)} per million input and output tokens"
    )


def spent(
    model: str, input_tokens: int, output_tokens: int, costs: JudgeCosts
) -> float | None:
    """The dollars that tokens of an api model cost at its list prices; None
    for a model the costs file has no prices for.

    Example:
        >>> spent("claude-haiku-5-5", 1_000_000, 2_000_000, read_judge_costs())
        1.1
    """
    prices = costs.prices.get(model)
    if prices is None:
        return None
    return (input_tokens * prices.input + output_tokens * prices.output) / MILLION


def _no_prices(model: str, costs: JudgeCosts) -> str:
    return f"unknown: add the prices of {model} to {shown_path(costs.path)}"


def spent_text(
    model: str, input_tokens: int, output_tokens: int, costs: JudgeCosts
) -> str:
    """What tokens of an api model cost at its list prices, in words.

    Example:
        >>> costs = read_judge_costs()
        >>> spent_text("claude-haiku-5-5", 1_000_000, 2_000_000, costs)[:40]
        "$1.10, at claude-haiku-5-5's list prices"
    """
    amount = spent(model, input_tokens, output_tokens, costs)
    if amount is None:
        return _no_prices(model, costs)
    return f"{dollars(amount)}, at {prices_text(model, costs.prices[model])}"


def estimate_judging(
    backend: str,
    model: str,
    requests: Sequence[Request],
    calls: int,
    costs: JudgeCosts,
) -> list[tuple[str, str]]:
    """The tokens and cost of calls to a judge, as (name, text) lines.

    requests are those still to send, and calls how many calls they take,
    one each. Input tokens are their instructions' and user messages'
    characters over characters_per_token; output tokens are
    output_tokens_per_call a call. The api backend's cost is at the model's
    list prices, unknown for a model with none; the local backend's is the
    GPU's time, calls over calls_per_second, at its dollars_per_hour.

    Example:
        >>> costs = read_judge_costs()
        >>> lines = estimate_judging("local", "granite-4.1-8b", [], 7200, costs)
        >>> dict(lines)["gpu-hours"]
        'about 1.00: 60 minutes at 2 calls a second, assumed'
    """
    characters = sum(len(request.system) + len(request.user) for request in requests)
    tokens = round(characters / costs.characters_per_token)
    out = calls * costs.output_tokens_per_call
    lines = [
        (
            "input tokens",
            f"about {tokens:,}: {characters:,} characters at "
            f"{costs.characters_per_token:g} a token, assumed",
        ),
        (
            "output tokens",
            f"about {out:,}: {costs.output_tokens_per_call:,} a call, the judge's "
            "thinking included, assumed",
        ),
    ]
    if backend == "local":
        hours = calls / costs.calls_per_second / 3600
        shown = f"{hours:.2f}" if hours >= 0.01 else f"{hours:.1g}"
        time = estimate.duration(hours * 3600)
        rate = f"{costs.calls_per_second:g} calls a second, assumed"
        low, high = (hours * price for price in costs.dollars_per_hour)
        cost = (
            _about(high) if high < 0.005 else f"about {dollars(low)} to {dollars(high)}"
        )
        prices = " to ".join(dollars(price) for price in costs.dollars_per_hour)
        return lines + [
            ("gpu-hours", f"about {shown}: {time} at {rate}"),
            ("cost", f"{cost}, on an {costs.gpu} at {prices} an hour"),
        ]
    listed = costs.prices.get(model) if backend == "api" else None
    if listed is None:
        if backend == "api":
            return lines + [("cost", _no_prices(model, costs))]
        return lines + [("cost", f"unknown: the {backend} backend has no prices")]
    amount = (tokens * listed.input + out * listed.output) / MILLION
    return lines + [("cost", f"{_about(amount)}, at {prices_text(model, listed)}")]
