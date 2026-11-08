"""What an experiment will take: reply tokens, GPU-hours and dollars.

configs/costs.yaml names the GPU, its price in dollars an hour (a range,
from the cheapest tier to the dearest), and each model's speed in reply
tokens a second over a whole batch. Once a run has measured a model's
speed in some data type, <runs dir>/speeds.json has it, with the time the
model took to load, and estimates use those instead. Every reply is taken
to run to max_new_tokens, so the reply tokens are an upper bound; hours
and cost are estimates at the speed given, which depends on how full the
batches are, and the speed line says what it was measured on.
"""

import fcntl
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pushlab import PROJECT_ROOT
from pushlab.runs.folder import read_json, write_json
from pushlab.schedules.yaml_text import load_yaml
from pushlab.store.folders import shown_path

COSTS_FILE = PROJECT_ROOT / "configs" / "costs.yaml"
# In the runs dir itself: the speed each model last ran at, for estimates.
SPEEDS_FILE = "speeds.json"
COSTS_KEYS = ("gpu", "dollars_per_hour", "reply_tokens_per_second")


@dataclass(frozen=True)
class Costs:
    """The GPU, its price range in dollars an hour, and models' speeds."""

    gpu: str
    dollars_per_hour: tuple[float, float]
    reply_tokens_per_second: dict[str, float]
    path: Path


@dataclass(frozen=True)
class Speed:
    """Reply tokens a second, and where the number comes from."""

    reply_tokens_per_second: float
    source: str
    load_seconds: float = 0.0  # once per session, as measured

    def text(self) -> str:
        """The speed in words.

        Example:
            >>> Speed(44.0, "from configs/costs.yaml").text()
            '44 reply tokens a second, from configs/costs.yaml'
        """
        return (
            f"{self.reply_tokens_per_second:.3g} reply tokens a second, {self.source}"
        )


def _positive(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def read_costs(path: Path | None = None) -> Costs:
    """Read and check configs/costs.yaml, or the file path names.

    Example:
        >>> read_costs().dollars_per_hour
        (0.34, 0.74)
    """
    path = COSTS_FILE if path is None else path
    try:
        data = load_yaml(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read {path}: {error}") from None
    if not isinstance(data, dict):
        raise ValueError(f"{path} must be a mapping of {', '.join(COSTS_KEYS)}")
    if missing := [key for key in COSTS_KEYS if key not in data]:
        raise ValueError(f"{path} needs {', '.join(missing)}")
    prices = data["dollars_per_hour"]
    if not (
        isinstance(prices, list)
        and len(prices) == 2
        and all(_positive(price) for price in prices)
        and prices[0] <= prices[1]
    ):
        raise ValueError(
            f"{path}: dollars_per_hour must be two prices above 0, the lower "
            f"first, such as [0.34, 0.74], not {prices!r}"
        )
    speeds = data["reply_tokens_per_second"]
    if not isinstance(speeds, dict):
        raise ValueError(f"{path}: reply_tokens_per_second must map models to speeds")
    for model, speed in speeds.items():
        if not _positive(speed):
            raise ValueError(
                f"{path}: reply_tokens_per_second for {model} must be a number "
                f"above 0, not {speed!r}"
            )
    return Costs(
        str(data["gpu"]),
        (float(prices[0]), float(prices[1])),
        {str(model): float(speed) for model, speed in speeds.items()},
        path,
    )


def _measured(speed: object) -> bool:
    """Whether speed is a measured speed as save_speed writes it."""
    return (
        isinstance(speed, dict)
        and _positive(speed.get("reply_tokens_per_second"))
        and _positive(speed.get("rows_per_batch", speed.get("batch_size")))
        and isinstance(speed.get("experiment"), str)
        and isinstance(speed.get("measured_at"), str)
    )


def read_speeds(runs_dir: Path) -> dict[str, dict[str, Any]]:
    """The speeds runs have measured, by model and data type, such as
    "qwen3.5-4b/bfloat16"; empty when none has.

    ValueError says when the file is not one save_speed wrote.

    Example:
        >>> read_speeds(Path("/no/such/folder"))
        {}
    """
    path = runs_dir / SPEEDS_FILE
    speeds = read_json(path) or {}
    if not (isinstance(speeds, dict) and all(map(_measured, speeds.values()))):
        raise ValueError(
            f"{path} is not a speeds file as pushlab run writes it; delete it, "
            "and the next run measures the speed again"
        )
    return speeds


def speeds_problem(runs_dir: Path) -> str | None:
    """Why the speeds file is left out of estimates, or None when it is not.

    Example:
        >>> speeds_problem(Path("/no/such/folder")) is None
        True
    """
    try:
        read_speeds(runs_dir)
    except (OSError, ValueError) as error:
        return f"{error}. Estimates leave it out"
    return None


def save_speed(runs_dir: Path, key: str, speed: dict[str, Any]) -> None:
    """Record the speed a run measured, in place of the last one for key.

    Runners of two experiments may finish at once, so the file is read and
    rewritten holding a lock of its own.

    Example:
        >>> save_speed(runs_dir, "qwen3.5-4b/bfloat16", speed)  # doctest: +SKIP
    """
    handle = os.open(runs_dir / f".{SPEEDS_FILE}.lock", os.O_RDONLY | os.O_CREAT, 0o644)
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        write_json(runs_dir / SPEEDS_FILE, read_speeds(runs_dir) | {key: speed})
    finally:
        os.close(handle)


def speed_for(model: str, dtype: str, runs_dir: Path, costs: Costs) -> Speed | None:
    """The speed to estimate with: the last measured, else the costs file's.

    None when neither knows the model.

    Example:
        >>> speed_for("qwen3.5-4b", "bfloat16", Path("/no/such"), read_costs()).text()
        '44 reply tokens a second, from configs/costs.yaml'
    """
    try:
        measured = read_speeds(runs_dir).get(f"{model}/{dtype}")
    except (OSError, ValueError):  # a broken speeds file (speeds_problem says)
        measured = None
    if measured:
        where = measured.get("gpu") or "the CPU"
        rows = measured.get("rows_per_batch", measured["batch_size"])
        load = measured.get("load_seconds")
        return Speed(
            float(measured["reply_tokens_per_second"]),
            f"measured by {measured['experiment']} on {where} with {rows:.3g} "
            f"{'row' if rows == 1 else 'rows'} a batch, {measured['measured_at'][:10]}",
            float(load) if _positive(load) else 0.0,
        )
    if model in costs.reply_tokens_per_second:
        rate = costs.reply_tokens_per_second[model]
        return Speed(rate, f"from {shown_path(costs.path)}")
    return None


def duration(seconds: float) -> str:
    """A length of time in words: seconds, minutes or hours.

    Example:
        >>> duration(40), duration(4000), duration(30_000)
        ('40 s', '67 minutes', '8.3 hours')
    """
    if seconds < 120:
        return f"{seconds:.0f} s"
    if seconds < 7200:
        return f"{seconds / 60:.0f} minutes"
    return f"{seconds / 3600:.1f} hours"


def exact(value: float) -> str:
    """A number, short when that is exact.

    Example:
        >>> exact(1.0), exact(0.25), exact(0.1234567)
        ('1', '0.25', '0.1234567')
    """
    short = format(value, "g")
    return short if float(short) == value else repr(value)


def _dollars(amount: float) -> str:
    return f"${amount:.2f}"


def estimate_lines(
    runs: int,
    max_new_tokens: int,
    speed: Speed | None,
    costs: Costs,
    max_gpu_hours: float,
    used_hours: float = 0.0,
) -> list[tuple[str, str]]:
    """The reply tokens, speed, GPU-hours and cost of runs still to run.

    runs are the rows the calls will run. used_hours are already spent on
    the experiment, against max_gpu_hours; the measured load time is added
    once, for the session that runs them, unless there is nothing to run.

    Example:
        >>> lines = estimate_lines(50, 96, Speed(44.0, "from x"), read_costs(), 0.25)
        >>> dict(lines)["gpu-hours"]
        'about 0.03, within max_gpu_hours 0.25'
    """
    tokens = runs * max_new_tokens
    lines = [("reply tokens", f"at most {tokens:,}, {max_new_tokens:,} for each run")]
    if speed is None:
        lines.append(
            ("speed", f"unknown: add this model's to {shown_path(costs.path)}")
        )
        return lines
    load = speed.load_seconds if runs else 0.0  # nothing to run: nothing loads
    hours = (load + tokens / speed.reply_tokens_per_second) / 3600
    left = max_gpu_hours - used_hours
    cap = f"max_gpu_hours {exact(max_gpu_hours)}"
    if used_hours:
        cap = f"the {max(left, 0):.2f} left of {cap}"
    if hours <= left:
        verdict = f"within {cap}"
    else:
        share = max(left, 0) / hours
        verdict = f"more than {cap}: it would stop after about {share:.0%} of the runs"
    low, high = (hours * price for price in costs.dollars_per_hour)
    if high < 0.005:
        cost = "under $0.01"
    else:
        cost = f"about {_dollars(low)} to {_dollars(high)}"
    prices = " to ".join(_dollars(price) for price in costs.dollars_per_hour)
    return lines + [
        ("speed", speed.text()),
        ("gpu-hours", f"about {hours:.2f}, {verdict}"),
        ("cost", f"{cost}, on an {costs.gpu} at {prices} an hour"),
    ]
