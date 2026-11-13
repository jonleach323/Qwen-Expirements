"""Made-up golden cases for the tests of pushlab.runs.golden and pushlab
golden record (decision 0023).

golden_data gives one prompt run the six ways tests/golden/qwen3-8b.yaml
runs each of its prompts, with a neuron and a set of three neurons of a
model's layers 0 and 1, so it suits the registry's tiny model and tiny-vl
alike. set_key and drop_key make one change to such data, by a path such as
cases.1.push.alpha. Nothing here comes from the Neuron Persona Library.
"""

from collections.abc import Callable
from typing import Any

IDS = [
    "hi-greedy",
    "hi-neuron-fixed",
    "hi-neuron-per-prompt",
    "hi-set-per-prompt",
    "hi-sampled-seed-1",
    "hi-sampled-seed-2",
]


def golden_data(model: str = "tiny") -> dict[str, Any]:
    """Made-up cases of model as plain data, in IDS's order: greedy and
    unpushed; its L1:N5 at dose 1, fixed and per_prompt; its set trio,
    per_prompt; sampled at 0.7 with seeds 1 and 2. Five reply tokens, float32."""
    greedy = {"strategy": "greedy"}
    neuron, trio = f"{model}:L1:N5", f"{model}:set:trio"

    def case(name: str, decoding: dict, push: tuple[str, str] | None = None) -> dict:
        item: dict[str, Any] = {"id": f"hi-{name}", "prompt": "hi"}
        item["decoding"] = dict(decoding)
        if push is not None:
            item["push"] = {"unit": push[0], "alpha": 1.0, "calibration": push[1]}
        return item

    def sampled(seed: int) -> dict:
        return {"strategy": "sample", "temperature": 0.7, "seed": seed}

    return {
        "version": 1,
        "model": model,
        "dtype": "float32",
        "max_new_tokens": 5,
        "prompts": {"hi": "Say hi."},
        "sets": {
            trio: {
                "entries": [[0, 5, 1.0], [1, 9, -0.5], [1, 17, 0.75]],
                "size": 0.2,
                "source": "made up for the tests",
            }
        },
        "cases": [
            case("greedy", greedy),
            case("neuron-fixed", greedy, (neuron, "fixed")),
            case("neuron-per-prompt", greedy, (neuron, "per_prompt")),
            case("set-per-prompt", greedy, (trio, "per_prompt")),
            case("sampled-seed-1", sampled(1)),
            case("sampled-seed-2", sampled(2)),
        ],
    }


def _parent(data: Any, path: list[str]) -> Any:
    for step in path:
        data = data[int(step)] if step.isdigit() else data[step]
    return data


def set_key(key: str, value: object) -> Callable[[dict], None]:
    """A change that sets key, a path such as cases.1.push.alpha, to value."""

    def change(data: dict) -> None:
        *path, last = key.split(".")
        _parent(data, path)[int(last) if last.isdigit() else last] = value

    return change


def drop_key(key: str) -> Callable[[dict], None]:
    """A change that removes key, a path such as cases.4.decoding.seed."""

    def change(data: dict) -> None:
        *path, last = key.split(".")
        del _parent(data, path)[last]

    return change
