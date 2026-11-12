"""Fixtures for the experiment-file and runner tests.

A two-prompt set and specs on it, for expand; an experiment file for
tiny-vl and a stand-in costs file, for pushlab run and status.
"""

from collections.abc import Callable
from pathlib import Path

import pytest

# Three pulse lengths of the recipe's hostile neuron, each with two controls.
PULSES = """\
conditions:
  - label: pulse
    shape: pulse
    params: {unit: qwen3.5-4b:L14:N8949, alpha: 1.0, start: 0}
    grid: {length: [4, 16, 64]}
    controls: 2
    control_seed: 7
"""


@pytest.fixture
def two_prompts(tmp_path: Path) -> Path:
    """A prompt set of two prompts, p-1 and p-2."""
    path = tmp_path / "two.jsonl"
    path.write_text(
        '{"prompt_id": "p-1", "version": 1, "prompt": "Say hi."}\n'
        '{"prompt_id": "p-2", "version": 1, "prompt": "Count to three."}\n',
        encoding="utf-8",
    )
    return path


@pytest.fixture
def pulses() -> str:
    """The conditions: three pulse lengths, each with two controls (seed 7)."""
    return PULSES


@pytest.fixture
def spec_text(two_prompts: Path) -> Callable[..., str]:
    """Experiment YAML on the two prompts: sampled at 0.7 with seeds 0 and 1,
    64 tokens, batches of 8, the conditions given (three pulse lengths by
    default), and any setting replaced by keyword, as YAML text."""

    def text(conditions: str = PULSES, **settings: str) -> str:
        lines = {
            "name": "pulse-lengths",
            "model": "qwen3.5-4b",
            "prompts": str(two_prompts),
            "decoding": "{strategy: sample, temperature: 0.7}",
            "seeds": "[0, 1]",
            "max_new_tokens": "64",
            "batch_size": "8",
            "max_gpu_hours": "1",
        } | settings
        return "".join(f"{key}: {value}\n" for key, value in lines.items()) + conditions

    return text


# No push and a pulse on each of the five smoke prompts: 10 runs, in five
# batches of 2 (a prompt's two runs together).
TINY_CONDITIONS = """\
conditions:
  - label: none
    segments: []
  - label: pulse
    segments: [{unit: tiny-vl:L2:N100, alpha: 1.0}]
"""


@pytest.fixture
def tiny_spec(tiny_vl_calibration: Path, tmp_path: Path) -> Callable[..., Path]:
    """Write an experiment file for tiny-vl, downloaded and calibrated, and
    return its path: TINY_CONDITIONS on prompts/smoke.jsonl, greedy in
    float32, 4 reply tokens, batches of 2 and max_gpu_hours 1. Keyword
    arguments replace settings, as YAML text; file names the file."""

    def write(
        conditions: str = TINY_CONDITIONS, file: str = "tiny.yaml", **settings: str
    ) -> Path:
        lines = {
            "name": "tiny-pulses",
            "model": "tiny-vl",
            "prompts": "prompts/smoke.jsonl",
            "decoding": "{strategy: greedy, dtype: float32}",
            "seeds": "[0]",
            "max_new_tokens": "4",
            "batch_size": "2",
            "max_gpu_hours": "1",
        } | settings
        path = tmp_path / file
        text = "".join(f"{key}: {value}\n" for key, value in lines.items())
        path.write_text(text + conditions, encoding="utf-8")
        return path

    return write


@pytest.fixture
def costs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A costs file that knows tiny-vl's speed, standing in for configs/costs.yaml."""
    from pushlab.runs import estimate

    path = tmp_path / "costs.yaml"
    path.write_text(
        "gpu: RTX 4090\n"
        "dollars_per_hour: [0.34, 0.74]\n"
        "reply_tokens_per_second:\n"
        "  tiny-vl: 0.01\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(estimate, "COSTS_FILE", path)
    return path


@pytest.fixture
def failing_generate(monkeypatch: pytest.MonkeyPatch) -> Callable[..., None]:
    """A function that makes generate() raise error in every call holding a
    pushed run of prompt; calls without one run as usual."""
    from pushlab.engine import generate as generate_module

    real = generate_module.generate

    def fail(prompt: str, error: Exception) -> None:
        def failing(model, tokenizer, prompts, *args, schedules=None, **kwargs):  # type: ignore[no-untyped-def]
            rows = zip(prompts, schedules or [], strict=True)
            if any(text == prompt and schedule.segments for text, schedule in rows):
                raise error
            return real(model, tokenizer, prompts, *args, schedules=schedules, **kwargs)

        monkeypatch.setattr(generate_module, "generate", failing)

    return fail


# Mayner's data in his formats, made up: see mayner_data's docstring.
@pytest.fixture
def mayner_data(tmp_path: Path) -> Path:
    """A data folder holding made-up data in Will Mayner's formats, as
    `pushlab fetch mayner` leaves his bundle there; returns the folder,
    tmp_path / "data". None of it is his data.

    data/mayner/bundle/ has, at his bundle's own paths:

    - split/top_neurons.parquet, as his step 5 writes it: the top 60 neurons
      of the roles coral_reef, pirate and sommelier, 60 different neurons per
      role, all in layers 0 to 18 (pirate's rank 1 is layer 18 neuron 12287,
      the last), ranked by |contribution|, with share = 100 * contribution /
      |v_r| for a split |v_r| of 25.05, 10.01 and 40.0;
    - split/concentration_neurons.parquet, as step 5 writes it: each role's
      share of |v_r| from its top k neurons, 100 * (the sum of their
      contributions) / |v_r|, for k from 1 to 60;
    - vectors/without/coral_reef.pt, pirate.pt and default.pt, as his step 4
      writes them: bfloat16 vectors of 36 layers by 8, whose layer-18
      differences from the default are exactly 25.0 and 10.0 long (and
      other lengths at other layers). sommelier has none, so its |v_r| must
      come from the shares;
    - role_steering_sizes/judged_replies.parquet, as his step 6b writes it,
      10 replies per condition (5 questions, 2 samples), each with the text
      "(a reply)". The top neurons' replies (kind neurons, draw 0) have the
      scores, "-" for none:
        pirate k 10:    3 3 2 1 0 2 3 - 2 1  (9 scored, 6 of them 2 or 3)
        pirate k 50:    3 3 3 2 3 3 2 3 1 3  (10 scored, 9)
        sommelier k 10: 0 1 0 0 2 1 0 0 1 0  (10 scored, 1)
        sommelier k 50: 2 3 1 2 2 0 3 2 1 2  (10 scored, 7)
      and rows an importer must leave out, all scored 0 or all 3: pirate's
      random 10, specific 50, shared 50, neurons 20 and neurons 10 of draw
      1, and coral_reef's random 10. coral_reef has no top neurons' scores.

    data/mayner/FETCHED.json records the download, at 2026-11-12T20:31:05Z,
    from files_url https://huggingface.co/datasets/wmayner/persona-neurons/resolve/main.
    """
    # Imported here, as the fixtures above import what they alone need.
    import json

    import numpy as np
    import pandas as pd
    import torch

    from pushlab.discovery.sources import (
        MAYNER,
        MAYNER_COMMIT,
        MAYNER_FILES,
        MAYNER_REPOSITORY,
    )

    data = tmp_path / "data"
    bundle = data / "mayner" / "bundle"
    for folder in ("split", "vectors/without", "role_steering_sizes"):
        (bundle / folder).mkdir(parents=True)

    # split/: each role's top 60 of the 19 x 12288 neurons of layers 0 to 18.
    layers, neurons, ranks = 19, 12288, 60
    split_norms = {"coral_reef": 25.05, "pirate": 10.01, "sommelier": 40.0}
    draws = np.random.default_rng(20261112)
    tops, curves = [], []
    for role, norm in split_norms.items():
        flat = draws.choice(layers * neurons - 1, size=ranks, replace=False)
        if role == "pirate":
            flat[0] = layers * neurons - 1  # layer 18, neuron 12287
        layer, neuron = np.divmod(flat, neurons)
        size = np.sort(draws.uniform(0.05, 2.0, ranks))[::-1]
        contribution = (size * draws.choice([-1.0, 1.0], ranks)).astype(np.float32)
        delta = draws.uniform(0.2, 3.0, ranks) * draws.choice([-1.0, 1.0], ranks)
        # His split.top_neurons, then the role and delta: share is float32
        # times 100 over a float64 |v_r|, so float64.
        top = pd.DataFrame(
            {
                "rank": np.arange(1, ranks + 1),
                "layer": layer,
                "neuron": neuron,
                "contribution": contribution,
                "share": 100 * contribution / np.float64(norm),
            }
        )
        tops.append(top.assign(role=role, delta=delta.astype(np.float32)))
        # His split.top_curve: the cumulative sums in float64, as a %.
        curve = 100 * np.cumsum(contribution.astype(np.float64)) / norm
        curves.append(
            pd.DataFrame(
                {
                    "role": role,
                    "basis": "neurons",
                    "k": np.arange(1, ranks + 1),
                    "share": curve,
                }
            )
        )
    split = bundle / "split"
    pd.concat(tops, ignore_index=True).to_parquet(
        split / "top_neurons.parquet", index=False
    )
    pd.concat(curves, ignore_index=True).to_parquet(
        split / "concentration_neurons.parquet", index=False
    )

    # vectors/without/: layer 18 of each role minus the default's is (15, 20)
    # or (6, 8) in two places, 25 and 10 long, all exact in bfloat16.
    generator = torch.Generator().manual_seed(20261112)
    default = torch.randn(36, 8, generator=generator)
    default[18] = torch.tensor([1.0, -2.0, 0.5, 3.0, -1.0, 2.0, 0.25, -0.5])
    differences = {
        "coral_reef": [15.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "pirate": [0.0, 0.0, 6.0, 8.0, 0.0, 0.0, 0.0, 0.0],
    }
    vectors = bundle / "vectors" / "without"
    for role, difference in differences.items():
        vector = default + torch.randn(36, 8, generator=generator)
        vector[18] = default[18] + torch.tensor(difference)
        saved = {"vector": vector.bfloat16(), "type": "mean", "role": role}
        torch.save(saved, vectors / f"{role}.pt")
    saved = {"vector": default.bfloat16(), "type": "mean", "role": "default"}
    torch.save(saved, vectors / "default.pt")

    # role_steering_sizes/: his REPLY_COLUMNS, 10 replies per condition.
    scores = {
        ("pirate", "neurons", 10, 0): [3, 3, 2, 1, 0, 2, 3, None, 2, 1],
        ("pirate", "neurons", 50, 0): [3, 3, 3, 2, 3, 3, 2, 3, 1, 3],
        ("sommelier", "neurons", 10, 0): [0, 1, 0, 0, 2, 1, 0, 0, 1, 0],
        ("sommelier", "neurons", 50, 0): [2, 3, 1, 2, 2, 0, 3, 2, 1, 2],
        # Other kinds, sizes and draws, which an importer leaves out.
        ("pirate", "random", 10, 0): [3] * 10,
        ("pirate", "specific", 50, 0): [0] * 10,
        ("pirate", "shared", 50, 0): [0] * 10,
        ("pirate", "neurons", 20, 0): [3] * 10,
        ("pirate", "neurons", 10, 1): [0] * 10,
        ("coral_reef", "random", 10, 0): [3] * 10,
    }
    replies = []
    for (role, kind, k, draw), marks in scores.items():
        for i, score in enumerate(marks):
            question, sample = divmod(i, 2)
            replies.append(
                {
                    "role": role,
                    "condition_id": f"{role}|{kind}|k{k}|d{draw}",
                    "kind": kind,
                    "k": k,
                    "draw": draw,
                    "strength": 4.0,
                    "question_index": question,
                    "sample": sample,
                    "seed": 20261103 + 10 * question + sample,
                    "response": "(a reply)",
                    "score": score,
                }
            )
    pd.DataFrame(replies).to_parquet(
        bundle / "role_steering_sizes" / "judged_replies.parquet", index=False
    )

    # The record `pushlab fetch mayner` writes (decision 0020).
    record = {
        "source": MAYNER.name,
        "title": MAYNER.title,
        "url": MAYNER.url,
        "files_url": f"{MAYNER.url}/resolve/main",
        "licence": MAYNER.licence,
        "credit": MAYNER.credit,
        "downloaded_at": "2026-11-12T20:31:05Z",
        "only": list(MAYNER_FILES),
        "script": {
            "repository": MAYNER_REPOSITORY,
            "commit": MAYNER_COMMIT,
            "command": f"python download_data.py {MAYNER.url} --only "
            + " ".join(MAYNER_FILES),
        },
        "manifest_sha256": "5e" * 32,
    }
    (data / "mayner" / "FETCHED.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    return data
