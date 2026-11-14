"""|v_r| from Mayner's vector files, as his step 6 takes it (decision 0019)."""

import doctest
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from pushlab.units import mayner, mayner_vectors
from pushlab.units.mayner_vectors import (
    TARGET_LAYER,
    load_vector,
    vector_norm_from_vectors,
)

# Shaped like his vectors/without/*.pt: one row per layer of Qwen3-8B's 36,
# here 8 wide, stored in bfloat16 as his step 4 stores them.
LAYERS, WIDTH = 36, 8


def _vector_file(path: Path, seed: int, scale: float = 10.0) -> Path:
    """A vector file as his step 4 writes one: a dict with a bf16 "vector"."""
    generator = torch.Generator().manual_seed(seed)
    vector = (torch.randn(LAYERS, WIDTH, generator=generator) * scale).bfloat16()
    torch.save({"vector": vector, "type": "mean", "role": path.stem}, path)
    return path


@pytest.fixture
def vectors(tmp_path: Path) -> tuple[Path, Path]:
    """A role's vector file and the default's."""
    return _vector_file(tmp_path / "pirate.pt", 1), _vector_file(
        tmp_path / "default.pt", 2
    )


def test_the_norm_is_his_float64_difference_at_layer_18(
    vectors: tuple[Path, Path],
) -> None:
    role, default = (torch.load(path, weights_only=True)["vector"] for path in vectors)
    # His load_vector converts to float64; run_prepare subtracts at the
    # target layer; role_inputs takes numpy's norm.
    by_hand = float(np.linalg.norm((role.double()[18] - default.double()[18]).numpy()))
    assert TARGET_LAYER == 18
    assert vector_norm_from_vectors(*vectors) == by_hand
    # Subtracting the bfloat16 vectors first gives another number.
    in_bfloat16 = float(np.linalg.norm((role[18] - default[18]).double().numpy()))
    assert in_bfloat16 != by_hand


def test_another_layer_can_be_named(vectors: tuple[Path, Path]) -> None:
    role, default = (load_vector(path) for path in vectors)
    for layer in (0, 35):
        expected = float(
            np.linalg.norm((role.double()[layer] - default.double()[layer]).numpy())
        )
        assert vector_norm_from_vectors(*vectors, layer=layer) == expected


@pytest.mark.parametrize("layer", [-1, 36, True, 18.0, "18"])
def test_a_layer_outside_the_vectors_is_refused(
    vectors: tuple[Path, Path], layer: object
) -> None:
    with pytest.raises(
        ValueError, match="layer must be a layer of the vectors, 0 to 35"
    ):
        vector_norm_from_vectors(*vectors, layer=layer)  # type: ignore[arg-type]


def test_load_vector_keeps_the_stored_type(vectors: tuple[Path, Path]) -> None:
    vector = load_vector(vectors[0])
    assert vector.dtype == torch.bfloat16 and vector.shape == (LAYERS, WIDTH)


def test_a_missing_file_says_where_the_files_come_from(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Mayner's data bundle"):
        load_vector(tmp_path / "pirate.pt")


class _Payload:
    """Something only a full unpickler would rebuild."""


def test_a_file_weights_only_cannot_read_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "pirate.pt"
    torch.save({"vector": _Payload()}, path)
    with pytest.raises(ValueError, match="not a plain tensor file"):
        load_vector(path)


@pytest.mark.parametrize(
    "content",
    [{"vectors": torch.zeros(LAYERS, WIDTH)}, {"vector": torch.zeros(WIDTH)}, [1, 2]],
    ids=["no vector", "one row", "a list"],
)
def test_a_file_without_a_vector_of_rows_is_refused(
    tmp_path: Path, content: object
) -> None:
    path = tmp_path / "pirate.pt"
    torch.save(content, path)
    with pytest.raises(ValueError, match="must hold a 'vector'"):
        load_vector(path)


def test_vectors_of_different_shapes_are_refused(tmp_path: Path) -> None:
    role = _vector_file(tmp_path / "pirate.pt", 1)
    torch.save({"vector": torch.zeros(LAYERS, WIDTH + 1)}, tmp_path / "default.pt")
    with pytest.raises(ValueError, match="they must match"):
        vector_norm_from_vectors(role, tmp_path / "default.pt")


def test_the_same_file_twice_gives_no_length_and_is_refused(
    vectors: tuple[Path, Path],
) -> None:
    with pytest.raises(ValueError, match="are they the same file"):
        vector_norm_from_vectors(vectors[0], vectors[0])


def test_a_set_sized_from_the_files_pushes_his_strength(
    vectors: tuple[Path, Path],
) -> None:
    norm = vector_norm_from_vectors(*vectors)
    rows = [
        (1, 0, 5, 0.25, 1.0, "pirate", 1.5),
        (2, 1, 9, -0.125, -0.5, "pirate", -2.0),
    ]
    table = pd.DataFrame(rows, columns=mayner.TOP_NEURONS_COLUMNS)
    pirate = mayner.top_neurons_set(table, "pirate", 2, norm, model_name="tiny")
    assert pirate.size == mayner.STRENGTH * norm


def test_docstring_examples_run() -> None:
    result = doctest.testmod(mayner_vectors)
    assert result.attempted > 0
    assert result.failed == 0
