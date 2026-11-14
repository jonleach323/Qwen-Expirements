"""Candidates from Mayner's persona-neurons data (decision 0020), imported from
the mayner_data fixture: made-up data in his formats (tests/unit/conftest.py),
whose docstring gives the numbers worked out by hand here."""

import csv
import doctest
import hashlib
import json
import shutil
from collections.abc import Callable
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pushlab.discovery import mayner_import
from pushlab.discovery.candidates import read_candidates, write_candidates
from pushlab.discovery.mayner_import import (
    concentration_shares,
    import_mayner,
    origin,
    reply_scores,
    scored_in_role,
    vector_norm,
)
from pushlab.discovery.sources import MAYNER
from pushlab.models.discovery import ModelInfo

QWEN3_8B = ModelInfo(layers=36, hidden_size=4096, mlp_size=12288)
ROLES = ("coral_reef", "pirate", "sommelier")
KEYS = [
    f"mayner/{role}-top{k}"
    for role, k in product(("coral-reef", "pirate", "sommelier"), (10, 50))
]
# The fixture's |v_r|: layer 18's differences in the vector files, 25 and 10
# long, and the split's, which its shares give.
VECTOR_NORMS = {"coral_reef": 25.0, "pirate": 10.0}
SPLIT_NORMS = {"coral_reef": 25.05, "pirate": 10.01, "sommelier": 40.0}
# His judge's scores of the top neurons, from the fixture's docstring:
# (replies scored, the share of them scored 2 or 3).
SCORES = {
    ("pirate", 10): (9, 6 / 9),
    ("pirate", 50): (10, 9 / 10),
    ("sommelier", 10): (10, 1 / 10),
    ("sommelier", 50): (10, 7 / 10),
}
TOP_URL = f"{MAYNER.url}/resolve/main/split/top_neurons.parquet"
TOP_SOURCE = "wmayner/persona-neurons split/top_neurons.parquet"
DOWNLOADED_AT = "2026-11-12T20:31:05Z"


def _bundle(data: Path) -> Path:
    return data / "mayner" / "bundle"


def _write_manifest(bundle: Path) -> None:
    """Write bundle/MANIFEST as his bundle's: a line per file, path TAB bytes
    TAB sha256, each as the file is now, and a file of his bundle that the
    importer never reads, which is not there."""
    lines = ["# made up for pushlab's tests"]
    for path in sorted(p for p in bundle.rglob("*") if p.is_file()):
        data = path.read_bytes()
        name = path.relative_to(bundle).as_posix()
        lines.append(f"{name}\t{len(data)}\t{hashlib.sha256(data).hexdigest()}")
    lines.append(f"split/arrays.npz\t1024\t{'0' * 64}")
    text = "\n".join(lines) + "\n"
    (bundle / "MANIFEST").write_text(text, encoding="utf-8")


def _top(data: Path, role: str, k: int) -> pd.DataFrame:
    """The role's rows of rank 1 to k in the fixture's split/top_neurons.parquet."""
    table = pd.read_parquet(_bundle(data) / "split" / "top_neurons.parquet")
    return table[table["role"] == role].sort_values("rank").head(k)


def test_each_role_gives_its_top_10_and_50_in_name_order(mayner_data: Path) -> None:
    found = import_mayner(mayner_data / "mayner")
    assert [candidate.key for candidate in found] == KEYS
    for candidate, (role, k) in zip(found, product(ROLES, (10, 50)), strict=True):
        rows = _top(mayner_data, role, k)[["layer", "neuron", "delta"]]
        unit = candidate.unit
        assert candidate.persona == role
        assert str(unit) == f"qwen3-8b:set:{role.replace('_', '-')}-top{k}"
        assert unit.entries == tuple(rows.itertuples(index=False, name=None))
        source = f"{TOP_SOURCE}: role {role}, ranks 1 to {k}"
        assert candidate.source_ref == unit.source == source
        assert candidate.source_file == "mayner/bundle/split/top_neurons.parquet"
        assert candidate.source_url == TOP_URL
        assert candidate.downloaded_at == DOWNLOADED_AT
        assert candidate.source == "mayner"
        assert (candidate.direction, candidate.strength) == (1, 1.0)
        assert (candidate.sensitive, candidate.category) == (False, "role")
        assert candidate.tags == ()
        assert candidate.coherence is candidate.weirdness is None
        assert candidate.verified_pool is None


def test_sizes_are_4_times_vr_from_the_vectors_or_else_the_shares(
    mayner_data: Path,
) -> None:
    for candidate in import_mayner(mayner_data / "mayner"):
        role = candidate.persona
        if role in VECTOR_NORMS:  # exactly his strength
            assert candidate.unit.size == 4 * VECTOR_NORMS[role]
            assert candidate.size_from == "vectors/without"
        else:
            norm = SPLIT_NORMS[role]
            assert candidate.unit.size == pytest.approx(4 * norm, rel=1e-6)
            assert candidate.size_from == "split shares"
    # Without the default's vector, every role's |v_r| comes from the shares.
    (_bundle(mayner_data) / "vectors" / "without" / "default.pt").unlink()
    for candidate in import_mayner(mayner_data / "mayner"):
        norm = SPLIT_NORMS[candidate.persona]
        assert candidate.unit.size == pytest.approx(4 * norm, rel=1e-6)
        assert candidate.size_from == "split shares"


def test_mayners_own_scores_are_kept_for_reference(mayner_data: Path) -> None:
    for candidate in import_mayner(mayner_data / "mayner"):
        role, k = candidate.persona, len(candidate.unit.entries)
        scored, in_role = SCORES.get((role, k), (None, None))
        assert (candidate.mayner_scored, candidate.mayner_in_role) == (scored, in_role)
        # His step 5's share of |v_r| from the top k (split.top_curve).
        contributions = _top(mayner_data, role, k)["contribution"]
        added = np.cumsum(contributions.to_numpy(np.float64))[-1]
        share = 100 * added / SPLIT_NORMS[role]
        assert candidate.mayner_share == pytest.approx(share, rel=1e-12)


def test_the_sizes_asked_for_give_the_sets_in_that_order(mayner_data: Path) -> None:
    found = import_mayner(mayner_data / "mayner", sizes=(10,))
    assert [candidate.key for candidate in found] == KEYS[::2]
    found = import_mayner(mayner_data / "mayner", sizes=(50, 1))
    assert [candidate.unit.name for candidate in found[:2]] == [
        "coral-reef-top50",
        "coral-reef-top1",
    ]
    # He judged no set of 1 in the fixture, but step 5 measured its share.
    top_1 = found[1]
    assert top_1.mayner_scored is top_1.mayner_in_role is None
    first = float(_top(mayner_data, "coral_reef", 1)["contribution"].iloc[0])
    share = 100 * first / SPLIT_NORMS["coral_reef"]
    assert top_1.mayner_share == pytest.approx(share, rel=1e-12)


def test_every_unit_is_in_qwen3_8b(mayner_data: Path) -> None:
    found = import_mayner(mayner_data / "mayner")
    for candidate in found:
        candidate.unit.validate(QWEN3_8B)
    # pirate's rank 1 is the last neuron of layer 18, both counted from 0.
    assert found[2].unit.entries[0][:2] == (18, 12287)


def test_without_the_optional_files_his_scores_are_none(mayner_data: Path) -> None:
    folder = mayner_data / "mayner"
    (folder / "bundle" / "role_steering_sizes" / "judged_replies.parquet").unlink()
    (folder / "bundle" / "split" / "concentration_neurons.parquet").unlink()
    (folder / "FETCHED.json").unlink()
    shutil.rmtree(folder / "bundle" / "vectors")
    found = import_mayner(folder)
    assert [candidate.key for candidate in found] == KEYS
    for candidate in found:
        assert candidate.mayner_scored is candidate.mayner_in_role is None
        assert candidate.mayner_share is None
        assert candidate.size_from == "split shares"
        assert (candidate.source_url, candidate.downloaded_at) == (TOP_URL, "")


def test_a_missing_top_neurons_file_is_refused(
    mayner_data: Path, tmp_path: Path
) -> None:
    (_bundle(mayner_data) / "split" / "top_neurons.parquet").unlink()
    refusal = "there is no split/top_neurons.parquet in .*`pushlab fetch mayner`"
    with pytest.raises(FileNotFoundError, match=refusal):
        import_mayner(mayner_data / "mayner")
    with pytest.raises(FileNotFoundError, match=refusal):
        import_mayner(tmp_path / "nothing")


def test_files_as_his_manifest_lists_them_import_as_without_it(
    mayner_data: Path,
) -> None:
    folder = mayner_data / "mayner"
    without = import_mayner(folder)
    _write_manifest(folder / "bundle")
    assert import_mayner(folder) == without


@pytest.mark.parametrize(
    ("path", "change", "fault"),
    [
        # Changed in place, its size kept.
        (
            "vectors/without/pirate.pt",
            lambda data: data[:-1] + bytes([data[-1] ^ 1]),
            "another SHA-256",
        ),
        # Cut short, as a download that stopped would leave it.
        (
            "split/concentration_neurons.parquet",
            lambda data: data[:100],
            "100 bytes, not {size:,}",
        ),
        # Gone: one the importer can do without, but the MANIFEST lists.
        ("role_steering_sizes/judged_replies.parquet", None, "missing"),
    ],
)
def test_a_file_not_as_his_manifest_lists_it_is_refused(
    mayner_data: Path, path: str, change: Callable[[bytes], bytes] | None, fault: str
) -> None:
    bundle = _bundle(mayner_data)
    _write_manifest(bundle)
    size = (bundle / path).stat().st_size
    if change is None:
        (bundle / path).unlink()
    else:
        (bundle / path).write_bytes(change((bundle / path).read_bytes()))
    with pytest.raises(ValueError) as error:
        import_mayner(mayner_data / "mayner")
    assert str(error.value) == (
        f"1 of the files of Mayner's bundle in {bundle} that the importer reads "
        f"are not as his MANIFEST lists them: {path} ({fault.format(size=size)}); "
        "run `uv run pushlab fetch mayner` again on the GPU pod, which downloads "
        "again each file that does not match (a fetch that failed can leave a "
        "file that failed his check in place)"
    )


def test_the_replies_text_is_never_read(
    mayner_data: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real, read = pd.read_parquet, []

    def spy(path: Path, *args: object, **kwargs: object) -> pd.DataFrame:
        read.append((Path(path).name, kwargs.get("columns")))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_parquet", spy)
    import_mayner(mayner_data / "mayner")
    assert sorted(name for name, _ in read) == [
        "concentration_neurons.parquet",
        "judged_replies.parquet",
        "top_neurons.parquet",
    ]
    assert all(columns and "response" not in columns for _, columns in read)


def test_roles_come_in_name_order_whatever_the_tables_order(
    mayner_data: Path,
) -> None:
    path = _bundle(mayner_data) / "split" / "top_neurons.parquet"
    in_order = import_mayner(mayner_data / "mayner")
    pd.read_parquet(path).iloc[::-1].to_parquet(path, index=False)
    assert import_mayner(mayner_data / "mayner") == in_order


def test_each_reader_takes_only_its_rows(mayner_data: Path) -> None:
    bundle = _bundle(mayner_data)
    replies = bundle / "role_steering_sizes" / "judged_replies.parquet"
    assert reply_scores(replies, (10, 50)) == SCORES
    assert reply_scores(replies, (20,)) == {("pirate", 20): (10, 1.0)}
    path = bundle / "split" / "concentration_neurons.parquet"
    shares = concentration_shares(path, (10, 50))
    assert sorted(shares) == list(product(ROLES, (10, 50)))
    # Rows of his other basis (his step 5's concentration.parquet has both),
    # and a share that is not a number, are left out.
    table = pd.read_parquet(path)
    table.loc[(table["role"] == "pirate") & (table["k"] == 50), "share"] = np.nan
    components = table.assign(basis="principal components", share=99.0)
    pd.concat([table, components]).to_parquet(path, index=False)
    del shares[("pirate", 50)]
    assert concentration_shares(path, (10, 50)) == shares
    pirate = _top(mayner_data, "pirate", 60)
    assert vector_norm(bundle, "pirate", pirate) == (10.0, "vectors/without")
    sommelier = _top(mayner_data, "sommelier", 60)
    norm, where = vector_norm(bundle, "sommelier", sommelier)
    assert (norm, where) == (pytest.approx(40.0, rel=1e-6), "split shares")


def test_the_record_says_where_the_table_came_from_and_when(mayner_data: Path) -> None:
    folder = mayner_data / "mayner"
    assert origin(folder) == (TOP_URL, DOWNLOADED_AT)
    path = folder / "FETCHED.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record |= {"files_url": "https://example.org/bundle/", "downloaded_at": None}
    path.write_text(json.dumps(record), encoding="utf-8")
    url = "https://example.org/bundle/split/top_neurons.parquet"
    assert origin(folder) == (url, "")
    assert {(c.source_url, c.downloaded_at) for c in import_mayner(folder)} == {
        (url, "")
    }
    path.write_text(json.dumps(record | {"downloaded_at": 20261112}), encoding="utf-8")
    with pytest.raises(ValueError, match="downloaded_at must be text, not 20261112"):
        origin(folder)


@pytest.mark.parametrize("sizes", [(), (0,), (10, 10), (True,), (2.5,), "10", None])
def test_sizes_must_be_different_whole_numbers_from_1(
    mayner_data: Path, sizes: object
) -> None:
    with pytest.raises(ValueError, match="sizes must be different whole numbers"):
        import_mayner(mayner_data / "mayner", sizes=sizes)  # type: ignore[arg-type]


def test_a_size_above_a_roles_neurons_is_refused(mayner_data: Path) -> None:
    with pytest.raises(ValueError, match="k must be a whole number from 1 to 60"):
        import_mayner(mayner_data / "mayner", sizes=(61,))


@pytest.mark.parametrize("model", ["qwen3.5-4b", "tiny", "qwen3-32b", None])
def test_only_a_model_of_qwen3_8b_has_his_neurons(
    mayner_data: Path, model: object
) -> None:
    refusal = "lists the neurons of Qwen/Qwen3-8B, so its candidates are for qwen3-8b"
    with pytest.raises(ValueError, match=refusal):
        import_mayner(mayner_data / "mayner", model=model)  # type: ignore[arg-type]


def test_a_table_not_in_his_format_is_refused(mayner_data: Path) -> None:
    folder = mayner_data / "mayner"
    top = folder / "bundle" / "split" / "top_neurons.parquet"
    table = pd.read_parquet(top)
    broken = {
        "lacks the column delta, so it is not": table.drop(columns="delta"),
        "has no rows; delete it and download it again": table.iloc[:0],
        "has a row with no role": table.assign(
            role=table["role"].where(table.index > 0)
        ),
    }
    for refusal, written in broken.items():
        written.to_parquet(top, index=False)
        with pytest.raises(ValueError, match=refusal):
            import_mayner(folder)
    top.write_text("not a table", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot be read as a parquet table"):
        import_mayner(folder)


def test_a_score_his_judge_cannot_give_is_refused(mayner_data: Path) -> None:
    for score in (4, -1, "3", True):
        with pytest.raises(ValueError, match="judge is a number from 0 to 3, not"):
            scored_in_role([3, score])
    path = _bundle(mayner_data) / "role_steering_sizes" / "judged_replies.parquet"
    replies = pd.read_parquet(path)
    replies.loc[0, "score"] = 7.0  # pirate's first top-10 reply
    replies.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="role pirate, k 10: a score of Mayner's"):
        import_mayner(mayner_data / "mayner")


def test_the_candidates_survive_a_candidates_file_credited(
    mayner_data: Path, tmp_path: Path
) -> None:
    found = import_mayner(mayner_data / "mayner")
    path = write_candidates(tmp_path / "candidates.csv", found)
    assert read_candidates(path) == found
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    # His credit, and how pushlab adapted his data, as CC BY 4.0 asks.
    assert {(row["licence"], row["credit"]) for row in rows} == {
        ("CC BY 4.0", MAYNER.list_credit)
    }


def test_docstring_examples_run() -> None:
    result = doctest.testmod(mayner_import)
    assert result.attempted > 0
    assert result.failed == 0
