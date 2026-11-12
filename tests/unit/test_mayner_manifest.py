"""Mayner's bundle checked against his MANIFEST (pushlab.discovery.mayner_manifest,
decision 0020), on small made-up bundles written here. import_mayner's own
checks of the mayner_data fixture's bundle are in test_mayner_import.py."""

import doctest
import hashlib
from pathlib import Path

import pytest

from pushlab.discovery import mayner_manifest
from pushlab.discovery.mayner_manifest import (
    FETCH_AGAIN,
    SHOWN,
    check_bundle,
    manifest_rows,
)

DIGEST = "5e" * 32


def _bundle(tmp_path: Path, files: dict[str, bytes], *extra: str) -> Path:
    """A bundle folder holding files, at their paths, and a MANIFEST that
    lists each as it is, then the lines extra."""
    bundle = tmp_path / "bundle"
    lines = ["# made up for pushlab's tests", ""]
    for path, data in files.items():
        (bundle / path).parent.mkdir(parents=True, exist_ok=True)
        (bundle / path).write_bytes(data)
        lines.append(f"{path}\t{len(data)}\t{hashlib.sha256(data).hexdigest()}")
    text = "\n".join([*lines, *extra]) + "\n"
    (bundle / "MANIFEST").write_text(text, encoding="utf-8")
    return bundle


def _refusal(bundle: Path, count: int, named: str) -> str:
    return (
        f"{count} of the files of Mayner's bundle in {bundle} that the importer "
        f"reads are not as his MANIFEST lists them: {named}; {FETCH_AGAIN}"
    )


def test_a_manifest_lists_each_files_path_bytes_and_sha256() -> None:
    text = (
        "# his bundle\n\n"
        f"split/top_neurons.parquet\t6553301\t{DIGEST}\r\n"
        f"vectors/without/pirate.pt\t296482\t{'0' * 64}\n"
    )
    assert manifest_rows(text) == [
        ("split/top_neurons.parquet", 6553301, DIGEST),
        ("vectors/without/pirate.pt", 296482, "0" * 64),
    ]
    assert manifest_rows("") == []


@pytest.mark.parametrize(
    "line",
    [
        "split/top_neurons.parquet 6553301 " + DIGEST,  # spaces, not tabs
        "split/top_neurons.parquet\t6553301",
        f"split/top_neurons.parquet\t6553301\t{DIGEST}\textra",
        f"split/top_neurons.parquet\tsix\t{DIGEST}",
        f"split/top_neurons.parquet\t-1\t{DIGEST}",
        "split/top_neurons.parquet\t6553301\tnot-a-sha256",
        f"split/top_neurons.parquet\t6553301\t{DIGEST.upper()}",
        f"\t6553301\t{DIGEST}",
        f"/etc/passwd\t6553301\t{DIGEST}",
        f"vectors/without/../../../secrets\t6553301\t{DIGEST}",
    ],
)
def test_a_line_in_another_shape_is_refused(line: str) -> None:
    with pytest.raises(ValueError) as error:
        manifest_rows(f"# his bundle\n{line}\n")
    assert str(error.value) == (
        "line 2 is not a file's path in the bundle, its bytes and its SHA-256, "
        f"separated by tabs: {line!r}"
    )


def test_without_a_manifest_nothing_is_checked(tmp_path: Path) -> None:
    (tmp_path / "split").mkdir()
    (tmp_path / "split" / "top_neurons.parquet").write_text("x", encoding="utf-8")
    assert check_bundle(tmp_path) is None


def test_only_the_files_under_the_prefixes_are_checked(tmp_path: Path) -> None:
    files = {
        "split/top_neurons.parquet": b"top",
        "vectors/without/pirate.pt": b"pirate",
        "split/arrays.npz": b"arrays",
    }
    # The MANIFEST also lists a file of his bundle the importer never reads.
    bundle = _bundle(tmp_path, files, f"necessity/paired.parquet\t7491\t{DIGEST}")
    (bundle / "split" / "arrays.npz").write_bytes(b"changed")
    check_bundle(bundle)  # the importer's files are as listed
    (bundle / "vectors" / "without" / "pirate.pt").write_bytes(b"Pirate")
    with pytest.raises(ValueError) as error:
        check_bundle(bundle)
    assert str(error.value) == _refusal(
        bundle, 1, "vectors/without/pirate.pt (another SHA-256)"
    )
    # With prefixes of its own, it checks what they pick.
    check_bundle(bundle, ["split/top_neurons.parquet"])
    with pytest.raises(ValueError, match=r"split/arrays\.npz \(7 bytes, not 6\)"):
        check_bundle(bundle, ["split/"])


def test_a_refusal_names_the_first_ten_wrong_files(tmp_path: Path) -> None:
    roles = [f"role{i:02d}" for i in range(13)]
    bundle = _bundle(tmp_path, {f"vectors/without/{r}.pt": b"v" for r in roles})
    for role in roles[1:]:
        (bundle / "vectors" / "without" / f"{role}.pt").unlink()
    with pytest.raises(ValueError) as error:
        check_bundle(bundle)
    named = ", ".join(f"vectors/without/{r}.pt (missing)" for r in roles[1:11])
    assert str(error.value) == _refusal(bundle, 12, f"{named} and 2 more")
    assert SHOWN == 10


@pytest.mark.parametrize(
    "manifest", [b"split/top_neurons.parquet\t3\n", b"\xff\xfe not text\n"]
)
def test_a_manifest_that_cannot_be_read_is_refused(
    tmp_path: Path, manifest: bytes
) -> None:
    (tmp_path / "MANIFEST").write_bytes(manifest)
    with pytest.raises(ValueError) as error:
        check_bundle(tmp_path)
    assert str(error.value).startswith(f"{tmp_path / 'MANIFEST'}: ")
    assert str(error.value).endswith(f"; {FETCH_AGAIN}")


def test_the_next_step_is_to_fetch_his_files_again_on_the_pod() -> None:
    assert FETCH_AGAIN.startswith(
        "run `uv run pushlab fetch mayner` again on the GPU pod"
    )
    assert "a fetch that failed can leave a file that failed his check" in FETCH_AGAIN


def test_docstring_examples_run() -> None:
    result = doctest.testmod(mayner_manifest)
    assert result.attempted > 0
    assert result.failed == 0
