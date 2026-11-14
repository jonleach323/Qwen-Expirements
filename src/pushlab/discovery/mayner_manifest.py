"""Mayner's bundle checked against his MANIFEST before mayner_import reads it.

His bundle (huggingface.co/datasets/wmayner/persona-neurons) lists each of
its files in a MANIFEST, one line per file: its path in the bundle, its size
in bytes and its SHA-256, separated by tabs, as his download_data.py reads
it (blank lines and lines that start with # list no file). `pushlab fetch
mayner` saves the MANIFEST in the bundle's folder and runs his script, which
checks each file it copies against it (decision 0020). A fetch that failed
can still leave a file that failed his check in place, though, and a file
can change or go after a fetch, so check_bundle checks the files the
importer reads against the MANIFEST again, at each import.
"""

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from pushlab.discovery.sources import MAYNER_FILES

# The file `pushlab fetch mayner` saves in the bundle's folder.
MANIFEST_FILE = "MANIFEST"
# How many of the files that are not as the MANIFEST lists them a refusal names.
SHOWN = 10
# A line of his MANIFEST: a path, its bytes and its SHA-256 in 64 hex digits.
MANIFEST_LINE = re.compile(r"([^\t]+)\t([0-9]+)\t([0-9a-f]{64})")
FETCH_AGAIN = (
    "run `uv run pushlab fetch mayner` again on the GPU pod, which downloads "
    "again each file that does not match (a fetch that failed can leave a file "
    "that failed his check in place)"
)


def manifest_rows(text: str) -> list[tuple[str, int, str]]:
    """The files a MANIFEST lists: each one's path in the bundle, bytes and SHA-256.

    text is the MANIFEST's text: one line per file, path TAB bytes TAB
    sha256, as his download_data.py reads it; blank lines and lines starting
    with # are skipped. A line in another shape, or whose path leaves the
    bundle, is refused with ValueError.

    Example:
        >>> line = "\\t".join(["split/top_neurons.parquet", "6553301", "5e" * 32])
        >>> [(path, size) for path, size, _ in manifest_rows(f"# made up\\n{line}\\n")]
        [('split/top_neurons.parquet', 6553301)]
    """
    rows = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.startswith("#"):
            continue
        match = MANIFEST_LINE.fullmatch(line)
        place = PurePosixPath(match[1]) if match else None
        if place is None or place.is_absolute() or ".." in place.parts:
            raise ValueError(
                f"line {number} is not a file's path in the bundle, its bytes and "
                f"its SHA-256, separated by tabs: {line!r}"
            )
        rows.append((match[1], int(match[2]), match[3]))
    return rows


def _fault(path: Path, size: int, digest: str) -> str:
    """How the file at path differs from the MANIFEST's size and SHA-256 for it;
    "" if it does not."""
    if not path.is_file():
        return "missing"
    if (found := path.stat().st_size) != size:
        return f"{found:,} bytes, not {size:,}"
    with path.open("rb") as file:  # hashed a block at a time, never whole
        if hashlib.file_digest(file, "sha256").hexdigest() != digest:
            return "another SHA-256"
    return ""


def check_bundle(bundle: Path, prefixes: Iterable[str] = MAYNER_FILES) -> None:
    """Refuse the bundle if a file its MANIFEST lists under prefixes is not as listed.

    bundle is the folder `pushlab fetch mayner` fills, with the MANIFEST it
    saves there. prefixes pick the files to check as his script's --only
    does, by the start of their paths: by default sources.MAYNER_FILES, the
    files the importer reads. A file the MANIFEST lists under them that is
    missing, or whose size or SHA-256 is not the MANIFEST's, is refused with
    ValueError, which names the first SHOWN of them and says to fetch them
    again. Without a MANIFEST, as when the files were copied in by hand,
    nothing is checked.

    Example:
        >>> check_bundle(Path("/no/such/bundle"))  # no MANIFEST, nothing checked
    """
    bundle = Path(bundle)
    manifest = bundle / MANIFEST_FILE
    if not manifest.is_file():
        return
    try:
        rows = manifest_rows(manifest.read_text(encoding="utf-8"))
    except ValueError as error:  # UnicodeDecodeError is a ValueError too
        raise ValueError(f"{manifest}: {error}; {FETCH_AGAIN}") from None
    picked = tuple(prefixes)
    wrong = [
        f"{path} ({fault})"
        for path, size, digest in rows
        if path.startswith(picked) and (fault := _fault(bundle / path, size, digest))
    ]
    if wrong:
        named = ", ".join(wrong[:SHOWN])
        if len(wrong) > SHOWN:
            named += f" and {len(wrong) - SHOWN:,} more"
        raise ValueError(
            f"{len(wrong):,} of the files of Mayner's bundle in {bundle} that the "
            f"importer reads are not as his MANIFEST lists them: {named}; "
            f"{FETCH_AGAIN}"
        )
