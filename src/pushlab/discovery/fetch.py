"""Download the two sources of persona candidates into the data folder.

`pushlab fetch library` runs fetch_library, here, and `pushlab fetch mayner`
runs fetch_mayner.fetch_mayner (decision 0020). Each source goes into its
own folder of the data folder (/workspace/data/library/qwen and
/workspace/data/mayner on the GPU pod), beside FETCHED.json, the record of
where its files came from and when. Files already there are skipped, so
running a fetch again downloads only what is missing.

- fetch_library downloads the Neuron Persona Library's three files for
  Qwen3-8B from neuronpersonas.com, about 26 MB. Each goes into a .part
  file first and must be the JSON asked for (fetch_http.check_download);
  only once all of them are whole and checked do they replace the files
  there, together, and the record follows at once. A fetch that fails
  changes nothing.
- fetch_mayner checks out Will Mayner's code from github.com and runs his
  own download script, which copies from huggingface.co the files of his
  bundle that the importers read, 279 files of about 105 MB (all_files: his
  whole bundle, 638 files of 1.02 GB). Its steps are in fetch_mayner.py.

Both use the network through fetch_http.py and the records' steps here.
The downloads are large, so they run on the pod, whose network must reach
neuronpersonas.com, github.com and huggingface.co, and the hosts under
hf.co, such as us.aws.cdn.hf.co, through which Hugging Face serves the
files (its CDN). The tests hand in a fake download, a fake run of his
script and a local git repository instead.
"""

import hashlib
import shlex
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pushlab.discovery.fetch_http import (
    Download,
    check_download,
    http_download,
    last_modified,
    part_of,
)
from pushlab.discovery.sources import (
    LIBRARY,
    LIBRARY_FILES,
    RECORD_FILE,
    Source,
    read_record,
    source_dir,
    write_record,
)

# The keys each of the library's files has, as a JSON object (decision
# 0020): library.json's table, the verified pool of combos.json and the
# sensitive neurons of sensitive.json. A download without them is refused.
LIBRARY_KEYS = {
    "library.json": ("fields", "rows"),
    "combos.json": ("pool_fields", "pool"),
    "sensitive.json": ("neurons",),
}

# The time now, with its time zone.
Clock = Callable[[], datetime]


def utc_stamp(moment: datetime) -> str:
    """A time as the fetch records write it: UTC, to the second.

    Example:
        >>> from datetime import timedelta, timezone
        >>> paris = timezone(timedelta(hours=1))
        >>> utc_stamp(datetime(2026, 11, 12, 21, 31, 5, tzinfo=paris))
        '2026-11-12T20:31:05Z'
    """
    if moment.utcoffset() is None:
        raise ValueError(
            f"{moment} has no time zone; give the time with one, such as "
            "datetime.now(UTC)"
        )
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_now() -> datetime:
    """The time now, in UTC: the fetches' clock unless they are given another.

    Example:
        >>> utc_now().tzinfo is UTC
        True
    """
    return datetime.now(UTC)


def sha256_of(path: Path) -> str:
    """The SHA-256 of a file's bytes, as hex.

    Example:
        >>> sha256_of(Path("/dev/null"))[:16]  # no bytes at all
        'e3b0c44298fc1c14'
    """
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def source_fields(source: Source) -> dict[str, str]:
    """What every fetch record says of its source.

    Example:
        >>> source_fields(LIBRARY)["url"]
        'https://neuronpersonas.com/data/qwen'
    """
    return {
        "source": source.name,
        "title": source.title,
        "url": source.url,
        "licence": source.licence,
        "credit": source.credit,
    }


def credit_line(source: Source) -> str:
    """The line a fetch ends with: its source's licence and credit.

    Example:
        >>> credit_line(LIBRARY).startswith("library: licence: ")
        True
    """
    return f"{source.name}: licence: {source.licence}. Credit: {source.credit}"


def fetch_again(source: str, root: Path | None = None, *options: str) -> str:
    """The command that runs a fetch again, as a failed one names it.

    root is the data folder the fetch was given (--data-dir), if any.

    Example:
        >>> fetch_again("library", None, "--refresh")
        'uv run pushlab fetch library --refresh'
        >>> fetch_again("mayner", Path("my data"), "--all")
        "uv run pushlab fetch mayner --all --data-dir 'my data'"
    """
    folder = [] if root is None else ["--data-dir", str(root)]
    return shlex.join(["uv", "run", "pushlab", "fetch", source, *options, *folder])


def _describe(
    path: Path,
    url: str,
    downloaded_at: str | None = None,
    modified: str | None = None,
) -> dict[str, Any]:
    """A library file's entry in the record."""
    return {
        "url": url,
        "bytes": path.stat().st_size,
        "sha256": sha256_of(path),
        "downloaded_at": downloaded_at,
        "last_modified": modified,
    }


def _matches(entry: object, found: Mapping[str, Any]) -> bool:
    """True if a record's entry still describes a file: same size and SHA-256."""
    return isinstance(entry, dict) and all(
        entry.get(key) == found[key] for key in ("bytes", "sha256")
    )


def _kept(path: Path, url: str, entry: object) -> dict[str, Any]:
    """The entry of a file left as it was: its old entry while that still
    describes it, else one with its download time null, unknown, as for a
    file copied in by hand."""
    found = _describe(path, url)
    return found | entry if _matches(entry, found) else found


def _old_files(folder: Path, root: Path | None) -> dict[str, Any]:
    """The file entries of the library's record in folder ({} without one)."""
    try:
        record = read_record(folder)
    except ValueError as error:
        raise ValueError(
            f"{error}. Delete it and run `{fetch_again(LIBRARY.name, root)}` "
            "again: the files already there are kept, their download times "
            "unknown"
        ) from None
    files = record.get("files")
    return files if isinstance(files, dict) else {}


def _warn_changed(path: Path, root: Path | None) -> None:
    """Warn that a file kept as it was no longer matches its record's entry."""
    print(
        f"pushlab fetch: warning: {path} is not the file FETCHED.json recorded: "
        "its size or SHA-256 differs, as for a file changed or replaced since "
        "its download, so its download time is now unknown. To download a "
        f"fresh copy, run `{fetch_again(LIBRARY.name, root, '--refresh')}`",
        file=sys.stderr,
    )


def download_parts(
    download: Download, folder: Path, names: Sequence[str], now: Clock
) -> dict[str, dict[str, Any]]:
    """Download each of the library's files named into its .part file in
    folder; return their record entries once all are whole and checked.

    Each must hold the JSON object that LIBRARY_KEYS describes for it
    (fetch_http.check_download). If a download fails, or brings anything
    else, every .part file is deleted, the files in folder are left as they
    were, and OSError says what went wrong. Prints each file's size.

    Example:
        >>> folder, names = Path("/workspace/data/library/qwen"), LIBRARY_FILES
        >>> download_parts(http_download, folder, names, utc_now)  # doctest: +SKIP
        library: library.json: downloaded 23,948,454 bytes from https://...
        ...
    """
    folder.mkdir(parents=True, exist_ok=True)
    entries: dict[str, dict[str, Any]] = {}
    try:
        for name in names:
            url, part = f"{LIBRARY.url}/{name}", part_of(folder / name)
            try:
                headers = dict(download(url, part) or {})
            except Exception as error:
                raise OSError(f"could not download {url}: {error}") from error
            check_download(part, url, LIBRARY_KEYS[name], headers)
            stamp = utc_stamp(now())
            entries[name] = _describe(part, url, stamp, last_modified(headers))
            size = entries[name]["bytes"]
            print(f"library: {name}: downloaded {size:,} bytes from {url}")
    except BaseException:  # Ctrl-C too
        for name in names:
            part_of(folder / name).unlink(missing_ok=True)
        raise
    return entries


def put_in_place(
    folder: Path,
    new: Mapping[str, dict[str, Any]],
    kept: Mapping[str, dict[str, Any]],
    old: Mapping[str, Any],
    now: Clock,
) -> dict[str, Any]:
    """Rename each new file's .part over its file in folder, then write the
    library's record there and return it.

    new holds the entries of the files downloaded (download_parts), kept
    those of the files left as they were, and old the entries of the record
    before. The record is written right after the renames, which are quick;
    should they stop partway, it still describes the files there, the ones
    not replaced by their old entries, and the .part files left are deleted.

    Example:
        >>> folder = Path("/workspace/data/library/qwen")
        >>> put_in_place(folder, new, {}, {}, utc_now)  # doctest: +SKIP
        {'source': 'library', ..., 'files': {'library.json': {...}, ...}}
    """
    files = dict(kept)
    try:
        for name, entry in new.items():
            part_of(folder / name).replace(folder / name)
            files[name] = entry
    finally:
        for name in new:
            part_of(folder / name).unlink(missing_ok=True)
            path = folder / name
            if name not in files and path.is_file():
                files[name] = _kept(path, f"{LIBRARY.url}/{name}", old.get(name))
        record = source_fields(LIBRARY) | {
            "updated_at": utc_stamp(now()),
            "files": {name: files[name] for name in LIBRARY_FILES if name in files},
        }
        write_record(folder, record)
    return record


def fetch_library(
    root: Path | None = None,
    *,
    refresh: bool = False,
    download: Download | None = None,
    now: Clock | None = None,
) -> dict[str, Any]:
    """Download the library's files for Qwen3-8B; write and return its record.

    Each of LIBRARY_FILES comes from LIBRARY.url/<name> into the library's
    folder of root (the data folder by default). A file already there is
    skipped unless refresh, keeping its record entry while its size and
    SHA-256 still match it; one without an entry gets one with downloaded_at
    null, and so does one changed since, with a warning that suggests
    --refresh. The others are downloaded into .part files and checked
    (download_parts); only once all are whole do they replace the files
    there, together, and the record is written at once (put_in_place). If
    any fails, nothing is changed, and OSError says what went wrong and the
    command to run again.

    The record, FETCHED.json, has the library's name, licence and credit,
    updated_at, and for each file its URL, bytes, SHA-256, downloaded_at
    (UTC) and the server's Last-Modified. Prints which files it downloaded
    and which it skipped. download(url, dest) and now() stand in for the
    network and the clock in tests.

    Example:
        >>> record = fetch_library(Path("/workspace/data"))  # doctest: +SKIP
        library: library.json: downloaded 23,948,454 bytes from https://...
        ...
    """
    download = download or http_download
    now = now or utc_now
    folder = source_dir(LIBRARY, root)
    old = _old_files(folder, root)
    kept: dict[str, dict[str, Any]] = {}
    for name in LIBRARY_FILES:
        path = folder / name
        if refresh or not path.is_file():
            continue
        kept[name] = _kept(path, f"{LIBRARY.url}/{name}", old.get(name))
        print(f"library: {name}: already in {folder}, skipped")
        if name in old and not _matches(old[name], kept[name]):
            _warn_changed(path, root)
    wanted = [name for name in LIBRARY_FILES if name not in kept]
    try:
        new = download_parts(download, folder, wanted, now)
    except OSError as error:
        again = fetch_again(LIBRARY.name, root, *(["--refresh"] if refresh else []))
        raise OSError(
            f"{error}. Nothing was changed: the files in {folder} and their "
            "record are as they were. Check that this machine reaches "
            f"neuronpersonas.com and that {LIBRARY.url} still serves the "
            f"files, then run `{again}` again"
        ) from error
    record = put_in_place(folder, new, kept, old, now)
    print(f"library: recorded in {folder / RECORD_FILE}")
    print(credit_line(LIBRARY))
    return record
