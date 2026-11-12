"""Download the two sources of persona candidates into the data folder.

`pushlab fetch library` and `pushlab fetch mayner` run fetch_library and
fetch_mayner (decision 0020). Each source goes into its own folder of the
data folder (/workspace/data/library/qwen and /workspace/data/mayner on the
GPU pod), beside FETCHED.json, the record of where its files came from and
when. Files already there are skipped, so running a fetch again downloads
only what is missing.

- fetch_library downloads the Neuron Persona Library's three files for
  Qwen3-8B, about 26 MB. Each goes into a .part file first and is renamed
  once whole, so a file in the folder is never half a download.
- fetch_mayner checks out Will Mayner's code at the pinned commit and runs
  his own download script, which copies the files of his bundle that the
  importers read, about 110 MB (all_files: the whole 1 GB), into bundle/,
  checking each against his MANIFEST and skipping those already there. Its
  steps for his code are in fetch_mayner.py.

The downloads are large, so they run on the pod. The tests hand in a fake
download, a fake run of his script and a local git repository instead.
"""

import hashlib
import os
import shlex
import shutil
import urllib.request
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pushlab.discovery.fetch_mayner import (
    Git,
    Run,
    checkout_code,
    mayner_command,
    run_command,
)
from pushlab.discovery.sources import (
    LIBRARY,
    LIBRARY_FILES,
    MAYNER,
    MAYNER_BUNDLE,
    MAYNER_CODE,
    MAYNER_COMMIT,
    MAYNER_FILES,
    MAYNER_REPOSITORY,
    RECORD_FILE,
    Source,
    read_record,
    source_dir,
    write_record,
)

# Sent with every download: a plain name, nothing about who runs it.
USER_AGENT = "pushlab"
# Seconds a download waits for the server before it gives up.
TIMEOUT = 60
# Bytes copied at a time.
CHUNK = 1 << 20

# Downloads a URL into a file; returns the response's headers.
Download = Callable[[str, Path], Mapping[str, str]]
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


def _utc_now() -> datetime:
    """The time now, in UTC."""
    return datetime.now(UTC)


def sha256_of(path: Path) -> str:
    """The SHA-256 of a file's bytes, as hex.

    Example:
        >>> sha256_of(Path(os.devnull))[:16]  # no bytes at all
        'e3b0c44298fc1c14'
    """
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def _header(headers: Mapping[str, str], name: str) -> str | None:
    """The value of a header whose name matches name in any case, or None."""
    for key, value in headers.items():
        if key.lower() == name.lower():
            return value
    return None


def last_modified(headers: Mapping[str, str]) -> str | None:
    """The Last-Modified header among a response's headers, in any case.

    Example:
        >>> last_modified({"last-modified": "Wed, 11 Nov 2026 09:00:00 GMT"})
        'Wed, 11 Nov 2026 09:00:00 GMT'
        >>> last_modified({}) is None
        True
    """
    return _header(headers, "Last-Modified")


def http_download(url: str, dest: Path) -> dict[str, str]:
    """Download url into dest with urllib; return the response's headers.

    The request names itself only "pushlab", and each wait for the server
    lasts at most TIMEOUT seconds. A response that ends before its
    Content-Length raises OSError.

    Example:
        >>> url = LIBRARY.url + "/sensitive.json"
        >>> http_download(url, Path("sensitive.json"))  # doctest: +SKIP
        {'Content-Type': 'application/json', 'Last-Modified': '...', ...}
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with (
        urllib.request.urlopen(request, timeout=TIMEOUT) as response,
        Path(dest).open("wb") as file,
    ):
        shutil.copyfileobj(response, file, CHUNK)
        headers = dict(response.headers.items())
    expected = (_header(headers, "Content-Length") or "").strip()
    got = Path(dest).stat().st_size
    if expected.isdigit() and int(expected) != got:
        raise OSError(
            f"{url} stopped after {got:,} of its {int(expected):,} bytes; run "
            "the fetch again"
        )
    return headers


def download_whole(download: Download, url: str, dest: Path) -> dict[str, str]:
    """Download url into dest with download, through a .part file beside it.

    The .part file becomes dest only once download has returned, so a file
    at dest is always whole. If download fails, the .part file is deleted,
    dest stays as it was, and OSError names the URL. Returns the headers.

    Example:
        >>> url = LIBRARY.url + "/combos.json"
        >>> download_whole(http_download, url, Path("combos.json"))  # doctest: +SKIP
        {'Content-Type': 'application/json', ...}
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    try:
        headers = download(url, partial)
    except Exception as error:
        partial.unlink(missing_ok=True)
        raise OSError(
            f"could not download {url}: {error}. Check the connection, and "
            "that the URL still works, then run the fetch again; the files "
            "already downloaded are kept"
        ) from error
    except BaseException:  # such as Ctrl-C
        partial.unlink(missing_ok=True)
        raise
    partial.replace(dest)
    return dict(headers or {})


def _source_fields(source: Source) -> dict[str, str]:
    """What every record says of its source."""
    return {
        "source": source.name,
        "title": source.title,
        "url": source.url,
        "licence": source.licence,
        "credit": source.credit,
    }


def _credit_line(source: Source) -> str:
    return f"{source.name}: licence: {source.licence}. Credit: {source.credit}"


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


def _kept(path: Path, url: str, entry: object) -> dict[str, Any]:
    """The entry of a file left as it was: its old entry while that still
    describes it (same size and SHA-256), else one with its download time
    null, unknown, as for a file copied in by hand."""
    found = _describe(path, url)
    if isinstance(entry, dict) and all(
        entry.get(key) == found[key] for key in ("bytes", "sha256")
    ):
        return found | entry
    return found


def _old_files(folder: Path) -> dict[str, Any]:
    """The file entries of the library's record in folder ({} without one)."""
    try:
        record = read_record(folder)
    except ValueError as error:
        raise ValueError(
            f"{error}. Delete it and run `pushlab fetch library` again: the "
            "files already there are kept, their download times unknown"
        ) from None
    files = record.get("files")
    return files if isinstance(files, dict) else {}


def fetch_library(
    root: Path | None = None,
    *,
    refresh: bool = False,
    download: Download | None = None,
    now: Clock | None = None,
) -> dict[str, Any]:
    """Download the library's files for Qwen3-8B; write and return its record.

    Each of LIBRARY_FILES comes from LIBRARY.url/<name> into the library's
    folder of root (the data folder by default). The record, FETCHED.json,
    has the library's name, licence and credit, updated_at, and for each
    file its URL, bytes, SHA-256, downloaded_at (UTC) and the server's
    Last-Modified. A file already there is skipped unless refresh, keeping
    its entry; one without an entry yet, or changed since, gets one with
    downloaded_at null. If a download fails, OSError names its URL, and the
    record still lists the files that are there. Prints which files it
    downloaded and which it skipped. download(url, dest) and now() stand in
    for the network and the clock in tests.

    Example:
        >>> record = fetch_library(Path("/workspace/data"))  # doctest: +SKIP
        library: library.json: downloaded 23,912,064 bytes from https://...
        ...
    """
    download = download or http_download
    now = now or _utc_now
    folder = source_dir(LIBRARY, root)
    old = _old_files(folder)
    files: dict[str, dict[str, Any]] = {}
    try:
        for name in LIBRARY_FILES:
            path, url = folder / name, f"{LIBRARY.url}/{name}"
            if path.is_file() and not refresh:
                files[name] = _kept(path, url, old.get(name))
                print(f"library: {name}: already in {folder}, skipped")
                continue
            headers = download_whole(download, url, path)
            stamp = utc_stamp(now())
            files[name] = _describe(path, url, stamp, last_modified(headers))
            size = files[name]["bytes"]
            print(f"library: {name}: downloaded {size:,} bytes from {url}")
    finally:
        # Whatever stopped the loop, the record lists every file that is there.
        for name in LIBRARY_FILES:
            path = folder / name
            if name not in files and path.is_file():
                files[name] = _kept(path, f"{LIBRARY.url}/{name}", old.get(name))
        record = _source_fields(LIBRARY) | {
            "updated_at": utc_stamp(now()),
            "files": {name: files[name] for name in LIBRARY_FILES if name in files},
        }
        written = write_record(folder, record)
    print(f"library: recorded in {written}")
    print(_credit_line(LIBRARY))
    return record


def fetch_mayner(
    root: Path | None = None,
    *,
    all_files: bool = False,
    run: Run | None = None,
    download: Download | None = None,
    now: Clock | None = None,
    git: Git | None = None,
) -> dict[str, Any]:
    """Download Mayner's bundle with his own script; write and return the record.

    His code goes into code/ of Mayner's folder of root (the data folder by
    default), at MAYNER_COMMIT; a checkout with a changed file of his is
    refused (checkout_code). His bundle's MANIFEST, a few kilobytes, is
    downloaded into bundle/ every time, for its SHA-256. Then his
    download_data.py copies into bundle/ the files the importers read
    (MAYNER_FILES), or all of them with all_files, checking each against the
    MANIFEST and skipping those already there. If it fails, RuntimeError
    says to run the fetch again.

    The record, FETCHED.json, has his data's name, licence and credit,
    files_url, downloaded_at (UTC: when his script finished, every file
    then matching the MANIFEST), the --only prefixes (null for all), his
    repository, the commit and the command that ran, and the MANIFEST's
    SHA-256. The old record is deleted before his script runs, so a record
    always describes a finished download. run, download, now and git stand
    in for his script, the network, the clock and git in tests.

    Example:
        >>> record = fetch_mayner(Path("/workspace/data"))  # doctest: +SKIP
        mayner: his code is in /workspace/data/mayner/code, at commit 64c9d36...
        ...
    """
    run = run or run_command
    download = download or http_download
    now = now or _utc_now
    folder = source_dir(MAYNER, root).absolute()
    code, bundle = folder / MAYNER_CODE, folder / MAYNER_BUNDLE
    checkout_code(code, MAYNER_REPOSITORY, MAYNER_COMMIT, git)
    print(f"mayner: his code is in {code}, at commit {MAYNER_COMMIT}")
    files_url = f"{MAYNER.url}/resolve/main"
    manifest = bundle / "MANIFEST"
    download_whole(download, f"{files_url}/MANIFEST", manifest)
    (folder / RECORD_FILE).unlink(missing_ok=True)
    command = mayner_command(code, all_files)
    shown = f"PN_DATA_DIR={shlex.quote(str(bundle))} {shlex.join(command)}"
    # Flushed, so this line comes before his script's own output in a log.
    print(f"mayner: running {shown}", flush=True)
    status = run(command, cwd=code, env={**os.environ, "PN_DATA_DIR": str(bundle)})
    if status != 0:
        raise RuntimeError(
            f"his download script stopped with exit code {status}. The files "
            "that matched his MANIFEST are kept; run `pushlab fetch mayner` "
            "again to fetch the rest"
        )
    record = _source_fields(MAYNER) | {
        "files_url": files_url,
        "downloaded_at": utc_stamp(now()),
        "only": None if all_files else list(MAYNER_FILES),
        "script": {
            "repository": MAYNER_REPOSITORY,
            "commit": MAYNER_COMMIT,
            "command": shlex.join(command),
        },
        "manifest_sha256": sha256_of(manifest),
    }
    written = write_record(folder, record)
    print(f"mayner: recorded in {written}")
    print(_credit_line(MAYNER))
    return record
