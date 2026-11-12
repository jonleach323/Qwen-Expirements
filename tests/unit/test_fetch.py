"""Downloading the library's files with their record (decision 0020), and
the download steps both sources share.

Nothing here touches the network: a fake download, or a fake urlopen, stands
in. All the bytes are made up; none of them is the library's data. Mayner's
bundle is tested in test_fetch_mayner.py.
"""

import doctest
import hashlib
import io
import re
import urllib.request
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from pushlab.discovery import fetch
from pushlab.discovery.fetch import (
    download_whole,
    fetch_library,
    http_download,
    utc_stamp,
)
from pushlab.discovery.sources import (
    LIBRARY,
    LIBRARY_FILES,
    RECORD_FILE,
    read_record,
    source_dir,
)

FIRST = datetime(2026, 11, 12, 20, 31, 5, tzinfo=UTC)
LATER = datetime(2026, 11, 13, 8, 0, 0, tzinfo=UTC)
MODIFIED = "Wed, 11 Nov 2026 09:00:00 GMT"


class FakeServer:
    """Stands in for the network: serves made-up bytes for each URL with the
    headers given; for a URL in fail it writes half the bytes, then fails."""

    def __init__(
        self,
        contents: dict[str, bytes],
        headers: dict[str, str] | None = None,
        fail: frozenset[str] = frozenset(),
    ) -> None:
        self.contents, self.headers, self.fail = contents, headers or {}, fail
        self.requested: list[str] = []

    def __call__(self, url: str, dest: Path) -> dict[str, str]:
        self.requested.append(url)
        body = self.contents[url]
        if url in self.fail:
            dest.write_bytes(body[: len(body) // 2])
            raise ConnectionResetError("connection reset by peer")
        dest.write_bytes(body)
        return dict(self.headers)


def _library(version: int = 1) -> dict[str, bytes]:
    """Made-up bytes for each of the library's URLs."""
    return {
        f"{LIBRARY.url}/{name}": f"made-up {name}, version {version}\n".encode()
        for name in LIBRARY_FILES
    }


def _entry(
    name: str, body: bytes, downloaded_at: str | None, modified: str | None = None
) -> dict[str, object]:
    """The record's entry for the library file name, holding body."""
    return {
        "url": f"{LIBRARY.url}/{name}",
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "downloaded_at": downloaded_at,
        "last_modified": modified,
    }


def test_the_librarys_files_are_downloaded_into_the_data_folder_and_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setenv("PUSHLAB_DATA_DIR", str(tmp_path))
    server = FakeServer(_library(), headers={"last-modified": MODIFIED})
    record = fetch_library(download=server, now=lambda: FIRST)
    folder = tmp_path / "library" / "qwen"
    assert server.requested == [f"{LIBRARY.url}/{name}" for name in LIBRARY_FILES]
    assert {path.name for path in folder.iterdir()} == {*LIBRARY_FILES, RECORD_FILE}
    stamp = "2026-11-12T20:31:05Z"
    assert record == {
        "source": "library",
        "title": "Neuron Persona Library",
        "url": "https://neuronpersonas.com/data/qwen",
        "licence": LIBRARY.licence,
        "credit": LIBRARY.credit,
        "updated_at": stamp,
        "files": {
            name: _entry(name, _library()[f"{LIBRARY.url}/{name}"], stamp, MODIFIED)
            for name in LIBRARY_FILES
        },
    }
    assert read_record(folder) == record
    for name in LIBRARY_FILES:
        assert (folder / name).read_bytes() == _library()[f"{LIBRARY.url}/{name}"]
    out = capsys.readouterr().out
    assert "library.json: downloaded" in out and "skipped" not in out
    assert LIBRARY.credit in out


def test_files_already_there_are_skipped_and_keep_their_entries(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    first = fetch_library(tmp_path, download=FakeServer(_library()), now=lambda: FIRST)
    capsys.readouterr()
    server = FakeServer(_library(2))
    again = fetch_library(tmp_path, download=server, now=lambda: LATER)
    assert server.requested == []
    assert again["files"] == first["files"]
    assert again["updated_at"] == "2026-11-13T08:00:00Z"
    assert capsys.readouterr().out.count("skipped") == len(LIBRARY_FILES)


def test_a_file_copied_in_or_changed_by_hand_has_its_download_time_unknown(
    tmp_path: Path,
) -> None:
    folder = source_dir(LIBRARY, tmp_path)
    folder.mkdir(parents=True)
    (folder / "sensitive.json").write_bytes(b"made-up, copied in by hand\n")
    server = FakeServer(_library())
    record = fetch_library(tmp_path, download=server, now=lambda: FIRST)
    assert server.requested == [
        f"{LIBRARY.url}/library.json",
        f"{LIBRARY.url}/combos.json",
    ]
    copied = _entry("sensitive.json", b"made-up, copied in by hand\n", None)
    assert record["files"]["sensitive.json"] == copied
    # A file changed after its download no longer matches its entry.
    (folder / "combos.json").write_bytes(b"made-up, edited\n")
    record = fetch_library(tmp_path, download=server, now=lambda: LATER)
    edited = _entry("combos.json", b"made-up, edited\n", None)
    assert record["files"]["combos.json"] == edited
    assert record["files"]["library.json"]["downloaded_at"] == "2026-11-12T20:31:05Z"


def test_refresh_downloads_every_file_again(tmp_path: Path) -> None:
    fetch_library(tmp_path, download=FakeServer(_library()), now=lambda: FIRST)
    server = FakeServer(_library(2), headers={"Last-Modified": MODIFIED})
    record = fetch_library(tmp_path, refresh=True, download=server, now=lambda: LATER)
    assert len(server.requested) == len(LIBRARY_FILES)
    for name in LIBRARY_FILES:
        body = _library(2)[f"{LIBRARY.url}/{name}"]
        assert (source_dir(LIBRARY, tmp_path) / name).read_bytes() == body
        expected = _entry(name, body, "2026-11-13T08:00:00Z", MODIFIED)
        assert record["files"][name] == expected


def test_a_failed_download_leaves_no_partial_file_and_no_entry(tmp_path: Path) -> None:
    folder = source_dir(LIBRARY, tmp_path)
    combos = f"{LIBRARY.url}/combos.json"
    server = FakeServer(_library(), fail=frozenset({combos}))
    problem = re.escape(f"could not download {combos}: connection reset")
    with pytest.raises(OSError, match=problem):
        fetch_library(tmp_path, download=server, now=lambda: FIRST)
    assert sorted(path.name for path in folder.iterdir()) == [
        RECORD_FILE,
        "library.json",
    ]
    assert list(read_record(folder)["files"]) == ["library.json"]


def test_a_failed_refresh_keeps_the_file_there_and_its_entry(tmp_path: Path) -> None:
    folder = source_dir(LIBRARY, tmp_path)
    first = fetch_library(tmp_path, download=FakeServer(_library()), now=lambda: FIRST)
    combos = f"{LIBRARY.url}/combos.json"
    server = FakeServer(_library(2), fail=frozenset({combos}))
    with pytest.raises(OSError, match="could not download"):
        fetch_library(tmp_path, refresh=True, download=server, now=lambda: LATER)
    files = read_record(folder)["files"]
    assert files["combos.json"] == first["files"]["combos.json"]
    assert files["sensitive.json"] == first["files"]["sensitive.json"]
    assert files["library.json"]["downloaded_at"] == "2026-11-13T08:00:00Z"
    assert (folder / "combos.json").read_bytes() == _library()[combos]
    assert not (folder / "combos.json.part").exists()


def test_a_broken_library_record_says_what_to_do(tmp_path: Path) -> None:
    folder = source_dir(LIBRARY, tmp_path)
    folder.mkdir(parents=True)
    (folder / RECORD_FILE).write_text("{")
    with pytest.raises(ValueError, match="Delete it and run `pushlab fetch library`"):
        fetch_library(tmp_path, download=FakeServer(_library()), now=lambda: FIRST)


def test_times_are_written_in_utc_to_the_second() -> None:
    plus_one = timezone(timedelta(hours=1))
    moment = datetime(2026, 11, 12, 0, 30, 0, 999, tzinfo=plus_one)
    assert utc_stamp(moment) == "2026-11-11T23:30:00Z"
    with pytest.raises(ValueError, match="no time zone"):
        utc_stamp(datetime(2026, 11, 12))


class FakeResponse(io.BytesIO):
    """What urlopen returns, standing in: a body and its headers."""

    def __init__(self, body: bytes, headers: dict[str, str]) -> None:
        super().__init__(body)
        self.headers = headers


def test_http_download_saves_the_body_and_names_itself_only_pushlab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    asked = []

    def urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
        asked.append((request.full_url, request.get_header("User-agent"), timeout))
        headers = {"Content-Length": "12", "Last-Modified": MODIFIED}
        return FakeResponse(b"made-up body", headers)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    headers = http_download("https://example.invalid/a.json", tmp_path / "a.json")
    assert asked == [("https://example.invalid/a.json", "pushlab", fetch.TIMEOUT)]
    assert (tmp_path / "a.json").read_bytes() == b"made-up body"
    assert fetch.last_modified(headers) == MODIFIED


def test_http_download_refuses_a_body_cut_short(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def urlopen(request: urllib.request.Request, timeout: float) -> FakeResponse:
        return FakeResponse(b"made-up", {"content-length": "100"})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    with pytest.raises(OSError, match="stopped after 7 of its 100 bytes"):
        http_download("https://example.invalid/a.json", tmp_path / "a.json")


def test_a_download_becomes_the_file_only_once_whole(tmp_path: Path) -> None:
    dest, url = tmp_path / "f.json", "https://example.invalid/f.json"
    part = tmp_path / "f.json.part"
    dest.write_bytes(b"made-up, old")

    def broken(url: str, part: Path) -> dict[str, str]:
        part.write_bytes(b"made-")
        raise TimeoutError("timed out")

    def interrupted(url: str, part: Path) -> dict[str, str]:
        part.write_bytes(b"made-")
        raise KeyboardInterrupt

    with pytest.raises(OSError, match=f"could not download {url}: timed out"):
        download_whole(broken, url, dest)
    with pytest.raises(KeyboardInterrupt):
        download_whole(interrupted, url, dest)
    assert dest.read_bytes() == b"made-up, old" and not part.exists()

    def whole(url: str, part: Path) -> dict[str, str]:
        part.write_bytes(b"made-up, new")
        return {"ETag": "1"}

    assert download_whole(whole, url, tmp_path / "new" / "f.json") == {"ETag": "1"}
    assert (tmp_path / "new" / "f.json").read_bytes() == b"made-up, new"


def test_docstring_examples_run() -> None:
    result = doctest.testmod(fetch)
    assert result.attempted > 0
    assert result.failed == 0
