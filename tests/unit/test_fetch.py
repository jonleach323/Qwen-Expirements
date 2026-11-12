"""Downloading the library's files with their record (decision 0020), and
the steps of the records both sources share.

Nothing here touches the network: a fake download stands in. All the bytes
are made up; none of them is the library's data. The network's side, and
what a download must hold, are tested in test_fetch_http.py, and Mayner's
bundle in test_fetch_mayner.py.
"""

import doctest
import hashlib
import json
import re
import shlex
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest

from pushlab.discovery import fetch
from pushlab.discovery.fetch import (
    LIBRARY_KEYS,
    download_parts,
    fetch_again,
    fetch_library,
    put_in_place,
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
# A made-up sign-in page, as a captive portal sends for any URL.
SIGN_IN = b"<!DOCTYPE html><html><title>Sign in to the made-up Wi-Fi</title></html>"


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
    """Made-up bytes for each of the library's URLs: a JSON object with the
    file's keys, holding none of the library's data."""
    return {
        f"{LIBRARY.url}/{name}": json.dumps(
            {key: [] for key in LIBRARY_KEYS[name]} | {"made_up": version}
        ).encode()
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


def _state(folder: Path) -> dict[str, bytes]:
    """Every file in folder, .part files and the record among them, by name."""
    return {path.name: path.read_bytes() for path in sorted(folder.iterdir())}


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
    captured = capsys.readouterr()
    assert captured.out.count("skipped") == len(LIBRARY_FILES)
    assert captured.err == ""


def test_a_file_copied_in_or_changed_by_hand_has_its_download_time_unknown(
    tmp_path: Path, capsys: pytest.CaptureFixture
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
    assert capsys.readouterr().err == ""  # it had no entry to differ from
    # A file changed after its download no longer matches its entry: warned of.
    (folder / "combos.json").write_bytes(b"made-up, edited\n")
    record = fetch_library(tmp_path, download=server, now=lambda: LATER)
    edited = _entry("combos.json", b"made-up, edited\n", None)
    assert record["files"]["combos.json"] == edited
    assert record["files"]["library.json"]["downloaded_at"] == "2026-11-12T20:31:05Z"
    err = " ".join(capsys.readouterr().err.split())
    assert err.startswith(f"pushlab fetch: warning: {folder / 'combos.json'} is not")
    assert "run `uv run pushlab fetch library --refresh" in err
    assert "library.json" not in err and "sensitive.json" not in err


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


def test_a_failed_first_fetch_leaves_no_file_and_no_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # library.json downloads whole, but goes in place only with the others.
    monkeypatch.setenv("PUSHLAB_DATA_DIR", str(tmp_path))
    folder = source_dir(LIBRARY, tmp_path)
    combos = f"{LIBRARY.url}/combos.json"
    server = FakeServer(_library(), fail=frozenset({combos}))
    with pytest.raises(OSError) as raised:
        fetch_library(download=server, now=lambda: FIRST)
    message = str(raised.value)
    assert message.startswith(f"could not download {combos}: connection reset")
    assert "Nothing was changed" in message
    assert message.endswith("run `uv run pushlab fetch library` again")
    assert server.requested == [f"{LIBRARY.url}/library.json", combos]
    assert _state(folder) == {}


@pytest.mark.parametrize("failing", ["combos.json", "sensitive.json"])
def test_a_failed_refresh_leaves_the_old_files_and_their_record(
    failing: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The second or the third file fails, after the first downloaded whole.
    monkeypatch.setenv("PUSHLAB_DATA_DIR", str(tmp_path))
    folder = source_dir(LIBRARY, tmp_path)
    fetch_library(download=FakeServer(_library()), now=lambda: FIRST)
    before = _state(folder)
    url = f"{LIBRARY.url}/{failing}"
    server = FakeServer(_library(2), fail=frozenset({url}))
    again = re.escape("run `uv run pushlab fetch library --refresh` again")
    with pytest.raises(OSError, match=f"could not download {url}.*{again}$"):
        fetch_library(refresh=True, download=server, now=lambda: LATER)
    assert server.requested[-1] == url
    assert _state(folder) == before  # the old three, the old record, no .part


@pytest.mark.parametrize(
    ("bodies", "headers", "problem"),
    [
        (
            dict.fromkeys(_library(2), SIGN_IN),
            {"Content-Type": "text/html; charset=utf-8"},
            f"{LIBRARY.url}/library.json did not send the JSON expected: it is "
            "not JSON (Expecting value: line 1 column 1 (char 0)); it begins "
            "'<!DOCTYPE html><html><title>Sign in to the made-up Wi-Fi</ti'; the "
            "server sent it as text/html; charset=utf-8, a web page",
        ),
        (
            _library(2) | {f"{LIBRARY.url}/combos.json": b'{"error": "moved"}'},
            {"Content-Type": "application/json"},
            f"{LIBRARY.url}/combos.json did not send the JSON expected: its JSON "
            "object has no 'pool_fields' or 'pool'; the server sent it as "
            "application/json. Nothing was changed",
        ),
    ],
)
def test_something_other_than_the_files_json_changes_nothing(
    bodies: dict[str, bytes], headers: dict[str, str], problem: str, tmp_path: Path
) -> None:
    folder = source_dir(LIBRARY, tmp_path)
    fetch_library(tmp_path, download=FakeServer(_library()), now=lambda: FIRST)
    before = _state(folder)
    server = FakeServer(bodies, headers=headers)
    with pytest.raises(OSError) as raised:
        fetch_library(tmp_path, refresh=True, download=server, now=lambda: LATER)
    assert problem in str(raised.value)
    # A fetch given its data folder names it in the command to run again.
    again = f"--refresh --data-dir {shlex.quote(str(tmp_path))}"
    assert str(raised.value).endswith(f"`uv run pushlab fetch library {again}` again")
    assert _state(folder) == before


def test_ctrl_c_during_the_downloads_deletes_every_part_file(tmp_path: Path) -> None:
    folder = source_dir(LIBRARY, tmp_path)
    fetch_library(tmp_path, download=FakeServer(_library()), now=lambda: FIRST)
    before = _state(folder)
    server = FakeServer(_library(2))

    def stopped(url: str, dest: Path) -> dict[str, str]:
        if url.endswith("/sensitive.json"):
            dest.write_bytes(b"made-")
            raise KeyboardInterrupt
        return server(url, dest)

    with pytest.raises(KeyboardInterrupt):
        fetch_library(tmp_path, refresh=True, download=stopped, now=lambda: LATER)
    assert len(server.requested) == 2
    assert _state(folder) == before


def test_download_parts_leaves_each_file_in_its_part_file_for_now(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    server = FakeServer(_library(), headers={"Last-Modified": MODIFIED})
    names = ["combos.json", "sensitive.json"]
    entries = download_parts(server, tmp_path / "qwen", names, lambda: FIRST)
    bodies = {name: _library()[f"{LIBRARY.url}/{name}"] for name in names}
    stamp = "2026-11-12T20:31:05Z"
    assert entries == {
        name: _entry(name, bodies[name], stamp, MODIFIED) for name in names
    }
    assert _state(tmp_path / "qwen") == {f"{n}.part": bodies[n] for n in names}
    size = len(bodies["sensitive.json"])
    assert (
        f"library: sensitive.json: downloaded {size:,} bytes" in capsys.readouterr().out
    )


def test_renames_stopped_partway_still_leave_a_true_record(tmp_path: Path) -> None:
    folder = source_dir(LIBRARY, tmp_path)
    first = fetch_library(tmp_path, download=FakeServer(_library()), now=lambda: FIRST)
    old = first["files"]
    server = FakeServer(_library(2))
    new = download_parts(server, folder, ["library.json"], lambda: LATER)
    # combos.json's .part is missing, so its rename fails after library.json's.
    new = new | {"combos.json": dict(old["combos.json"], downloaded_at="later")}
    kept = {"sensitive.json": old["sensitive.json"]}
    with pytest.raises(FileNotFoundError):
        put_in_place(folder, new, kept, old, lambda: LATER)
    files = read_record(folder)["files"]
    assert files["library.json"] == new["library.json"]
    assert files["combos.json"] == old["combos.json"]
    assert files["sensitive.json"] == old["sensitive.json"]
    assert sorted(_state(folder)) == sorted([*LIBRARY_FILES, RECORD_FILE])


def test_a_broken_library_record_says_what_to_do(tmp_path: Path) -> None:
    folder = source_dir(LIBRARY, tmp_path)
    folder.mkdir(parents=True)
    (folder / RECORD_FILE).write_text("{")
    again = re.escape(
        "Delete it and run `uv run pushlab fetch library --data-dir "
        f"{shlex.quote(str(tmp_path))}`"
    )
    with pytest.raises(ValueError, match=again):
        fetch_library(tmp_path, download=FakeServer(_library()), now=lambda: FIRST)


def test_a_failed_fetch_names_the_command_to_run_again() -> None:
    assert fetch_again("library") == "uv run pushlab fetch library"
    data = Path("/made/up/it's data")
    assert fetch_again("library", data, "--refresh") == (
        "uv run pushlab fetch library --refresh --data-dir '/made/up/it'\"'\"'s data'"
    )


def test_every_file_of_the_library_has_the_keys_its_download_must_have() -> None:
    assert set(LIBRARY_KEYS) == set(LIBRARY_FILES)
    assert LIBRARY_KEYS["library.json"] == ("fields", "rows")
    assert LIBRARY_KEYS["combos.json"] == ("pool_fields", "pool")
    assert LIBRARY_KEYS["sensitive.json"] == ("neurons",)


def test_times_are_written_in_utc_to_the_second() -> None:
    plus_one = timezone(timedelta(hours=1))
    moment = datetime(2026, 11, 12, 0, 30, 0, 999, tzinfo=plus_one)
    assert utc_stamp(moment) == "2026-11-11T23:30:00Z"
    with pytest.raises(ValueError, match="no time zone"):
        utc_stamp(datetime(2026, 11, 12))


def test_docstring_examples_run() -> None:
    result = doctest.testmod(fetch)
    assert result.attempted > 0
    assert result.failed == 0
