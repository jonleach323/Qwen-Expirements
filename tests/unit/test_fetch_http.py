"""The network side of pushlab fetch (pushlab.discovery.fetch_http): a URL
into a file, whole or not at all, and a check that a downloaded file of the
library is JSON, an object, with its file's keys.

Nothing here touches the network: a fake urlopen, or a fake download, stands
in. All the bytes are made up; none of them is the library's data. A fetch
that brings something else is tested in test_fetch.py.
"""

import doctest
import io
import json
import urllib.request
from pathlib import Path

import pytest

from pushlab.discovery import fetch_http
from pushlab.discovery.fetch_http import (
    check_download,
    download_whole,
    http_download,
    json_problem,
)

URL = "https://example.invalid/data/qwen/combos.json"
# The keys combos.json has.
KEYS = ("pool_fields", "pool")
MODIFIED = "Wed, 11 Nov 2026 09:00:00 GMT"


def _file(tmp_path: Path, body: bytes) -> Path:
    """A downloaded .part file holding body."""
    path = tmp_path / "combos.json.part"
    path.write_bytes(body)
    return path


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
    assert asked == [("https://example.invalid/a.json", "pushlab", fetch_http.TIMEOUT)]
    assert (tmp_path / "a.json").read_bytes() == b"made-up body"
    assert fetch_http.last_modified(headers) == MODIFIED


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


@pytest.mark.parametrize(
    ("body", "problem"),
    [
        (b'{"pool_fields": [], "pool": [], "more": 1}', None),
        (b"", "it is empty"),
        (
            b"<html><body>Moved</body></html>",
            "it is not JSON (Expecting value: line 1 column 1 (char 0)); it "
            "begins '<html><body>Moved</body></html>'",
        ),
        (
            b'{"pool": [1, 2',
            "it is not JSON (Expecting ',' delimiter: line 1 column 15 (char "
            "14)); it begins '{\"pool\": [1, 2'",
        ),
        (b"[1, 2]", "it holds an array, not a JSON object"),
        (b"null", "it holds null, not a JSON object"),
        (b'{"pool": []}', "its JSON object has no 'pool_fields'"),
        (b'{"error": "moved"}', "its JSON object has no 'pool_fields' or 'pool'"),
    ],
)
def test_json_problem_says_what_is_wrong(
    body: bytes, problem: str | None, tmp_path: Path
) -> None:
    assert json_problem(_file(tmp_path, body), KEYS) == problem


@pytest.mark.parametrize("body", [b"\xff\xfe\xff", b"[" * 200_000])
def test_bytes_json_cannot_read_are_not_json_rather_than_a_crash(
    body: bytes, tmp_path: Path
) -> None:
    # Bytes that are no text, and arrays nested past Python's recursion limit.
    found = json_problem(_file(tmp_path, body), ["pool"])
    assert found is not None and found.startswith("it is not JSON (")


def test_only_the_first_characters_of_a_file_are_shown(tmp_path: Path) -> None:
    found = json_problem(_file(tmp_path, b"<p>" + b"x" * 1000), ["pool"])
    assert found is not None
    assert found.endswith(repr("<p>" + "x" * (fetch_http.SHOWN - 3)))


def test_a_file_with_its_keys_is_kept(tmp_path: Path) -> None:
    body = json.dumps({"pool_fields": ["id"], "pool": []}).encode()
    path = _file(tmp_path, body)
    check_download(path, URL, KEYS, {"Content-Type": "application/json"})
    assert path.read_bytes() == body


@pytest.mark.parametrize(
    ("headers", "served"),
    [
        (
            {"content-type": "text/html; charset=utf-8"},
            "; the server sent it as text/html; charset=utf-8, a web page, such as "
            "a network's sign-in page or a notice that the file has moved",
        ),
        (
            {"Content-Type": "application/octet-stream"},
            "; the server sent it as application/octet-stream",
        ),
        ({}, ""),
    ],
)
def test_anything_else_is_deleted_and_refused_naming_the_url_and_its_type(
    headers: dict[str, str], served: str, tmp_path: Path
) -> None:
    path = _file(tmp_path, b"<html>Sign in</html>")
    with pytest.raises(OSError) as raised:
        check_download(path, URL, KEYS, headers)
    assert str(raised.value) == (
        f"{URL} did not send the JSON expected: it is not JSON (Expecting value: "
        f"line 1 column 1 (char 0)); it begins '<html>Sign in</html>'{served}"
    )
    assert not path.exists()


def test_docstring_examples_run() -> None:
    result = doctest.testmod(fetch_http)
    assert result.attempted > 0
    assert result.failed == 0
