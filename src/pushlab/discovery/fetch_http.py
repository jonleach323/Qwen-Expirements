"""The network side of `pushlab fetch`: one file over HTTPS, whole and checked.

fetch.fetch_library and fetch_mayner.fetch_mayner download with these
(decision 0020):

- http_download gets a URL into a file. The request names itself only
  "pushlab", each wait for the server lasts at most TIMEOUT seconds, and a
  response cut short of its Content-Length is refused.
- download_whole downloads into a .part file beside the file, which becomes
  the file only once whole, so a file there is never half a download.
- check_download makes sure a downloaded file of the library is what was
  asked for: JSON, an object, with the keys its file has. Anything else,
  such as a network's sign-in page (a captive portal) or a notice that the
  file has moved, is refused: the file is deleted, and the error says what
  came instead and the Content-Type the server gave it.

The tests hand in a fake urlopen or a fake download, so they never go online.
"""

import json
import shutil
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

# Sent with every download: a plain name, nothing about who runs it.
USER_AGENT = "pushlab"
# Seconds a download waits for the server before it gives up.
TIMEOUT = 60
# Bytes copied at a time.
CHUNK = 1 << 20
# What JSON calls the values json.load makes, other than objects.
JSON_KINDS = {
    list: "an array",
    str: "a string",
    int: "a number",
    float: "a number",
    bool: "true or false",
    type(None): "null",
}
# Characters shown of a file that is not JSON: enough to tell a web page.
SHOWN = 60

# Downloads a URL into a file; returns the response's headers.
Download = Callable[[str, Path], Mapping[str, str]]


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
        >>> url = "https://neuronpersonas.com/data/qwen/sensitive.json"
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
        raise OSError(f"{url} stopped after {got:,} of its {int(expected):,} bytes")
    return headers


def part_of(path: Path) -> Path:
    """Where a download to path goes until it is whole: path with .part added.

    Example:
        >>> part_of(Path("data/library.json"))
        PosixPath('data/library.json.part')
    """
    return path.with_name(path.name + ".part")


def download_whole(download: Download, url: str, dest: Path) -> dict[str, str]:
    """Download url into dest with download, through a .part file beside it.

    The .part file becomes dest only once download has returned, so a file
    at dest is always whole. If download fails, the .part file is deleted,
    dest stays as it was, and OSError names the URL. Returns the headers.

    Example:
        >>> url = "https://neuronpersonas.com/data/qwen/combos.json"
        >>> download_whole(http_download, url, Path("combos.json"))  # doctest: +SKIP
        {'Content-Type': 'application/json', ...}
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = part_of(dest)
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


def json_problem(path: Path, keys: Sequence[str]) -> str | None:
    """What is wrong with the file at path, which should hold a JSON object
    with keys: None if nothing is.

    Example:
        >>> json_problem(Path("/dev/null"), ["neurons"])
        'it is empty'
    """
    try:
        with Path(path).open("rb") as file:
            value = json.load(file)
    except (ValueError, RecursionError) as error:  # Unicode errors among them
        with Path(path).open("rb") as file:
            start = file.read(SHOWN).decode("utf-8", errors="replace")
        if not start:
            return "it is empty"
        return f"it is not JSON ({error}); it begins {start!r}"
    if not isinstance(value, dict):
        return f"it holds {JSON_KINDS.get(type(value), 'a value')}, not a JSON object"
    missing = [key for key in keys if key not in value]
    if missing:
        return f"its JSON object has no {' or '.join(map(repr, missing))}"
    return None


def check_download(
    path: Path, url: str, keys: Sequence[str], headers: Mapping[str, str]
) -> None:
    """Make sure the file downloaded from url into path holds a JSON object
    with keys; if not, delete it and raise OSError.

    The error names the URL, what was wrong (json_problem) and the
    response's Content-Type, from its headers, when it had one: a web page,
    such as a network's sign-in page or a notice that the file moved, is not
    the JSON asked for.

    Example:
        >>> url = "https://neuronpersonas.com/data/qwen/sensitive.json"
        >>> part, headers = Path("sensitive.json.part"), {"Content-Type": "text/html"}
        >>> check_download(part, url, ["neurons"], headers)  # doctest: +SKIP
        Traceback (most recent call last):
        OSError: https://neuronpersonas.com/data/qwen/sensitive.json did not ...
    """
    problem = json_problem(path, keys)
    if problem is None:
        return
    Path(path).unlink(missing_ok=True)
    kind = _header(headers, "Content-Type")
    served = f"; the server sent it as {kind}" if kind else ""
    if kind and "html" in kind.lower():
        served += (
            ", a web page, such as a network's sign-in page or a notice that "
            "the file has moved"
        )
    raise OSError(f"{url} did not send the JSON expected: {problem}{served}")
