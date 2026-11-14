"""pushlab fetch (pushlab.discovery.fetch_command): what it runs, the summary
it prints of the files downloaded and skipped, and what it refuses.

Nothing here goes online: fakes stand in for fetch_library and fetch_mayner,
or for the network, git and Mayner's script under the real ones. All the
bytes are made up; none of them is either source's data. The command's
options are tested in test_cli_discovery.py.
"""

import doctest
import json
import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from pushlab.cli import main
from pushlab.discovery import fetch, fetch_command
from pushlab.discovery import fetch_mayner as mayner_code
from pushlab.discovery.fetch import LIBRARY_KEYS
from pushlab.discovery.fetch_command import run_fetch
from pushlab.discovery.sources import (
    LIBRARY,
    LIBRARY_FILES,
    MAYNER,
    MAYNER_FILES,
    read_record,
    source_dir,
    write_record,
)

# The files of Mayner's bundle the stand-ins for his script bring: the one
# file of each of his --only prefixes, and 3 under vectors/without/.
BUNDLE_FILES = [
    "role_steering_sizes/judged_replies.parquet",
    "split/concentration_neurons.parquet",
    "split/top_neurons.parquet",
    "vectors/without/default.pt",
    "vectors/without/pirate.pt",
    "vectors/without/sommelier.pt",
]


class FakeLibrary:
    """Stands in for fetch_library: keeps each call's arguments, and writes
    made-up files where it would (those not there yet, all with refresh)
    and its record; or, given error, raises it."""

    def __init__(self, error: BaseException | None = None) -> None:
        self.error, self.calls = error, []

    def __call__(self, root: Path | None = None, *, refresh: bool = False) -> dict:
        self.calls.append((root, refresh))
        if self.error is not None:
            raise self.error
        folder = source_dir(LIBRARY, root)
        folder.mkdir(parents=True, exist_ok=True)
        for name in LIBRARY_FILES:
            if refresh or not (folder / name).is_file():
                (folder / name).write_text(f"made-up {name}, downloaded\n")
        write_record(folder, {"source": "library"})
        return {"source": "library"}


def _bring(bundle: Path, names: Sequence[str]) -> None:
    """Write made-up files of these names into bundle, as his script does,
    but none already there."""
    for name in names:
        if not (bundle / name).is_file():
            (bundle / name).parent.mkdir(parents=True, exist_ok=True)
            (bundle / name).write_bytes(b"made-up")


class FakeMayner:
    """Stands in for fetch_mayner as FakeLibrary does for fetch_library: it
    brings the files named that are not there yet, and his MANIFEST."""

    def __init__(
        self, names: Sequence[str] = BUNDLE_FILES, error: BaseException | None = None
    ) -> None:
        self.names, self.error, self.calls = names, error, []

    def __call__(self, root: Path | None = None, *, all_files: bool = False) -> dict:
        self.calls.append((root, all_files))
        if self.error is not None:
            raise self.error
        bundle = source_dir(MAYNER, root) / "bundle"
        _bring(bundle, self.names)
        (bundle / "MANIFEST").write_text("made-up MANIFEST\n")
        write_record(source_dir(MAYNER, root), {"source": "mayner"})
        return {"source": "mayner"}


def _fetch(source: str, data: Path, *options: str) -> int:
    return main(["fetch", source, "--data-dir", str(data), *options])


def test_fetch_library_says_what_it_downloaded_and_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    fake = FakeLibrary()
    monkeypatch.setattr(fetch_command, "fetch_library", fake)
    folder = tmp_path / "library" / "qwen"
    folder.mkdir(parents=True)
    (folder / "combos.json").write_text("made-up combos.json, copied in\n")
    assert _fetch("library", tmp_path) == 0
    record = f"the record of where they came from and when is {folder}/FETCHED.json"
    assert capsys.readouterr().out.splitlines() == [
        f"library: downloaded library.json and sensitive.json into {folder}",
        "library: skipped combos.json, already there",
        f"library: {record}",
    ]
    assert _fetch("library", tmp_path, "--refresh") == 0
    assert capsys.readouterr().out.splitlines() == [
        "library: downloaded combos.json, library.json and sensitive.json into "
        f"{folder}",
        f"library: {record}",
    ]
    assert fake.calls == [(tmp_path, False), (tmp_path, True)]


def test_fetch_mayner_says_what_his_script_downloaded_and_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    fake = FakeMayner()
    monkeypatch.setattr(fetch_command, "fetch_mayner", fake)
    bundle = tmp_path / "mayner" / "bundle"
    record = tmp_path / "mayner" / "FETCHED.json"
    last = f"mayner: the record of where they came from and when is {record}"
    assert _fetch("mayner", tmp_path) == 0
    captured = capsys.readouterr()
    assert captured.out.splitlines() == [
        f"mayner: downloaded 6 files into {bundle}",
        last,
    ]
    assert captured.err == ""
    assert _fetch("mayner", tmp_path) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"mayner: downloaded nothing into {bundle}",
        "mayner: skipped 6 files, already there",
        last,
    ]
    (bundle / "split" / "top_neurons.parquet").unlink()  # deleted by hand
    assert _fetch("mayner", tmp_path, "--all") == 0
    assert capsys.readouterr().out.splitlines() == [
        f"mayner: downloaded split/top_neurons.parquet into {bundle}",
        "mayner: skipped role_steering_sizes/judged_replies.parquet, "
        "split/concentration_neurons.parquet, vectors/without/default.pt, "
        "vectors/without/pirate.pt and vectors/without/sommelier.pt, already there",
        last,
    ]
    assert fake.calls == [(tmp_path, False), (tmp_path, False), (tmp_path, True)]


def test_a_file_written_again_is_told_from_one_left_alone(tmp_path: Path) -> None:
    for name in ("a.json", "b.json", "c.txt"):
        (tmp_path / name).write_text("made-up\n")
    wanted = fetch_command.files_now(tmp_path, lambda name: name.endswith(".json"))
    assert list(wanted) == ["a.json", "b.json"]
    (tmp_path / "b.json").write_text("made-up, downloaded again\n")
    replaced = (tmp_path / "a.json").with_name("a.json.part")
    replaced.write_text("made-up\n")
    replaced.replace(tmp_path / "a.json")  # the same size, but a new file
    again = fetch_command.files_now(tmp_path, lambda name: name.endswith(".json"))
    assert again["a.json"] != wanted["a.json"]
    assert again["b.json"] != wanted["b.json"]
    left = fetch_command.files_now(tmp_path, lambda name: name.endswith(".json"))
    assert left == again


def test_a_prefix_his_script_brought_nothing_under_is_warned_of(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    no_vectors = [name for name in BUNDLE_FILES if not name.startswith("vectors/")]
    monkeypatch.setattr(fetch_command, "fetch_mayner", FakeMayner(no_vectors))
    assert _fetch("mayner", tmp_path) == 0
    bundle = tmp_path / "mayner" / "bundle"
    assert capsys.readouterr().err.splitlines() == [
        f"pushlab fetch: warning: his bundle in {bundle} has no file under "
        "vectors/without/, although his script was asked for it; his MANIFEST may "
        "have moved it"
    ]
    assert _fetch("mayner", tmp_path, "--all") == 0  # nothing asked for by name
    assert capsys.readouterr().err == ""


def test_the_data_folder_defaults_to_pushlab_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    fake = FakeLibrary()
    monkeypatch.setattr(fetch_command, "fetch_library", fake)
    monkeypatch.setenv("PUSHLAB_DATA_DIR", str(tmp_path))
    assert main(["fetch", "library"]) == 0
    assert fake.calls == [(None, False)]
    assert f"into {tmp_path / 'library' / 'qwen'}" in capsys.readouterr().out


@pytest.mark.parametrize("source", ["library", "mayner"])
@pytest.mark.parametrize(
    "error",
    [
        OSError("could not download https://example.invalid/f: timed out"),
        ValueError("FETCHED.json is not valid JSON. Delete it and fetch again"),
        RuntimeError("his download script stopped with exit code 1"),
    ],
)
def test_a_failed_fetch_is_refused_with_its_reason(
    source: str,
    error: Exception,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setattr(fetch_command, "fetch_library", FakeLibrary(error))
    monkeypatch.setattr(fetch_command, "fetch_mayner", FakeMayner(error=error))
    assert _fetch(source, tmp_path) == 1
    captured = capsys.readouterr()
    assert captured.err == f"pushlab fetch: {error}\n"
    assert "the record of" not in captured.out


def test_git_missing_and_ctrl_c_are_explained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    no_git = FileNotFoundError(2, "No such file or directory", "git")
    monkeypatch.setattr(fetch_command, "fetch_mayner", FakeMayner(error=no_git))
    assert _fetch("mayner", tmp_path) == 1
    assert "pushlab fetch: git is not installed" in capsys.readouterr().err
    stopped = FakeLibrary(KeyboardInterrupt())
    monkeypatch.setattr(fetch_command, "fetch_library", stopped)
    assert _fetch("library", tmp_path) == 130
    err = " ".join(capsys.readouterr().err.split())
    assert err.startswith("pushlab fetch: stopped by Ctrl-C. The library's new files")
    assert "so its folder is as it was" in err
    monkeypatch.setattr(
        fetch_command, "fetch_mayner", FakeMayner(error=KeyboardInterrupt())
    )
    assert _fetch("mayner", tmp_path) == 130
    err = " ".join(capsys.readouterr().err.split())
    assert err.startswith("pushlab fetch: stopped by Ctrl-C. The files his script")
    assert err.endswith("before running `uv run pushlab candidates`")


@pytest.mark.parametrize(
    ("source", "options", "refusal"),
    [
        ("zoo", {}, "there is no source 'zoo'; give library or mayner"),
        ("mayner", {"refresh": True}, "--refresh goes with `pushlab fetch library`"),
        ("library", {"all_files": True}, "--all goes with `pushlab fetch mayner`"),
    ],
)
def test_options_that_do_not_go_with_the_source_are_refused(
    source: str,
    options: Mapping[str, bool],
    refusal: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    monkeypatch.setattr(fetch_command, "fetch_library", FakeLibrary())
    monkeypatch.setattr(fetch_command, "fetch_mayner", FakeMayner())
    assert run_fetch(source, tmp_path, **options) == 1
    assert refusal in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_fetch_library_through_the_real_steps_with_a_stand_in_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    requested = []

    def download(url: str, dest: Path) -> dict[str, str]:
        # A made-up object with the file's keys, as the library's files have.
        requested.append(url)
        keys = LIBRARY_KEYS[url.rsplit("/", 1)[1]]
        dest.write_text(json.dumps({key: [] for key in keys} | {"made_up": url}))
        return {"Last-Modified": "Wed, 11 Nov 2026 09:00:00 GMT"}

    monkeypatch.setattr(fetch, "http_download", download)
    folder = tmp_path / "library" / "qwen"
    assert _fetch("library", tmp_path) == 0
    assert requested == [f"{LIBRARY.url}/{name}" for name in LIBRARY_FILES]
    files = read_record(folder)["files"]
    assert {name: files[name]["url"] for name in files} == {
        name: f"{LIBRARY.url}/{name}" for name in LIBRARY_FILES
    }
    names = "combos.json, library.json and sensitive.json"
    out = capsys.readouterr().out
    assert f"library: downloaded {names} into {folder}" in out
    assert LIBRARY.credit in out
    assert _fetch("library", tmp_path) == 0
    assert len(requested) == len(LIBRARY_FILES)
    assert f"library: skipped {names}, already there" in capsys.readouterr().out


def test_a_refresh_that_fails_through_the_command_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    broken = []

    def download(url: str, dest: Path) -> dict[str, str]:
        if broken and url.endswith("/sensitive.json"):
            raise ConnectionResetError("connection reset by peer")
        keys = LIBRARY_KEYS[url.rsplit("/", 1)[1]]
        dest.write_text(json.dumps({key: [] for key in keys} | {"made_up": 1}))
        return {}

    monkeypatch.setattr(fetch, "http_download", download)
    folder = tmp_path / "library" / "qwen"
    assert _fetch("library", tmp_path) == 0
    before = {path.name: path.read_bytes() for path in folder.iterdir()}
    capsys.readouterr()
    broken.append(True)
    assert _fetch("library", tmp_path, "--refresh") == 1
    captured = capsys.readouterr()
    err = " ".join(captured.err.split())
    assert err.startswith(f"pushlab fetch: could not download {LIBRARY.url}/sens")
    data = shlex.quote(str(tmp_path))
    again = f"uv run pushlab fetch library --refresh --data-dir {data}"
    assert err.endswith(f"run `{again}` again")
    assert "the record of" not in captured.out
    assert {path.name: path.read_bytes() for path in folder.iterdir()} == before


@pytest.mark.parametrize("all_files", [False, True])
def test_fetch_mayner_through_the_real_steps_with_stand_ins(
    all_files: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
) -> None:
    commands = []

    def script(command: list[str], *, cwd: Path, env: Mapping[str, str]) -> int:
        commands.append(command)
        _bring(Path(env["PN_DATA_DIR"]), BUNDLE_FILES)
        return 0

    def download(url: str, dest: Path) -> dict[str, str]:
        dest.write_bytes(b"made-up MANIFEST\n")
        return {}

    monkeypatch.setattr(mayner_code, "checkout_code", lambda *args: None)
    monkeypatch.setattr(mayner_code, "http_download", download)
    monkeypatch.setattr(mayner_code, "run_command", script)
    assert _fetch("mayner", tmp_path, *(["--all"] if all_files else [])) == 0
    record = read_record(tmp_path / "mayner")
    assert record["only"] == (None if all_files else list(MAYNER_FILES))
    assert ("--only" in commands[0]) is not all_files
    out = capsys.readouterr().out
    assert f"mayner: downloaded 6 files into {tmp_path / 'mayner' / 'bundle'}" in out
    assert MAYNER.credit in out


def test_docstring_examples_run() -> None:
    result = doctest.testmod(fetch_command)
    assert result.attempted > 0
    assert result.failed == 0
