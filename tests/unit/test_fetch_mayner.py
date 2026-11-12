"""Downloading Mayner's bundle with his own script (decision 0020): his code
checked out at the pinned commit, his download script run, and the record.

A local git repository stands in for his, and fakes for the network and for
running his script, so nothing here goes online. All of it is made up.
"""

import doctest
import hashlib
import os
import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from pushlab.discovery import fetch
from pushlab.discovery import fetch_mayner as mayner_code
from pushlab.discovery.fetch_mayner import checkout_code, run_command, run_git
from pushlab.discovery.sources import (
    MAYNER,
    MAYNER_COMMIT,
    MAYNER_FILES,
    MAYNER_REPOSITORY,
    RECORD_FILE,
    read_record,
)

FIRST = datetime(2026, 11, 12, 20, 31, 5, tzinfo=UTC)
LATER = datetime(2026, 11, 13, 8, 0, 0, tzinfo=UTC)
MANIFEST_URL = f"{MAYNER.url}/resolve/main/MANIFEST"
# A made-up MANIFEST in his format: path, bytes and sha256, tab-separated.
MANIFEST = b"split/top_neurons.parquet\t12\t" + b"ab" * 32 + b"\n"
# Set-up commits in the stand-in repository: never signed, by a made-up name.
GIT = ["git", "-c", "user.name=pushlab", "-c", "user.email=pushlab@localhost"]
GIT += ["-c", "commit.gpgsign=false"]


class FakeManifest:
    """Stands in for the network: keeps each URL it is asked for and writes
    the made-up MANIFEST, or, with fail, breaks off with an error."""

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.requested: list[str] = []

    def __call__(self, url: str, dest: Path) -> dict[str, str]:
        self.requested.append(url)
        if self.fail:
            raise ConnectionResetError("connection reset by peer")
        dest.write_bytes(MANIFEST)
        return {}


def _git(folder: Path, *args: str) -> str:
    """Run git in folder for a test's own set-up; return what it printed."""
    command = [*GIT, "-C", str(folder), *args]
    return subprocess.run(
        command, check=True, capture_output=True, text=True
    ).stdout.strip()


def _commit(repository: Path, text: str) -> str:
    """Commit a made-up download_data.py holding text; return the commit."""
    (repository / "download_data.py").write_text(text, encoding="utf-8")
    _git(repository, "add", "download_data.py")
    _git(repository, "commit", "--quiet", "-m", text.strip())
    return _git(repository, "rev-parse", "HEAD")


@pytest.fixture
def origin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, list[str]]:
    """A local repository standing in for Mayner's, with two commits of a
    made-up download_data.py. MAYNER_REPOSITORY names it and MAYNER_COMMIT
    pins the first. Returns the repository and its two commits."""
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    repository = tmp_path / "origin"
    repository.mkdir()
    _git(repository, "init", "--quiet", "--initial-branch=main")
    commits = [_commit(repository, f"# made up, version {n}\n") for n in (1, 2)]
    monkeypatch.setattr(fetch, "MAYNER_REPOSITORY", str(repository))
    monkeypatch.setattr(fetch, "MAYNER_COMMIT", commits[0])
    return repository, commits


class FakeRun:
    """Stands in for running his script: keeps each command, folder and
    environment, and returns status."""

    def __init__(self, status: int = 0) -> None:
        self.status = status
        self.calls: list[tuple[list[str], Path, dict[str, str]]] = []

    def __call__(
        self, command: Sequence[str], *, cwd: Path, env: Mapping[str, str]
    ) -> int:
        self.calls.append((list(command), cwd, dict(env)))
        return self.status


@pytest.mark.parametrize("all_files", [False, True])
def test_mayner_checks_out_his_code_and_runs_his_own_script(
    tmp_path: Path,
    origin: tuple[Path, list[str]],
    monkeypatch: pytest.MonkeyPatch,
    all_files: bool,
) -> None:
    repository, commits = origin
    monkeypatch.setenv("PUSHLAB_TEST_VARIABLE", "passed on")
    run, server = FakeRun(), FakeManifest()
    record = fetch.fetch_mayner(
        tmp_path, all_files=all_files, run=run, download=server, now=lambda: FIRST
    )
    folder = tmp_path / "mayner"
    code, bundle = folder / "code", folder / "bundle"
    assert _git(code, "rev-parse", "HEAD") == commits[0]
    script = [sys.executable, "-I", str(code / "download_data.py"), MAYNER.url]
    command = script if all_files else [*script, "--only", *MAYNER_FILES]
    [(ran, cwd, env)] = run.calls
    assert ran == command and cwd == code
    assert env["PN_DATA_DIR"] == str(bundle)
    assert env["PUSHLAB_TEST_VARIABLE"] == "passed on"
    assert server.requested == [MANIFEST_URL]
    assert (bundle / "MANIFEST").read_bytes() == MANIFEST
    assert record == {
        "source": "mayner",
        "title": "persona-neurons",
        "url": "https://huggingface.co/datasets/wmayner/persona-neurons",
        "files_url": f"{MAYNER.url}/resolve/main",
        "licence": "CC BY 4.0",
        "credit": MAYNER.credit,
        "downloaded_at": "2026-11-12T20:31:05Z",
        "only": None if all_files else list(MAYNER_FILES),
        "script": {
            "repository": str(repository),
            "commit": commits[0],
            "command": shlex.join(command),
        },
        "manifest_sha256": hashlib.sha256(MANIFEST).hexdigest(),
    }
    assert read_record(folder) == record


def _fake_git(
    commands: list[list[str]], head: str, changed: str = ""
) -> Callable[[list[str]], str]:
    """A git that only keeps the commands it is given: status prints changed
    and rev-parse HEAD prints head."""

    def git(command: list[str]) -> str:
        commands.append(command)
        if command[-2:] == ["rev-parse", "HEAD"]:
            return head
        return changed if "status" in command else ""

    return git


def test_his_code_is_cloned_from_his_repository_and_checked_out_at_the_pin(
    tmp_path: Path,
) -> None:
    commands: list[list[str]] = []
    code = tmp_path / "code"
    checkout_code(
        code, MAYNER_REPOSITORY, MAYNER_COMMIT, _fake_git(commands, MAYNER_COMMIT)
    )
    url = "https://github.com/wmayner/persona-neurons.git"
    here, detached = ["git", "-C", str(code)], ["-c", "advice.detachedHead=false"]
    assert commands == [
        ["git", "clone", "--quiet", url, str(code)],
        [*here, "status", "--porcelain", "--untracked-files=no"],
        [*here, *detached, "checkout", "--quiet", MAYNER_COMMIT],
        [*here, "rev-parse", "HEAD"],
    ]


def test_a_checkout_must_end_unchanged_at_the_pinned_commit(tmp_path: Path) -> None:
    code, pin = tmp_path / "code", MAYNER_COMMIT
    # As git status --porcelain lists a changed file, and a changed one staged.
    changed = " M download_data.py\nM  persona_neurons/config.py"
    names = "download_data.py, persona_neurons/config.py"
    with pytest.raises(RuntimeError, match=f"his code: {names}. His"):
        checkout_code(code, MAYNER_REPOSITORY, pin, _fake_git([], pin, changed))
    with pytest.raises(RuntimeError, match=f"at commit {'0' * 40}, not {pin}"):
        checkout_code(code, MAYNER_REPOSITORY, pin, _fake_git([], "0" * 40))


def test_a_change_to_his_files_is_refused_naming_them(
    tmp_path: Path, origin: tuple[Path, list[str]]
) -> None:
    repository, commits = origin
    code = tmp_path / "mayner" / "code"
    checkout_code(code, str(repository), commits[0])
    (code / "download_data.py").write_text("print('changed')\n")
    (code / "notes.txt").write_text("made up, and not his: git does not track it\n")
    run, server = FakeRun(), FakeManifest()
    with pytest.raises(RuntimeError, match=r"his code: download_data\.py\. His"):
        fetch.fetch_mayner(tmp_path, run=run, download=server, now=lambda: FIRST)
    assert run.calls == [] and server.requested == []
    assert not (tmp_path / "mayner" / RECORD_FILE).exists()
    # Undone, the change no longer stops it; the untracked file never did.
    _git(code, "checkout", "--", "download_data.py")
    checkout_code(code, str(repository), commits[0])


def test_a_checkout_at_another_commit_is_moved_to_the_pinned_one(
    tmp_path: Path, origin: tuple[Path, list[str]]
) -> None:
    repository, commits = origin
    code = tmp_path / "code"
    _git(tmp_path, "clone", "--quiet", str(repository), str(code))
    assert _git(code, "rev-parse", "HEAD") == commits[1]
    checkout_code(code, str(repository), commits[0])
    assert _git(code, "rev-parse", "HEAD") == commits[0]


def test_a_commit_the_checkout_lacks_is_fetched_from_origin(
    tmp_path: Path, origin: tuple[Path, list[str]]
) -> None:
    repository, commits = origin
    code = tmp_path / "code"
    checkout_code(code, str(repository), commits[0])
    newer = _commit(repository, "# made up, version 3\n")
    checkout_code(code, str(repository), newer)
    assert _git(code, "rev-parse", "HEAD") == newer


def test_his_script_gets_whole_paths_even_for_a_relative_data_folder(
    tmp_path: Path, origin: tuple[Path, list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    # His script runs in his code's folder, so a relative PN_DATA_DIR would
    # put the bundle inside it.
    monkeypatch.chdir(tmp_path)
    run = FakeRun()
    fetch.fetch_mayner(
        Path("data"), run=run, download=FakeManifest(), now=lambda: FIRST
    )
    [(command, cwd, env)] = run.calls
    folder = tmp_path / "data" / "mayner"
    assert env["PN_DATA_DIR"] == str(folder / "bundle")
    assert command[2] == str(folder / "code" / "download_data.py")
    assert cwd == folder / "code"


def test_a_failed_fetch_of_mayners_data_says_to_run_it_again(
    tmp_path: Path, origin: tuple[Path, list[str]]
) -> None:
    record = tmp_path / "mayner" / RECORD_FILE
    server = FakeManifest()
    fetch.fetch_mayner(tmp_path, run=FakeRun(), download=server, now=lambda: FIRST)
    # A MANIFEST that cannot be downloaded changes nothing.
    offline = FakeManifest(fail=True)
    with pytest.raises(OSError, match="could not download"):
        fetch.fetch_mayner(tmp_path, run=FakeRun(), download=offline, now=lambda: LATER)
    assert read_record(record.parent)["downloaded_at"] == "2026-11-12T20:31:05Z"
    # His script failing halfway leaves no record of a finished download.
    with pytest.raises(RuntimeError, match="exit code 1.*`pushlab fetch mayner` again"):
        fetch.fetch_mayner(tmp_path, run=FakeRun(1), download=server, now=lambda: LATER)
    assert not record.exists()


def test_run_git_returns_what_git_printed_or_raises_with_its_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    assert run_git(["git", "--version"]).startswith("git version")
    with pytest.raises(RuntimeError, match="not a git repository"):
        run_git(["git", "-C", str(tmp_path), "rev-parse", "HEAD"])


def test_run_command_runs_in_its_folder_with_its_environment(
    tmp_path: Path, capfd: pytest.CaptureFixture
) -> None:
    script = "import os, sys; print(os.getcwd()); sys.exit(int(os.environ['STATUS']))"
    env = {**os.environ, "STATUS": "3"}
    assert run_command([sys.executable, "-c", script], cwd=tmp_path, env=env) == 3
    assert Path(capfd.readouterr().out.strip()).resolve() == tmp_path.resolve()


def test_docstring_examples_run() -> None:
    result = doctest.testmod(mayner_code)
    assert result.attempted > 0
    assert result.failed == 0
