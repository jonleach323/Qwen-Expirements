"""Mayner's data, for `pushlab fetch mayner`: his code, his script, the record.

fetch_mayner downloads Will Mayner's persona-neurons bundle with his own
download script, as decision 0020 says: the files the importers read, 279
files of about 105 MB, or with all_files his whole bundle, 638 files of 1.02
GB. Its steps:

- checkout_code makes a folder a git checkout of his repository, from
  github.com, at the pinned commit, his files unchanged, as his own
  persona_neurons/upstream.py does for the code it pins;
- mayner_command is the command that runs his download_data.py, unchanged,
  with a timeout on each connection it opens (SCRIPT_TIMEOUT), and
  run_command runs it with its output shown, in the environment script_env
  gives it: only what it needs, so no secret of pushlab's reaches it.

His script downloads from huggingface.co, which serves the files through
the hosts under hf.co, such as us.aws.cdn.hf.co (its CDN): the pod's network
must reach those too. git commands go through run_git, so tests can hand in
a fake instead. The steps it shares with the library's fetch are in fetch.py
and fetch_http.py.
"""

import os
import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from pushlab.discovery.fetch import (
    Clock,
    credit_line,
    fetch_again,
    sha256_of,
    source_fields,
    utc_now,
    utc_stamp,
)
from pushlab.discovery.fetch_http import Download, download_whole, http_download
from pushlab.discovery.sources import (
    MAYNER,
    MAYNER_BUNDLE,
    MAYNER_CODE,
    MAYNER_COMMIT,
    MAYNER_FILES,
    MAYNER_REPOSITORY,
    RECORD_FILE,
    source_dir,
    write_record,
)

# Runs a git command; returns what it printed, raising RuntimeError if it fails.
Git = Callable[[list[str]], str]
# Runs a command, with cwd= and env= keywords; returns its exit code.
Run = Callable[..., int]

# Seconds each connection of his script waits for the server before it gives
# up: his script sets no timeout, so a stalled download would hang forever.
SCRIPT_TIMEOUT = 60
# Python code that runs his script, unchanged, its path the first argument,
# with SCRIPT_TIMEOUT on every connection. runpy sets __file__ to the path,
# from which his script puts its own folder on sys.path, as when run directly.
WRAPPER = (
    "import runpy, socket, sys; "
    f"socket.setdefaulttimeout({SCRIPT_TIMEOUT}); "
    "sys.argv = sys.argv[1:]; "
    'runpy.run_path(sys.argv[0], run_name="__main__")'
)
# The variables of pushlab's environment that his script gets, when set:
# where programs are, the home folder, language, temporary files and time
# zone, and the proxy and certificate settings its downloads may need. No
# others, so no secret of pushlab's, such as the pod's HF_TOKEN, reaches it.
SCRIPT_ENV = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "TMPDIR",
    "TZ",
    "http_proxy",
    "https_proxy",
    "no_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
)


def run_git(command: Sequence[str]) -> str:
    """Run a git command, its output captured; return what it printed.

    A command that fails raises RuntimeError with git's own message.

    Example:
        >>> run_git(["git", "--version"]).startswith("git version")
        True
    """
    try:
        result = subprocess.run(
            list(command), check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as error:
        said = (error.stderr or error.stdout or "").strip()
        raise RuntimeError(
            f"{shlex.join(command)} failed: {said}. Fix what git says and run "
            "the fetch again"
        ) from None
    # Only the last newline goes: `status --porcelain` starts lines with spaces.
    return result.stdout.rstrip("\n")


def _has_commit(code: Path, commit: str, git: Git) -> bool:
    """True if the checkout in code already has commit."""
    try:
        git(["git", "-C", str(code), "cat-file", "-e", f"{commit}^{{commit}}"])
    except RuntimeError:
        return False
    return True


def checkout_code(
    code: Path, repository: str, commit: str, git: Git | None = None
) -> None:
    """Make code a checkout of repository at commit, with its files unchanged.

    It clones repository into code if code is not a checkout yet, or fetches
    from origin if the checkout lacks the commit. A checkout in which a file
    git tracks was changed is refused, naming the files: his code must run
    as he wrote it. Files git does not track are left alone. Then it checks
    the commit out, detached, and makes sure HEAD is that commit. git runs
    each git command (run_git by default).

    Example:
        >>> from pushlab.discovery.sources import MAYNER_COMMIT, MAYNER_REPOSITORY
        >>> code = Path("/workspace/data/mayner/code")
        >>> checkout_code(code, MAYNER_REPOSITORY, MAYNER_COMMIT)  # doctest: +SKIP
    """
    git = git or run_git
    here = ["git", "-C", str(code)]
    if not (code / ".git").exists():
        code.parent.mkdir(parents=True, exist_ok=True)
        git(["git", "clone", "--quiet", repository, str(code)])
    elif not _has_commit(code, commit, git):
        git([*here, "fetch", "--quiet", "origin"])
    changed = git([*here, "status", "--porcelain", "--untracked-files=no"])
    names = [line[3:] for line in changed.splitlines() if line.strip()]
    if names:
        raise RuntimeError(
            f"{code} has changes to files of his code: {', '.join(names)}. His "
            f"code must run unchanged: delete {code} (it is cloned again) or "
            "undo the changes, then run the fetch again"
        )
    git([*here, "-c", "advice.detachedHead=false", "checkout", "--quiet", commit])
    head = git([*here, "rev-parse", "HEAD"]).strip()
    if head != commit:
        raise RuntimeError(
            f"{code} is at commit {head}, not {commit}; delete {code} and run "
            "the fetch again"
        )


def mayner_command(code: Path, all_files: bool = False) -> list[str]:
    """The command that runs his download script from the checkout in code.

    It runs his download_data.py, unchanged, through WRAPPER, which gives
    each connection it opens a timeout of SCRIPT_TIMEOUT seconds, with
    pushlab's own Python in isolated mode (-I: no PYTHON* variables, nothing
    from the current folder on its path), for the files of his bundle the
    importers read (MAYNER_FILES, his --only prefixes), or all of them with
    all_files. The script's own folder is its only addition to the path; it
    needs only the standard library.

    Example:
        >>> command = mayner_command(Path("/data/mayner/code"))
        >>> command[0] == sys.executable, command[3] == WRAPPER
        (True, True)
        >>> command[1:3], command[4]
        (['-I', '-c'], '/data/mayner/code/download_data.py')
        >>> command[5] == MAYNER.url
        True
        >>> command[6:8]
        ['--only', 'split/top_neurons.parquet']
        >>> "--only" in mayner_command(Path("/data/mayner/code"), all_files=True)
        False
    """
    script = [sys.executable, "-I", "-c", WRAPPER, str(code / "download_data.py")]
    command = [*script, MAYNER.url]
    return command if all_files else [*command, "--only", *MAYNER_FILES]


def script_env(
    bundle: Path, environ: Mapping[str, str] | None = None
) -> dict[str, str]:
    """The environment his script runs in: PN_DATA_DIR, the folder it fills,
    and those of SCRIPT_ENV that environ (os.environ by default) sets.

    Example:
        >>> environ = {"HOME": "/root", "HF_TOKEN": "made-up", "TZ": "UTC"}
        >>> script_env(Path("/data/mayner/bundle"), environ)
        {'HOME': '/root', 'TZ': 'UTC', 'PN_DATA_DIR': '/data/mayner/bundle'}
    """
    environ = os.environ if environ is None else environ
    kept = {name: environ[name] for name in SCRIPT_ENV if name in environ}
    return kept | {"PN_DATA_DIR": str(bundle)}


def run_command(command: Sequence[str], *, cwd: Path, env: Mapping[str, str]) -> int:
    """Run a command in folder cwd with environment env; return its exit code.

    Its output goes straight to the terminal, so his script's progress shows.

    Example:
        >>> run_command([sys.executable, "-c", "exit(3)"], cwd=Path("/"), env={})
        3
    """
    return subprocess.run(list(command), cwd=cwd, env=dict(env), check=False).returncode


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
    refused (checkout_code). His bundle's MANIFEST, about 63 KB, is
    downloaded into bundle/ every time, for its SHA-256. Then his
    download_data.py (mayner_command, in the environment script_env gives
    it) copies into bundle/ the files the importers read (MAYNER_FILES), or
    all of them with all_files, checking each against the MANIFEST and
    skipping those already there. If it fails, RuntimeError says that files
    which failed his check may be left in place, and to run the fetch again
    before `pushlab candidates`.

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
    now = now or utc_now
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
    status = run(command, cwd=code, env=script_env(bundle))
    if status != 0:
        again = fetch_again(MAYNER.name, root, *(["--all"] if all_files else []))
        raise RuntimeError(
            f"his download script stopped with exit code {status}. The files "
            "that matched his MANIFEST are kept; files that failed his check may "
            "be left in place, and are downloaded again on the next run. If "
            "downloads failed or timed out, check that this machine reaches "
            "huggingface.co and the hosts under hf.co, such as us.aws.cdn.hf.co, "
            "through which Hugging Face serves the files. Finish the fetch before "
            f"running `uv run pushlab candidates`: run `{again}` again"
        )
    record = source_fields(MAYNER) | {
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
    print(credit_line(MAYNER))
    return record
