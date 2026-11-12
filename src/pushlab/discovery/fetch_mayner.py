"""Mayner's code, for `pushlab fetch mayner`: its checkout and his script.

fetch.fetch_mayner downloads Will Mayner's persona-neurons bundle with his
own download script, as decision 0020 says. These are its steps that deal
with his code:

- checkout_code makes a folder a git checkout of his repository at the
  pinned commit, his files unchanged, as his own persona_neurons/upstream.py
  does for the code it pins;
- mayner_command is the command that runs his download_data.py, and
  run_command runs it with its output shown.

git commands go through run_git, so tests can hand in a fake instead.
"""

import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from pushlab.discovery.sources import MAYNER, MAYNER_FILES

# Runs a git command; returns what it printed, raising RuntimeError if it fails.
Git = Callable[[list[str]], str]
# Runs a command, with cwd= and env= keywords; returns its exit code.
Run = Callable[..., int]


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

    It runs with pushlab's own Python in isolated mode (-I: no PYTHON*
    variables, nothing from the current folder on its path), for the files
    of his bundle the importers read (MAYNER_FILES, his --only prefixes),
    or all of them with all_files. The script's own folder is its only
    addition to the path; it needs only the standard library.

    Example:
        >>> command = mayner_command(Path("/data/mayner/code"))
        >>> command[0] == sys.executable
        True
        >>> command[1:3]
        ['-I', '/data/mayner/code/download_data.py']
        >>> command[3] == MAYNER.url
        True
        >>> command[4:6]
        ['--only', 'split/top_neurons.parquet']
        >>> "--only" in mayner_command(Path("/data/mayner/code"), all_files=True)
        False
    """
    command = [sys.executable, "-I", str(code / "download_data.py"), MAYNER.url]
    return command if all_files else [*command, "--only", *MAYNER_FILES]


def run_command(command: Sequence[str], *, cwd: Path, env: Mapping[str, str]) -> int:
    """Run a command in folder cwd with environment env; return its exit code.

    Its output goes straight to the terminal, so his script's progress shows.

    Example:
        >>> run_command([sys.executable, "-c", "exit(3)"], cwd=Path("/"), env={})
        3
    """
    return subprocess.run(list(command), cwd=cwd, env=dict(env), check=False).returncode
