"""Run folders: where a command saves its runs.parquet and manifest.json.

A command checks its --out folder before it loads a model, so a mistake
costs nothing, and refuses to replace a run that is already there unless it
is given --overwrite.
"""

from pathlib import Path

from pushlab import PROJECT_ROOT
from pushlab.store.records import MANIFEST_FILE, RUNS_FILE


def file_in_the_way(out: Path) -> Path | None:
    """The file, if any, at out or above it that stops out being a folder.

    A link to something missing counts too, since no folder can be made there.

    Example:
        >>> file_in_the_way(Path("/no/such/folder")) is None
        True
    """
    for path in (out, *out.parents):
        if path.is_symlink() or path.exists():  # exists() alone misses a broken link
            return None if path.is_dir() else path
    return None


def already_there(out: Path) -> str:
    """Why a folder that already holds a run is refused.

    Example:
        >>> already_there(Path("runs/a"))  # doctest: +ELLIPSIS
        'runs/a already holds a run; choose another --out, or pass --overwrite ...'
    """
    return (
        f"{out} already holds a run; choose another --out, or pass --overwrite "
        "to replace it"
    )


def holds_a_run(out: Path) -> bool:
    """True if out has a runs.parquet or a manifest.json.

    Example:
        >>> holds_a_run(Path("/no/such/folder"))
        False
    """
    return any((out / name).exists() for name in (RUNS_FILE, MANIFEST_FILE))


def folder_problem(out: Path, overwrite: bool) -> str | None:
    """Why out cannot take a command's runs, or None when it can.

    Example:
        >>> folder_problem(Path("/no/such/folder"), overwrite=False) is None
        True
    """
    if (in_the_way := file_in_the_way(out)) is not None:
        what = "a file" if in_the_way.exists() else "a link to nothing"
        return f"--out must be a folder, but {in_the_way} is {what}"
    if not overwrite and holds_a_run(out):
        return already_there(out)
    return None


def shown_path(path: Path) -> str:
    """path relative to the repository when it is inside it, as manifests name files.

    Example:
        >>> shown_path(PROJECT_ROOT / "prompts" / "smoke.jsonl")
        'prompts/smoke.jsonl'
        >>> shown_path(Path("/workspace/calibration/x.json"))
        '/workspace/calibration/x.json'
    """
    resolved = path.resolve()
    if resolved.is_relative_to(PROJECT_ROOT):
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    return str(path)
