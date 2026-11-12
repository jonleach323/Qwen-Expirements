"""`pushlab fetch`: download a source of persona candidates into the data folder.

`pushlab fetch library` runs fetch.fetch_library and `pushlab fetch mayner`
runs fetch.fetch_mayner (decision 0020); both print each step as it goes.
This is the command around them. When the fetch has finished, it says which
of the source's files it downloaded and which were already there and
skipped, telling them apart by each file's size, inode and modification time
before and after, and where the record, FETCHED.json, is. A download, git
step or script that fails is refused with its reason on stderr and exit
status 1; Ctrl-C stops it with exit status 130, the whole files kept.
"""

import sys
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

from pushlab.discovery.fetch import fetch_library, fetch_mayner
from pushlab.discovery.sources import (
    LIBRARY,
    LIBRARY_FILES,
    MAYNER,
    MAYNER_BUNDLE,
    MAYNER_FILES,
    RECORD_FILE,
    SOURCES,
    source_dir,
)

# The most file names a summary lists; for more, it gives their number.
NAMES_SHOWN = 5
# His bundle's list of its files, which fetch_mayner downloads every time.
MANIFEST = "MANIFEST"
# What tells a new copy of a file from the one there before: its size, inode
# and modification time.
FileState = tuple[int, int, int]


def _refuse(message: str) -> int:
    print(f"pushlab fetch: {message}", file=sys.stderr)
    return 1


def files_now(folder: Path, wanted: Callable[[str], bool]) -> dict[str, FileState]:
    """The files under folder whose path in it wanted keeps, as they are now.

    Each file's path in folder, such as split/top_neurons.parquet, maps to
    its size, inode and modification time, which change when it is
    downloaded again.

    Example:
        >>> files_now(Path("/no/such/folder"), lambda name: True)
        {}
    """
    if not folder.is_dir():
        return {}
    found = {}
    for path in sorted(folder.rglob("*")):
        name = path.relative_to(folder).as_posix()
        if path.is_file() and wanted(name):
            stat = path.stat()
            found[name] = (stat.st_size, stat.st_ino, stat.st_mtime_ns)
    return found


def mayner_wanted(all_files: bool = False) -> Callable[[str], bool]:
    """Which files of his bundle a fetch of Mayner's data asks his script for.

    Those under MAYNER_FILES, his script's --only prefixes, or with
    all_files every one; never his MANIFEST, which is downloaded every time.

    Example:
        >>> wanted = mayner_wanted()
        >>> wanted("vectors/without/pirate.pt"), wanted("split/other.parquet")
        (True, False)
        >>> mayner_wanted(all_files=True)("MANIFEST")
        False
    """

    def wanted(name: str) -> bool:
        return name != MANIFEST and (all_files or name.startswith(MAYNER_FILES))

    return wanted


def missing_prefixes(names: Iterable[str], prefixes: Sequence[str]) -> list[str]:
    """The prefixes that none of the names starts with, in their order.

    Example:
        >>> missing_prefixes(["split/top_neurons.parquet"], ["split/", "vectors/"])
        ['vectors/']
    """
    names = list(names)
    return [
        prefix for prefix in prefixes if not any(n.startswith(prefix) for n in names)
    ]


def listed(names: Sequence[str]) -> str:
    """Files in a sentence: their names, up to NAMES_SHOWN, else their number.

    Example:
        >>> listed(["library.json", "combos.json", "sensitive.json"])
        'library.json, combos.json and sensitive.json'
        >>> listed([])
        'nothing'
        >>> listed([f"vectors/without/role{i}.pt" for i in range(276)])
        '276 files'
    """
    if not names:
        return "nothing"
    if len(names) > NAMES_SHOWN:
        return f"{len(names):,} files"
    if len(names) == 1:
        return names[0]
    return f"{', '.join(names[:-1])} and {names[-1]}"


def summary(
    source: str,
    folder: Path,
    downloaded: Sequence[str],
    skipped: Sequence[str],
    record: Path,
) -> list[str]:
    """What `pushlab fetch` prints once a fetch has finished.

    downloaded are the source's files that were downloaded into folder,
    skipped those already there, left as they were, and record the path of
    FETCHED.json.

    Example:
        >>> folder = Path("data/qwen")
        >>> record = folder / "FETCHED.json"
        >>> lines = summary(
        ...     "library", folder, ["library.json"], ["combos.json"], record
        ... )
        >>> print("\\n".join(lines))
        library: downloaded library.json into data/qwen
        library: skipped combos.json, already there
        library: the record of where they came from and when is data/qwen/FETCHED.json
    """
    lines = [f"{source}: downloaded {listed(downloaded)} into {folder}"]
    if skipped:
        lines.append(f"{source}: skipped {listed(skipped)}, already there")
    lines.append(f"{source}: the record of where they came from and when is {record}")
    return lines


def _reason(error: Exception) -> str:
    """Why the fetch failed, as the command says it."""
    if isinstance(error, FileNotFoundError) and error.filename == "git":
        return (
            "git is not installed, and `pushlab fetch mayner` needs it to check "
            "out Mayner's code; install git, then run the fetch again"
        )
    return str(error)


def run_fetch(
    source: str,
    data_dir: Path | None = None,
    *,
    refresh: bool = False,
    all_files: bool = False,
) -> int:
    """Download source, "library" or "mayner", into data_dir; return the exit status.

    data_dir is the data folder ($PUSHLAB_DATA_DIR, else /workspace/data, by
    default). refresh, for the library, downloads its files again even if
    they are there; all_files, for Mayner's data, downloads his whole bundle.
    Returns 0 once the record is written, with a summary of the files
    downloaded and skipped; 1, with the reason on stderr, when the fetch
    fails or is refused; and 130 after Ctrl-C.

    Example:
        >>> run_fetch("library", Path("/workspace/data"))  # doctest: +SKIP
        0
    """
    if source not in SOURCES:
        return _refuse(f"there is no source {source!r}; give library or mayner")
    if refresh and source != LIBRARY.name:
        return _refuse(
            "--refresh goes with `pushlab fetch library`; Mayner's script "
            "downloads again only the files that do not match his MANIFEST"
        )
    if all_files and source != MAYNER.name:
        return _refuse("--all goes with `pushlab fetch mayner`")
    folder = source_dir(SOURCES[source], data_dir)
    if source == LIBRARY.name:
        files, wanted = folder, lambda name: name in LIBRARY_FILES
    else:
        files, wanted = folder / MAYNER_BUNDLE, mayner_wanted(all_files)
    before = files_now(files, wanted)
    try:
        if source == LIBRARY.name:
            fetch_library(data_dir, refresh=refresh)
        else:
            fetch_mayner(data_dir, all_files=all_files)
    except KeyboardInterrupt:
        print(
            "pushlab fetch: stopped by Ctrl-C. The files downloaded whole are "
            "kept; run the same command again to fetch the rest",
            file=sys.stderr,
        )
        return 130
    except (OSError, ValueError, RuntimeError) as error:
        return _refuse(_reason(error))
    after = files_now(files, wanted)
    downloaded = [name for name, state in after.items() if before.get(name) != state]
    skipped = [name for name, state in after.items() if before.get(name) == state]
    for line in summary(source, files, downloaded, skipped, folder / RECORD_FILE):
        print(line)
    if source == MAYNER.name and not all_files:
        for prefix in missing_prefixes(after, MAYNER_FILES):
            print(
                f"pushlab fetch: warning: his bundle in {files} has no file under "
                f"{prefix}, although his script was asked for it; his MANIFEST "
                "may have moved it",
                file=sys.stderr,
            )
    return 0
