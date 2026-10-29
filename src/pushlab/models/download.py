"""Download model weights from Hugging Face once, for the GPU pod.

scripts/setup_pod.sh runs it as

    python -m pushlab.models.download configs/models.yaml /workspace/models

Each model named in the YAML file goes to its own folder, such as
/workspace/models/qwen3.5-4b. The registry says where the model comes from.
A model whose folder already holds a finished download of the same Hugging
Face ID and revision is skipped, so running it again only fetches what is
missing. After an interruption, the files that finished are kept and a file
that was still downloading starts again. Hugging Face reads the token from
the HF_TOKEN environment variable.
"""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

import yaml
from huggingface_hub import HfApi, get_cached_repo_tree, snapshot_download
from huggingface_hub.errors import CachedRepoTreeNotFoundError, HfHubHTTPError

from pushlab.models.registry import REGISTRY, ModelSpec

# Written into a model's folder once its download has finished.
MARKER = ".pushlab-download.json"

# Downloads one model into a folder and returns the commit it got.
Downloader = Callable[[ModelSpec, Path], str]


class IncompleteDownloadError(OSError):
    """Hugging Face handed back a model folder that is missing files."""


def parse_model_names(text: str) -> list[str]:
    """Return the model names listed under `models:` in a YAML document.

    Every name must be a registry entry that comes from Hugging Face, so the
    tiny test model is refused. A name listed twice is kept once.

    Example:
        >>> parse_model_names("models: [qwen3.5-4b, qwen3-8b]")
        ['qwen3.5-4b', 'qwen3-8b']
    """
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ValueError(f"not valid YAML: {error}") from error
    names = data.get("models") if isinstance(data, dict) else None
    if not names or not isinstance(names, list):
        raise ValueError("expected a list of model names under 'models:'")
    for name in names:
        if not isinstance(name, str) or name not in REGISTRY:
            known = ", ".join(sorted(REGISTRY))
            raise ValueError(f"{name!r} is not in the model registry ({known})")
        if REGISTRY[name].kind == "test":
            raise ValueError(f"{name!r} is built by the tests, not downloaded")
    return list(dict.fromkeys(names))


def is_downloaded(spec: ModelSpec, folder: Path) -> bool:
    """True if folder holds a finished download of spec's ID and revision.

    Example:
        >>> is_downloaded(REGISTRY["qwen3.5-4b"], Path("/no/such/folder"))
        False
    """
    try:
        record = json.loads((folder / MARKER).read_text())
    except (OSError, ValueError):
        return False
    return (
        isinstance(record, dict)
        and record.get("hf_id") == spec.hf_id
        and record.get("revision") == spec.revision
    )


def remove_unfinished_files(folder: Path) -> int:
    """Delete the partial files an interrupted download left; return how many.

    Hugging Face writes each file to a temporary .incomplete file first and
    deletes it when a download fails, but a killed process, such as one on a
    stopped pod, leaves it behind. Call it only while nothing else downloads
    into the folder.

    Example:
        >>> remove_unfinished_files(Path("/no/such/folder"))
        0
    """
    partial = list((folder / ".cache" / "huggingface").rglob("*.incomplete"))
    for path in partial:
        path.unlink(missing_ok=True)
    return len(partial)


def hub_download(spec: ModelSpec, folder: Path) -> str:
    """Download spec's model from Hugging Face into folder; return the commit.

    The revision is turned into a commit first, so every file comes from the
    same commit even if the revision moves on during the download. When the
    Hub cannot list that commit's files, snapshot_download returns the folder
    as it is, perhaps half full, with only a warning. So the folder is checked
    against the commit's file list before the download counts as finished.

    Example:
        >>> spec = REGISTRY["qwen3.5-4b"]
        >>> hub_download(spec, Path("/workspace/models/qwen3.5-4b"))  # doctest: +SKIP
        '...'
    """
    commit = HfApi().model_info(spec.hf_id, revision=spec.revision).sha
    if commit is None:
        raise ValueError(f"Hugging Face gave no commit for {spec.hf_id}")
    if removed := remove_unfinished_files(folder):
        print(
            f"{spec.name}: removed {removed} partial files of an interrupted download"
        )
    snapshot_download(spec.hf_id, revision=commit, local_dir=folder)
    try:
        files = get_cached_repo_tree(spec.hf_id, revision=commit, local_dir=folder)
    except CachedRepoTreeNotFoundError as error:
        raise IncompleteDownloadError(
            f"Hugging Face did not list the files of {spec.hf_id} at {commit}"
        ) from error
    missing = [file.path for file in files if not (folder / file.path).is_file()]
    if missing:
        raise IncompleteDownloadError(
            f"{len(missing)} files of {spec.hf_id} are missing, such as {missing[0]}"
        )
    return commit


def ensure_models(
    names: Sequence[str], models_dir: Path, download: Downloader = hub_download
) -> list[Path]:
    """Download each named model into models_dir/<name> unless it is there.

    Returns the model folders in the order given. A folder only counts as
    finished once download returns, so an interrupted download is retried.

    Example:
        >>> ensure_models(["qwen3.5-4b"], Path("models"))  # doctest: +SKIP
        qwen3.5-4b: downloading Qwen/Qwen3.5-4B at main into models/qwen3.5-4b
        qwen3.5-4b: finished, commit ...
        [PosixPath('models/qwen3.5-4b')]
    """
    folders = []
    for name in names:
        spec = REGISTRY[name]
        folder = models_dir / name
        folders.append(folder)
        if is_downloaded(spec, folder):
            print(f"{name}: already downloaded into {folder}, skipping")
            continue
        print(f"{name}: downloading {spec.hf_id} at {spec.revision} into {folder}")
        folder.mkdir(parents=True, exist_ok=True)
        commit = download(spec, folder)
        record = {"hf_id": spec.hf_id, "revision": spec.revision, "commit": commit}
        (folder / MARKER).write_text(json.dumps(record, indent=2) + "\n")
        print(f"{name}: finished, commit {commit}")
    return folders


def main(argv: Sequence[str] | None = None) -> int:
    """Download the models listed in a YAML file; return the exit status.

    Example:
        >>> main(["configs/models.yaml", "/workspace/models"])  # doctest: +SKIP
        0
    """
    parser = argparse.ArgumentParser(
        prog="python -m pushlab.models.download",
        description="Download the models listed in a YAML file, skipping "
        "models that are already there.",
    )
    parser.add_argument("config", type=Path, help="such as configs/models.yaml")
    parser.add_argument("models_dir", type=Path, help="gets one folder per model")
    args = parser.parse_args(argv)
    try:
        names = parse_model_names(args.config.read_text())
    except (OSError, ValueError) as error:
        print(f"Cannot use {args.config}: {error}", file=sys.stderr)
        return 1
    try:
        ensure_models(names, args.models_dir)
    except HfHubHTTPError as error:
        print(f"Hugging Face refused the download: {error}", file=sys.stderr)
        print(
            "If it mentions 401, check that HF_TOKEN holds a valid token "
            "(docs/runbooks/pod.md).",
            file=sys.stderr,
        )
        return 1
    except Exception as error:  # a network failure or an unfinished download
        print(f"The download stopped: {type(error).__name__}: {error}", file=sys.stderr)
        print(
            "Check the pod's internet connection and run the script again; "
            "finished files are kept.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
