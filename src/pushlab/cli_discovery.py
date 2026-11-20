"""The options of the commands that gather persona candidates (decision 0020):
pushlab fetch, which downloads a source into the data folder, and pushlab
candidates, which makes one list of both sources' candidates.

pushlab.cli_parser builds the whole parser from these and the others. Only
light modules are imported here, so that every --help answers at once.
"""

import argparse
from pathlib import Path

from pushlab.discovery.sources import LIBRARY, LIBRARY_FILES, MAYNER, POD_DATA_DIR
from pushlab.models.registry import REGISTRY

# The model whose neurons both sources list: the library's files under
# /data/qwen/ and Mayner's data (mayner_import.MODEL_ID) are Qwen3-8B's.
SOURCES_MODEL_ID = "Qwen/Qwen3-8B"


def models_with_sources() -> list[str]:
    """The registry names of the model that the persona sources describe.

    Example:
        >>> models_with_sources()
        ['qwen3-8b']
    """
    return sorted(
        name for name, spec in REGISTRY.items() if spec.hf_id == SOURCES_MODEL_ID
    )


def add_data_dir(command: argparse.ArgumentParser) -> None:
    """Add the --data-dir option that fetch and candidates share.

    Example:
        >>> parser = argparse.ArgumentParser()
        >>> add_data_dir(parser)
        >>> parser.parse_args(["--data-dir", "data"]).data_dir
        PosixPath('data')
    """
    command.add_argument(
        "--data-dir",
        type=Path,
        metavar="DIR",
        help="the data folder, DATA_DIR (default: $PUSHLAB_DATA_DIR, else "
        f"{POD_DATA_DIR})",
    )


def add_fetch(commands: argparse._SubParsersAction) -> None:
    """Add `pushlab fetch library` and `pushlab fetch mayner` to commands.

    Example:
        >>> parser = argparse.ArgumentParser()
        >>> add_fetch(parser.add_subparsers(dest="command"))
        >>> args = parser.parse_args(["fetch", "mayner", "--all"])
        >>> args.source, args.all_files, args.refresh
        ('mayner', True, False)
    """
    fetch = commands.add_parser(
        "fetch",
        help="download a source of persona candidates into the data folder, on "
        "the GPU pod",
        description="Download one of the two sources of persona candidates for "
        "Qwen3-8B into its folder of the data folder, DATA_DIR "
        f"($PUSHLAB_DATA_DIR, else {POD_DATA_DIR}): `pushlab fetch library` "
        "downloads the Neuron Persona Library's files into DATA_DIR/library/qwen, "
        "and `pushlab fetch mayner` Will Mayner's persona-neurons data into "
        "DATA_DIR/mayner. Files already there are skipped, and FETCHED.json "
        "beside them records where each came from and when. The downloads are "
        "large, so run them on the GPU pod; `pushlab candidates` then reads them.",
    )
    sources = fetch.add_subparsers(dest="source", required=True)
    library = sources.add_parser(
        "library",
        help="the Neuron Persona Library's three files for Qwen3-8B, about 26 MB",
        description="Download the Neuron Persona Library's files for Qwen3-8B, "
        f"{', '.join(LIBRARY_FILES)}, from {LIBRARY.url}, about 26 MB in all, "
        "into DATA_DIR/library/qwen. A file already there is skipped unless "
        "--refresh. The files it downloads replace the old ones together, once "
        "all have arrived whole and are the JSON expected, so a failed fetch "
        "changes nothing and a web page served instead is refused. FETCHED.json "
        "there "
        "records where each file came from and when: its URL, size, SHA-256, "
        "download time (UTC) and the server's Last-Modified. The library states "
        "no licence and asks to be cited, so its data stays out of git. Run it "
        "on the GPU pod.",
    )
    add_data_dir(library)
    library.add_argument(
        "--refresh",
        action="store_true",
        help="download every file again, even those already there",
    )
    mayner = sources.add_parser(
        "mayner",
        help="Will Mayner's persona-neurons data, with his own download script, "
        "about 105 MB",
        description=f"Download Will Mayner's persona-neurons data ({MAYNER.url}) "
        "into DATA_DIR/mayner with his own download script. His code is checked "
        "out with git, at a pinned commit, into DATA_DIR/mayner/code; then his "
        "download_data.py copies the files of his bundle that `pushlab "
        "candidates` reads, 279 files of about 105 MB (with --all his whole "
        "bundle, 638 files of about 1 GB), into DATA_DIR/mayner/bundle, checking "
        "each against his MANIFEST "
        "and skipping those already there. FETCHED.json in DATA_DIR/mayner "
        "records where they came from and when, with the commit and the command "
        f"that ran. His data is under {MAYNER.licence}: credit him wherever it is "
        "used, as `pushlab candidates` does on each row of its list. Run it on "
        "the GPU pod; it needs git and access to github.com, huggingface.co and "
        "Hugging Face's file servers (hosts under hf.co, such as "
        "us.aws.cdn.hf.co).",
    )
    add_data_dir(mayner)
    mayner.add_argument(
        "--all",
        dest="all_files",
        action="store_true",
        help="download his whole bundle, about 1 GB, not only the files `pushlab "
        "candidates` reads",
    )
    # Each source's option on both, so that every fetch's arguments have both.
    library.set_defaults(all_files=False)
    mayner.set_defaults(refresh=False)


def add_candidates(commands: argparse._SubParsersAction) -> None:
    """Add `pushlab candidates` and its options to commands.

    Example:
        >>> parser = argparse.ArgumentParser()
        >>> add_candidates(parser.add_subparsers(dest="command"))
        >>> argv = ["candidates", "--model", "qwen3-8b", "--out", "c.csv"]
        >>> args = parser.parse_args(argv)
        >>> args.out, args.include_sensitive, args.overwrite
        (PosixPath('c.csv'), False, False)
    """
    candidates = commands.add_parser(
        "candidates",
        help="one list of both sources' persona candidates, sensitive ones left out",
        description="Make one list of persona candidates from the two sources "
        "that `pushlab fetch` downloaded into the data folder: each persona of "
        "the Neuron Persona Library, one neuron pushed one way, and each of Will "
        "Mayner's roles as sets of neurons, its top 10 and its top 50 unless "
        "--set-sizes says otherwise. Every "
        "unit is checked against the model's shape. The personas a source calls "
        "sensitive are dropped, unless --include-sensitive, and then the "
        "duplicate names within each source, keeping the best of each (the "
        "library's verified pool first, then the more coherent). Each row of the "
        "CSV file says where its candidate came from: the source, the file and "
        "the place in it, its URL and download time, and the source's licence "
        "and credit. It prints how many candidates each source gave, dropped and "
        "kept, and each source's credit. It loads no model.",
    )
    candidates.add_argument(
        "--model",
        required=True,
        choices=models_with_sources(),
        help="registry name of the model the candidates are for: one the sources "
        "list the neurons of",
    )
    candidates.add_argument(
        "--out",
        required=True,
        type=Path,
        metavar="FILE",
        help="the CSV file to write, such as units/candidates_qwen3-8b.csv, which "
        "git ignores: the library's data stays out of git",
    )
    add_data_dir(candidates)
    candidates.add_argument(
        "--include-sensitive",
        action="store_true",
        help="keep the personas a source calls sensitive; their rows say "
        "sensitive true",
    )
    candidates.add_argument(
        "--overwrite", action="store_true", help="replace the file if it exists"
    )
    candidates.add_argument(
        "--set-sizes",
        type=set_sizes,
        metavar="K,K",
        help="how many of each Mayner role's top neurons each of its sets takes, "
        "comma-separated, such as 10,50,100 (default: 10,50); his table holds "
        "each role's top 1,000",
    )


def set_sizes(text: str) -> tuple[int, ...]:
    """--set-sizes as numbers: different whole numbers from 1.

    Example:
        >>> set_sizes("10, 50,100")
        (10, 50, 100)
    """
    try:
        sizes = tuple(int(part) for part in text.split(","))
    except ValueError:
        sizes = ()
    if not sizes or min(sizes) < 1 or len(set(sizes)) != len(sizes):
        raise argparse.ArgumentTypeError(
            f"give different whole numbers from 1, such as 10,50,100; not {text!r}"
        )
    return sizes
