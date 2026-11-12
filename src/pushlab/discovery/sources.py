"""Where persona candidates come from: the two sources, their files and credits.

Two sources supply persona candidates for Qwen3-8B (decision 0020):

- the Neuron Persona Library (neuronpersonas.com), whose per-model files for
  Qwen3-8B are under https://neuronpersonas.com/data/qwen/. It states no
  licence and asks to be cited, so none of its data goes into git;
- Will Mayner's persona-neurons data
  (huggingface.co/datasets/wmayner/persona-neurons), under CC BY 4.0, which
  must be credited wherever it is used.

`pushlab fetch` downloads each into its own folder of the data folder,
/workspace/data on the GPU pod (PUSHLAB_DATA_DIR names another), and writes a
record there, FETCHED.json, of where each file came from and when. The
importers read the files and the record, so every candidate says where it
came from.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

POD_DATA_DIR = Path("/workspace/data")
RECORD_FILE = "FETCHED.json"


@dataclass(frozen=True)
class Source:
    """One source of candidates: its name, home page, licence and credit line.

    folder is its folder in the data folder; url is where its files are
    served from. credit is how the source asks to be credited, which a fetch
    record of its unchanged files carries; adapted, if any, says what a list
    made from them changes, as CC BY 4.0 asks of adapted material.
    """

    name: str
    title: str
    folder: str
    url: str
    licence: str
    credit: str
    adapted: str = ""

    @property
    def list_credit(self) -> str:
        """The credit a candidate list made from the source gives it.

        Example:
            >>> LIBRARY.list_credit == LIBRARY.credit
            True
            >>> MAYNER.list_credit.endswith(MAYNER.adapted)
            True
        """
        return f"{self.credit}; {self.adapted}" if self.adapted else self.credit


# The library's own citation, from its "Cite this" section.
LIBRARY = Source(
    name="library",
    title="Neuron Persona Library",
    folder="library/qwen",
    url="https://neuronpersonas.com/data/qwen",
    licence="none stated; the library asks to be cited",
    credit=(
        "Fonseca Rivera, J. (2026). Neuron Persona Library: What Language Models "
        "Say They Are When One Neuron Is Pushed. https://neuronpersonas.com "
        "(Qwen3-8B files)"
    ),
)
MAYNER = Source(
    name="mayner",
    title="persona-neurons",
    folder="mayner",
    url="https://huggingface.co/datasets/wmayner/persona-neurons",
    licence="CC BY 4.0",
    credit=(
        "Will Mayner, persona-neurons (Qwen3-8B), "
        "huggingface.co/datasets/wmayner/persona-neurons, CC BY 4.0 "
        "(https://creativecommons.org/licenses/by/4.0/)"
    ),
    adapted=(
        "adapted by pushlab: each role's top 10 and top 50 neurons as sets "
        "sized 4 x |v_r| from his vectors, with his judge's in-role share"
    ),
)
SOURCES = {source.name: source for source in (LIBRARY, MAYNER)}

# The library's files for Qwen3-8B that the importer reads.
LIBRARY_FILES = ("library.json", "combos.json", "sensitive.json")

# Mayner's code, whose download script fetches his bundle: the commit is
# pinned, as his own code pins the Assistant Axis repository's.
MAYNER_REPOSITORY = "https://github.com/wmayner/persona-neurons.git"
MAYNER_COMMIT = "64c9d36f5bdde14bbd5b736cbd83eb6015e031eb"
# Inside MAYNER.folder: his code, and his bundle's files at their own paths.
MAYNER_CODE = "code"
MAYNER_BUNDLE = "bundle"
# The bundle's files the importer reads (his download script's --only
# prefixes): each role's top neurons, the role vectors that give the exact
# |v_r|, and his scores for sets of 10 and 50 neurons.
MAYNER_FILES = (
    "split/top_neurons.parquet",
    "split/concentration_neurons.parquet",
    "vectors/without/",
    "role_steering_sizes/judged_replies.parquet",
)


def data_dir() -> Path:
    """The data folder: PUSHLAB_DATA_DIR if set, else the pod's /workspace/data.

    Example:
        >>> data_dir()  # doctest: +SKIP
        PosixPath('/workspace/data')
    """
    return Path(os.environ.get("PUSHLAB_DATA_DIR") or POD_DATA_DIR)


def source_dir(source: Source, root: Path | None = None) -> Path:
    """The source's folder in the data folder (root, or data_dir()).

    Example:
        >>> source_dir(LIBRARY, Path("/workspace/data"))
        PosixPath('/workspace/data/library/qwen')
    """
    return (root if root is not None else data_dir()) / source.folder


def read_record(folder: Path) -> dict[str, Any]:
    """The fetch record in folder, or {} if there is none yet.

    Example:
        >>> read_record(Path("/no/such/folder"))
        {}
    """
    path = folder / RECORD_FILE
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as error:
        raise ValueError(f"{path} is not valid JSON: {error}") from None
    if not isinstance(record, dict):
        raise ValueError(f"{path} must hold a JSON object")
    return record


def write_record(folder: Path, record: dict[str, Any]) -> Path:
    """Write the fetch record into folder, in one step; return its path.

    Example:
        >>> write_record(folder, {"source": "mayner"})  # doctest: +SKIP
        PosixPath('/workspace/data/mayner/FETCHED.json')
    """
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / RECORD_FILE
    partial = path.with_name(path.name + ".part")
    text = json.dumps(record, indent=2, sort_keys=True, allow_nan=False)
    partial.write_text(text + "\n", encoding="utf-8")
    partial.replace(path)
    return path
