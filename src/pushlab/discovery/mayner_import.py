"""Candidates from Will Mayner's persona-neurons data: each role's top neurons as sets.

Mayner's data (huggingface.co/datasets/wmayner/persona-neurons) is under CC BY
4.0, so it is credited wherever it is used: every row of a candidates file
made from it carries sources.MAYNER.list_credit, his credit and how pushlab
adapted his data. It splits the persona vector of each of 275 roles in
Qwen3-8B into the contributions of the MLP neurons of layers 0 to 18.
`pushlab fetch mayner` copies the files of his data bundle that are read
here into data/mayner/bundle/, at the bundle's own paths, with his MANIFEST
of the bundle's files there too and FETCHED.json beside it, the record of
the download (decision 0020). When the MANIFEST is there, every file it
lists that is read here must still be as it lists it, the right size and
SHA-256 (mayner_manifest.check_bundle); without it, nothing is checked.

import_mayner makes every role of split/top_neurons.parquet, in name order, a
candidate at each set size, 10 and 50 neurons unless told otherwise: the
NeuronSet of the role's top k neurons at Mayner's strength, named such as
coral-reef-top10, which a dose of 1 pushes as his code steers
(pushlab.units.mayner.top_neurons_set; decision 0019). A set's size is 4
times |v_r|, the length of the role's persona vector. It comes from the role
vectors when they are there, the role's and the default's files in
vectors/without/, which give exactly his strength (size_from
"vectors/without"), and from the split's shares otherwise, 0.1 to 0.2% off
his strength (size_from "split shares").

Each candidate keeps Mayner's own measures of its set, for reference, each
None where his files have none:

- mayner_scored and mayner_in_role, from step 6b's
  role_steering_sizes/judged_replies.parquet: of the replies his judge scored
  when he steered with the role's top k neurons (kind "neurons", draw 0), how
  many there are and the share of them scored 2 or 3, from 0 to 1, his
  in-role measure (steps/6_role_steering.py, in_role_summary). Only the
  columns needed are read, never the replies' text. His scores of the top 100
  are in step 6's role_steering/judged_replies.parquet, which is not fetched,
  so a set of 100 has none.
- mayner_share, from step 5's split/concentration_neurons.parquet: the % of
  |v_r| the role's top k neurons contribute.

Every candidate of his data has category "role", pushlab's own label for one
of his roles: his data gives none. His data flags no role as sensitive, so
every candidate from it has sensitive False, and leaving sensitive personas
out drops none of them. Each candidate says where it came from: source_file
mayner/bundle/split/top_neurons.parquet, source_ref the set's own source (the
role and the ranks taken), and from FETCHED.json the file's URL and when the
bundle was downloaded.
"""

import math
import numbers
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from pushlab.discovery.candidates import Candidate
from pushlab.discovery.mayner_manifest import check_bundle
from pushlab.discovery.sources import (
    MAYNER,
    MAYNER_BUNDLE,
    MAYNER_FILES,
    RECORD_FILE,
    read_record,
)
from pushlab.models.registry import REGISTRY
from pushlab.units import mayner
from pushlab.units.mayner_vectors import vector_norm_from_vectors

# The set sizes made of each role unless told otherwise: its top 10 and 50.
SIZES = (10, 50)
# The model whose neurons his data lists (his persona_neurons/config.py).
MODEL_ID = "Qwen/Qwen3-8B"

# The files of his bundle read here, at their paths in the bundle.
TOP_NEURONS_FILE = mayner.TOP_NEURONS_FILE
CONCENTRATION_FILE = "split/concentration_neurons.parquet"
REPLIES_FILE = "role_steering_sizes/judged_replies.parquet"
VECTORS_FOLDER = "vectors/without"
DEFAULT_VECTOR = "default.pt"
# The columns read from his tables: never a reply's text.
REPLY_COLUMNS = ("role", "kind", "k", "draw", "score")
CONCENTRATION_COLUMNS = ("role", "basis", "k", "share")

# His judge scores a reply from 0 to 3; 2 or 3 counts as in role.
TOP_SCORE = 3
IN_ROLE_SCORE = 2

# Where a set's |v_r| came from, its candidate's size_from.
FROM_VECTORS = VECTORS_FOLDER
FROM_SHARES = "split shares"

FETCH_AGAIN = "delete it and download it again with `pushlab fetch mayner`"


def _read(path: Path, columns: tuple[str, ...]) -> pd.DataFrame | None:
    """Only these columns of the parquet table at path, or None if there is none."""
    if not path.is_file():
        return None
    try:
        names = pq.read_schema(path).names
    except (OSError, ValueError) as error:  # pyarrow's ArrowInvalid is a ValueError
        raise ValueError(
            f"{path} cannot be read as a parquet table ({error}); {FETCH_AGAIN}"
        ) from None
    if missing := [column for column in columns if column not in names]:
        raise ValueError(
            f"{path} lacks the column {', '.join(missing)}, so it is not the table "
            f"of Mayner's bundle; {FETCH_AGAIN}"
        )
    return pd.read_parquet(path, columns=list(columns))


def scored_in_role(scores: Iterable[object]) -> tuple[int, float] | None:
    """How many replies have a score, and the share of them scored 2 or 3.

    That share, from 0 to 1, is Mayner's in-role measure
    (steps/6_role_steering.py, in_role_summary): replies his judge could not
    score, None or NaN, are left out. None when no reply has a score.

    Example:
        >>> scored_in_role([3, 2, 1, None, float("nan")])
        (3, 0.6666666666666666)
        >>> scored_in_role([None]) is None
        True
    """
    kept = [score for score in scores if not pd.isna(score)]
    for score in kept:
        if (
            not isinstance(score, numbers.Real)
            or isinstance(score, bool)
            or not 0 <= score <= TOP_SCORE
        ):
            raise ValueError(
                f"a score of Mayner's judge is a number from 0 to {TOP_SCORE}, "
                f"not {score!r}"
            )
    if not kept:
        return None
    return len(kept), sum(score >= IN_ROLE_SCORE for score in kept) / len(kept)


def reply_scores(
    path: Path, sizes: Iterable[int]
) -> dict[tuple[str, int], tuple[int, float]]:
    """Mayner's judge's scores of each role's top k neurons, for each k in sizes.

    path is his role_steering_sizes/judged_replies.parquet (step 6b), one row
    per reply. The replies of a role's top k neurons are its rows of kind
    "neurons", that k and draw 0. For each role and k with a scored reply, the
    result maps (role, k) to scored_in_role of their scores. Only the columns
    REPLY_COLUMNS are read, never the replies' text. Without the file, the
    result is {}.

    Example:
        >>> reply_scores(Path("/no/such/judged_replies.parquet"), SIZES)
        {}
    """
    replies = _read(Path(path), REPLY_COLUMNS)
    if replies is None:
        return {}
    picked = replies[
        (replies["kind"] == "neurons")
        & (replies["draw"] == 0)
        & replies["k"].isin(list(sizes))
    ]
    scores: dict[tuple[str, int], tuple[int, float]] = {}
    for (role, k), group in picked.groupby(["role", "k"], sort=True):
        try:
            counted = scored_in_role(group["score"])
        except ValueError as error:
            raise ValueError(f"{path}, role {role}, k {k}: {error}") from None
        if counted is not None:
            scores[(str(role), int(k))] = counted
    return scores


def concentration_shares(
    path: Path, sizes: Iterable[int]
) -> dict[tuple[str, int], float]:
    """The % of |v_r| each role's top k neurons contribute, for each k in sizes.

    path is his split/concentration_neurons.parquet (step 5), one row per
    role, basis and k; its rows of basis "neurons" map (role, k) to their
    share. A share that is not a finite number is left out. Without the file,
    the result is {}.

    Example:
        >>> concentration_shares(Path("/no/such/concentration.parquet"), SIZES)
        {}
    """
    table = _read(Path(path), CONCENTRATION_COLUMNS)
    if table is None:
        return {}
    picked = table[(table["basis"] == "neurons") & table["k"].isin(list(sizes))]
    rows = zip(picked["role"], picked["k"], picked["share"], strict=True)
    return {
        (str(role), int(k)): float(share)
        for role, k, share in rows
        if math.isfinite(share)
    }


def origin(folder: Path) -> tuple[str, str]:
    """The URL split/top_neurons.parquet came from, and when it was downloaded.

    Both come from FETCHED.json, the record `pushlab fetch mayner` writes in
    folder: its files_url with the file's path added, and its downloaded_at,
    a UTC time such as 2026-11-12T20:31:05Z. Without a record, or without
    them, the URL is the one his bundle is served from and the time is "",
    unknown.

    Example:
        >>> url, downloaded_at = origin(Path("/no/such/folder"))
        >>> url
        'https://huggingface.co/datasets/wmayner/persona-neurons/resolve/main/split/top_neurons.parquet'
        >>> downloaded_at
        ''
    """
    folder = Path(folder)
    record = read_record(folder)
    found = {}
    for key in ("files_url", "downloaded_at"):
        value = record.get(key)
        if value is not None and not isinstance(value, str):
            raise ValueError(
                f"{folder / RECORD_FILE}: {key} must be text, not {value!r}; "
                "download the data again with `pushlab fetch mayner`"
            )
        found[key] = value or ""
    files_url = found["files_url"] or f"{MAYNER.url}/resolve/main"
    return f"{files_url.rstrip('/')}/{TOP_NEURONS_FILE}", found["downloaded_at"]


def vector_norm(bundle: Path, role: str, rows: pd.DataFrame) -> tuple[float, str]:
    """|v_r| of a role, which sizes its sets, and where it came from.

    When bundle/vectors/without/ holds both the role's vector file and the
    default's, |v_r| is the length of their difference at layer 18, exactly
    as Mayner's step 6 takes it (mayner_vectors.vector_norm_from_vectors),
    and it came from "vectors/without". Otherwise it is read back from the
    shares of rows, the role's rows of split/top_neurons.parquet, 0.1 to 0.2%
    off his (mayner.vector_norm_from_shares), and it came from "split shares".

    Example:
        >>> rows = pd.DataFrame(
        ...     [(1, 9, 10, 0.5, 5.0, "pirate", 1.5)],
        ...     columns=mayner.TOP_NEURONS_COLUMNS,
        ... )
        >>> vector_norm(Path("/no/such/bundle"), "pirate", rows)  # 100 * 0.5 / 5.0
        (10.0, 'split shares')
    """
    folder = Path(bundle) / VECTORS_FOLDER
    role_file, default_file = folder / f"{role}.pt", folder / DEFAULT_VECTOR
    # A role's name is a file's name in the folder, never a path out of it.
    if role_file.parent == folder and role_file.is_file() and default_file.is_file():
        return vector_norm_from_vectors(role_file, default_file), FROM_VECTORS
    return mayner.vector_norm_from_shares(rows, role), FROM_SHARES


def _check_model(model: object) -> None:
    """Refuse model unless it is a registry name of Mayner's model, Qwen3-8B."""
    spec = REGISTRY.get(model) if isinstance(model, str) else None
    if spec is None or spec.hf_id != MODEL_ID:
        names = [name for name, other in REGISTRY.items() if other.hf_id == MODEL_ID]
        raise ValueError(
            f"Mayner's data lists the neurons of {MODEL_ID}, so its candidates are "
            f"for {' or '.join(sorted(names))}, not {model!r}"
        )


def _sizes(sizes: object) -> tuple[int, ...]:
    """sizes, refused unless they are different whole numbers from 1."""
    if (
        not isinstance(sizes, tuple | list)
        or not sizes
        or not all(
            isinstance(k, numbers.Integral) and not isinstance(k, bool) and k >= 1
            for k in sizes
        )
        or len(set(sizes)) != len(sizes)
    ):
        raise ValueError(
            "sizes must be different whole numbers from 1, how many top neurons "
            f"each of a role's sets takes, such as (10, 50); not {sizes!r}"
        )
    return tuple(int(k) for k in sizes)


def import_mayner(
    folder: Path, model: str = "qwen3-8b", sizes: tuple[int, ...] = SIZES
) -> list[Candidate]:
    """Every role of Mayner's data as candidates: its top k neurons, each k in sizes.

    folder is data/mayner, as `pushlab fetch mayner` fills it: his bundle's
    files in bundle/ and the record FETCHED.json. The bundle's
    split/top_neurons.parquet must be there; the vector files, the judged
    replies and the concentration table are used when they are (see the
    module's docstring). If bundle/ has his MANIFEST, a file it lists among
    those read here (sources.MAYNER_FILES) that is missing or not as it
    lists it is refused with ValueError, which says to fetch it again. The
    candidates come role by role in name order, each role's in the order of
    sizes. model is the registry name the sets are made for, which must name
    Qwen3-8B, the model whose neurons his data lists.

    Example:
        >>> found = import_mayner(Path("/workspace/data/mayner"))  # doctest: +SKIP
        >>> [c.key for c in found if c.persona == "pirate"]  # doctest: +SKIP
        ['mayner/pirate-top10', 'mayner/pirate-top50']
        >>> import_mayner(Path("/no/such"))  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        FileNotFoundError: there is no split/top_neurons.parquet in /no/such/bundle; ...
    """
    _check_model(model)
    sizes = _sizes(sizes)
    folder = Path(folder)
    bundle = folder / MAYNER_BUNDLE
    check_bundle(bundle, MAYNER_FILES)
    path = bundle / TOP_NEURONS_FILE
    table = _read(path, mayner.TOP_NEURONS_COLUMNS)
    if table is None:
        raise FileNotFoundError(
            f"there is no {TOP_NEURONS_FILE} in {bundle}; download Mayner's data "
            "with `pushlab fetch mayner` (on the GPU pod), which puts it in the "
            "data folder's mayner/bundle/"
        )
    if table.empty or table["role"].isna().any():
        what = "no rows" if table.empty else "a row with no role"
        raise ValueError(f"{path} has {what}; {FETCH_AGAIN}")
    replies = reply_scores(bundle / REPLIES_FILE, sizes)
    shares = concentration_shares(bundle / CONCENTRATION_FILE, sizes)
    source_url, downloaded_at = origin(folder)
    source_file = f"{MAYNER.folder}/{MAYNER_BUNDLE}/{TOP_NEURONS_FILE}"
    candidates = []
    # One pass over the table, which holds 275 roles' top 1,000 on the pod.
    for role, rows in table.groupby("role", sort=True):
        norm, size_from = vector_norm(bundle, role, rows)
        for k in sizes:
            unit = mayner.top_neurons_set(rows, role, k, norm, model_name=model)
            scored, in_role = replies.get((role, k), (None, None))
            candidates.append(
                Candidate(
                    source=MAYNER.name,
                    persona=role,
                    unit=unit,
                    direction=1,
                    strength=1.0,  # dose 1 pushes his strength
                    sensitive=False,  # his data flags no role
                    source_file=source_file,
                    source_ref=unit.source,
                    source_url=source_url,
                    downloaded_at=downloaded_at,
                    category="role",
                    size_from=size_from,
                    mayner_in_role=in_role,
                    mayner_scored=scored,
                    mayner_share=shares.get((role, k)),
                )
            )
    return candidates
