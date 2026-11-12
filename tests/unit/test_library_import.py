"""The Neuron Persona Library's files as persona candidates (decision 0020).

The fixture, tests/fixtures/library/qwen, holds made-up rows in the library's
real format: twelve rows of library.json, seven of them personas, a pool of
three, and four sensitive neurons.
"""

import doctest
import hashlib
import json
import re
import shutil
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from pushlab.discovery import library_import
from pushlab.discovery.candidates import Candidate, to_row
from pushlab.discovery.library_files import stamped_sizes
from pushlab.discovery.library_import import import_library
from pushlab.discovery.sources import LIBRARY, LIBRARY_FILES, RECORD_FILE
from pushlab.models.discovery import ModelInfo
from pushlab.units.neuron import NeuronUnit

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "library" / "qwen"
QWEN3_8B = ModelInfo(layers=36, hidden_size=4096, mlp_size=12288)
LIBRARY_URL = "https://neuronpersonas.com/data/qwen/library.json"
COMBOS_URL = "https://neuronpersonas.com/data/qwen/combos.json"
# When the fixture's FETCHED.json says library.json and combos.json came.
LIBRARY_TIME = "2026-11-12T20:31:05Z"
COMBOS_TIME = "2026-11-12T20:31:07Z"


def _row(
    persona: str,
    layer: int,
    index: int,
    sign: int,
    *,
    a: float,
    w: int,
    c: int,
    tags: tuple[str, ...],
    ref: str,
    sensitive: bool = False,
    pool: bool = False,
) -> Candidate:
    """The candidate of a persona row of the fixture's library.json."""
    return Candidate(
        source="library",
        persona=persona,
        unit=NeuronUnit("qwen3-8b", layer, index),
        direction=sign,
        strength=a,
        sensitive=sensitive,
        source_file="library/qwen/library.json",
        source_ref=ref,
        source_url=LIBRARY_URL,
        downloaded_at=LIBRARY_TIME,
        category="persona",
        tags=tags,
        coherence=c,
        weirdness=w,
        verified_pool=pool,
    )


# The pool persona that library.json has no row for, built from its pool row.
BRASS_OWL = Candidate(
    source="library",
    persona="Brass Owl",
    unit=NeuronUnit("qwen3-8b", 33, 4096),
    direction=-1,
    strength=0.75,
    sensitive=False,
    source_file="library/qwen/combos.json",
    source_ref="pool L33N4096-",
    source_url=COMBOS_URL,
    downloaded_at=COMBOS_TIME,
    category="",
    tags=(),
    coherence=8,
    weirdness=9,
    verified_pool=True,
)
# The fixture's candidates: library.json's persona rows in its order, then the
# pool's persona that has no row.
EXPECTED = [
    _row(
        "Storm Wraith",
        18,
        2468,
        -1,
        a=1.25,
        w=10,
        c=4,
        tags=("gale-born", "lamp-snuffing", "thundery"),
        ref="L18 N2468 s-1",
        sensitive=True,
    ),
    _row(
        "Teapot",
        3,
        1207,
        1,
        a=1.0,
        w=8,
        c=7,
        tags=("spout-proud", "brew-ready", "chipper"),
        ref="L3 N1207 s+1",
        pool=True,
    ),
    _row(
        "Lighthouse Keeper",
        22,
        1357,
        1,
        a=1.0,
        w=8,
        c=9,
        tags=("lamp-tending", "salt-weathered", "solitary-shift"),
        ref="L22 N1357 s+1",
        sensitive=True,
    ),
    _row(
        "Lighthouse Keeper",
        12,
        4410,
        -1,
        a=0.75,
        w=8,
        c=7,
        tags=("lamp-tending", "salt-weathered", "ship-counting"),
        ref="L12 N4410 s-1",
    ),
    _row(
        "Snowfield Clerk",
        35,
        12287,
        -1,
        a=0.5,
        w=8,
        c=7,
        tags=("rime-slow", "winter-counting", "unhurried-ice"),
        ref="L35 N12287 s-1",
        pool=True,
    ),
    _row(
        "Teapot",
        20,
        77,
        1,
        a=0.75,
        w=6,
        c=9,
        tags=("spout-proud", "brew-ready", "unruffled"),
        ref="L20 N77 s+1",
    ),
    _row(
        "Rain Cloud",
        18,
        2468,
        1,
        a=1.0,
        w=7,
        c=8,
        tags=("drizzle-soft", "hill-drifting", "mild-mannered"),
        ref="L18 N2468 s+1",
    ),
    BRASS_OWL,
]


def _copy(tmp_path: Path) -> Path:
    """A copy of the fixture's folder, to change."""
    return Path(shutil.copytree(FIXTURE, tmp_path / "library" / "qwen"))


def _edit(folder: Path, file: str, change: Callable[[Any], object]) -> None:
    """Change the JSON of folder/file in place with change(data), as if the
    library had published it so and `pushlab fetch library` had fetched it:
    sensitive.json's stamp and the fetch record follow the files' new sizes,
    and the record their SHA-256."""
    path = folder / file
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")
    sensitive = folder / "sensitive.json"
    text = sensitive.read_text(encoding="utf-8")
    for name in LIBRARY_FILES[:2]:
        size = (folder / name).stat().st_size
        text = re.sub(rf"\|{re.escape(name)}:[0-9]+:", f"|{name}:{size}:", text)
    sensitive.write_text(text, encoding="utf-8")
    record = json.loads((folder / RECORD_FILE).read_text(encoding="utf-8"))
    for name, entry in record["files"].items():
        held = (folder / name).read_bytes()
        entry |= {"bytes": len(held), "sha256": hashlib.sha256(held).hexdigest()}
    (folder / RECORD_FILE).write_text(json.dumps(record), encoding="utf-8")


def _set_row(data: dict, row: int, field: str, value: object) -> None:
    """Set a value of a row of library.json, found by its field's name."""
    data["rows"][row][data["fields"].index(field)] = value


def _set_pool(data: dict, row: int, field: str, value: object) -> None:
    """Set a value of a row of combos.json's pool, found by its field's name."""
    data["pool"][row][data["pool_fields"].index(field)] = value


def test_the_fixture_imports_into_the_expected_candidates() -> None:
    assert import_library(FIXTURE) == EXPECTED


def test_only_persona_rows_and_verified_pool_personas_are_candidates() -> None:
    found = import_library(FIXTURE)
    # The fixture's rows of the other categories: default_assistant,
    # altered_assistant, incoherent, and two of other.
    others = {"Stock Helper", "Sleepy Assistant", "Word Soup", "Unsorted"}
    assert others.isdisjoint(candidate.persona for candidate in found)
    assert "Blank Reply" not in {candidate.persona for candidate in found}
    assert [candidate.category for candidate in found] == ["persona"] * 7 + [""]
    assert len({candidate.key for candidate in found}) == len(found) == 8


def test_pool_personas_are_flagged_and_one_without_a_row_is_built() -> None:
    found = {candidate.key: candidate for candidate in import_library(FIXTURE)}
    pooled = [key for key, candidate in found.items() if candidate.verified_pool]
    assert pooled == ["library/L3N1207+", "library/L35N12287-", "library/L33N4096-"]
    others = [found[key].verified_pool for key in found if key not in pooled]
    assert others == [False] * 5
    assert found["library/L33N4096-"] == BRASS_OWL
    # The pool's Teapot is library.json's row, its tags and all.
    assert found["library/L3N1207+"].source_file == "library/qwen/library.json"
    assert found["library/L3N1207+"].tags == ("spout-proud", "brew-ready", "chipper")


def test_sensitive_neurons_are_flagged_in_their_direction_only() -> None:
    found = {candidate.key: candidate for candidate in import_library(FIXTURE)}
    flagged = [key for key, candidate in found.items() if candidate.sensitive]
    assert flagged == ["library/L18N2468-", "library/L22N1357+"]
    # Rain Cloud is the same neuron as Storm Wraith, pushed the other way.
    assert found["library/L18N2468+"].sensitive is False
    # The other Lighthouse Keeper shares the sensitive one's name, not its neuron.
    assert found["library/L12N4410-"].sensitive is False


def test_each_candidates_url_and_download_time_come_from_the_fetch_record() -> None:
    record = json.loads((FIXTURE / RECORD_FILE).read_text(encoding="utf-8"))
    for candidate in import_library(FIXTURE):
        file = candidate.source_file.removeprefix("library/qwen/")
        entry = record["files"][file]
        assert candidate.source_url == entry["url"]
        assert candidate.downloaded_at == entry["downloaded_at"]
    assert {candidate.downloaded_at for candidate in EXPECTED} == {
        LIBRARY_TIME,
        COMBOS_TIME,
    }


def test_every_candidates_unit_is_a_neuron_of_qwen3_8b() -> None:
    for candidate in import_library(FIXTURE):
        candidate.unit.validate(QWEN3_8B)
        assert str(candidate.unit).startswith("qwen3-8b:L")


def test_values_are_found_by_their_names_not_their_places(tmp_path: Path) -> None:
    folder = _copy(tmp_path)

    def reverse(data: dict, names_key: str, rows_key: str) -> None:
        names = data[names_key]
        data[rows_key] = [row[::-1] for row in data[rows_key]]
        data[names_key] = names[::-1]

    def reverse_library(data: dict) -> None:
        reverse(data, "fields", "rows")
        categories = data["categories"]
        data["categories"] = categories[::-1]
        cat = data["fields"].index("cat")
        for row in data["rows"]:
            row[cat] = len(categories) - 1 - row[cat]

    _edit(folder, "library.json", reverse_library)
    _edit(folder, "combos.json", lambda data: reverse(data, "pool_fields", "pool"))
    assert import_library(folder) == EXPECTED


def test_a_neuron_and_direction_given_twice_count_once_as_their_first_row(
    tmp_path: Path,
) -> None:
    folder = _copy(tmp_path)

    def repeat_rows(data: dict) -> None:
        # Storm Wraith's neuron and direction again, under another name.
        again = list(data["rows"][0])
        data["rows"].append(again)
        _set_row(data, -1, "persona", "Thunder Wraith")
        # Unsorted's again, now as a persona: its first row is of other.
        unsorted = list(data["rows"][10])
        data["rows"].append(unsorted)
        _set_row(data, -1, "persona", "Binary Ghost")
        _set_row(data, -1, "cat", data["categories"].index("persona"))

    def repeat_pool(data: dict) -> None:
        data["pool"].append(list(data["pool"][1]))
        _set_pool(data, -1, "name", "Iron Owl")

    _edit(folder, "library.json", repeat_rows)
    _edit(folder, "combos.json", repeat_pool)
    assert import_library(folder) == EXPECTED


def test_a_pool_persona_whose_row_is_no_persona_comes_from_its_pool_row(
    tmp_path: Path,
) -> None:
    folder = _copy(tmp_path)
    # Sleepy Assistant's row (L9 N3021 s-1) is of category altered_assistant.
    night_owl = ["L9N3021-", 9, 3021, -1, 1.0, "Night Owl", "made up", 7, 8, "Hoo."]
    night_owl += [["exact", "exact", "exact"], True]
    _edit(folder, "combos.json", lambda data: data["pool"].append(night_owl))
    expected = replace(
        BRASS_OWL,
        persona="Night Owl",
        unit=NeuronUnit("qwen3-8b", 9, 3021),
        strength=1.0,
        source_ref="pool L9N3021-",
        coherence=8,
        weirdness=7,
    )
    assert import_library(folder) == [*EXPECTED, expected]


def test_a_pool_row_the_library_did_not_verify_is_not_in_the_pool(
    tmp_path: Path,
) -> None:
    folder = _copy(tmp_path)

    def unverify(data: dict) -> None:
        _set_pool(data, 0, "verified", False)  # Teapot, which has a row
        _set_pool(data, 1, "verified", False)  # Brass Owl, which has none

    _edit(folder, "combos.json", unverify)
    found = {candidate.key: candidate for candidate in import_library(folder)}
    assert found["library/L3N1207+"].verified_pool is False
    assert "library/L33N4096-" not in found
    assert found["library/L35N12287-"].verified_pool is True
    assert len(found) == 7


def test_the_replies_text_is_never_kept() -> None:
    library = json.loads((FIXTURE / "library.json").read_text(encoding="utf-8"))
    combos = json.loads((FIXTURE / "combos.json").read_text(encoding="utf-8"))
    text = library["fields"].index("text")
    pool_text = [combos["pool_fields"].index(name) for name in ("desc", "text")]
    replies = [row[text] for row in library["rows"]]
    replies += [row[at] for row in combos["pool"] for at in pool_text]
    found = import_library(FIXTURE)
    written = "\n".join(value for c in found for value in to_row(c).values())
    assert not [reply for reply in replies if reply in written]


def test_the_fixtures_record_matches_its_files() -> None:
    record = json.loads((FIXTURE / RECORD_FILE).read_text(encoding="utf-8"))
    assert (record["source"], record["url"]) == (LIBRARY.name, LIBRARY.url)
    assert set(record["files"]) == set(LIBRARY_FILES)
    for file in LIBRARY_FILES:
        data = (FIXTURE / file).read_bytes()
        entry = record["files"][file]
        assert entry["url"] == f"{LIBRARY.url}/{file}"
        assert entry["bytes"] == len(data)
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()


def test_the_fixtures_stamp_gives_the_sizes_of_its_files() -> None:
    sensitive = json.loads((FIXTURE / "sensitive.json").read_text(encoding="utf-8"))
    sizes = [(file, (FIXTURE / file).stat().st_size) for file in LIBRARY_FILES[:2]]
    assert stamped_sizes(sensitive["from"]) == sizes


def test_docstring_examples_run() -> None:
    result = doctest.testmod(library_import)
    assert result.attempted > 0
    assert result.failed == 0
