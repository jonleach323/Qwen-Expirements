"""`pushlab candidates`: one list of both sources' persona candidates (decision
0020), made from the library's fixture files (tests/fixtures/library/qwen)
and the mayner_data fixture (tests/unit/conftest.py), both made up in the
sources' real formats.

The library's fixture gives 8 candidates, 2 of them sensitive (Storm Wraith,
and the Lighthouse Keeper of L22N1357+), and two names twice (Teapot and
Lighthouse Keeper); Mayner's gives 3 roles at 2 sizes, none sensitive.
"""

import csv
import doctest
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from pushlab import PROJECT_ROOT
from pushlab.cli import main
from pushlab.discovery import candidates_command
from pushlab.discovery.candidates import Candidate, read_candidates, same_name
from pushlab.discovery.candidates_command import (
    IGNORED,
    Tally,
    check_units,
    clean,
    run_candidates,
    tally_lines,
)
from pushlab.discovery.library_import import import_library
from pushlab.discovery.mayner_import import import_mayner
from pushlab.discovery.sources import LIBRARY, MAYNER
from pushlab.models.discovery import ModelInfo
from pushlab.units.neuron import NeuronUnit
from pushlab.units.neuron_set import NeuronSet

LIBRARY_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "library" / "qwen"
QWEN3_8B = ModelInfo(layers=36, hidden_size=4096, mlp_size=12288)
# The library's candidates kept by default, in its order: the sensitive ones
# dropped, then the Teapot of L20N77+, as the one of L3N1207+ is verified.
LIBRARY_KEPT = [
    "library/L3N1207+",
    "library/L12N4410-",
    "library/L35N12287-",
    "library/L18N2468+",
    "library/L33N4096-",
]
# With --include-sensitive: Storm Wraith too, and the Lighthouse Keeper of
# L22N1357+, more coherent, instead of the one of L12N4410-.
LIBRARY_ALL = [
    "library/L18N2468-",
    "library/L3N1207+",
    "library/L22N1357+",
    "library/L35N12287-",
    "library/L18N2468+",
    "library/L33N4096-",
]
MAYNER_KEYS = [
    f"mayner/{role}-top{k}"
    for role in ("coral-reef", "pirate", "sommelier")
    for k in (10, 50)
]
HEADER = "  source   read  sensitive  duplicates  kept"


@pytest.fixture
def data(mayner_data: Path) -> Path:
    """A data folder with both sources fetched: the library's fixture files in
    library/qwen, beside mayner_data's made-up bundle in mayner/."""
    shutil.copytree(LIBRARY_FIXTURE, mayner_data / "library" / "qwen")
    return mayner_data


def _candidates(data: Path, out: Path, *options: str) -> int:
    """Run `pushlab candidates` for qwen3-8b on the data folder."""
    argv = ["candidates", "--model", "qwen3-8b", "--out", str(out)]
    return main([*argv, "--data-dir", str(data), *options])


def _imported(data: Path, keys: list[str]) -> list[Candidate]:
    """The library's candidates of these keys, then all of Mayner's."""
    library = {c.key: c for c in import_library(data / "library" / "qwen")}
    return [library[key] for key in keys] + import_mayner(data / "mayner")


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _no_import(*args: object, **kwargs: object) -> None:
    raise AssertionError("nothing may be imported when the command refuses first")


def test_the_list_holds_both_sources_without_the_sensitive_ones(
    data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "units" / "candidates_qwen3-8b.csv"
    assert _candidates(data, out) == 0
    found = read_candidates(out)
    assert [candidate.key for candidate in found] == LIBRARY_KEPT + MAYNER_KEYS
    assert found == _imported(data, LIBRARY_KEPT)
    assert not any(candidate.sensitive for candidate in found)
    for candidate in found:
        candidate.unit.validate(QWEN3_8B)
    lines = capsys.readouterr().out.splitlines()
    table = lines.index(HEADER)
    assert lines[table + 1 : table + 4] == [
        "  library     8          2           1     5",
        "  mayner      6          0           0     6",
        "  total      14          2           1    11",
    ]
    assert lines[table + 4] == f"Saved 11 candidates in {out}"


def test_include_sensitive_keeps_them_and_says_so(
    data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "all.csv"
    assert _candidates(data, out, "--include-sensitive") == 0
    found = read_candidates(out)
    assert [candidate.key for candidate in found] == LIBRARY_ALL + MAYNER_KEYS
    assert found == _imported(data, LIBRARY_ALL)
    flagged = [candidate.key for candidate in found if candidate.sensitive]
    assert flagged == ["library/L18N2468-", "library/L22N1357+"]
    assert {row["sensitive"] for row in _rows(out)} == {"true", "false"}
    for candidate in found:
        candidate.unit.validate(QWEN3_8B)
    printed = capsys.readouterr().out
    assert "  library     8          0           2     6" in printed
    assert "Kept 2 candidates a source calls sensitive" in printed


def test_every_row_credits_its_source_and_the_credits_are_printed(
    data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "c.csv"
    assert _candidates(data, out) == 0
    rows = _rows(out)
    credits = {(row["source"], row["licence"], row["credit"]) for row in rows}
    assert credits == {
        ("library", LIBRARY.licence, LIBRARY.credit),
        ("mayner", "CC BY 4.0", MAYNER.list_credit),
    }
    lines = capsys.readouterr().out.splitlines()
    assert f"  mayner: licence: CC BY 4.0. Credit: {MAYNER.list_credit}" in lines
    assert f"  library: licence: {LIBRARY.licence}. Credit: {LIBRARY.credit}" in lines
    assert lines[-1] == (
        f"The library states no licence, so keep the list out of git (git ignores "
        f"{IGNORED})."
    )
    # The pattern the command names is the one git ignores.
    assert f"/{IGNORED}" in (PROJECT_ROOT / ".gitignore").read_text().splitlines()


def _restamp(folder: Path) -> None:
    """Give sensitive.json's "from" stamp the sizes of the library's files as
    they are now, as the library's own copy would after an edit."""
    path = folder / "sensitive.json"
    sensitive = json.loads(path.read_text(encoding="utf-8"))
    parts = sensitive["from"].split("|")
    for i, part in enumerate(parts):
        name, _, rest = part.partition(":")
        if name in ("library.json", "combos.json"):
            _, _, modified = rest.partition(":")
            parts[i] = f"{name}:{(folder / name).stat().st_size}:{modified}"
    sensitive["from"] = "|".join(parts)
    path.write_text(json.dumps(sensitive), encoding="utf-8")


def test_duplicate_names_go_within_a_source_but_not_across_them(
    data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The library's Rain Cloud renamed Pirate, as one of Mayner's roles is
    # named, and a role Pirate added to his table, a copy of pirate's rows.
    path = data / "library" / "qwen" / "library.json"
    library = json.loads(path.read_text(encoding="utf-8"))
    persona = library["fields"].index("persona")
    for row in library["rows"]:
        row[persona] = "Pirate" if row[persona] == "Rain Cloud" else row[persona]
    path.write_text(json.dumps(library), encoding="utf-8")
    _restamp(path.parent)
    top = data / "mayner" / "bundle" / "split" / "top_neurons.parquet"
    table = pd.read_parquet(top)
    pirate = table[table["role"] == "pirate"].assign(role="Pirate")
    pd.concat([table, pirate], ignore_index=True).to_parquet(top, index=False)
    out = tmp_path / "c.csv"
    assert _candidates(data, out) == 0
    found = read_candidates(out)
    pirates = [c.key for c in found if same_name(c.persona) == "pirate"]
    assert pirates == [
        "library/L18N2468+",
        "mayner/pirate-top10",
        "mayner/pirate-top50",
    ]
    keys = [candidate.key for candidate in found]
    assert sorted(keys) == sorted(LIBRARY_KEPT + MAYNER_KEYS)
    assert "  mayner      8          0           2     6" in capsys.readouterr().out


def test_the_data_folder_defaults_to_pushlab_data_dir(
    data: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PUSHLAB_DATA_DIR", str(data))
    out = tmp_path / "c.csv"
    assert main(["candidates", "--model", "qwen3-8b", "--out", str(out)]) == 0
    assert len(read_candidates(out)) == 11


def test_a_list_that_cannot_be_written_is_refused(
    data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "units").write_text("a file, not a folder\n", encoding="utf-8")
    out = tmp_path / "units" / "c.csv"
    assert _candidates(data, out) == 1
    assert f"pushlab candidates: cannot write {out}: " in capsys.readouterr().err


def test_an_existing_list_is_refused_before_anything_is_imported(
    data: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out = tmp_path / "c.csv"
    out.write_text("an earlier list\n", encoding="utf-8")
    monkeypatch.setattr(candidates_command, "import_library", _no_import)
    monkeypatch.setattr(candidates_command, "import_mayner", _no_import)
    assert _candidates(data, out) == 1
    assert f"{out} already exists; pass --overwrite" in capsys.readouterr().err
    assert out.read_text(encoding="utf-8") == "an earlier list\n"
    assert _candidates(data, tmp_path) == 1
    assert f"--out must be a file, but {tmp_path} is a folder" in (
        capsys.readouterr().err
    )
    monkeypatch.undo()
    assert _candidates(data, out, "--overwrite") == 0
    assert len(read_candidates(out)) == 11


@pytest.mark.parametrize(
    ("fetched", "refusals"),
    [
        (["library"], ["`pushlab fetch mayner`"]),
        (["mayner"], ["`uv run pushlab fetch library`"]),
        ([], ["`uv run pushlab fetch library`", "`pushlab fetch mayner`"]),
    ],
)
def test_a_source_not_fetched_says_which_fetch_to_run(
    data: Path,
    tmp_path: Path,
    fetched: list[str],
    refusals: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    for source in ("library", "mayner"):
        if source not in fetched:
            shutil.rmtree(data / source)
    out = tmp_path / "c.csv"
    assert _candidates(data, out) == 1
    errors = capsys.readouterr().err.splitlines()
    assert len(errors) == len(refusals)
    for error, refusal in zip(errors, refusals, strict=True):
        assert error.startswith("pushlab candidates: ") and refusal in error
    assert not out.exists()


def test_a_source_not_in_its_format_is_refused(
    data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (data / "library" / "qwen" / "library.json").write_text("{", encoding="utf-8")
    out = tmp_path / "c.csv"
    assert _candidates(data, out) == 1
    assert "library.json is not valid JSON" in capsys.readouterr().err
    assert not out.exists()


def test_a_model_the_sources_have_no_neurons_of_is_refused(
    data: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "c.csv"
    assert run_candidates("qwen3.5-4b", out, data) == 1
    error = capsys.readouterr().err
    assert "list no neurons of 'qwen3.5-4b'" in error
    assert "give --model qwen3-8b" in error
    with pytest.raises(SystemExit) as exit_info:
        main(["candidates", "--model", "qwen3.5-4b", "--out", str(out)])
    assert exit_info.value.code == 2
    assert "invalid choice: 'qwen3.5-4b'" in capsys.readouterr().err
    assert not out.exists()


def test_a_unit_outside_the_models_shape_is_refused(
    data: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    good = import_library(data / "library" / "qwen")
    # Qwen3-8B's layers are 0 to 35, its neurons 0 to 12287.
    deep = replace(good[0], unit=NeuronUnit("qwen3-8b", 36, 7))
    wide = replace(good[1], unit=NeuronUnit("qwen3-8b", 3, 12288))
    monkeypatch.setattr(
        candidates_command, "import_library", lambda *args: [deep, *good, wide]
    )
    # And a set of Mayner's whose last neuron is in layer 36.
    sets = import_mayner(data / "mayner")
    unit = sets[2].unit
    entries = [*unit.entries[:-1], (36, 0, 1.0)]
    far = NeuronSet("qwen3-8b", unit.name, entries, unit.source, unit.size)
    sets[2] = replace(sets[2], unit=far)
    monkeypatch.setattr(candidates_command, "import_mayner", lambda *args: sets)
    out = tmp_path / "c.csv"
    assert _candidates(data, out) == 1
    errors = capsys.readouterr().err.splitlines()
    assert errors == [
        "pushlab candidates: a unit is not one of qwen3-8b's: library/L36N7-, from "
        "library/qwen/library.json: qwen3-8b:L36:N7: layer 36 is out of range; the "
        "model's layers are 0 to 35",
        "pushlab candidates: a unit is not one of qwen3-8b's: library/L3N12288+, "
        "from library/qwen/library.json: qwen3-8b:L3:N12288: neuron 12288 is out of "
        "range; each layer's neurons are 0 to 12287",
        "pushlab candidates: a unit is not one of qwen3-8b's: mayner/pirate-top10, "
        "from mayner/bundle/split/top_neurons.parquet: qwen3-8b:set:pirate-top10: "
        "qwen3-8b:L36:N0: layer 36 is out of range; the model's layers are 0 to 35",
    ]
    assert not out.exists()


def test_many_wrong_units_are_named_up_to_a_limit(
    data: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    good = import_library(data / "library" / "qwen")[0]
    deep = [replace(good, unit=NeuronUnit("qwen3-8b", 36, i)) for i in range(12)]
    monkeypatch.setattr(candidates_command, "import_library", lambda *args: deep)
    assert _candidates(data, tmp_path / "c.csv") == 1
    errors = capsys.readouterr().err.splitlines()
    assert len(errors) == candidates_command.PROBLEMS_SHOWN + 1
    assert errors[-1] == "pushlab candidates: and 2 more like them"


def test_a_unit_of_another_model_is_named() -> None:
    other = Candidate(
        "library", "Owl", NeuronUnit("qwen3.5-4b", 3, 7), 1, 1.0, False, "f", "r"
    )
    assert check_units([other], "qwen3-8b") == [
        "library/L3N7+, from f: qwen3.5-4b:L3:N7 is a unit of qwen3.5-4b, not qwen3-8b"
    ]


def test_sensitive_ones_go_before_duplicates() -> None:
    found = import_library(LIBRARY_FIXTURE)
    kept, tallies = clean(found)
    assert [candidate.key for candidate in kept] == LIBRARY_KEPT
    assert tallies == {"library": Tally(8, 2, 1, 5), "mayner": Tally(0, 0, 0, 0)}
    kept, tallies = clean(found, include_sensitive=True)
    assert [candidate.key for candidate in kept] == LIBRARY_ALL
    assert tallies["library"] == Tally(8, 0, 2, 6)


def test_the_counts_table_lines_up_large_numbers() -> None:
    tallies = {
        "library": Tally(61234, 344, 9012, 51878),
        "mayner": Tally(550, 0, 0, 550),
    }
    assert tally_lines(tallies) == [
        "source     read  sensitive  duplicates    kept",
        "library  61,234        344       9,012  51,878",
        "mayner      550          0           0     550",
        "total    61,784        344       9,012  52,428",
    ]


def test_docstring_examples_run() -> None:
    result = doctest.testmod(candidates_command)
    assert result.attempted > 0
    assert result.failed == 0
