"""The categories file of pushlab panel select (pushlab.panel.categories,
decision 0023): written for you to fill in, read back as a spreadsheet saves
it, and written again, keeping what you gave, while a finalist lacks one.
Keys and names are made up.
"""

import codecs
import doctest
from pathlib import Path

import pytest

from pushlab import PROJECT_ROOT
from pushlab.panel import categories
from pushlab.panel.categories import check_categories, read_categories, write_template

ALLOWED = ("people", "animals", "fictional", "objects", "mystical")
FINALISTS = [
    ("library/L9N1-", "Brass Owl"),
    ("mayner/teapot-top2", "Velvet Teapot"),
    ("library/L2N100+", "=Salt Captain"),  # a spreadsheet would run it
]


def test_the_template_has_a_row_per_finalist_its_category_blank(
    tmp_path: Path,
) -> None:
    path = tmp_path / "experiments" / "categories.csv"
    assert read_categories(path, ALLOWED) is None
    assert write_template(path, FINALISTS, None) == path
    data = path.read_bytes()
    assert data.startswith(codecs.BOM_UTF8)
    assert data[3:].decode("utf-8").splitlines() == [
        "key,persona,category",
        "library/L9N1-,Brass Owl,",
        "mayner/teapot-top2,Velvet Teapot,",
        "library/L2N100+,'=Salt Captain,",
    ]
    found = read_categories(path, ALLOWED)
    assert found is not None and found.categories == {}
    assert found.unfilled([key for key, _ in FINALISTS]) == [k for k, _ in FINALISTS]
    assert [p.name for p in path.parent.iterdir()] == ["categories.csv"]


def test_categories_are_read_as_a_spreadsheet_saves_them(tmp_path: Path) -> None:
    path = tmp_path / "categories.csv"
    text = (
        "Category ; KEY;persona\r\n"
        " Animals ;library/L9N1-;Brass Owl\r\n"
        "\r\n"
        "OBJECTS; mayner/teapot-top2 ;Velvet Tea\xe9pot\r\n"
    )
    path.write_bytes(text.encode("cp1252"))
    found = read_categories(path, ALLOWED)
    assert found is not None
    assert found.categories == {
        "library/L9N1-": "animals",
        "mayner/teapot-top2": "objects",
    }
    assert found.unfilled([key for key, _ in FINALISTS]) == ["library/L2N100+"]


def test_an_unfinished_file_is_written_again_keeping_what_it_gave(
    tmp_path: Path,
) -> None:
    path = tmp_path / "categories.csv"
    path.write_text(
        "key,persona,category\n"
        "library/L9N1-,Brass Owl,animals\n"
        "library/L7N7+,Old One,mystical\n"  # no finalist now, but kept
        "library/L5N5+,Older One,\n",  # no finalist and no category: dropped
        encoding="utf-8",
    )
    known = read_categories(path, ALLOWED)
    write_template(path, FINALISTS, known)
    assert path.read_text(encoding="utf-8-sig").splitlines() == [
        "key,persona,category",
        "library/L9N1-,Brass Owl,animals",
        "mayner/teapot-top2,Velvet Teapot,",
        "library/L2N100+,'=Salt Captain,",
        "library/L7N7+,Old One,mystical",
    ]


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        (
            "key,category\nlibrary/L9N1-,animal\nmayner/x,robots\n",
            "'animal' (row 2) and 'robots' (row 3) are not categories; a category "
            "is one of people, animals",
        ),
        ("key,category\nlibrary/L9N1-,people\nlibrary/L9N1-,animals\n", "rows 2 and 3"),
        ("persona,category\nBrass Owl,people\n", "needs a key and a category column"),
        ("key,category,key\n", "has two columns named key"),
    ],
)
def test_a_file_that_cannot_be_used_is_refused(
    tmp_path: Path, text: str, problem: str
) -> None:
    path = tmp_path / "categories.csv"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        read_categories(path, ALLOWED)
    assert problem in str(caught.value)


def test_a_categories_file_belongs_outside_the_repository(tmp_path: Path) -> None:
    check_categories(tmp_path / "categories.csv")
    with pytest.raises(ValueError, match="is in the repository: it names candidates"):
        check_categories(PROJECT_ROOT / "units" / "categories.csv")
    with pytest.raises(ValueError, match="must be a file"):
        check_categories(tmp_path)


def test_the_examples_in_the_docstrings_hold() -> None:
    result = doctest.testmod(categories)
    assert result.failed == 0 and result.attempted > 0
