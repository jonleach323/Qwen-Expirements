"""Personas files (pushlab.scoring.personas): the personas a judge asks about."""

import doctest
from pathlib import Path

import pytest

from pushlab.scoring import personas as personas_module
from pushlab.scoring.personas import (
    Persona,
    names_no_persona,
    pick_persona,
    read_personas,
)

OWLS = """\
personas:
  - name: Brass Owl
    description: A pompous mechanical owl who lectures in riddles.
  - name: Snow Owl
"""


def _file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "personas.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_a_personas_file_is_read_in_order(tmp_path: Path) -> None:
    assert read_personas(_file(tmp_path, OWLS)) == [
        Persona("Brass Owl", "A pompous mechanical owl who lectures in riddles."),
        Persona("Snow Owl", ""),
    ]


@pytest.mark.parametrize(
    ("text", "problem"),
    [
        ("- name: Owl\n", "one key, personas"),
        ("personas: []\n", "at least one persona"),
        ("personas:\n  - Owl\n", r"personas\[0\] must be a mapping with a name"),
        ("personas:\n  - {name: Owl, colour: red}\n", "unknown keys colour"),
        ("personas:\n  - {name: Owl}\n  - {name: ' owl'}\n", "one line with no spaces"),
        (
            "personas:\n  - {name: Brass  Owl}\n  - {name: brass owl}\n",
            "repeats the name",
        ),
        ("personas:\n  - {name: None}\n", 'cannot be named "none"'),
        ("personas:\n  - {name: ''}\n", "must be text"),
        ("personas:\n  - {name: Owl, description: 3}\n", "description must be text"),
        ('personas:\n  - {name: "Owl\\nCat"}\n', "one line"),
        ("personas: [\n", "cannot read the personas file"),
    ],
)
def test_a_malformed_personas_file_is_refused(
    tmp_path: Path, text: str, problem: str
) -> None:
    with pytest.raises(ValueError, match=problem):
        read_personas(_file(tmp_path, text))


def test_a_persona_is_picked_by_name_or_as_the_only_one() -> None:
    owl, snow = Persona("Brass Owl"), Persona("Snow Owl")
    assert pick_persona([owl], None) == owl
    assert pick_persona([owl, snow], "Snow Owl") == snow
    with pytest.raises(ValueError, match="choose one persona with --persona"):
        pick_persona([owl, snow], None)
    with pytest.raises(ValueError, match="no persona is named 'snow owl'"):
        pick_persona([owl, snow], "snow owl")


def test_docstring_examples_run() -> None:
    result = doctest.testmod(personas_module)
    assert result.attempted > 0
    assert result.failed == 0


@pytest.mark.parametrize(
    ("name", "answer"),
    [
        ("none", True),
        ("None", True),
        (" NONE ", True),
        ("Nonesuch", False),
        ("", False),
    ],
)
def test_names_no_persona_is_the_judges_answer_in_any_case(
    name: str, answer: bool
) -> None:
    assert names_no_persona(name) is answer
