"""Personas a judge asks about: a name and a short description each.

A personas file is YAML with one key, personas, a list of mappings with a
name and, optionally, a description (decision 0021):

    personas:
      - name: Brass Owl
        description: A pompous mechanical owl who lectures in riddles.

The speaks_as rubric asks about one persona, and which_persona chooses among
all of a file's. The same personas are asked about for every reply of an
experiment, whatever condition made it, so the question never tells the
judge which condition that was. A file of persona names from the Neuron
Persona Library stays out of git, as its data does (decision 0020).
"""

import re
from dataclasses import dataclass
from pathlib import Path

from pushlab.schedules.yaml_text import load_yaml

# which_persona's answer when the text speaks as none of the personas.
NONE = "none"
PERSONA_KEYS = ("name", "description")


def names_no_persona(name: str) -> bool:
    """Whether name is NONE, in any case and with any spaces around it: the
    judge's answer for no persona, which no persona can have, though a
    source, such as the Neuron Persona Library, may name one so.

    Example:
        >>> names_no_persona(" None"), names_no_persona("Nonesuch")
        (True, False)
    """
    return name.strip().casefold() == NONE


@dataclass(frozen=True)
class Persona:
    """A persona: its name, as the judge must write it back, and what it is.

    Example:
        >>> Persona("Brass Owl", "A pompous mechanical owl.").name
        'Brass Owl'
    """

    name: str
    description: str = ""

    def __post_init__(self) -> None:
        """Refuse a name that is blank, has a line break or is "none"."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"a persona's name must be text, not {self.name!r}")
        if self.name != self.name.strip() or re.search(r"[\r\n\t]", self.name):
            raise ValueError(
                f"a persona's name must be one line with no spaces around it, "
                f"not {self.name!r}"
            )
        if names_no_persona(self.name):
            raise ValueError(
                f'a persona cannot be named "{NONE}": it is the answer for no persona'
            )
        if not isinstance(self.description, str):
            raise ValueError(
                f"{self.name}'s description must be text, not {self.description!r}"
            )


def read_personas(path: Path) -> list[Persona]:
    """The personas in a personas file, in its order.

    ValueError names what is wrong: a file that is not YAML, a key other
    than personas, a persona that is not a mapping of name and description,
    and two personas whose names differ only in case or spacing.

    Example:
        >>> read_personas(Path("experiments/personas/owls.yaml"))  # doctest: +SKIP
        [Persona(name='Brass Owl', description='A pompous mechanical owl.')]
    """
    try:
        data = load_yaml(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read the personas file {path}: {error}") from None
    if not isinstance(data, dict) or set(data) != {"personas"}:
        raise ValueError(f"{path} must hold one key, personas, a list of personas")
    entries = data["personas"]
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: personas must be a list of at least one persona")
    personas, seen = [], {}
    for place, entry in enumerate(entries):
        where = f"{path}: personas[{place}]"
        if not isinstance(entry, dict) or "name" not in entry:
            raise ValueError(f"{where} must be a mapping with a name")
        if unknown := sorted(set(entry) - set(PERSONA_KEYS)):
            raise ValueError(
                f"{where} has unknown keys {', '.join(map(str, unknown))}; a persona "
                "takes name and description"
            )
        description = entry.get("description", "")
        try:
            persona = Persona(entry["name"], description)
        except ValueError as error:
            raise ValueError(f"{where}: {error}") from None
        same = " ".join(persona.name.split()).casefold()
        if same in seen:
            raise ValueError(
                f"{where} repeats the name of personas[{seen[same]}]: {persona.name!r}"
            )
        seen[same] = place
        personas.append(persona)
    return personas


def pick_persona(personas: list[Persona], name: str | None) -> Persona:
    """The persona of personas called name; with no name, the only one.

    ValueError says when no persona has that name, listing the names, or
    when name is None and there is more than one to choose from.

    Example:
        >>> owls = [Persona("Brass Owl"), Persona("Snow Owl")]
        >>> pick_persona(owls, "Snow Owl").name
        'Snow Owl'
    """
    if name is None:
        if len(personas) == 1:
            return personas[0]
        listed = ", ".join(repr(persona.name) for persona in personas)
        raise ValueError(f"choose one persona with --persona: {listed}")
    for persona in personas:
        if persona.name == name:
            return persona
    listed = ", ".join(repr(persona.name) for persona in personas)
    raise ValueError(f"no persona is named {name!r}; the file has {listed}")
