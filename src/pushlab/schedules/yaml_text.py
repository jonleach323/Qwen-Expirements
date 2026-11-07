"""YAML text as schedule files are read and written: stricter than PyYAML's.

load_yaml differs from PyYAML's safe_load in two ways:
- A key given twice in one mapping is refused, not overwritten. Keys that a
  merge (<<) brings in may still be overridden, as YAML allows.
- Numbers and true/false follow YAML 1.2 rather than YAML 1.1: 010 is text,
  not 8 (octal); 1:20 is text, not 80 (base 60); 1e-1 is a number; yes, no,
  on and off are text; and so is a date such as 2026-11-07. An explicit tag
  such as !!int 010 is refused rather than read the YAML 1.1 way.
dump_yaml writes what load_yaml reads back unchanged: it quotes any text that
either YAML would read as something else, such as a name 1e-3.
"""

import re
from collections.abc import Callable
from typing import Any

import yaml

_BOOL, _INT, _FLOAT = (f"tag:yaml.org,2002:{tag}" for tag in ("bool", "int", "float"))
_TIMESTAMP = "tag:yaml.org,2002:timestamp"

# YAML 1.2's true/false, decimal integers and floats, in place of YAML 1.1's,
# which read yes and on as true, 010 as 8, 1:20 as 80 and 1e-1 as text; and,
# as in YAML 1.2, no dates: 2026-11-07 stays text. Each with its first letters.
_SCALARS = {
    _BOOL: (r"true|True|TRUE|false|False|FALSE", "tTfF"),
    _INT: (r"[-+]?(?:0|[1-9][0-9]*)", "-+0123456789"),
    _FLOAT: (
        r"[-+]?(?:(?:[0-9]+\.[0-9]*|\.[0-9]+)(?:[eE][-+]?[0-9]+)?|[0-9]+[eE][-+]?[0-9]+)"
        r"|[-+]?\.(?:inf|Inf|INF)|\.(?:nan|NaN|NAN)",
        "-+0123456789.",
    ),
}
_PATTERNS = {tag: re.compile(f"^(?:{text})$") for tag, (text, _) in _SCALARS.items()}
_KINDS = {_BOOL: "true or false", _INT: "a whole number", _FLOAT: "a number"}


class _ScheduleLoader(yaml.SafeLoader):
    """PyYAML's safe loader with YAML 1.2's scalars, refusing a key given twice."""

    def compose_mapping_node(self, anchor: str | None) -> yaml.MappingNode:
        # Checked as each mapping is read in, before a merge (<<) rewrites it:
        # so a mapping that is only merged in is checked too, and a key may
        # still override one that a merge brings in.
        node = super().compose_mapping_node(anchor)
        seen = set()
        for key_node, _ in node.value:
            if isinstance(key_node, yaml.ScalarNode):
                key = (key_node.tag, key_node.value)
                if key in seen:
                    raise yaml.composer.ComposerError(
                        "while reading a mapping",
                        node.start_mark,
                        f"duplicate key {key_node.value!r}",
                        key_node.start_mark,
                    )
                seen.add(key)
        return node


class _ScheduleDumper(yaml.SafeDumper):
    """PyYAML's safe dumper, which also quotes text that load_yaml would read
    as something else, such as the name 1e-3."""


def _strict(tag: str) -> Callable[[yaml.SafeLoader, yaml.Node], Any]:
    """PyYAML's constructor for tag, refusing text that YAML 1.2 does not
    read as one, so that an explicit tag such as !!int 010 is not 8."""
    construct = yaml.SafeLoader.yaml_constructors[tag]
    accepted = [_PATTERNS[tag], *([_PATTERNS[_INT]] if tag == _FLOAT else [])]

    def strict(loader: yaml.SafeLoader, node: yaml.Node) -> Any:
        text = loader.construct_scalar(node)
        if not any(pattern.match(text) for pattern in accepted):
            raise yaml.constructor.ConstructorError(
                None,
                None,
                f"{text!r} is not {_KINDS[tag]} as YAML 1.2 writes one",
                node.start_mark,
            )
        return construct(loader, node)

    return strict


_ScheduleLoader.yaml_implicit_resolvers = {
    first: [
        (tag, regexp) for tag, regexp in resolvers if tag not in (*_SCALARS, _TIMESTAMP)
    ]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
for _tag, (_, _first) in _SCALARS.items():
    _ScheduleLoader.add_implicit_resolver(_tag, _PATTERNS[_tag], list(_first))
    _ScheduleLoader.add_constructor(_tag, _strict(_tag))
    _ScheduleDumper.add_implicit_resolver(_tag, _PATTERNS[_tag], list(_first))


def load_yaml(text: str) -> Any:
    """Parse YAML text as schedule files are read (see the module's docstring).

    Example:
        >>> load_yaml("start: 010\\nend: 16\\nalpha: 1e-1\\ninclude_prompt: yes")
        {'start': '010', 'end': 16, 'alpha': 0.1, 'include_prompt': 'yes'}
        >>> load_yaml("alpha: 1\\nalpha: 2")  # doctest: +ELLIPSIS
        Traceback (most recent call last):
        ...
        ValueError: the text is not valid YAML: ...duplicate key 'alpha'...
    """
    try:
        return yaml.load(text, Loader=_ScheduleLoader)  # safe: a SafeLoader
    except (
        yaml.YAMLError,
        # What PyYAML's constructors raise on a value with an explicit tag
        # they cannot read, such as !!binary on a number, and on deep nesting.
        ValueError,
        TypeError,
        LookupError,
        AttributeError,
        RecursionError,
    ) as error:
        raise ValueError(f"the text is not valid YAML: {error}") from None


def dump_yaml(data: Any) -> str:
    """data as YAML text, keys in their order, which load_yaml reads back as it is.

    Example:
        >>> print(dump_yaml({"name": "1e-3", "alpha": 0.1, "end": None}), end="")
        name: '1e-3'
        alpha: 0.1
        end: null
        >>> load_yaml(dump_yaml({"name": "1e-3"}))
        {'name': '1e-3'}
    """
    return yaml.dump(data, Dumper=_ScheduleDumper, sort_keys=False)
