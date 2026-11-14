"""Model families: a judge must not judge its own family's writing.

A model may favour text that reads like its own, so a judge is refused when
it is of the family of the model whose replies it judges (decision 0021).
A model's family is the first word of letters of its name: qwen for
Qwen/Qwen3-8B and qwen3.5-4b, granite for granite-4.1-8b, claude for
claude-haiku-5-5. A few models made from another family's model go by a
name of their own, and ALIASES gives that family: QwQ and Marco-o1 are
Qwen models. A judge whose name holds the tested family as a word, as a
distilled "...-distill-qwen-7b" does, counts as of that family too.

The rule reads names only: a model served under another name than its own,
or made from another family's model under a name ALIASES does not know, is
not caught, so serve a local judge under its own name (docs/runbooks/judge.md).
"""

import re
from collections.abc import Iterable

from pushlab.models.registry import REGISTRY

# Names of models made from another family's model, and that family.
ALIASES = {"qwq": "qwen", "marco": "qwen"}


def family(model: str) -> str:
    """A model's family: the first word of letters of its name, after any
    owner/ and before any @revision, as ALIASES maps it.

    Example:
        >>> [family(m) for m in ("Qwen/Qwen3-8B", "qwen3.5-4b@85", "granite-4.1-8b")]
        ['qwen', 'qwen', 'granite']
        >>> family("QwQ-32B"), family("AIDC-AI/Marco-o1")
        ('qwen', 'qwen')
    """
    name = model.split("@", 1)[0].rsplit("/", 1)[-1].casefold()
    match = re.match(r"[a-z]+", name)
    word = match[0] if match else name
    return ALIASES.get(word, word)


def _families(model: str) -> set[str]:
    """The families of a model, by its name and, for one of the registry,
    its Hub ID."""
    name = model.split("@", 1)[0]
    found = {family(name)}
    if name in REGISTRY:
        found.add(family(REGISTRY[name].hf_id))
    return found


def same_family(judge_model: str, model: str) -> bool:
    """Whether a judge is of the family of the model whose text it judges:
    its family is one of the model's, or its name holds one as a word, as a
    distilled "...-distill-qwen-7b" does.

    Example:
        >>> same_family("granite-4.1-8b", "qwen3-8b")
        False
        >>> same_family("deepseek-r1-distill-qwen-7b", "qwen3-8b")
        True
    """
    families = _families(model)
    name = judge_model.casefold()
    return family(judge_model) in families or any(
        re.search(rf"(?<![a-z]){re.escape(word)}(?![a-z])", name) for word in families
    )


def check_family(judge_model: str, models: Iterable[str]) -> None:
    """Refuse, naming both, a judge of the family of any model given.

    Example:
        >>> try:
        ...     check_family("qwen2.5-7b", ["qwen3-8b"])
        ... except ValueError as error:
        ...     print(error)  # doctest: +ELLIPSIS
        the judge qwen2.5-7b is of the family of qwen3-8b (qwen): choose ...
    """
    for model in sorted(set(models)):
        if same_family(judge_model, model):
            words = "/".join(sorted(_families(model)))
            raise ValueError(
                f"the judge {judge_model} is of the family of {model} ({words}): "
                "choose a judge of another family"
            )
