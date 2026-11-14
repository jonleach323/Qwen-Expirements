"""pushlab panel stage preview: varied pairs of the panel, each pushed in four
shapes (decision 0023).

It reads the panel file that pushlab panel select wrote
(pushlab.panel.panel_file), and the candidates list it names, whose SHA-256
must still be the panel's, for each persona's unit: a set with its
definition. preview_pairs chooses the selection's preview_pairs pairs, each
persona in one at most: every pair of the panel ranked by how varied it is,
those of two categories first, then those of a neuron and a set, then those
with no layer in common, ties in an order drawn from configs/panel.yaml's
seed; then, best first, each pair whose two personas are both free. Each
pair is pushed in four conditions, each persona at its working dose in its
direction: a handoff from the first to the second at reply token
handoff_at; alternation every alternate_every tokens, the first first; a
pulse of the first for pulse_length tokens from reply token 0; and the
second from reply token onset_at (pushlab.schedules.shapes, each with its
defaults otherwise). They are labelled p1-handoff, p1-alternate, p1-pulse
and p1-onset, then p2-handoff and so on, and written as segments, so that a
set carries its definition, on the set's last preview_prompts prompts.
"""

import csv
import hashlib
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pushlab import PROJECT_ROOT
from pushlab.analysis.dose_text import aligned
from pushlab.analysis.panel import Panel
from pushlab.discovery.candidates import read_candidates
from pushlab.discovery.sources import LIBRARY
from pushlab.panel.panel_file import PanelFile, PanelPersona, read_panel_file
from pushlab.panel.settings import Selection
from pushlab.panel.stage_files import Stage, chosen_prompts, set_prompts
from pushlab.panel.stage_text import plain
from pushlab.runs.estimate import exact
from pushlab.schedules.model import Schedule
from pushlab.schedules.shapes import alternate, handoff, onset, pulse
from pushlab.schedules.validation import DEFAULT_CEILING
from pushlab.schedules.yaml_io import to_mapping
from pushlab.scoring.score_rows import counted
from pushlab.units.neuron_set import Unit, unit_layers

SHAPES = ("handoff", "alternate", "pulse", "onset")
Pair = tuple[PanelPersona, PanelPersona]


def panel_units(panel: PanelFile, where: Path | str = "the panel") -> dict[str, Unit]:
    """Each persona's unit, by its key, from the candidates list the panel
    names, a path from the repository's root or an absolute one.

    ValueError says why: a list that is missing or cannot be read, whose
    SHA-256 is not the panel's, that lacks a persona's key, or whose row for
    it pushes another unit or direction.
    """
    path = PROJECT_ROOT / panel.candidates_file  # an absolute path stays as it is
    try:
        found = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError(
            f"{where} was chosen from the candidates list {panel.candidates_file}, "
            f"which cannot be read: {error.strerror or error}"
        ) from None
    if found != panel.candidates_sha256:
        raise ValueError(
            f"{panel.candidates_file} has changed since {where} was chosen from it: "
            f"its SHA-256 is {found}, not {panel.candidates_sha256}; give the list "
            "the panel was chosen from, or choose the panel again"
        )
    try:
        by_key = {candidate.key: candidate for candidate in read_candidates(path)}
    except (OSError, ValueError, KeyError, TypeError, csv.Error) as error:
        raise ValueError(f"cannot read the candidates list {path}: {error}") from None
    units: dict[str, Unit] = {}
    for persona in panel.personas:
        candidate = by_key.get(persona.key)
        if candidate is None:
            raise ValueError(f"{panel.candidates_file} has no {persona.key}")
        if (str(candidate.unit), candidate.direction) != (
            persona.unit,
            persona.direction,
        ):
            raise ValueError(
                f"{where} pushes {persona.key} as {persona.unit} in direction "
                f"{persona.direction}, but {panel.candidates_file} has "
                f"{candidate.unit} in direction {candidate.direction}"
            )
        units[persona.key] = candidate.unit
    return units


def variety(pair: Pair, units: Mapping[str, Unit]) -> tuple[bool, bool, bool]:
    """How a pair varies: its personas' categories differ; one is a neuron
    and the other a set; their units share no layer.

    Example:
        >>> from types import SimpleNamespace as Persona
        >>> from pushlab.units.neuron import NeuronUnit
        >>> owl = Persona(key="o", category="animals", kind="neuron")
        >>> fox = Persona(key="f", category="animals", kind="neuron")
        >>> units = {
        ...     "o": NeuronUnit("qwen3-8b", 9, 1),
        ...     "f": NeuronUnit("qwen3-8b", 14, 2),
        ... }
        >>> variety((owl, fox), units)
        (False, False, True)
    """
    first, second = pair
    layers = set(unit_layers(units[first.key])) & set(unit_layers(units[second.key]))
    return first.category != second.category, first.kind != second.kind, not layers


def preview_pairs(
    personas: Sequence[PanelPersona], units: Mapping[str, Unit], count: int, seed: int
) -> list[Pair]:
    """count varied pairs of personas, or as many as they make, each persona
    in one at most, as the module's docstring says: the same personas,
    units and seed always give the same pairs, in the same order.

    Example:
        >>> preview_pairs([], {}, 6, seed=0)
        []
    """
    order = list(personas)
    random.Random(seed).shuffle(order)
    pairs = [(a, b) for i, a in enumerate(order) for b in order[i + 1 :]]
    # Most varied first; sorted() keeps the drawn order among equals.
    pairs.sort(key=lambda pair: tuple(not differs for differs in variety(pair, units)))
    chosen: list[Pair] = []
    used: set[str] = set()
    for first, second in pairs:
        if len(chosen) == count:
            break
        if first.key not in used and second.key not in used:
            chosen.append((first, second))
            used |= {first.key, second.key}
    return chosen


def pair_schedules(
    first: Unit, second: Unit, alphas: tuple[float, float], selection: Selection
) -> dict[str, Schedule]:
    """A pair's four schedules, by shape, as the module's docstring says;
    alphas are the first's and the second's signed doses."""
    a, b = alphas
    return {
        "handoff": handoff(first, second, a, b, selection.handoff_at),
        "alternate": alternate(first, second, a, b, selection.alternate_every),
        "pulse": pulse(first, a, 0, selection.pulse_length),
        "onset": onset(second, b, selection.onset_at),
    }


def _alpha(persona: PanelPersona) -> float:
    return persona.direction * persona.working_dose


def pair_conditions(
    number: int, pair: Pair, units: Mapping[str, Unit], selection: Selection
) -> list[dict[str, Any]]:
    """Pair number's four conditions, labelled p<number>-<shape>, each its
    schedule's segments written out, allowed above the dose ceiling when a
    persona's working dose is."""
    first, second = pair
    alphas = (_alpha(first), _alpha(second))
    over = max(map(abs, alphas)) > DEFAULT_CEILING
    schedules = pair_schedules(units[first.key], units[second.key], alphas, selection)
    conditions = []
    for shape, schedule in schedules.items():
        condition: dict[str, Any] = {
            "label": f"p{number}-{shape}",
            "segments": to_mapping(schedule)["segments"],
        }
        if over:
            condition["allow_over_ceiling"] = True
        conditions.append(condition)
    return conditions


def _layers(unit: Unit) -> str:
    return ", ".join(str(layer) for layer in unit_layers(unit))


def _describe(persona: PanelPersona, unit: Unit) -> str:
    layers = [str(layer) for layer in unit_layers(unit)]
    where = f"layer {layers[0]}"
    if len(layers) > 1:
        where = f"layers {', '.join(layers[:-1])} and {layers[-1]}"
    return (
        f"{persona.key} ({persona.persona}, {persona.category}, a {persona.kind} in "
        f"{where}) at {exact(_alpha(persona))}"
    )


def pairs_table(pairs: Sequence[Pair], units: Mapping[str, Unit]) -> list[str]:
    """The pairs as a table: a row per persona, its pair's label and its role
    first or second, and on the first's row what the pair varies in."""
    head = ["pair", "role", "candidate", "persona", "category", "kind", "layers"]
    table = [[*head, "alpha", "differ in"]]
    names = ("categories", "kinds", "layers")
    for number, pair in enumerate(pairs, 1):
        varied = zip(names, variety(pair, units), strict=True)
        differs = [name for name, yes in varied if yes]
        for role, persona in zip(("first", "second"), pair, strict=True):
            table.append(
                [f"p{number}", role, persona.key, plain(persona.persona)]
                + [persona.category, persona.kind, _layers(units[persona.key])]
                + [exact(_alpha(persona)), ""]
            )
        table[-2][-1] = ", ".join(differs) or "nothing"
    return aligned(table)


def stage_preview(source: Path, selection: Selection, panel: Panel) -> Stage:
    """Stage preview from the panel file at source: its preview_pairs pairs,
    each in four shapes, on the set's last preview_prompts prompts.

    ValueError when the panel cannot be read, has fewer than two personas,
    or its candidates list is not the one it was chosen from.
    """
    chosen = read_panel_file(source)
    units = panel_units(chosen, source)
    if len(chosen.personas) < 2:
        raise ValueError(
            f"{source} has {counted(len(chosen.personas), 'persona')}, and a preview "
            "needs two or more to pair"
        )
    pairs = preview_pairs(chosen.personas, units, selection.preview_pairs, panel.seed)
    prompts = chosen_prompts(
        set_prompts(selection), selection.preview_prompts, "preview_prompts", last=True
    )
    conditions = [
        condition
        for number, pair in enumerate(pairs, 1)
        for condition in pair_conditions(number, pair, units, selection)
    ]
    notes = {}
    for number, (first, second) in enumerate(pairs, 1):
        notes[len(SHAPES) * (number - 1)] = (
            f"Pair {number}: {_describe(first, units[first.key])}, then "
            f"{_describe(second, units[second.key])}."
        )
    about = (
        f"Stage preview of the persona panel (decision 0023), written by pushlab "
        f"panel stage preview from {source}: {len(pairs)} varied pairs of its "
        f"{len(chosen.personas)} personas, each persona at its working dose in its "
        f"direction, each pair in four shapes: a handoff from the first to the "
        f"second at reply token {selection.handoff_at}, alternation every "
        f"{selection.alternate_every} tokens, a pulse of the first for "
        f"{selection.pulse_length} tokens, and the second from reply token "
        f"{selection.onset_at}; on the last {len(prompts)} prompts of "
        f"{selection.prompts}. It names the panel's units, so it stays out of git "
        "(decision 0020)."
    )
    shown = [
        f"Read {source}: {len(chosen.personas)} personas of {chosen.model}, chosen "
        f"from {chosen.candidates_file}, whose SHA-256 is still the panel's.",
        "",
        f"Paired {len(pairs) * 2} of them, the most varied pairs first, drawn with "
        f"seed {panel.seed} ({panel.path}):",
        *pairs_table(pairs, units),
    ]
    return Stage(
        chosen.model,
        tuple(prompts),
        None,
        tuple(conditions),
        len(conditions) * len(prompts),
        about,
        condition_notes=notes,
        shown=tuple(shown),
        from_library=any(
            persona.source == LIBRARY.name for pair in pairs for persona in pair
        ),
    )
