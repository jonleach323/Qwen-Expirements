"""pushlab panel stage preview (pushlab.panel.preview, decision 0023): varied
pairs of the panel, each pushed in four shapes.

A panel file of made-up tiny-vl personas (tests/unit/panel_experiments.py)
stands in for one pushlab panel select writes: the owl (animals, a neuron
in layer 1), the captain (people, layer 2), the teapot (objects, a set in
layers 1 and 2), the fox (animals, layer 3, working above the dose
ceiling) and, in some tests, the oracle (mystical, layer 0).
"""

import doctest
import hashlib
import importlib.util
from pathlib import Path

import pytest
import torch

from pushlab.panel import preview
from pushlab.panel.panel_file import (
    Mean,
    PanelFile,
    PanelPersona,
    Share,
    write_panel_file,
)
from pushlab.panel.preview import preview_pairs, variety
from pushlab.runs.expand import expand
from pushlab.runs.spec import read_spec
from pushlab.schedules.yaml_text import load_yaml

_SPEC = importlib.util.spec_from_file_location(
    "panel_experiments", Path(__file__).with_name("panel_experiments.py")
)
made = importlib.util.module_from_spec(_SPEC)  # type: ignore[arg-type]
_SPEC.loader.exec_module(made)  # type: ignore[union-attr]

# Each persona's category and working dose, as select would find them.
CHOSEN = {
    "owl": ("animals", 1.0),
    "captain": ("people", 0.25),
    "teapot": ("objects", 1.0),
    "fox": ("animals", 1.5),
    "oracle": ("mystical", 0.75),
}
SHAPES = ("handoff", "alternate", "pulse", "onset")


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run on the CPU even with a GPU."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _persona(role: str) -> PanelPersona:
    candidate = made.candidates()[role]
    category, working = CHOSEN[role]
    return PanelPersona(
        key=candidate.key,
        persona=candidate.persona,
        category=category,
        kind=candidate.kind,
        source=candidate.source,
        unit=str(candidate.unit),
        direction=candidate.direction,
        reference_dose=candidate.strength,
        working_dose=working,
        collapse_dose=None,
        reliability=Share(5, 5, 1.0, 0.57, 1.0),
        coherence=Mean(5, 8.0, 7.5, 8.5),
        distinct=Share(2, 2, 1.0, 0.34, 1.0),
    )


@pytest.fixture
def panel(tiny_downloads: object, tmp_path: Path):  # type: ignore[no-untyped-def]
    """Write a panel of roles, the first four by default, chosen from the
    made-up list; its path, and the list's and the settings' paths."""

    def write(roles=("owl", "captain", "teapot", "fox")):  # type: ignore[no-untyped-def]
        listed = made.write_list(tmp_path)
        sha = hashlib.sha256(listed.read_bytes()).hexdigest()
        personas = tuple(_persona(role) for role in roles)
        chosen = PanelFile("tiny-vl", str(listed), sha, {"confirm": "x"}, personas)
        path = write_panel_file(tmp_path / "panel_tiny-vl.yaml", chosen)
        return path, listed, made.write_settings(tmp_path / "panel.yaml")

    return write


def _preview(path: Path, settings: Path, out: Path, *options: str) -> int:
    argv = ["--panel", str(settings), "--max-gpu-hours", "1", *options]
    return made.stage("preview", path, out, *argv)


def test_preview_pushes_varied_pairs_in_four_shapes(
    panel, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:  # type: ignore[no-untyped-def]
    path, listed, settings = panel()
    out = tmp_path / "experiments" / "preview.yaml"
    assert _preview(path, settings, out) == 0
    written = load_yaml(out.read_text(encoding="utf-8"))
    assert "candidates" not in written and written["calibration"] == "fixed"
    assert written["prompt_ids"] == ["smoke-04", "smoke-05"]  # the last 2
    labels = [condition["label"] for condition in written["conditions"]]
    assert labels == [f"p{n}-{shape}" for n in (1, 2) for shape in SHAPES]
    spec = read_spec(out)
    assert len(expand(spec).runs) == 2 * 4 * 2  # pairs x shapes x prompts
    schedules = {c.label: c.points[0][1] for c in spec.conditions}
    keys = {c.key: role for role, c in made.candidates().items()}
    units = {str(c.unit): c for c in made.candidates().values()}
    # The teapot and the fox differ in category, kind and layers: the first
    # pair. The owl and the captain differ in category and layers.
    pairs = []
    for number in (1, 2):
        handoff = schedules[f"p{number}-handoff"].segments
        first, second = (units[str(handoff[i].unit)] for i in (0, 2))
        pairs.append({keys[first.key], keys[second.key]})
        alpha_a = first.direction * CHOSEN[keys[first.key]][1]
        alpha_b = second.direction * CHOSEN[keys[second.key]][1]
        # A handoff at reply token 2: the first on the prompt pass and reply
        # tokens 0 and 1, the second from token 2.
        assert [(s.region, s.start, s.end, s.alpha_start) for s in handoff] == [
            ("prompt", 0, None, alpha_a),
            ("reply", 0, 2, alpha_a),
            ("reply", 2, None, alpha_b),
        ]
        alternate = schedules[f"p{number}-alternate"].segments
        assert [(s.unit, s.start, s.on_for, s.off_for) for s in alternate] == [
            (first.unit, 0, 3, 3),
            (second.unit, 3, 3, 3),
        ]
        (pulse,) = schedules[f"p{number}-pulse"].segments
        assert (pulse.unit, pulse.start, pulse.end) == (first.unit, 0, 4)
        (onset,) = schedules[f"p{number}-onset"].segments
        assert (onset.unit, onset.start, onset.end) == (second.unit, 5, None)
        assert onset.alpha_start == alpha_b
    assert pairs == [{"teapot", "fox"}, {"owl", "captain"}]
    # The set carries its definition; the fox's 1.5 is above the ceiling.
    teapot = made.candidates()["teapot"].unit
    assert teapot in schedules["p1-handoff"].units()
    assert all(schedules[f"p1-{shape}"].allow_over_ceiling for shape in SHAPES)
    assert not any(schedules[f"p2-{shape}"].allow_over_ceiling for shape in SHAPES)
    text = out.read_text(encoding="utf-8")
    assert "set: {entries: [[1, 5, 0.75], [2, 7, -0.5]], size: 0.2, source: " in text
    printed = capsys.readouterr().out.splitlines()
    shown = [" ".join(line.split()) for line in printed]
    assert any(line.endswith("categories, kinds, layers") for line in shown)
    assert any(line.endswith("categories, layers") for line in shown)
    report = f"  uv run pushlab report {out} --out /workspace/reports/preview.md"
    assert report in printed
    # The same panel and seed write the same file.
    assert _preview(path, settings, out, "--overwrite") == 0
    assert out.read_text(encoding="utf-8") == text


def test_preview_pairs_are_the_most_varied_each_persona_once(
    tiny_downloads: object,
) -> None:
    roles = ("owl", "captain", "teapot", "fox", "oracle")
    personas = [_persona(role) for role in roles]
    units = {c.key: c.unit for c in made.candidates().values()}
    found = {seed: preview_pairs(personas, units, 6, seed) for seed in range(8)}
    for pairs in found.values():
        assert len(pairs) == 2  # five personas make two pairs at most
        used = [persona.key for pair in pairs for persona in pair]
        assert len(used) == len(set(used))
        # The first pair differs in every way: the teapot, a set, with the
        # fox or the oracle, neurons of other layers and categories.
        assert variety(pairs[0], units) == (True, True, True)
        assert "Velvet Teapot" in {persona.persona for persona in pairs[0]}
    # A seed always draws the same pairs, and other seeds draw others.
    assert preview_pairs(personas, units, 6, 3) == found[3]
    drawn = {tuple((a.key, b.key) for a, b in pairs) for pairs in found.values()}
    assert len(drawn) > 1
    assert preview_pairs(personas, units, 1, 0) == found[0][:1]


def test_variety_ranks_categories_then_kinds_then_layers(
    tiny_downloads: object,
) -> None:
    roles = ("owl", "captain", "teapot", "fox")
    owl, captain, teapot, fox = (_persona(role) for role in roles)
    units = {c.key: c.unit for c in made.candidates().values()}
    assert variety((owl, fox), units) == (False, False, True)
    assert variety((owl, teapot), units) == (True, True, False)  # layer 1 in both
    assert variety((captain, owl), units) == (True, False, True)
    # Of a pair of a neuron and a set that share a layer, and one of two
    # neurons of different layers, both of two categories: the first.
    pairs = preview_pairs([owl, captain, teapot], units, 1, 0)
    assert {persona.key for persona in pairs[0]} in (
        {owl.key, teapot.key},
        {captain.key, teapot.key},
    )


def test_preview_takes_no_judge_as_it_reads_no_verdicts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "preview.yaml"
    with pytest.raises(SystemExit):
        made.stage("preview", tmp_path / "panel.yaml", out, "--judge", "fake:fake")
    assert "unrecognized arguments: --judge" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize(
    ("situation", "problem"),
    [
        ("changed list", "has changed since"),
        ("one persona", "has 1 persona, and a preview needs two or more to pair"),
        ("no panel", "cannot read the panel"),
        ("unknown key", "has no library/L9N9-"),
        ("other unit", "pushes library/L1N3+ as tiny-vl:L1:N4 in direction 1, but"),
    ],
)
def test_preview_refuses_a_panel_it_cannot_pair(
    situation: str,
    problem: str,
    panel,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:  # type: ignore[no-untyped-def]
    roles = ("owl",) if situation == "one persona" else ("owl", "captain")
    path, listed, settings = panel(roles)
    if situation == "changed list":
        listed.write_text(listed.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    if situation == "no panel":
        path.unlink()
    if situation in ("unknown key", "other unit"):
        text = path.read_text(encoding="utf-8")
        if situation == "unknown key":
            text = text.replace("library/L2N100-", "library/L9N9-")
        else:
            text = text.replace("unit: tiny-vl:L1:N3", "unit: tiny-vl:L1:N4")
        path.write_text(text, encoding="utf-8")
    out = tmp_path / "experiments" / "preview.yaml"
    assert _preview(path, settings, out) == 1
    err = capsys.readouterr().err
    assert err.startswith("pushlab panel stage preview: ") and problem in err
    assert not out.exists()


def test_the_examples_in_the_docstrings_hold() -> None:
    result = doctest.testmod(preview)
    assert result.failed == 0 and result.attempted > 0
