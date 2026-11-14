"""Made-up persona candidates, for the tests of experiment files that name
their units by rule (pushlab.runs.spec_candidates, each_candidate).

None of it is the Neuron Persona Library's data, which stays out of git
(decision 0020): the neurons, sets, strengths and names are invented, on
qwen3-8b and on the tests' tiny-vl. library_neurons spreads neurons over
layers, as a sample across layers needs; owl_set makes a set as Mayner's
are listed, dose 1 his strength.
"""

from collections.abc import Sequence

from pushlab.discovery.candidates import Candidate
from pushlab.units.neuron import NeuronUnit
from pushlab.units.neuron_set import NeuronSet

# An experiment of each_candidate on a list, for str.format with its file,
# the rule after the file in candidates:, such as ", kind: neuron", and its
# conditions, such as EACH.
EXPERIMENT = """\
name: owls
model: qwen3-8b
prompts: prompts/smoke.jsonl
decoding: {{strategy: greedy}}
seeds: [0]
max_new_tokens: 16
batch_size: 8
max_gpu_hours: 1
candidates: {{file: {file}{rule}}}
conditions:
{conditions}"""
EACH = "  - {label: screen, each_candidate: {}}\n"
NAMES = ("Brass Owl", "Salt Captain", "Velvet Teapot", "Moss Oracle", "Tin Fox")
# The library's strengths are often 1 or 1.25; some pass the ceiling of 1.4.
STRENGTHS = (1.0, 1.25, 0.75, 1.5, 0.5)


def neuron(
    layer: int,
    index: int,
    direction: int = 1,
    strength: float = 1.0,
    *,
    model: str = "qwen3-8b",
    persona: str = "Brass Owl",
) -> Candidate:
    """A made-up library candidate: one neuron, pushed in direction at
    strength, its key library/L<layer>N<index> and the direction's sign."""
    sign = "+1" if direction > 0 else "-1"
    return Candidate(
        "library",
        persona,
        NeuronUnit(model, layer, index),
        direction,
        strength,
        False,
        "library/qwen/library.json",
        f"invented L{layer} N{index} s{sign}",
        category="persona",
        coherence=7,
        weirdness=6,
    )


def owl_set(
    name: str,
    entries: Sequence[tuple[int, int, float]],
    size: float | None = 96.04,
    *,
    model: str = "qwen3-8b",
    persona: str = "brass owl",
) -> Candidate:
    """A made-up set listed as Mayner's are, direction +1 and strength 1,
    its key mayner/<name>."""
    source = f"invented role {persona}, top {len(entries)}"
    unit = NeuronSet(model, name, list(entries), source, size)
    return Candidate(
        "mayner",
        persona,
        unit,
        1,
        1.0,
        False,
        "mayner/bundle/split/top_neurons.parquet",
        source,
        category="role",
        size_from="vectors",
    )


def library_neurons(
    count: int, layers: Sequence[int], *, model: str = "qwen3-8b"
) -> list[Candidate]:
    """count made-up library neurons, candidate n in layers[n % len(layers)],
    neuron 100 + n, every other one pushed down, strengths from STRENGTHS:
    on layers [4, 9], library/L4N100+, library/L9N101-, library/L4N102+."""
    return [
        neuron(
            layers[n % len(layers)],
            100 + n,
            1 if n % 2 == 0 else -1,
            STRENGTHS[n % len(STRENGTHS)],
            model=model,
            persona=f"{NAMES[n % len(NAMES)]} {n}",
        )
        for n in range(count)
    ]


def mayner_sets(count: int) -> list[Candidate]:
    """count made-up sets of two neurons on qwen3-8b, across layers 17 and
    18: mayner/role-0-top2, mayner/role-1-top2 and so on."""
    return [
        owl_set(
            f"role-{n}-top2",
            [(17, 1000 + n, 0.5 + n / 10), (18, 2000 + n, -0.25)],
            90.0 + n,
            persona=f"role {n}",
        )
        for n in range(count)
    ]
