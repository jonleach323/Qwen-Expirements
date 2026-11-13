"""The measures that need no judge, per dose of a unit's ladder, as pushlab
doses shows them (decision 0022).

pushlab measure saves each reply's repetition and word variety, its grade
when the prompt set has an answers file beside it, and with --perplexity
its perplexity, each under the key of what it measured
(pushlab.scoring.measure_rows). current_measures names those keys as
pushlab measure would make them now: the panel's n-gram and window, the
answers file beside the prompt set as it is now, and the model the replies
were made with. dose_measures reads each reply's ok row of each such key
only, so a row made with another panel, answers file or model is never
mixed in, and gives each dose the mean repetition, the share text_flags
flags, the median perplexity and the share right.
"""

import hashlib
import statistics
from collections.abc import Sequence

from pushlab.analysis.panel import Panel
from pushlab.scoring.answers import answers_path
from pushlab.scoring.items import Experiment
from pushlab.scoring.measure_rows import (
    Measure,
    capability_measure,
    perplexity_measure,
    repetition_and_variety,
)
from pushlab.scoring.text_measures import TextMeasures, text_flags, text_units
from pushlab.store.records import RunRecord
from pushlab.store.scores import ScoreRow

# What a dose shows of the measures, with its decimal places: the mean
# repetition, the share flagged, the median perplexity and the share right.
SHOWN = {"repetition": 2, "flagged": 2, "perplexity": 1, "capability": 2}


def current_measures(experiment: Experiment, panel: Panel) -> dict[str, Measure]:
    """The measures pushlab measure would save on the experiment now, by
    scorer: repetition and word_variety at the panel's n-gram and window;
    capability, against the answers file beside the prompt set as it is
    now, if there is one; and perplexity, under the model the folder's
    replies were made with, in the experiment's data type.

    Example:
        >>> sorted(current_measures(experiment, read_panel()))  # doctest: +SKIP
        ['perplexity', 'repetition', 'word_variety']
    """
    measures = {measure.scorer: measure for measure in repetition_and_variety(panel)}
    answers = answers_path(experiment.spec.prompts_path)
    if answers.is_file():
        sha256 = hashlib.sha256(answers.read_bytes()).hexdigest()
        measures["capability"] = capability_measure(sha256)
    model = experiment.identity.get("model")  # <name>@<revision>, as run
    if isinstance(model, str) and model:
        dtype = experiment.spec.decoding["dtype"]
        measures["perplexity"] = perplexity_measure(model, dtype)
    return measures


def dose_measures(
    pairs: Sequence[tuple[float, RunRecord]],
    rows: Sequence[ScoreRow],
    measures: dict[str, Measure],
    version: int,
    panel: Panel,
) -> dict[float, dict[str, float]]:
    """Each dose's measures, from each reply's ok row under each of measures'
    keys on the folder's version, rows being newest last: the mean
    repetition, the share text_flags flags (where both text measures are
    saved), the median perplexity and the share right on the capability
    set. A dose leaves out a measure none of its replies has.

    Example:
        >>> from pushlab.analysis.panel import read_panel
        >>> dose_measures([], [], {}, 1, read_panel())
        {}
    """
    ok = {row.key: row for row in rows if row.status == "ok"}

    def saved(name: str, run: RunRecord) -> ScoreRow | None:
        measure = measures.get(name)
        return None if measure is None else ok.get(measure.key(run.run_id, version))

    values: dict[float, dict[str, list[float]]] = {}
    for dose, run in sorted(pairs, key=lambda pair: pair[0]):
        got = values.setdefault(dose, {name: [] for name in SHOWN})
        rep, var = saved("repetition", run), saved("word_variety", run)
        ppl, cap = saved("perplexity", run), saved("capability", run)
        if rep is not None and rep.value is not None:
            got["repetition"].append(rep.value)
            if var is not None and var.value is not None:
                units = len(text_units(run.output_text))
                flags = text_flags(TextMeasures(units, rep.value, var.value), panel)
                got["flagged"].append(float(bool(flags)))
        if ppl is not None and ppl.value is not None:
            got["perplexity"].append(ppl.value)
        if cap is not None:
            got["capability"].append(float(cap.label == "correct"))
    average = dict.fromkeys(SHOWN, statistics.fmean) | {"perplexity": statistics.median}
    return {
        dose: {name: average[name](each) for name, each in got.items() if each}
        for dose, got in values.items()
    }
