"""One call of a run: its inputs checked again, then one generate() call.

What a run is made from was checked before the model loaded
(pushlab.runs.plan); session_inputs reads the calibration and the prompt
set again, as the session uses them, and refuses them if they changed in
between. run_call then runs one call's runs together, every row where the
plan puts it, and returns the records of its due runs: ok with their
replies, or all failed with the error, so that one bad call never stops an
unattended run.
"""

import time
from dataclasses import dataclass, replace
from typing import Any

from pushlab.runs.plan import Call
from pushlab.runs.spec import ExperimentSpec
from pushlab.store.records import RunRecord


@dataclass(frozen=True)
class Inputs:
    """What a session generates with: the calibration's sizes, if anything is
    pushed, and each prompt's text by its ID."""

    norms: list[float] | None
    texts: dict[str, str]


def session_inputs(
    spec: ExperimentSpec, revision: str, identity: dict[str, Any], chosen: list[Call]
) -> Inputs:
    """Read the calibration and the prompts again, as the session uses them.

    ValueError says which changed since the folder's version was made from
    them, such as a prompt set edited while the model loaded.

    Example:
        >>> session_inputs(
        ...     spec, loaded.revision, identity, chosen
        ... ).texts  # doctest: +SKIP
        {'smoke-01': 'Give me a simple recipe for pancakes.', ...}
    """
    from pushlab.prompts import read_prompt_set
    from pushlab.store.environment import file_sha256
    from pushlab.units import calibrate

    if f"{spec.model}@{revision}" != identity["model"]:
        raise ValueError(f"{spec.model}'s download changed while it loaded; run again")
    norms = None
    if any(run.record.schedule for call in chosen for run in call.runs):
        calibration = calibrate.load_calibration(spec.model, revision)
        path = calibrate.calibration_path(spec.model, revision)
        if file_sha256(path) != identity["calibration_sha256"]:
            raise ValueError(f"{path} changed while it was read; run again")
        norms = list(calibration.norms)
    prompt_set = read_prompt_set(spec.prompts_path)
    if prompt_set.sha256 != identity["prompts_sha256"]:
        raise ValueError(f"{spec.prompts_path} changed while it was read; run again")
    return Inputs(
        norms, dict(zip(prompt_set.prompt_ids, prompt_set.prompts, strict=True))
    )


@dataclass(frozen=True)
class Outcome:
    """What one call gave: the due runs' records, the error if it raised, and
    its generation seconds and reply tokens over all its rows."""

    records: list[RunRecord]
    error: str | None
    seconds: float
    reply_tokens: int


def _schedule(record: RunRecord) -> Any:
    from pushlab.schedules.yaml_io import from_mapping

    segments = [
        {k: v for k, v in s.items() if k not in ("control_for", "control_seed")}
        for s in record.schedule
    ]
    return from_mapping({"name": "run", "segments": segments})


def run_call(
    call: Call,
    stem: str,
    spec: ExperimentSpec,
    loaded: Any,
    inputs: Inputs,
    generate: Any,
) -> Outcome:
    """Run every row of one call, and return its due runs' records.

    Each record's batch_key is where it ran: stem, its shard's name without
    .parquet, and its row there, such as v1-b000001-1:0.

    Example:
        >>> run_call(
        ...     call, "v1-b000001-1", spec, loaded, inputs, generate
        ... ).error  # doctest: +SKIP
    """
    from pushlab.engine.decoding import Decoding

    planned = [run.record for run in call.runs]
    first = planned[0]
    for record in planned:  # what lets runs share a batch
        if (record.seed, record.decoding, record.model) != (
            first.seed,
            first.decoding,
            first.model,
        ):
            raise ValueError(f"{call.key} mixes runs that cannot share a batch")
    model = f"{spec.model}@{loaded.revision}"
    sample = spec.decoding["strategy"] == "sample"
    start = time.perf_counter()
    try:
        decoding = Decoding(
            strategy=spec.decoding["strategy"],
            max_new_tokens=spec.max_new_tokens,
            temperature=spec.decoding["temperature"] if sample else None,
            top_p=spec.decoding["top_p"],
            seed=first.seed if sample else None,
        )
        replies = generate(
            loaded.model,
            loaded.tokenizer,
            [inputs.texts[record.prompt_id] for record in planned],
            [],
            decoding,
            inputs.norms,
            schedules=[_schedule(record) for record in planned],
        )
    except Exception as error:  # the call's due runs are saved as failed
        message = f"{type(error).__name__}: {error}"
        if "out of memory" in str(error).lower():
            message += (
                f"; lower batch_size ({spec.batch_size}) in the experiment file, "
                "through a pull request"
            )
        failed = [
            replace(record, model=model, batch_key=f"{stem}:{row}", status="failed")
            for row, record in enumerate(planned)
            if record.run_id in call.due
        ]
        return Outcome(failed, message, time.perf_counter() - start, 0)
    seconds = time.perf_counter() - start
    done = [
        replace(
            record,
            model=model,
            batch_key=f"{stem}:{row}",
            output_text=reply.text,
            output_tokens=list(reply.token_ids),
            status="ok",
        )
        for row, (record, reply) in enumerate(zip(planned, replies, strict=True))
        if record.run_id in call.due
    ]
    tokens = sum(len(reply.token_ids) for reply in replies)
    return Outcome(done, None, seconds, tokens)
