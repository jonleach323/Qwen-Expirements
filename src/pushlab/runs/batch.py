"""One batch of a run: its inputs checked again, then one generate() call.

What a run is made from was checked before the model loaded
(pushlab.runs.plan); session_inputs reads the calibration and the prompt
set again, as the session uses them, and refuses them if they changed in
between. run_batch then runs a planned batch's runs together and returns
their records: ok with their replies, or all failed with the error, so
that one bad batch never stops an unattended run.
"""

import time
from dataclasses import replace
from typing import Any

from pushlab.runs.plan import Batch
from pushlab.runs.spec import ExperimentSpec
from pushlab.store.records import RunRecord


def session_inputs(
    spec: ExperimentSpec, revision: str, identity: dict[str, Any], chosen: list[Batch]
) -> str | tuple[list[float] | None, dict[str, str]]:
    """The calibration's sizes and the prompts' texts, or why the inputs read
    now are not the ones the folder's version was made from."""
    from pushlab.prompts import read_prompt_set
    from pushlab.store.environment import file_sha256
    from pushlab.units import calibrate

    if f"{spec.model}@{revision}" != identity["model"]:
        return f"{spec.model}'s download changed while it loaded; run again"
    norms = None
    if any(run.record.schedule for batch in chosen for run in batch.runs):
        calibration = calibrate.load_calibration(spec.model, revision)
        path = calibrate.calibration_path(spec.model, revision)
        if file_sha256(path) != identity["calibration_sha256"]:
            return f"{path} changed while it was read; run again"
        norms = list(calibration.norms)
    prompt_set = read_prompt_set(spec.prompts_path)
    if prompt_set.sha256 != identity["prompts_sha256"]:
        return f"{spec.prompts_path} changed while it was read; run again"
    return norms, dict(zip(prompt_set.prompt_ids, prompt_set.prompts, strict=True))


def run_batch(
    batch: Batch,
    spec: ExperimentSpec,
    loaded: Any,
    norms: list[float] | None,
    texts: dict[str, str],
    generate: Any,
) -> tuple[list[RunRecord], str | None, float, int]:
    """Run one batch: its records, its error or None, its seconds, its tokens.

    Each record's batch_key is where it ran: the batch, and its row there.
    """
    from pushlab.engine.decoding import Decoding
    from pushlab.schedules.yaml_io import from_mapping

    planned = [run.record for run in batch.runs]
    first = planned[0]
    for record in planned:  # what lets runs share a batch
        if (record.prompt_id, record.seed, record.decoding, record.model) != (
            first.prompt_id,
            first.seed,
            first.decoding,
            first.model,
        ):
            raise ValueError(f"{batch.key} mixes runs that cannot share a batch")
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
        schedules = [
            from_mapping(
                {
                    "name": "run",
                    "segments": [
                        {
                            k: v
                            for k, v in s.items()
                            if k not in ("control_for", "control_seed")
                        }
                        for s in record.schedule
                    ],
                }
            )
            for record in planned
        ]
        prompts = [texts[first.prompt_id]] * len(planned)
        replies = generate(
            loaded.model,
            loaded.tokenizer,
            prompts,
            [],
            decoding,
            norms,
            schedules=schedules,
        )
    except Exception as error:  # the batch's runs are saved as failed
        seconds = time.perf_counter() - start
        message = f"{type(error).__name__}: {error}"
        if "out of memory" in str(error).lower():
            message += f"; lower batch_size in the experiment file ({spec.batch_size})"
        failed = [
            replace(r, model=model, batch_key=f"{batch.key}:{i}", status="failed")
            for i, r in enumerate(planned)
        ]
        return failed, message, seconds, 0
    seconds = time.perf_counter() - start
    done = [
        replace(
            record,
            model=model,
            batch_key=f"{batch.key}:{row}",
            output_text=reply.text,
            output_tokens=list(reply.token_ids),
            status="ok",
        )
        for row, (record, reply) in enumerate(zip(planned, replies, strict=True))
    ]
    return done, None, seconds, sum(len(reply.token_ids) for reply in replies)
