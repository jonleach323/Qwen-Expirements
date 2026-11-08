"""One call of a run: its inputs checked again, then one generate() call.

What a run is made from was checked before the model loaded
(pushlab.runs.plan); session_inputs reads the calibration and the prompt
set again, as the session uses them, and refuses them if they changed in
between. run_call then runs one call's runs together, every row where the
plan puts it, and returns the records of its due runs: ok with their
replies and their steps (pushlab.store.steps), or all failed with the
error, when generate raised or the steps cannot be saved (an amount no
16-bit float holds, as a huge dose gives), so that one bad call never
stops an unattended run.
"""

import time
from dataclasses import dataclass, field, replace
from typing import Any

from pushlab.runs.plan import Call
from pushlab.runs.spec import ExperimentSpec
from pushlab.store.records import RunRecord
from pushlab.store.steps import MAX_FLOAT16, StepRecord, steps_table

# What a call's error adds when its steps hold an amount no 16-bit float can.
LOWER_THE_DOSE = (
    f"lower the dose, as 16-bit steps hold amounts up to {MAX_FLOAT16:,.0f}"
)


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
    """What one call gave: the due runs' records and their steps, the error if
    it raised or its steps cannot be saved, and its generation seconds and
    reply tokens over all its rows."""

    records: list[RunRecord]
    error: str | None
    seconds: float
    reply_tokens: int
    steps: list[StepRecord] = field(default_factory=list)


def reply_steps(run_id: str, reply: Any) -> list[StepRecord]:
    """A reply's steps: each token, its log-probability and the pushes on
    the pass that chose it (Generation.amounts and logprobs).

    Example:
        >>> from pushlab.engine.generate import Generation
        >>> reply = Generation("Hi", (7, 8), "max_new_tokens", (), ((), ()))
        >>> reply = replace(reply, logprobs=(-1.0, -2.0))
        >>> [step.logprob for step in reply_steps("x-000001", reply)]
        [-1.0, -2.0]
    """
    return [
        StepRecord(
            run_id,
            step,
            "reply",
            token,
            logprob,
            tuple(str(pushed.unit) for pushed in pushes),
            tuple(pushed.amount for pushed in pushes),
        )
        for step, (token, logprob, pushes) in enumerate(
            zip(reply.token_ids, reply.logprobs, reply.amounts, strict=True)
        )
    ]


def saveable_steps(
    replies: list[tuple[str, Any]],
) -> tuple[list[StepRecord], str | None]:
    """The steps of replies, given as (run ID, reply) pairs, checked as
    steps_table checks them before they are saved; or no steps and why they
    cannot be saved, with how to fix it.

    Example:
        >>> from pushlab.engine.generate import Generation
        >>> from pushlab.schedules.compile import UnitAmount
        >>> from pushlab.units.neuron import NeuronUnit
        >>> pushed = (UnitAmount(NeuronUnit("tiny", 1, 5), 70_000.0),)
        >>> reply = replace(
        ...     Generation("Hi", (7, 8), "max_new_tokens", (1,), ((), pushed)),
        ...     logprobs=(-1.0, -2.0),
        ... )
        >>> steps, why = saveable_steps([("x-000001", reply)])
        >>> steps, why.endswith(LOWER_THE_DOSE)
        ([], True)
    """
    try:
        steps = [
            step for run_id, reply in replies for step in reply_steps(run_id, reply)
        ]
        steps_table(steps)  # as saving them will check them
    except ValueError as error:
        why = f"the batch's steps cannot be saved: {error}"
        amounts = [
            pushed.amount
            for _, reply in replies
            for pushes in reply.amounts
            for pushed in pushes
        ]
        if not all(abs(amount) <= MAX_FLOAT16 for amount in amounts):  # or NaN
            why += f"; {LOWER_THE_DOSE}"
        return [], why
    return steps, None


def record_schedule(record: RunRecord) -> Any:
    """A run record's schedule as a Schedule: its segments, without the
    controls' notes (control_for, control_seed), which no Segment holds.

    Example:
        >>> from pushlab.store.records import RunRecord
        >>> segment = {"unit": "tiny:L1:N5", "alpha": 1.0, "control_for": None}
        >>> record = RunRecord(
        ...     "x-000001",
        ...     "x",
        ...     "tiny@main",
        ...     "p",
        ...     [segment],
        ...     {},
        ...     0,
        ...     "",
        ...     "",
        ...     "",
        ...     [],
        ...     "ok",
        ... )
        >>> [str(s.unit) for s in record_schedule(record).segments]
        ['tiny:L1:N5']
    """
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
            schedules=[record_schedule(record) for record in planned],
        )
    except Exception as error:  # the call's due runs are saved as failed
        message = f"{type(error).__name__}: {error}"
        if "out of memory" in str(error).lower():
            message += (
                f"; lower batch_size ({spec.batch_size}) in the experiment file, "
                "through a pull request"
            )
        failed = _failed(call, planned, stem, model)
        return Outcome(failed, message, time.perf_counter() - start, 0)
    seconds = time.perf_counter() - start
    tokens = sum(len(reply.token_ids) for reply in replies)
    done, due = [], []
    for row, (record, reply) in enumerate(zip(planned, replies, strict=True)):
        if record.run_id in call.due:
            done.append(
                replace(
                    record,
                    model=model,
                    batch_key=f"{stem}:{row}",
                    output_text=reply.text,
                    output_tokens=list(reply.token_ids),
                    status="ok",
                )
            )
            due.append((record.run_id, reply))
    # Checked now, so that steps no file can hold fail this call, not the run.
    steps, why = saveable_steps(due)
    if why is not None:
        return Outcome(_failed(call, planned, stem, model), why, seconds, tokens)
    return Outcome(done, None, seconds, tokens, steps)


def _failed(
    call: Call, planned: list[RunRecord], stem: str, model: str
) -> list[RunRecord]:
    """The call's due runs, failed, each at its row."""
    return [
        replace(record, model=model, batch_key=f"{stem}:{row}", status="failed")
        for row, record in enumerate(planned)
        if record.run_id in call.due
    ]
