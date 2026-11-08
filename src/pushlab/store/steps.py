"""The steps table: one row per run and reply token, stored compactly.

Step j is reply token j and the pass that chose it, as decision 0008 asks of
a per-token record: the prompt pass for token 0, and the pass that read
token j - 1 for token j. So every column of a row is about one choice:
- token_id, the token chosen;
- logprob, its log-probability on that pass, from the model's next-token
  scores before any temperature or top-p, saved as a 16-bit float;
- active_units and active_amounts, the units pushed on that pass and the
  amount added to each one's neuron (decision 0013), the amounts 16-bit
  floats too. A unit pushed at dose 0 is on the list, with amount 0.
- region, where the token is: "reply" for now; thinking and answer come
  with thinking detection.

So a push on reply token k, which first changes token k + 1, shows at step
k + 1: a pulse on reply tokens 0 to 15 shows at steps 1 to 16. Steps are
small integers (16 bits, so a reply can have up to 32,768 tokens), token
IDs 32-bit, and unit IDs text that Parquet stores once per file: 1,000
replies of 1,000 tokens take 5 to 15 MiB, by how they are batched into
files (decision 0016).

Example:
    >>> pushed = (("tiny:L2:N100",), (1.5,))
    >>> steps_table([StepRecord("x-000001", 0, "reply", 7, -0.5, *pushed)]).num_rows
    1
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REGIONS = ("reply",)
MAX_STEP = 2**15 - 1  # the largest 16-bit step
MAX_FLOAT16 = 65504.0  # the largest finite 16-bit float

STEPS_SCHEMA = pa.schema(
    [
        pa.field("run_id", pa.string(), nullable=False),
        pa.field("step", pa.int16(), nullable=False),
        pa.field("region", pa.string(), nullable=False),
        pa.field("token_id", pa.int32(), nullable=False),
        pa.field("logprob", pa.float16(), nullable=False),
        pa.field("active_units", pa.list_(pa.string()), nullable=False),
        pa.field("active_amounts", pa.list_(pa.float16()), nullable=False),
    ]
)


@dataclass(frozen=True)
class StepRecord:
    """One reply token of one run, and the pushes on the pass that chose it.

    active_units are unit IDs, such as "qwen3.5-4b:L14:N8949", and
    active_amounts the amounts added to their neurons, in the same order.

    Example:
        >>> StepRecord("x-000001", 1, "reply", 7, -2.0, (), ()).step
        1
    """

    run_id: str
    step: int
    region: str
    token_id: int
    logprob: float
    active_units: tuple[str, ...]
    active_amounts: tuple[float, ...]


def _check(step: StepRecord) -> None:
    where = f"step {step.step!r} of {step.run_id!r}"
    if not isinstance(step.run_id, str) or not step.run_id:
        raise ValueError(f"a step's run_id must be non-empty text, not {step.run_id!r}")
    if type(step.step) is not int or not 0 <= step.step <= MAX_STEP:
        raise ValueError(f"{where}: step must be a whole number from 0 to {MAX_STEP}")
    if step.region not in REGIONS:
        raise ValueError(f"{where}: region must be one of {', '.join(REGIONS)}")
    if type(step.token_id) is not int or not 0 <= step.token_id < 2**31:
        raise ValueError(f"{where}: token_id must be a token ID, not {step.token_id!r}")
    if not (_nan(step.logprob) or (_float16(step.logprob) and step.logprob <= 0)):
        raise ValueError(
            f"{where}: logprob must be a log-probability, at most 0 (or NaN, "
            f"where the model's scores were not finite), not {step.logprob!r}"
        )
    units, amounts = step.active_units, step.active_amounts
    if not all(isinstance(unit, str) and unit for unit in units):
        raise ValueError(f"{where}: active_units must be unit IDs, not {units!r}")
    if len(set(units)) != len(units):
        raise ValueError(f"{where}: active_units names a unit twice: {units!r}")
    if len(amounts) != len(units) or not all(map(_float16, amounts)):
        raise ValueError(
            f"{where}: active_amounts must give each active unit a finite amount "
            f"a 16-bit float can hold, not {amounts!r}"
        )


def _nan(value: object) -> bool:
    return isinstance(value, float) and math.isnan(value)


def _float16(value: object) -> bool:
    """Whether value is a finite number a 16-bit float can hold."""
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(value)
        and abs(value) <= MAX_FLOAT16
    )


def steps_table(steps: Sequence[StepRecord]) -> pa.Table:
    """Check steps and make them a table in STEPS_SCHEMA.

    ValueError names the first step that is not one: a step out of range, an
    unknown region, a logprob above 0, amounts that do not match the units,
    and the like.

    Example:
        >>> steps_table([]).schema == STEPS_SCHEMA
        True
    """
    for step in steps:
        _check(step)
    return pa.table(
        {
            "run_id": [step.run_id for step in steps],
            "step": pa.array([step.step for step in steps], pa.int16()),
            "region": [step.region for step in steps],
            "token_id": pa.array([step.token_id for step in steps], pa.int32()),
            "logprob": pa.array(
                np.array([step.logprob for step in steps], np.float16), pa.float16()
            ),
            "active_units": pa.array(
                [list(step.active_units) for step in steps], pa.list_(pa.string())
            ),
            "active_amounts": pa.array(
                [np.array(step.active_amounts, np.float16) for step in steps],
                pa.list_(pa.float16()),
            ),
        },
        schema=STEPS_SCHEMA,
    )


def _kind(type_: pa.DataType) -> str:
    """A column type in plain words, such as float16 or list of float16.

    Example:
        >>> _kind(pa.list_(pa.float16())), _kind(pa.float64()), _kind(pa.int16())
        ('list of float16', 'float64', 'int16')
    """
    if pa.types.is_floating(type_):
        return f"float{type_.bit_width}"
    if pa.types.is_list(type_):
        return f"list of {_kind(type_.value_type)}"
    return str(type_)


def _differences(schema: pa.Schema) -> str:
    """How the columns of schema, which has STEPS_SCHEMA's names, differ
    from STEPS_SCHEMA's in type, or in whether they may be null."""

    def shown(field: pa.Field) -> str:
        return _kind(field.type) + (" or null" if field.nullable else "")

    return "; ".join(
        f"{want.name} is {shown(field)}, not {shown(want)}"
        for field, want in zip(schema, STEPS_SCHEMA, strict=True)
        if not field.equals(want)
    )


def read_steps_file(path: Path) -> list[StepRecord]:
    """The steps in a Parquet file that steps_table made, in its order.

    The 16-bit floats come back as Python floats, so a logprob of -0.1 reads
    back as -0.0999755859375. ValueError says when the file's columns are
    not STEPS_SCHEMA's: other names, other types, or columns that may be
    null. So steps saved as 64-bit numbers, as pandas or DuckDB would save
    them, are not a steps table, though they hold the same columns.

    Example:
        >>> read_steps_file(Path("runs/x/steps/v1-b000001-1.parquet"))  # doctest: +SKIP
        [StepRecord(run_id='x-000001', step=0, ...), ...]
    """
    # Absolute, so pyarrow never reads a folder name such as run:1 as a URL.
    table = pq.read_table(Path(path).absolute())
    if table.schema.names != STEPS_SCHEMA.names:
        raise ValueError(
            f"{path} is not a steps table: its columns are "
            f"{', '.join(table.schema.names)}"
        )
    if not table.schema.equals(STEPS_SCHEMA):
        raise ValueError(f"{path} is not a steps table: {_differences(table.schema)}")
    columns = table.to_pydict()
    return [
        StepRecord(
            run_id,
            step,
            region,
            token_id,
            float(logprob),
            tuple(units),
            tuple(float(amount) for amount in amounts),
        )
        for run_id, step, region, token_id, logprob, units, amounts in zip(
            *(columns[name] for name in STEPS_SCHEMA.names), strict=True
        )
    ]
