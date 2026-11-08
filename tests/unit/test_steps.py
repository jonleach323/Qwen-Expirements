"""The steps table (pushlab.store.steps): its schema, its checks, its size."""

import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pushlab.store.steps import (
    MAX_STEP,
    STEPS_SCHEMA,
    StepRecord,
    read_steps_file,
    steps_table,
)

UNIT_A, UNIT_B = "qwen3.5-4b:L14:N8949", "qwen3.5-4b:L13:N7956"


def _step(**changes: object) -> StepRecord:
    fields = {
        "run_id": "x-000001",
        "step": 0,
        "region": "reply",
        "token_id": 13048,
        "logprob": -0.5,
        "active_units": (UNIT_A,),
        "active_amounts": (1.25,),
    } | changes
    return StepRecord(**fields)  # type: ignore[arg-type]


def test_steps_read_back_as_written_with_16_bit_floats(tmp_path: Path) -> None:
    steps = [
        _step(),
        _step(step=1, logprob=-0.1, active_units=(), active_amounts=()),
        _step(
            step=2,
            token_id=248_045,
            logprob=-12.0,
            active_units=(UNIT_A, UNIT_B),
            active_amounts=(0.0, -3.5),
        ),
    ]
    path = tmp_path / "steps.parquet"
    pq.write_table(steps_table(steps), path)
    back = read_steps_file(path)
    # 0.5, 1.25, 12, 3.5 and 0 are exact in 16 bits; 0.1 is not.
    assert back[0] == steps[0] and back[2] == steps[2]
    assert back[1].logprob == float(np.float16(-0.1)) != -0.1
    assert back[1].active_units == () and back[1].active_amounts == ()


def test_the_columns_are_small_integers_and_16_bit_floats() -> None:
    table = steps_table([_step()])
    assert table.schema == STEPS_SCHEMA
    types = {field.name: field.type for field in STEPS_SCHEMA}
    assert types["step"] == pa.int16() and types["token_id"] == pa.int32()
    assert types["logprob"] == pa.float16()
    assert types["active_amounts"] == pa.list_(pa.float16())
    assert types["active_units"] == pa.list_(pa.string())


def test_a_logprob_may_be_nan_where_the_scores_were_not_finite() -> None:
    (logprob,) = steps_table([_step(logprob=math.nan)]).column("logprob").to_pylist()
    assert math.isnan(logprob)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"run_id": ""}, "run_id must be non-empty text"),
        ({"step": -1}, "step must be a whole number from 0 to 32767"),
        ({"step": MAX_STEP + 1}, "step must be a whole number from 0 to 32767"),
        ({"step": True}, "step must be a whole number"),
        ({"region": "thinking"}, "region must be one of reply"),
        ({"token_id": -1}, "token_id must be a token ID"),
        ({"logprob": 0.25}, "logprob must be a log-probability, at most 0"),
        ({"logprob": -math.inf}, "logprob must be a log-probability"),
        ({"logprob": -70_000.0}, "logprob must be a log-probability"),
        ({"active_units": ("",)}, "active_units must be unit IDs"),
        ({"active_units": (UNIT_A, UNIT_A), "active_amounts": (1.0, 1.0)}, "twice"),
        ({"active_amounts": ()}, "active_amounts must give each active unit"),
        ({"active_amounts": (math.inf,)}, "a 16-bit float can hold"),
        ({"active_amounts": (1e6,)}, "a 16-bit float can hold"),
    ],
)
def test_a_step_that_is_not_one_is_refused_by_name(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message) as error:
        steps_table([_step(), _step(**({"step": 1} | changes))])
    if "run_id" not in changes:
        assert "step" in str(error.value) and "'x-000001'" in str(error.value)


def test_a_file_with_other_columns_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "other.parquet"
    pq.write_table(pa.table({"run_id": ["x-000001"]}), path)
    with pytest.raises(
        ValueError, match="is not a steps table: its columns are run_id"
    ):
        read_steps_file(path)


def test_a_file_with_the_columns_but_other_types_is_refused(tmp_path: Path) -> None:
    """Steps saved as 64-bit numbers, as pandas or DuckDB would save them,
    are not the compact steps table, though every column is there."""
    path = tmp_path / "wide.parquet"
    columns = steps_table([_step()]).to_pydict()
    wide = {"step": pa.int64(), "token_id": pa.int64(), "logprob": pa.float64()}
    wide |= {"active_amounts": pa.list_(pa.float64())}
    table = pa.table(
        {name: pa.array(values, wide.get(name)) for name, values in columns.items()}
    )
    assert table.schema.names == STEPS_SCHEMA.names
    pq.write_table(table, path)
    with pytest.raises(ValueError, match="is not a steps table") as error:
        read_steps_file(path)
    message = str(error.value)
    # Every column of such a file may also be null, which no step's may be.
    assert "step is int64 or null, not int16" in message
    assert "logprob is float64 or null, not float16" in message
    assert "active_amounts is list of float64 or null, not list of float16" in message
    assert "run_id is string or null, not string" in message
    # The file itself is fine once its types are the steps table's.
    pq.write_table(table.cast(STEPS_SCHEMA), path)
    assert read_steps_file(path) == [_step()]


def test_1000_runs_of_1000_tokens_stay_well_under_100_mb(tmp_path: Path) -> None:
    """A thousand replies of a thousand tokens each, half of them pushed by
    one or two units, as a large experiment's steps would be."""
    rng = np.random.default_rng(0)
    runs, tokens = 1000, 1000
    path = tmp_path / "steps.parquet"
    tables = []
    for run in range(runs):
        token_ids = rng.integers(0, 248_320, tokens)
        logprobs = -rng.exponential(2.0, tokens)
        pushed = run % 2 == 1
        units = [UNIT_A, UNIT_B] if run % 4 == 3 else [UNIT_A]
        steps = [
            StepRecord(
                f"exp-{run + 1:06d}",
                step,
                "reply",
                int(token_ids[step]),
                float(logprobs[step]),
                tuple(units) if pushed and step > 0 else (),
                tuple(1.115 * (i + 1) for i in range(len(units)))
                if pushed and step > 0
                else (),
            )
            for step in range(tokens)
        ]
        tables.append(steps_table(steps))
    pq.write_table(pa.concat_tables(tables), path)
    size = path.stat().st_size
    assert size < 20 * 2**20  # well under 100 MB: under 20 MiB
    assert pq.read_metadata(path).num_rows == runs * tokens


def test_a_file_reads_through_a_relative_path_pyarrow_could_take_for_a_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    folder = Path("runs:1")  # as pushlab run --runs-dir runs:1 names it
    folder.mkdir()
    pq.write_table(steps_table([_step()]), (folder / "steps.parquet").absolute())
    assert read_steps_file(folder / "steps.parquet") == [_step()]
