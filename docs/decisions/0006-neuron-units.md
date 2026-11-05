# 0006: A neuron is named `model:L<layer>:N<index>`, and its dose is scaled by its layer's residual size

Date: 2026-11-04 · Sprint: S1 · Status: accepted

## Context

- S1 starts the push engine with its smallest piece: one MLP neuron as a
  unit that can be pushed (`src/pushlab/units/neuron.py`), and the amount
  a dose adds to it (`src/pushlab/units/dose.py`). Later S1 steps measure
  the residual sizes, add the amounts with hooks and run the evil-neuron
  smoke test. The random controls in S3 must avoid known units.
- The development plan names a unit `model:layer:index`, for example
  `qwen3.5-4b:L14:N8949` ("Identifiers"). The research plan's dose rule is
  c = α · r̃_L / ‖W_down[:, n]‖, where r̃_L is the median residual size at
  the neuron's layer L ("Concepts").
- The published recipe and its replications count layers and neurons from
  0. occcat tested it: reading the recipe's layers as 1-indexed gave no
  persona shift. hedwigai writes "Layers and neurons are numbered from 0",
  and Mr-Neutr0n's table is headed "Zero-based layer".

## Decision

- **ID format.** `<registry name>:L<layer>:N<index>`: a capital L and N,
  and the numbers in ASCII digits with no sign, spaces or leading zeros.
  `str(unit)` writes exactly this and `NeuronUnit.parse` reads only this.
  Every other spelling (`L014`, `l14`, a space, a trailing newline,
  Arabic-Indic digits) is refused, not corrected.
- **A unit names a registered model.** Creating a unit refuses a model
  that is not in `src/pushlab/models/registry.py`, so a typo such as
  `Qwen3.5-4B` fails at once with the list of valid names. It also refuses
  a layer or index that is not a Python int from 0: `True`, `14.0` and
  NumPy integers are refused too.
- **Three checks, each in its own place.**
  - Creating a unit: the spelling and the registry.
  - `validate(model_info)`: the range only, against a model's shape. For
    Qwen3.5-4B, layers 0 to 31 and neurons 0 to 9215. It needs no loaded
    model, so a dry run can check an experiment file.
  - `write_direction(model)`: the model must have the registry's shape for
    the unit's model; then the range; then the weights must not be
    quantized.
- **The write direction** is the neuron's `down_proj` column: a new
  float32 vector, hidden-size long, on the weight's device. It is a copy.
  On a float32 model, `.to(torch.float32)` alone would return a view of
  the weight, and changing the vector would change the model. bfloat16
  converts to float32 exactly.
- **Lengths are computed in float32**, as occcat and Mr-Neutr0n did.
  bfloat16 keeps about 3 significant digits, so a length kept in bfloat16
  can be off by a few tenths of a percent (up to 0.36% across the tiny
  test model's columns). In float32 the error is about 1e-7.
- **`amount(unit, alpha, model, norms)`** returns a Python float,
  `alpha * (norms[layer] / length)`. Dividing first makes `amount(a)`
  exactly `a * amount(1)`, so amounts scaled from dose 1 agree bit for bit.
  - alpha must be a finite number.
  - **A negative dose pushes the other way**: it subtracts along the write
    direction, as the recipe's −1 on L13 N7956 does.
    `amount(-a) == -amount(a)` exactly.
  - norms is a list (or tuple) with one finite number above 0 per layer,
    from layer 0. Every layer is checked, not only the pushed one. Until
    S1's calibration step measures them, callers pass their own.
  - A write direction of length 0 is refused.
- **What dose 1 means.** Adding c to the neuron adds c · W_down[:, n] to
  the residual stream, a vector of length |α| · r̃_L. At dose 1 it is as
  long as the layer's typical residual. For L14 N8949 the replications
  measured, in bfloat16:

  | Replication | Median at layer 14 | Column length | Amount at dose +1 |
  | --- | --- | --- | --- |
  | occcat | 10.488 | 0.46472 | +22.569 |
  | Mr-Neutr0n, at model commit 851bf6e (the registry's pin) | 10.322 | 0.46472 | +22.211 |

  For L13 N7956 at dose −1 they give −23.322 and −23.292. pushlab's own
  calibration should come out close to these.
- **`units/known.csv`** lists published units that random controls must
  avoid, whichever way they were pushed. Its columns are `unit_id`,
  `sign` (`+` or `-`, the plan's signs for a schedule segment), `label`
  and `sources`: https links separated by single spaces, since a link
  cannot contain one. It starts with the evil-Qwen pair, L14 N8949 (+) and
  L13 N7956 (−), with the four sources the research plan cites. It holds
  no doses; those belong in `experiments/`.

## Why this shape

- **Explicit arguments, not a bound unit.** The development plan sketches
  `write_direction()` and `amount(alpha)` on a unit bound to its model and
  residual sizes, and `docs/prompts/S1/05-constant-pushes.md` speaks of a
  "bound neuron". The S1 task asked for `write_direction(model)` and
  `amount(unit, alpha, model, norms)` instead. A unit then stays a plain
  value that can be compared, hashed and written to a CSV, and nothing
  holds on to a model or to norms. A bound neuron in later steps is a
  unit, `loaded.model` and the calibrated norms, passed together.
- **Refuse other spellings rather than correct them.** If `L014` were read
  as `L14`, one neuron could appear under two IDs, and a text comparison
  (a join, or "is this a known unit?") would miss it.
- **The shape check sits in `write_direction`, not in `validate`.** L14
  N8949 is in range for Qwen3-8B too, so a range check alone would read
  the wrong model's column without a word. `validate` stays a plain range
  check, which a dry run can use with the registry's shape.
- **`write_direction` takes the model itself, not what `load_model`
  returns.** Accepting a `LoadedModel` would pull the loader, and with it
  transformers' model classes, into the units module. Passing one by
  mistake gives an error that says what to pass instead.
- **Quantized weights are refused.** An int8 or float8 column means
  nothing without its scales, so it would give a wrong dose silently. The
  plan's 8-bit work can lift this once it handles the scales.
- **A float, not a tensor.** It goes into JSON and run records as it is,
  and carries no device.

## Consequences

- `write_direction` cannot tell apart models of the same shape, such as
  `tiny` and `tiny-lm`, or a later base checkpoint shaped like Qwen3.5-4B.
  Code that holds the `LoadedModel` (the hooks and the runner) must also
  compare `unit.model_name` with `loaded.spec.name`.
- `amount` cannot tell where norms came from. The calibration cache must
  refuse a file made for another model, model commit, data type or text
  set. (Decision 0007 narrows this: the cache refuses another model or
  model commit, and the code that loads sizes for a run compares the rest.)
- Units of models not yet in the registry, such as Liu's Qwen3-4B and
  Qwen3.5-2B neurons, cannot be written down until those models are
  registered. tiny-vl and tiny-lm units exist only inside the
  `tiny_downloads` test fixture.
- Layer and index must be plain Python ints, so code that draws random
  neurons with NumPy must convert them with `int()`.
- `known.csv` has no loader yet; the random-control sampler adds one, with
  its tests. `tests/unit/test_known_units.py` checks every row.
