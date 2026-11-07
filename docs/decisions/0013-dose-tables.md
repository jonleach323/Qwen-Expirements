# 0013: A dose table gives every row its own amounts, and one hook per layer adds them

Date: 2026-11-07 · Sprint: S2 · Status: accepted

## Context

- S1's hooks give every row of a batch the same pushes; decision 0008 left
  "S2's per-row schedules" to "fill that column with each row's own
  amount". Decision 0012 built the schedule language.
- The S2 prompt pasted for this step ("prompt 2 of 7") asked for:
  `compile(schedules, model, norms, max_new_tokens)`, a DoseTable that
  answers, for any row and reply token or the prompt pass, which neurons
  get which amounts by S1's dose rule; hooks that read it on every pass and
  add each row's own amounts in one vectorized step per layer, never at
  padding; `generate()` taking one schedule per prompt and recording, for
  every row and token, the active units and amounts; S1's push windows
  kept working as schedules. Rows must never affect each other, an empty
  schedule must give S1's unsteered output, and memory must grow with the
  active units, not layers times neurons. A GPU test,
  `tests/gpu/test_isolation_smoke.py`, repeats the four-row check on
  Qwen3.5-4B.
- It is not `docs/prompts/S2/02-engine-follows-schedules.md` or
  `05-per-row-schedules.md`: no `steer --schedule` or `--dtype`, no seeds
  per run, no steps file. Those are left for later steps.

## Decision

**compile and the DoseTable** (`pushlab.schedules.compile`).
- A table has one row per schedule and `max_new_tokens` passes, counted as
  decision 0008 counts them: pass 0 is the prompt pass, and pass k + 1 the
  one that reads reply token k, up to max_new_tokens − 2, since no pass
  reads a reply's last token. `amounts(row, reply_token)` answers for any
  reply token, past the table's passes too.
- The amount of a unit on a pass is the sum, in the schedule's order, of
  each of its segments' dose on that pass times the unit's dose-1 amount,
  `amount(unit, 1.0, model, norms)`. That is bit for bit what S1 adds:
  `amount(unit, alpha)` is `alpha * (norm / length)`, and S1's hook adds
  pushes on one neuron one by one, in order.
- Every schedule is checked as `validate()` checks it, against the model's
  shape, with `ceiling=None`: the engine pushes what it is given, as S1's
  steer did, and schedule files meet the ceiling when they are read. All
  units must name one model; `amount()` checks each against the model and
  the norms. compile issues no warnings.
- Per pushed layer, a `LayerTable` holds the layer's pushed units, by
  neuron index; `amounts`, float64, of shape (passes, rows, units); `active`,
  bool, the same shape, true where a row pushes a unit on a pass, at dose 0
  too; and `busy`, one flag per pass. So its size is passes × rows × the
  units the layer pushes: 64 rows of 600 tokens over two units take 0.7 MB
  and compile in 0.4 s on a cloud CPU. `select(rows)` gives a batch's rows.

**The hooks** (`pushlab.engine.hooks.doses_active`).
- One forward pre-hook on the down_proj of each pushed layer, installed for
  a with block and removed after it, even after an error.
- On each pass it reads the StepState. If no row pushes anything in the
  layer, it returns nothing and down_proj gets its input as it was, as with
  S1; `busy` lives on the CPU, so this costs no GPU sync. Otherwise one
  call to `add_amounts` adds the pass's (rows, units) amounts.
- `add_amounts` (`pushlab.engine.push`) adds in float32 (float64 for a
  float64 model) and rounds once (decision 0008). Where a row does not
  push a neuron, at a padding position, or in a finished row, the value
  keeps every bit, −0.0 included; a neuron pushed at dose 0 still goes
  through the add. S1's hook now uses the same function, with one row of
  amounts that every row shares.
- The table is moved to the activations' device and add type once. It is
  float64 on the CPU, so each amount rounds to float32 exactly as S1's
  `torch.full` rounded it. A pass with a different number of rows, or past
  the table's last pass, is refused.

**generate** (`pushlab.engine.generate`).
- `generate(model, tokenizer, prompts, pushes, decoding, norms, *,
  schedules=None, group_by_length=False)`. `schedules` gives one Schedule
  per prompt, with `pushes` empty. S1's `pushes` stay a list of Push that
  every prompt gets: `schedule_of(pushes)` turns them into one schedule,
  each push into its segments (`Push.segments()`: the prompt pass if it is
  on, then its window).
- The schedules are compiled once per call, and each batch, one or one per
  token length, runs with its rows' part of the table.
- `Generation.amounts[j]` lists the units pushed on the pass that chose
  token j, each with its amount: the prompt pass for token 0, the pass that
  read token j − 1 for token j. So there is one entry per token, the
  record decision 0008 asks a per-token record to keep. `touched` counts
  each push's reply tokens with pushes, and each segment's with schedules.

**Why rows never touch each other.** A row's values change only by its own
amounts, and every value it does not push keeps every bit. Everything else
in the model works row by row. So in one batch layout a row's tokens and
scores depend on its own prompt and schedule alone, bit for bit, whatever
the other rows push; against its prompt run alone they are the same up to
the rounding a batch's other shape brings (decision 0009).

**The checks.** On the tiny models in float32: a probe on down_proj sees
every row's exact amount on every pass, along a ramp too, and nothing on
rows that push nothing, on padding or on finished rows. Four rows (no
push, a pulse, a handoff, an alternation) each match their prompt alone in
tokens, amounts and touched counts, with scores within 1e-5. Changing one
row's schedule leaves the others bit for bit, and empty schedules give
S1's unsteered run. A list of pushes gives exactly the tokens, scores and
amounts of its schedule. S1's push, generate and steer tests pass
unchanged.

**The GPU test.** `tests/gpu/test_isolation_smoke.py`, marked gpu and slow,
loads Qwen3.5-4B in bfloat16 and the calibration in `calibration/`. Four
smoke prompts of different lengths run as one left-padded batch, since S1
does not group by length unless asked, with no push, a pulse, a handoff and
an alternation of the recipe's pair (L14 N8949 at +1, L13 N7956 at −1).
Each prompt then runs alone with its schedule, made to read its batched
reply. The scores of every pass must stay within 1.0, the limit of
`test_generate_smoke.py` (decision 0009), and the amounts behind every
token must be equal. Its logic ran on tiny-vl with the model swapped in.

## Why this shape

- **`schedules` as a keyword of its own,** rather than a fourth argument
  that is sometimes pushes and sometimes schedules: steer and S1's tests
  hand generate a list of Push in that place, and each argument keeps one
  meaning.
- **Amounts summed segment by segment,** rather than doses summed first, so
  S1's pushes run through the table bit for bit as before.
- **A mask rather than adding 0,** so a row that pushes nothing keeps
  every bit, and isolation does not rest on x + 0.0 being x, which is false
  for x = −0.0.
- **One entry per token,** the pass that chose it: the plan's steps table,
  "units pushed at this step", can take it as it is.

## Consequences

- `pushlab steer` is unchanged: it hands generate its pushes, which now run
  as their schedule, with the same tokens, records and touched counts.
  `pushes_active` stays, for S1's tests and callers, though generate no
  longer uses it.
- `Generation.amounts` lives in memory only; storing it is for the step
  that stores runs. Sampling still seeds each row from its place.
- The pod's setup step 7 runs every GPU test whose name holds "smoke", so it
  now runs this one too: `tests/unit/test_setup_pod.py`'s list of them
  gains it. It loads the model once more and generates five times, so step
  7 takes longer. The first pod run's largest gap belongs in this note.
