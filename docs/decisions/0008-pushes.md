# 0008: A push adds its amount to a neuron's value at down_proj's input, on the passes its window names

Date: 2026-11-05 · Sprint: S1 · Status: accepted

## Context

- The research is about timing: pushes that start and stop, such as a
  pulse, a late onset or a handoff (research plan, "Schedules"). So even
  the first engine needs pushes that know which pass is running.
- The plan's engine adds the whole batch's amounts in one vectorized step
  per targeted layer, so that every row can follow its own schedule. A
  custom decoding loop keeps the step counter exact (development plan,
  "4. engine").
- The amount c comes from the dose rule (decision 0006), and the sizes it
  needs from calibration (decision 0007).
- The recipe pushes "at the down_proj at every input". Both replications do
  it at down_proj's input, the neuron's own value:
  - Mr-Neutr0n uses a forward pre-hook on down_proj, `h[..., neuron] +=
    delta`;
  - occcat wraps down_proj and adds in float32:
    `x = (x.astype(mx.float32) + add).astype(dtype)`.

## Decision

- **A Push** (`pushlab.engine.push.Push`) names a neuron, a dose α and a
  window:
  - `start`, the first reply token it applies to, counting from 0;
  - `end`, the first reply token it no longer applies to, or None for the
    rest of the reply;
  - `include_prompt`, whether it also applies on the prompt pass. It has
    no default, because the recipe's constant push needs True and a late
    window needs False.
- **A StepState** says which pass is running: the prompt pass, or the pass
  that reads reply token k. The decoding loop sets it before each pass,
  and it can mark positions of that pass as padding.
- **What a push on reply token k changes** is the computation at token k's
  position. So it first shows in the choice of token k + 1, and it stays in
  the cache for every later token. Only a push on the prompt pass reaches
  the choice of token 0. The recipe's constant push is therefore `start=0`,
  `end=None`, `include_prompt=True`.
- **Where it enters:** a forward pre-hook on the pushed layer's down_proj
  adds c to neuron n's value at every real position of an active pass.
  Since down_proj is linear and has no bias, this is the same as adding
  c · W_down[:, n] to the MLP's output. The tests check this on the tiny
  models in float32, and the real model's check is below.
- **One hook per pushed layer,** shared by that layer's pushes. Pushes add
  up: amounts for the same neuron are summed, and different neurons are
  pushed independently, in one layer or several.
- **Precision:** the amount and the value are added in float32, and the
  sum is rounded once to the activations' type, as occcat does. Rounding the
  amount to bfloat16 first would turn 22.21 into 22.25 before the add.
- **Only the pushed neurons change.** The hook copies down_proj's input
  and changes only the pushed neurons' columns, so nothing else moves, not
  even by rounding.
- **Padding** positions, as the StepState marks them, are never pushed.
- **Batch rows:** in S1 every row gets the same pushes. The hook already
  adds one amount per row, so S2's per-row schedules only have to fill that
  column with each row's own amount.
- **Cleanup:** the hooks are removed when the `with` block of
  `pushes_active` ends, even after an error. A push that is refused (a
  bad window, another model, wrong norms) installs no hook at all.
- **Guards:** every push in one call must name the same model, and
  `amount` checks each against the model's shape. Calibration refuses to
  run while pushes are active (decision 0007).

## The push checked on the real model

Qwen3.5-4B at commit `851bf6e` in bfloat16, on a cloud session's CPU,
pushing L14 N8949 at dose 1 with the sizes of decision 0007. The S0 prompt
ran first, then five reply tokens, one pass each:

- The push adds 23.49 to the neuron's value. Along its write direction that
  is a vector 10.92 long, exactly layer 14's size.
- The MLP's output with the push differs from the output without it plus
  c · W_down[:, n] by at most 0.32% of the push's length, which is
  bfloat16's rounding.
- At dose 0, every pass gives exactly the logits of a run with no hooks.

This meets the plan's done-when for units: "pushing a neuron through its
MLP value matches adding the same amount along its write direction, within
bfloat16 tolerance."

## Why this shape

- **down_proj's input rather than the MLP's output.** It is where the
  recipe and both replications push, and it is the neuron's own value, the
  thing that "pushing a neuron" means.
- **A copy rather than an in-place change,** so no other code that holds
  the tensor sees it change.
- **The pass is set by the loop, not counted by the hook.** Counting calls
  inside the hook would go wrong as soon as a pass is repeated, skipped or
  batched differently. The loop knows which pass it runs.

## Consequences

- The decoding loop must call `state.at_prompt(...)` before the prompt pass
  and `state.at_reply_token(k, ...)` before each reply pass, with the
  padding of that pass. A pass it forgets to announce runs with the
  previous state.
- A teacher-forced pass that reads several reply tokens at once
  (`score_text`, later) will need a reply token per position. StepState
  holds one per pass for now.
- `pushes_active` hands back each push's amount, for the run's record.
