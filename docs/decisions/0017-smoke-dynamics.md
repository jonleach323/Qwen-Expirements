# 0017: S2's smoke test of dynamics runs every schedule shape, each with a matched random control

Date: 2026-11-09 · Sprint: S2 · Status: accepted

## Context

- The S2 prompt pasted for this step ("prompt 6 of 7") asked for the files
  of the first dynamics experiment, which the owner runs on the GPU pod:
  - `prompts/neutral32.jsonl`, 32 neutral everyday prompts unlike the
    calibration and smoke sets;
  - `experiments/s2/smoke_dynamics.yaml`: qwen3.5-4b, those prompts,
    greedy, 96 reply tokens, batches of 16, max_gpu_hours 1; units E =
    L14 N8949 at +1.0 (the evil neuron) and W = L13 N7956 at −1.0 (the
    evil recipe's second neuron, which alone made replies warmer in
    hedwigai's check); 16 conditions with given labels; one matched
    random control for every condition but baseline;
  - a test that the spec expands to 31 conditions times 32 prompts, 992
    runs, all passing validation;
  - no new engine features: anything the schedule language cannot say was
    to be raised, not built.
- The research plan studies hostile personas on neutral prompts only.

## Decision

**Two questions settled by the owner before writing:**
- **Labels in lowercase.** The prompt's labels had capitals (constant-E,
  handoff-E-W-10), but a condition's label becomes its schedule's name,
  and names are lowercase letters and digits (decision 0012). The owner
  chose lowercase labels over changing the schedule language: constant-e,
  handoff-e-w-10 and so on. The file's comments and the report's legend
  name E and W.
- **The ramp rises, then stops.** "ramp-E-0-32 (E rising from 0 to 1 over
  tokens 0 to 32)" is the ramp shape with start 0 and end 32: dose 0 on
  reply token 0 rising in equal steps to 1 on token 31, then no push from
  token 32. A ramp holds no dose after its end, and the owner chose that
  reading over holding E at 1 or ending on token 32.

**The prompt set,** `prompts/neutral32.jsonl`, version 1, IDs `neutral32-01`
to `neutral32-32`: everyday practical questions and small requests, such
as storing bread, changing a flat tire or explaining tides. None repeats a
calibration or smoke prompt, and none touches feelings, health, money or
anything a hostile reply could make harmful. Each asks for a few steps or
an explanation, so that most replies run to the 96-token limit and the
late pushes (onsets at 10 and 30, a 64-token pulse, 32-token turns) fall
inside the reply.

**The conditions** (`experiments/s2/smoke_dynamics.yaml`, experiment
`s2-smoke-dynamics`). E pushes at +1.0 and W at −1.0 in every condition.
A push on reply token k first changes token k + 1 (decision 0012).

| Label | Shape | Prompt pass | Reply tokens |
| --- | --- | --- | --- |
| baseline | none | nothing | nothing |
| constant-e, constant-w | constant | the unit | the unit on every token |
| pulse-e-4, -16, -64 | pulse | nothing | E on 0 to 3, 0 to 15, 0 to 63 |
| onset-e-10, -30 | onset | nothing | E from 10, from 30 |
| handoff-e-w-10, -30 | handoff | E | E on 0 to 9 (0 to 29), then W |
| handoff-w-e-10, -30 | handoff | W | W on 0 to 9 (0 to 29), then E |
| alternate-e-w-2, -8, -32 | alternate | nothing | E then W, turns of 2, 8, 32 tokens, from 0 |
| ramp-e-0-32 | ramp | nothing | E from dose 0 on 0 to 1 on 31, then off |

constant and handoff push the prompt pass, the shapes' default and the
evil recipe's setup, so they can change token 0. pulse, onset, alternate
and ramp push reply tokens only. Every schedule passes `validate()` at the
ceiling of 1.4 with no warning: E and W sit in different layers, so no
layer's combined push is above 1.

**The controls.** Every condition but baseline has one: `controls: 1`,
`control_seed: 0`. A control is its condition's schedule with E swapped
for a random neuron of layer 14 and W for one of layer 13, at the same
doses on the same tokens, never a known unit (decision 0014). The draw
depends on the seed and the unit only, so with one seed every condition's
control uses the same two random neurons: the controls are a matched copy
of the whole experiment, shape for shape, on two random neurons.

**The count and the cost.** 16 conditions and 15 controls are 31
schedules, times 32 prompts: 992 runs, 480 of them controls, at most
95,232 reply tokens. expand plans 64 batches of 15 or 16 runs. The dry run
estimates about 0.60 GPU-hours at the costs file's 44 reply tokens a
second, within max_gpu_hours 1: about $0.20 to $0.44 on an RTX 4090.

**The checks** (`tests/unit/test_smoke_dynamics.py`, on the CPU, with no
model): the spec expands to 992 runs, 31 groups of the 32 prompts, the
labels in order, one control for each condition but baseline; its
settings are the planned ones; every schedule passes validation with no
warning; each condition pushes exactly what the table says on the prompt
pass and every reply token from 0 to 95; each control mirrors its run on
the same two random neurons of the right layers, never E, W or a known
unit; and the prompt set is 32 new prompts. A slip in the file, such as a
ramp ending at 33, a switch at 31 or a pulse starting at 1, fails a test.

## Why this shape

- **One seed for every control,** rather than one each: with a single
  control per condition, the same two random neurons give a clean
  shape-for-shape comparison, where fifteen different ones would add
  neuron-to-neuron differences between conditions.
- **Prompts that invite long replies:** a reply that ends at token 20
  never meets an onset at 30 or the second half of a 64-token pulse.

## Consequences

- The run is the owner's, on the pod (docs/runbooks/experiments.md, "Run
  S2's smoke test of dynamics"); then `pushlab verify` and `pushlab report`.
- It is a smoke test, not a measurement: one greedy seed, 32 prompts and
  one pair of random neurons show whether the shapes behave as planned
  and roughly where effects appear, not how large they are.
- In a report, units get letters in the order they first appear, so E and
  W show as A and B when an E condition comes first; the legend says which
  is which. Letting an experiment name its units in reports would be a new
  feature.
