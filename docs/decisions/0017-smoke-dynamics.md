# 0017: S2's smoke test of dynamics runs every schedule shape, each with a matched random control

Date: 2026-11-09 · Sprint: S2 · Status: accepted

## Context

- The S2 prompt pasted for this step ("prompt 6 of 7") asked for the files
  of the first dynamics experiment, which the owner runs on the GPU pod:
  - `prompts/neutral32.jsonl`, 32 neutral everyday prompts different from
    the calibration and smoke sets;
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
- S2's done-when (development plan) is "a 1,000-run smoke experiment
  survives a deliberate kill and resume, and per-row isolation holds on
  the GPU". This is S2's only experiment of about 1,000 runs.

## Decision

**Two questions settled by the owner before writing:**
- **Labels in lowercase.** The prompt's labels had capitals (constant-E,
  handoff-E-W-10), but a condition's label becomes its schedule's name,
  and names are lowercase letters and digits (decision 0012). The owner
  chose lowercase labels over changing the schedule language: constant-e,
  handoff-e-w-10 and so on. The file's comments say which unit E and W
  are; a report's legend names each unit's neuron under a letter.
- **The ramp rises, then stops.** "ramp-E-0-32 (E rising from 0 to 1 over
  tokens 0 to 32)" is the ramp shape with start 0 and end 32: dose 0 on
  reply token 0 rising in equal steps to 1 on token 31, then no push from
  token 32. A ramp holds no dose after its end, and the owner chose that
  reading over holding E at 1 or ending on token 32.

**The prompt set,** `prompts/neutral32.jsonl`, version 1, IDs `neutral32-01`
to `neutral32-32`, SHA-256 `5ca28963eaee920963c19db7a9b360fbd74eaa42a4dfdc37306087a32b844f45`:
everyday practical questions and requests, such as storing bread,
explaining tides or hanging a picture.
- None repeats a calibration or smoke prompt, even ignoring case and
  punctuation, but they are of the calibration prompts' everyday genre,
  and a few ask the same kind of thing about another object.
- None asks for dangerous help, as the research plan requires, or is
  about feelings, health or money. A few involve physical steps whose
  safety warnings a hostile reply could drop or turn round, such as
  changing a flat tire or cleaning a bathroom; such replies get the plan's
  content note.
- Each asks for steps, an explanation or a description with some detail,
  so that most replies run to the 96-token limit and the late pushes
  (onsets at 10 and 30, a 64-token pulse, 32-token turns) fall inside the
  reply. So 10 asks for a description from first light until the shops
  open, and 27 for a product description with a list of features, rather
  than something short, which could end early.

**The conditions** (`experiments/s2/smoke_dynamics.yaml`, experiment
`s2-smoke-dynamics`), in the file's order. E pushes at +1.0 and W at
−1.0, except that the ramp's E rises from 0 to 1. A push on reply token k
first changes token k + 1 (decision 0012).

| Label | Batch | Shape | Prompt pass | Reply tokens |
| --- | --- | --- | --- | --- |
| baseline | 1 | none | nothing | nothing |
| onset-e-10, -30 | 1 | onset | nothing | E from 10, from 30 |
| ramp-e-0-32 | 1 | ramp | nothing | E from dose 0 on 0 to 1 on 31, then off |
| pulse-e-4, -16, -64 | 1 | pulse | nothing | E on 0 to 3, 0 to 15, 0 to 63 |
| alternate-e-w-32 | 1 | alternate | nothing | E on 0 to 31, W on 32 to 63, E on 64 to 95 |
| constant-e, constant-w | 2 | constant | the unit | the unit on every token |
| handoff-e-w-10, -30 | 2 | handoff | E | E on 0 to 9 (0 to 29), then W |
| handoff-w-e-10, -30 | 2 | handoff | W | W on 0 to 9 (0 to 29), then E |
| alternate-e-w-2, -8 | 2 | alternate | nothing | E then W, turns of 2 and 8 tokens, from 0 |

- constant and handoff push the prompt pass, the shapes' default and the
  evil recipe's setup, so they can change token 0. That push is written
  into the model's memory of the whole prompt and acts on every later
  token: after a handoff's switch, the first unit still acts through it.
  So a handoff differs from an onset of its second unit, and constant-e
  from pulse-e-64, by the prompt pass as well as by the reply tokens.
- pulse, onset, alternate and ramp push reply tokens only.
- Every schedule passes `validate()` at the ceiling of 1.4 with no
  warning: E and W sit in different layers, so no layer's combined push
  is above 1.

**The order sets the batches.** A batch holds runs of one prompt, each run
with its control, in the file's order (decision 0014). So each prompt's 31
runs fill two batches: the baseline and seven conditions (15 runs), then
the other eight (16 runs). Runs that must match exactly must share a batch
layout (decision 0009): a batch of 15 rows can round differently from one
of 16. The order puts each pair that the schedules say must match, over
its first tokens, in one batch:

| Pair | Batch | Same reply tokens |
| --- | --- | --- |
| onset-e-10, onset-e-30 against baseline | 1 | 0 to 10, 0 to 30 |
| ramp-e-0-32 against baseline (its dose on token 0 is 0) | 1 | 0 to 1 |
| pulse-e-64 against baseline | 1 | 0 |
| pulse-e-4, pulse-e-16 against pulse-e-64 | 1 | 0 to 4, 0 to 16 |
| alternate-e-w-32 against pulse-e-64 | 1 | 0 to 32 |
| handoff-e-w-10, -30 against constant-e | 2 | 0 to 10, 0 to 30 |
| handoff-w-e-10, -30 against constant-w | 2 | 0 to 10, 0 to 30 |
| alternate-e-w-2 against alternate-e-w-8 | 2 | 0 to 2 |

- The controls match one another the same way, as they use the same two
  random neurons. The runbook's check of these pairs reads a finished run
  and prints, for each pair, on how many prompts the replies agree.
- No order keeps every such pair in one batch: the baseline, the onsets,
  the ramp, the pulses and the alternations are 19 runs, more than a batch
  of 16. What is left across batches is short: alternate-e-w-2 and -8
  against the first batch's conditions, through token 2 or token 8 at
  most.
- `batch_size: 32` would put each prompt's 31 runs in one batch, so that
  every comparison is exact. The prompt fixed 16; that is the owner's
  call.

**The controls.** Every condition but baseline has one: `controls: 1`,
`control_seed: 0`. A control is its condition's schedule with E swapped
for a random neuron of layer 14 and W for one of layer 13, at the same
doses on the same tokens, never a known unit or one the experiment
pushes (decision 0014).
- The draw depends on the seed, the unit and the units it must avoid,
  which are the same for every condition here. So every condition's
  control uses the same two random neurons, E′ = L14 N1929 and W′ = L13
  N3302: the controls are a matched copy of the whole experiment, shape
  for shape.
- E′ is S1's random control (`s1-random`, decision 0011) and the first
  control of `s2-pulse-lengths`. So every random-control result for layer
  14 so far rests on one neuron.
- The research plan asks for at least 5 random neurons per condition. One
  pair is one draw of the null, not a spread: enough for a smoke test,
  not for a measurement.

**The count and the cost.** 16 conditions and 15 controls are 31
schedules, times 32 prompts: 992 runs, 480 of them controls, at most
95,232 reply tokens, in 64 batches of 15 or 16 runs. The runner never
joins two of them, since 15 + 15 is more than 16. At the costs file's 44
reply tokens a second, the dry run estimates about 0.60 GPU-hours, within
max_gpu_hours 1: about $0.20 to $0.44 on an RTX 4090. Once a run on the
pod has measured the model's speed, the dry run uses that instead.

**The checks** (`tests/unit/test_smoke_dynamics.py`, on the CPU):
- the spec expands to 992 runs with no warning, 31 groups of the 32
  prompts, the labels in the file's order, one control for each
  condition but baseline;
- each prompt's runs fill the two batches above;
- its settings are the planned ones, and the spec's hash is pinned;
- every schedule passes validation with no warning;
- each condition pushes what the table says on the prompt pass and every
  reply token from 0 to 95 (with 96 reply tokens the last pass reads
  token 94);
- each control mirrors its run on E′ and W′, never E, W or a known unit;
- the runbook's pairs are pushed alike, in one batch, up to the token
  they name, and differ on the next pass;
- the runbook's check, run on a finished run of this file's conditions
  moved to the tiny test model, finds every pair alike on every prompt;
- the prompt set is 32 new prompts, its SHA-256 pinned.

A slip in the file, such as a ramp ending at 33, a switch at 31, a pulse
starting at 1 or a condition moved to the other batch, fails a test.

## Why this shape

- **One seed for every control,** rather than one each: with a single
  control per condition, the same two random neurons give a clean
  shape-for-shape comparison, where fifteen different ones would add
  neuron-to-neuron differences between conditions.
- **The order chosen for the batches,** not the prompt's order of the
  labels: it costs nothing, and it makes the comparisons a reader is most
  likely to make exact, the baseline against the onsets and the ramp, the
  pulses against one another, and the handoffs against their constants.
- **Prompts that invite long replies:** a reply that ends at token 20
  never meets an onset at 30 or the second half of a 64-token pulse.

## Consequences

- The run is the owner's, on the pod (docs/runbooks/experiments.md, "Run
  S2's smoke test of dynamics"). It is S2's done-when experiment, so the
  runbook kills it on purpose once at least 3 batches are saved and starts
  it again; then the owner checks the count, verifies it, runs the check
  of the pairs and reads the reports. The S1 review asks for the owner's
  labels of S1's 20 replies before any new replies are made, so the
  runbook asks for those first.
- It is a smoke test, not a measurement: one greedy seed, 32 prompts and
  one pair of random neurons show whether the shapes behave as planned
  and roughly where effects appear, not how large they are.
- Few replies may be openly hostile. At dose 1, S1 saw E turn 1 of 5
  smoke replies hostile (0 to 3 on any reading; decision 0011), against
  hedwigai's 17 of 20 on prompts mostly about values, and neutral32 is all
  everyday tasks. So the runbook asks the reader to look for where replies
  part from the baseline and from their controls, and to compare each
  shape with constant-e on the same prompt, rather than to count hostile
  replies.
- In a report, units get letters in the order they first appear, so E is
  A in a report whose first pushed unit is E, and W is A in one that
  starts with constant-w or a W-first handoff; the legend says which is
  which. Letting an experiment name its units in reports would be a new
  feature.
