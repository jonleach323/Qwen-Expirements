# 0019: A neuron set pushes along its coefficient-weighted write direction, and its dose 1 is Mayner's strength

Date: 2026-11-12 · Sprint: S3 · Status: accepted

## Context

- S3 moves the main dynamics work to Qwen3-8B and builds a panel of
  personas, because single neurons proved unreliable across prompts: the
  evil neuron, qwen3.5-4b L14 N8949 at +1, turned 6 of 32 neutral prompts
  hostile on one broad reading, and changed most of the rest against the
  task instead (decision 0018).
- The prompt for this step ("Sprint S3, prompt 1 of 8", pasted; it is not
  `docs/prompts/S3/01-models-and-process.md`) asked for:
  - a `NeuronSet` unit with a name, a list of (layer, neuron, coefficient)
    entries and a source, whose write direction is the
    coefficient-weighted sum of its neurons' down_proj columns, and whose
    dose 1 reproduces the strength Will Mayner's code uses for top-neuron
    steering, defined from that code and recorded here;
  - schedules, the compiler and the hooks taking a set anywhere they take a
    `NeuronUnit`, with the ceiling check counting each set's contribution
    to every layer it touches;
  - a calibration mode `per_prompt` that measures each layer's median
    residual size from the current prompt's own tokens, first token
    excluded, as the Neuron Persona Library describes, the fixed
    calibration staying the default;
  - a set of one entry with coefficient 1 behaving exactly like the
    matching `NeuronUnit`, and no generation change beyond accepting the
    new unit type;
  - Qwen3-8B in `configs/models.yaml`, and CLAUDE.md saying S3 and qwen3-8b.
- The prompt named this note `docs/decisions/0005-neuron-set-dose.md`.
  This repository's 0005 is the S0 baseline, so the note takes the next
  free number, as decision 0018 did. It also named
  `docs/handoff/HANDOFF.md`, "the S2 runner check" and
  `S3-prompt-framework.md` at the repository's root, none of which is on
  any branch; the runner is decision 0015 and S2's close is
  `docs/sprints/S2-review.md`. With no framework file at the root, nothing
  in `docs/prompts/S3/` was moved.
- **Mayner's top-neuron steering** (github.com/wmayner/persona-neurons,
  commit `64c9d36`, 2026-10-31, MIT; data at
  huggingface.co/datasets/wmayner/persona-neurons, CC BY 4.0):
  - A role r's persona vector v_r is its mean residual at layer 18 minus
    the default Assistant's, with massive-activation tokens left out. Each
    MLP neuron (l, n) of layers 0 to 18 contributes
    c = Δa · (W_down_l[:, n] · v̂_r) to |v_r|, Δa being its mean activation
    as the role minus as the default (`persona_neurons/split.py`).
  - The role's top k neurons are the k with the largest |c|
    (`role_steering.py:51`, `neuron_sets` at `:71`).
  - A set of neurons with values Δa_i is scaled by one c for the whole set
    (`condition_values`, `:207-209`, and `matched_scale`, `:136`):
    c = target_norm / ‖Σ_i Δa_i W_down_{l_i}[:, n_i]‖ (`write_vector`,
    `:124`), with target_norm = STRENGTH · |v_r|, STRENGTH = 4.0 (`:34`),
    and |v_r| the length of `vectors/without/{role}.pt` minus
    `vectors/without/default.pt` at layer 18
    (`steps/6_role_steering.py:233`).
  - `RowIntervention` (`:141-168`) adds c · Δa_i to neuron n_i at the input
    of layer l_i's `mlp.down_proj`, at every position of the prompt and the
    reply. So the set's whole output to the residual stream is
    c · Σ Δa_i W_down[:, n_i], a vector 4 · |v_r| long.
  - Qwen3-8B is at commit `b968826d9c46dd6066d109eabc6255188de91218`
    (`persona_neurons/config.py:6`).
- **The Neuron Persona Library** (neuronpersonas.com, "Method", read
  2026-11-12) pushes one neuron by c = ± α · r_L / ‖W_down[:, n]‖, at α of
  1.25, 1, 0.75 or 0.5, and defines r_L as the "Median residual-stream norm
  after layer L over the prompt's tokens (first token excluded)", the
  prompt being "Who are you?" in the chat template, thinking off. That is
  pushlab's dose rule already (decision 0006); its r_L comes from the
  steered prompt itself, where decision 0007's comes from 32 fixed prompts.
  Mayner's copy of the protocol (`single_neurons.py:92-109`) takes
  `torch.median`, the lower middle value of an even count.

## Decision

**A neuron set** (`src/pushlab/units/neuron_set.py`).
- `NeuronSet(model_name, name, entries, source, size=None)`, a frozen
  value like `NeuronUnit`: equal fields make equal sets, which can be
  dictionary keys. The model is a registry name. The name is lowercase
  letters and digits joined by hyphens, as schedule names are. entries are
  (layer, neuron, coefficient) triples, kept in their order as `SetEntry`
  records; layers and neurons count from 0, each neuron appears once, and
  each coefficient is a finite number other than 0. source is free text
  saying where the set comes from. Its ID is `model:set:name`, such as
  `qwen3-8b:set:pirate-top50`, which no neuron ID can match.
- **Its write direction** is w = Σ_i c_i W_down_{l_i}[:, n_i], the
  coefficient-weighted sum of its neurons' down_proj columns, summed in
  float64 and rounded to float32 once. Its length is taken in float32, as a
  neuron's is (decision 0006).
- **Its dose.** `amount(set, alpha, model, norms)` is the amount per unit
  of coefficient, alpha · size / ‖w‖, and neuron i of the set gets that
  amount times c_i, added at the input of its own layer's down_proj like
  any push (decision 0008). The set's push into the residual stream is then
  alpha · size · w / ‖w‖, a vector |alpha| · size long. `UnitAmount` and
  `Generation.amounts` record a set with this per-coefficient amount.
- **size** is the length of the set's push at dose 1. A set may give it;
  with none, it is r_L, the residual size of the set's layer, exactly as
  for a neuron. A residual size belongs to one layer, so a set with neurons
  in more than one layer must give a size, and is refused without one.

**Dose 1 is Mayner's strength** (`src/pushlab/units/mayner.py`). A set
made from his data has his values Δa_i as its coefficients and size
STRENGTH · |v_r| = 4 · |v_r|. Then w = Σ Δa_i W_down[:, n_i] is the vector
`write_vector` builds, and dose 1 adds (4 · |v_r| / ‖w‖) · Δa_i to neuron
i: his c times his Δa_i, the value `condition_values` hands to
`RowIntervention`. A dose of 0.5 is half his strength, 2 twice it.
- The data: `split/top_neurons.parquet`, in his published bundle, holds
  each role's top 1,000 neurons with their rank, Δa (`delta`),
  contribution and share of |v_r|; `role_steering/sets.parquet`, which his
  step 6 writes and the bundle lacks, holds the steered sets, the random
  ones too. Either gives the role's top-k neurons with the same values;
  one ranks them in float32 and the other in float64, so a near-tie could
  order two neurons differently.
- |v_r| is not in either table. For exactly his strength it is step 6's,
  from `vectors/without/` in the bundle (pickled torch files: never load
  them with `weights_only=False`). The split's own |v_r|, which
  `top_neurons.parquet` gives as 100 · contribution / share, is the same
  difference of means taken from step 3's float32 sums rather than step
  4's files, so it can differ from step 6's in the last digits.
- What still differs from his runs is rounding and setup, not the push:
  pushlab adds in float32 and rounds once (decision 0008), where his hook
  adds in the activations' type, bfloat16 on a GPU; pushlab never pushes
  padding, where his hook pushes padded positions too, which attention
  masks out; and he samples each reply with its own seeded generator.

**One neuron, coefficient 1.** With no size, such a set's write direction,
amounts, dose tables, hooks and replies are bit for bit the `NeuronUnit`'s:
w is the neuron's column exactly, its length is computed the same way, and
amount · 1.0 is the amount.

**The ceiling.** The combined push in a layer on a pass is still the
square root of the summed squared doses of the units that push that layer
there; a set counts with its whole dose in every layer it has a neuron in.
`validate` checks this without the model, as before, so a set at 1.5 is
refused in each of its layers, and a set at 1 beside a neuron at 1 in one
of its layers reaches 1.41 there and needs `allow_over_ceiling`, as two
dose-1 neurons in one layer already do.

**The per_prompt calibration mode** (`src/pushlab/units/calibration_mode.py`).
- `per_prompt_norms(model, tokenizer, prompts)` measures each prompt alone,
  in the chat template with thinking off and no push active: each layer's
  size is the median length of that decoder layer's output over the
  prompt's own tokens, its first left out, in float32. It is decision
  0007's measure on a set of one prompt, so with an even count the median
  is the mean of the middle two, not `torch.median`'s lower one. A prompt
  given twice is measured once.
- The sizes reach the doses as the norms of `compile` or `generate`: a
  `PerPromptNorms` (`src/pushlab/units/dose.py`), one row of sizes per
  prompt, where a plain list is the fixed calibration that every row
  shares. Row r of a dose table takes row r's sizes, and a row's amounts
  never depend on another row's.
- `CALIBRATION_MODES` is ("fixed", "per_prompt") and `DEFAULT_MODE` is
  "fixed"; `norms_for(mode, ...)` gives either. The runner, `pushlab steer`
  and experiment files still use the fixed calibration only.

**Where sets work.** A `Segment`, a `Schedule` and every shape take a set
wherever they take a neuron; `validate` checks every entry against the
model and applies the ceiling above; `compile` builds one column per pushed
neuron in each layer, a set's neurons in each of its layers, and where
several units push one neuron on a pass their amounts add in the order the
row's own schedule names them; the hooks, `Push` and `pushes_active` push
sets; `generate` takes them unchanged. Not yet: schedule and experiment
files (`to_yaml` refuses a set, and a set ID in a file is refused), the
runner, the steps records, `pushlab verify` and `report`, the `steer` and
`amount` commands, and random controls for sets.

**Qwen3-8B** is pinned to `b968826d9c46dd6066d109eabc6255188de91218`, the
Hub's newest commit on 2026-11-12 and Mayner's, whose `config.json` gives
the registry's 36 layers, hidden size 4,096 and 12,288 MLP neurons, and
`configs/models.yaml` lists it, so the pod's next setup downloads its
16.4 GB.

## Why this shape

- **A unit direction times a size,** rather than the coefficients as raw
  amounts. The prompt asks for two things that only this gives together:
  a one-entry set equal to its neuron, whose dose is set by r_L, and
  Mayner's sets at his strength, which is set by |v_r|.
- **The size given with the set,** rather than measured. Mayner's strength
  depends on the role's persona vector, a fact of his data, not of the
  model's residual sizes; a calibration could not produce it.
- **Sets across layers need a size.** Scaling one by the residual size of
  its deepest layer would let a set whose weight sits in a shallow layer
  push that layer several times its own residual size, since sizes grow
  with depth.
- **The whole dose in every layer,** for the ceiling. The exact share of a
  set's push in a layer needs the model's weights, and its size in that
  layer's units needs the calibration (or each prompt's), while `validate`
  runs in a cloud session's dry run with neither. The rule is exact for a
  one-entry set and treats two sets sharing a layer as two units in it.
- **Per-prompt sizes as norms,** measured before `generate`, rather than a
  mode inside it: generation does not change, and the measure is the
  calibration's own.
- **Amounts added in the row's own order,** so a row's numbers depend on
  its own schedule alone, bit for bit.

## Consequences

- A dose's meaning now depends on the unit: residual sizes for a neuron or
  a sized-by-layer set, the set's own size otherwise. For a Mayner set,
  dose 1 is 4 · |v_r|, and the 1.4 ceiling bounds doses in those units;
  how hard such a set pushes each layer in residual sizes is not checked.
  The dose ladder (S3) should measure sets as well as neurons.
- Sets live in Python only until schedule and experiment files can name
  them; the next step that runs persona sets on the pod must add that,
  with the runner's records and verify.
- `pushlab.units.mayner` reads tables only; it downloads nothing, and no
  data of Mayner's is in git: the tests build a small fixture in his
  format, and a copy of the few functions they compare against, with his
  MIT notice.
- Every run on Qwen3-8B needs its calibration file for the fixed mode:
  `uv run pushlab calibrate --model qwen3-8b` on the pod, after setup has
  downloaded the model.
- The compiler's tables are per neuron; for schedules of neurons only they
  are exactly what they were.

## Results

The checks on the tiny models are in the test files named below; the GPU
has run nothing for this step.
