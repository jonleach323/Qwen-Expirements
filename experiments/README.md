# Experiment files

One YAML file describes a whole experiment: the model, the prompts, how
replies are decoded, and the conditions to compare. `pushlab expand`
turns it into the exact list of runs it means, the same list every time,
without running anything or loading a model. Running the list comes in a
later step.

`p0/s0_baseline.yaml` is older: S0's one-run file, in a format of its own
that only `pushlab baseline` reads.

## An example

`p0/s2_pulse_lengths.yaml`:

```yaml
# Sprint 2's example experiment: the hostile neuron of the evil-Qwen pair
# pushed for 4, 16 or 64 reply tokens, against no push and against random
# neurons of its layer. `uv run pushlab expand
# experiments/p0/s2_pulse_lengths.yaml --summary` counts its runs; nothing
# runs it yet.
name: s2-pulse-lengths
model: qwen3.5-4b
prompts: prompts/smoke.jsonl
# Greedy decoding draws no random numbers, so it takes one seed.
decoding: {strategy: greedy}
seeds: [0]
max_new_tokens: 96
batch_size: 10
max_gpu_hours: 0.25
conditions:
  # No push: what every pulse is compared with.
  - label: none
    segments: []
  # +1 on reply tokens 0 to length - 1, so a pulse first changes token 1.
  - label: pulse
    shape: pulse
    params: {unit: qwen3.5-4b:L14:N8949, alpha: 1.0, start: 0}
    grid: {length: [4, 16, 64]}
    # Each pulse again on two random neurons of layer 14, as matched controls.
    controls: 2
    control_seed: 0
```

It means 50 runs: 5 prompts, 1 seed, and 4 conditions (no push, and a
pulse of each length), each pulse with 2 random controls.

```bash
uv run pushlab expand experiments/p0/s2_pulse_lengths.yaml --summary
```

prints the settings, then:

```text
condition        own  controls  total
none               5         0      5
pulse length=4     5        10     15
pulse length=16    5        10     15
pulse length=64    5        10     15
all               20        30     50

50 runs, of which 30 are controls, at most 4,800 reply tokens. Nothing was run.
```

Without `--summary` it prints every run on a line of its own: its ID,
condition, control number (0 for the condition's own run), prompt, seed,
batch and, for a control, the run it mirrors.

## The settings

Every setting is required, and any other key is refused.

| Key | What it is |
| --- | --- |
| `name` | The experiment's name: lowercase letters and digits joined by hyphens. Every run ID starts with it. |
| `model` | A model in `src/pushlab/models/registry.py`: `qwen3.5-4b` or `qwen3-8b`. |
| `prompts` | A prompt set (`prompts/*.jsonl`), as a path from the repository's root. |
| `decoding` | `strategy: greedy`, or `strategy: sample` with a `temperature` above 0 and, if you like, a `top_p` above 0 and at most 1 (default 1). `dtype` is `bfloat16` (the default) or `float32`. Greedy decoding takes no temperature or top_p. |
| `seeds` | A list of different whole numbers. Every run is repeated with each. Greedy decoding draws no random numbers, so it takes exactly one. |
| `max_new_tokens` | The most tokens a reply may have. |
| `batch_size` | The most runs that share a batch. |
| `max_gpu_hours` | The GPU time the experiment may take, a number above 0, for the runner to hold it to. |
| `conditions` | The conditions, at least one (below). |

Numbers are read as YAML 1.2 reads them: `010` is text, not 8, and
`1e-3` is a number. A key given twice is refused.

## Conditions

Each condition has a `label`, lowercase letters and digits joined by
hyphens and different from every other condition's, and a schedule
written in one of two ways, as schedule files are
(`src/pushlab/schedules/yaml_io.py`, decision 0012):

- Its segments, one mapping each: `segments: [{unit:
  qwen3.5-4b:L14:N8949, alpha: 1.0, start: 0, end: 16}]`. `segments: []`
  pushes nothing.
- A shape and its params: `shape: pulse` with `params: {unit: ..., alpha:
  1.0, start: 0, length: 16}`. The shapes are `constant`, `pulse`,
  `onset`, `handoff`, `alternate` and `ramp`
  (`src/pushlab/schedules/shapes.py` lists each one's params).

A condition may also have:

- `grid`: some of a shape's params, each given as a list of values, such
  as `grid: {length: [4, 16, 64]}`. The condition then runs one schedule
  per value, labelled `pulse length=4` and so on. With several params it
  runs every combination, the first param changing slowest. A param goes
  in `params` or in `grid`, not both.
- `allow_over_ceiling: true`, which lets the combined push in one layer go
  above the ceiling of 1.4 times the layer's residual size (decision
  0012). Without it, a condition above the ceiling is refused.
- `controls: n` and `control_seed: s`: n matched random controls, drawn
  from seed s. A control is a copy of the condition's schedule with each
  pushed neuron swapped for a random neuron of the same layer, at the same
  doses on the same reply tokens, so the only difference is which neuron
  is pushed. It is never a unit in `units/known.csv`, never a unit any
  condition of the experiment pushes, and a condition's n controls are n
  different neurons. Control k uses the same neurons for every prompt,
  seed and grid value. A condition that pushes nothing has no controls.

## What expand does

1. It checks every setting, naming any problem.
2. It checks every condition's schedule, and its controls', against the
   model and `max_new_tokens` with `validate()`, before it makes any run.
   The first problem stops it, with the condition's label and the rule
   it broke. A segment that will not take effect in a reply this long,
   such as one starting after the last token, gets a warning instead.
3. It reads the prompt set.
4. It makes one run per prompt, seed, condition (each grid value its
   own) and control, in that order: all of the first prompt's runs, its
   first seed first, and within a seed the conditions in the file's
   order, each condition's own run followed by its controls. Run IDs
   count up in that order, from `<name>-000001`.
5. It puts the runs into batches in order: a batch holds runs of one
   prompt and one seed, at most `batch_size` of them. Each run's
   `batch_key` is `b<batch>:<row>`, such as `b000002:0`.

Each run is a run record (`src/pushlab/store/records.py`) with no reply
yet and the status `planned`. Its schedule lists the segments it pushes,
each with `control_for` (for a control, the unit it replaces) and
`control_seed`. The spec's hash, its `spec sha256`, changes whenever any
setting does, but not with comments, spacing or the order of keys.
Decision 0014 explains these choices.
