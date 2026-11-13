# Experiment files

One YAML file describes a whole experiment: the model, the prompts, how
replies are decoded, and the conditions to compare. `pushlab expand`
turns it into the exact list of runs it means, without running anything
or loading a model. The list is the same every time, as long as the
prompt set and `units/known.csv`, which the random controls avoid, stay
the same. `pushlab run` runs the list batch by batch (decision 0015).

`p0/s0_baseline.yaml` is older: S0's one-run file, in a format of its own
that only `pushlab baseline` reads.

The experiment files:

| File | What it runs |
| --- | --- |
| `p0/s2_pulse_lengths.yaml` | The example below: the evil neuron pushed for 4, 16 or 64 reply tokens, on the five smoke prompts, 50 runs. |
| `s2/smoke_dynamics.yaml` | Sprint 2's smoke test of dynamics: every schedule shape on the evil neuron and its pair, each with one matched random control, on `prompts/neutral32.jsonl`, 992 runs (decision 0017). |
| `s2/evil_diagnosis.yaml` | Why the evil neuron changed the smoke test's replies so little: no push; the evil neuron at doses 1, 1.25 and 1.5; at dose 1 without the prompt pass; with its pair; occcat's L16 N8927; and one matched random control for dose 1; on `prompts/neutral32.jsonl`, 256 runs (decision 0018; `docs/runbooks/s2-evil-diagnosis.md`). |

`pushlab ladder` writes a dose ladder's file from a units file: one
condition per unit, `u01`, `u02` and so on, each a constant push of its
unit at every dose of the ladder, so units × doses × prompts runs
(decision 0022; `docs/runbooks/ladder.md`). A ladder of the Neuron
Persona Library's units holds its data, so it is written and kept outside
the repository, on the pod. Its units are single neurons: a set of
neurons needs its definition written out (below), which `pushlab panel
stage ladder` does (decision 0023).

## An example

`p0/s2_pulse_lengths.yaml`:

```yaml
# Sprint 2's example experiment: the hostile neuron of the evil-Qwen pair
# pushed for 4, 16 or 64 reply tokens, against no push and against random
# neurons of its layer. `uv run pushlab run
# experiments/p0/s2_pulse_lengths.yaml --dry-run` counts its runs and their
# cost; docs/runbooks/experiments.md says how to run it on the pod.
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

It has 2 conditions: no push, and a pulse whose grid gives it 3 lengths.
That makes 4 schedules, each run on 5 prompts with 1 seed, and each pulse
also runs on 2 random neurons: 50 runs in all.

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
(all)             20        30     50

50 runs, of which 30 are controls, at most 4,800 reply tokens. Nothing was run.
```

The settings above the table include the experiment's hashes (below)
and its batches. Without `--summary` it prints a line naming the
columns, then every run on a line of its own: its ID, condition, control
number (0 for the condition's own run), prompt, seed, batch and, for a
control, the run it mirrors. The columns are separated by tabs, and a
field holding a tab, a line break or a double quote is quoted, as CSV
does.

## The settings

Every setting is required but `group_by_length`, and any other key is
refused.

| Key | What it is |
| --- | --- |
| `name` | The experiment's name: lowercase letters and digits joined by hyphens. Every run ID starts with it. |
| `model` | A model in `src/pushlab/models/registry.py`: `qwen3.5-4b` or `qwen3-8b`, not the test suite's. |
| `prompts` | A prompt set (`prompts/*.jsonl`), as a path from the repository's root. |
| `decoding` | `strategy: greedy`, or `strategy: sample` with a `temperature` above 0 and, if you like, a `top_p` above 0 and at most 1 (default 1). `dtype` is `bfloat16` (the default) or `float32`. Greedy decoding takes no temperature or top_p. |
| `seeds` | A list of different whole numbers. Every run is repeated with each. Greedy decoding draws no random numbers, so it takes exactly one. |
| `max_new_tokens` | The most tokens a reply may have, at most 32,768: the longest reply the steps table holds. |
| `batch_size` | The most runs that share a batch. `expand` plans batches of one prompt and one seed, so they can be shorter (below); `pushlab run` joins planned batches that can share one, up to `batch_size`. |
| `group_by_length` | Optional, `true` by default: `pushlab run` joins planned batches of different prompts only when the prompts are of one length in tokens, so that no row is padded. `false` lets prompts of different lengths share a padded batch. |
| `max_gpu_hours` | The GPU time the experiment may take, a number above 0, for the runner to hold it to. |
| `conditions` | The conditions, at least one (below). |

Numbers are read as YAML 1.2 reads them: `010` is text, not 8, and
`1e-3` is a number. A key given twice is refused. A name or label that
YAML reads as a number, such as `2026`, needs quotes.

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

A segment can push a set of neurons, pushed as one unit (decision 0019),
such as one of Mayner's persona sets. Its `unit` is the set's ID,
`<model>:set:<name>`, and the set's definition goes beside it under `set`
(decision 0023):

```
segments:
  - unit: qwen3-8b:set:pirate-top50
    set: {entries: [[17, 1234, 0.52], [18, 99, -0.31]], size: 96.04, source: mayner role pirate}
    alpha: 1.0
```

- `entries` lists the set's neurons, each `[layer, neuron, coefficient]`,
  layers and neurons counted from 0, each neuron once and no coefficient 0.
- `size` is the length of the set's push at dose 1, such as Mayner's
  strength, or `null` for the residual size of the set's one layer, as a
  neuron's dose has; a set with neurons in more than one layer needs one.
- `source` says where the set comes from.

All three are needed, on every segment that pushes the set, so a run's
record holds the whole set and the experiment's hash covers every
coefficient; a YAML anchor saves writing it twice: `set: &pirate {...}`
on the first segment, `set: *pirate` on the next. A set's ID names one
set in the whole experiment: one ID with two definitions is refused, and
so is a definition beside a neuron's ID. A shape's params name units by
ID alone, so a schedule that pushes a set is written as segments, and a
grid cannot vary one. A condition that pushes a set cannot have random
controls yet.

A condition may also have:

- `grid`: some of a shape's params, each given as a list of values, such
  as `grid: {length: [4, 16, 64]}`. The condition then runs one schedule
  per value, labelled `pulse length=4` and so on. With several params it
  runs every combination, the first param changing slowest, so the order
  of a grid's keys matters. A param goes in `params` or in `grid`, not
  both. A dose is labelled as the decimal number its schedule holds, so
  `alpha: [1, -1]` gives `pulse alpha=1.0` and `pulse alpha=-1.0`, and
  two doses that are the same number are refused. A grid too large for
  six-digit run IDs is refused before its schedules are built.
- `allow_over_ceiling: true`, which lets the combined push in one layer go
  above the ceiling of 1.4 times the layer's residual size (decision
  0012). Without it, a condition above the ceiling is refused.
- `controls: n` and `control_seed: s`: n matched random controls, drawn
  from seed s. A control is a copy of the condition's schedule with each
  pushed neuron swapped for a random neuron of the same layer, at the same
  doses on the same reply tokens, so its schedule differs only in which
  neuron is pushed. It is never a unit in `units/known.csv`, never a unit
  any condition of the experiment pushes, alone or in a set, and a
  condition's n controls are n different neurons. Control k uses the same
  neurons for every prompt, seed and grid value, and runs in the same
  batch as the run it mirrors, unless a run and its controls are more than
  a batch holds. A condition that pushes nothing cannot have controls, nor
  yet one that pushes a set, and asking for more than a layer has neurons
  left is refused. With greedy decoding, a control and its run then
  differ only in the neuron pushed. With sampling, they do not yet share
  random numbers: `generate` draws each row's from its place in the batch,
  and a control sits in another row (decision 0014).

## What expand does

1. It checks every setting, naming any problem.
2. It reads the prompt set, and refuses an experiment of more than
   999,999 runs, which six-digit run IDs cannot number.
3. It checks every condition's schedule against the model and
   `max_new_tokens` with `validate()`, before it makes any run. The
   first problem stops it, with the condition's label and the rule it
   broke. A segment that will not take effect in a reply this long, such
   as one starting after the last token, gets a warning instead.
4. It draws the controls and checks each one's schedule the same way.
5. It makes one run per prompt, seed, condition (each grid value its
   own) and control, in that order: all of the first prompt's runs, its
   first seed first, and within a seed the conditions in the file's
   order, each condition's own run followed by its controls. Run IDs
   count up in that order, from `<name>-000001`.
6. It puts the runs into batches in order. A batch holds runs of one
   prompt and one seed, so no row is padded, and at most `batch_size` of
   them. A run and its controls stay in one batch, so the batch's shape
   rounds their arithmetic alike: when they do not fit in what is left of
   a batch, they start the next one. If they are more than `batch_size`,
   they are split, with a warning. Each run's `batch_key` is
   `b<batch>:<row>`, such as `b000002:0`. A prompt and seed with fewer
   runs than `batch_size` make a short batch: with one schedule and no
   controls, every batch holds one run. These are planned batches:
   `pushlab run` joins those with the same model, decoding settings and
   seed (and, with `group_by_length`, prompts of one length) into
   batches of up to `batch_size`, each planned batch whole, and saves
   each run's `batch_key` as where it ran (decision 0015).

Each run is a run record (`src/pushlab/store/records.py`) with no reply
yet and the status `planned`. Its schedule lists the segments it pushes,
a set's with its definition, each with `control_for` (for a control, the
unit it replaces) and `control_seed`.

Three hashes say what the runs came from. The `spec sha256` changes
whenever any setting does, but not with comments, spacing, the order of
keys (except a grid's), or 1 against 1.0. The `prompts sha256` is the
prompt set file's, and the `known sha256` is that of `units/known.csv`,
which the controls avoid, so a change there shows even when the spec is
the same. Decision 0014 explains these choices.
