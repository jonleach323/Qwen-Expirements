# 0014: An experiment file expands into planned runs, in a fixed order, with matched controls

Date: 2026-11-07 · Sprint: S2 · Status: accepted

## Context

- The development plan's "5. runs" module expands one experiment file
  into individual runs, `expand(spec) -> [RunRecord]`, with deterministic
  run IDs and random control neurons "drawn from the same layer with a
  seeded generator, excluding known units".
- The S2 prompt pasted for this step ("prompt 3 of 7") asked for:
  `experiments/README.md` with the format and an example;
  `src/pushlab/runs/spec.py`, an ExperimentSpec read from YAML with a
  name, model, prompts file, decoding settings, seeds, max_new_tokens,
  batch size, max_gpu_hours and conditions, each a label and a schedule
  written out or given as a shape with a parameter grid, any of them with
  n random controls drawn from a seed and never from `units/known.csv`;
  `src/pushlab/runs/expand.py`, `expand(spec)` returning RunRecords, one
  per condition, control, prompt and seed, with IDs
  `<experiment>-<6-digit counter>` in a fixed order and a hash of the
  whole spec; and `pushlab expand <spec> --summary`. The same spec must
  always give the same runs, IDs and controls, and every condition must
  pass `validate()` before any run is made.
- It is not `docs/prompts/S2/06-experiment-files.md` ("prompt 6 of 9"),
  which also asks for `pushlab run --dry-run`, a speed and price in the
  file, a cost estimate and the 1,000-run kill-and-resume experiment.
  None of those is in this step.

## Decision

**The file.** `experiments/README.md` describes it and shows
`experiments/p0/s2_pulse_lengths.yaml`, which a test keeps identical to
the README's example. Every key is required: `name`, `model` (a registry
model; the test suite's model is refused), `prompts` (a path from the
repository's root), `decoding` (`strategy` greedy or sample; sampling
needs a `temperature` and takes a `top_p`, default 1; `dtype` bfloat16,
the default, or float32), `seeds` (different whole numbers; greedy
decoding takes exactly one), `max_new_tokens`, `batch_size`,
`max_gpu_hours` and `conditions`. A condition has a `label` (the form of a
schedule name, different from every other condition's) and a schedule as
schedule files write one (decision 0012): `segments`, or `shape` and
`params`. It may add `grid`, `allow_over_ceiling`, `controls` and
`control_seed`. Unknown keys are refused at every level, and the text is
read with `load_yaml` (YAML 1.2 numbers, a key given twice refused).
Thinking stays off: the file has no key for it until thinking mode.

**The spec** (`pushlab.runs.spec`, `pushlab.runs.conditions`).
`read_spec(path)`, `spec_from_yaml(text)` and `spec_from_mapping(data)`
return an ExperimentSpec; its `conditions` are Conditions, each with
`points`, one `(label, schedule)` pair per grid value. A grid gives some
of a shape's params as lists; every combination is a point, the first
param changing slowest, labelled `pulse length=16 alpha=-1.0`. A dose's
value is labelled as the float its schedule holds, so `alpha: [1, -1]`
and `alpha: [1.0, -1.0]` give the same labels, and true, false and null
are written as YAML writes them; two points with one label, such as two
doses that are one float, are refused. Each schedule is named after its
condition. A problem is a ValueError naming the setting, the condition by
its place and label, and with `read_spec` the file. A missing name or
label is called missing, and one that YAML read as a number, true or
false comes with a hint to quote it. A condition that asks for controls
but pushes nothing is refused here, before any model is involved, and so
is a grid whose points alone would need more runs than six-digit IDs can
number, before its schedules are built. The classes check nothing
themselves: a spec built in code skips these checks.

**The hash.** `spec.sha256` is the SHA-256 of `spec.to_mapping()` as JSON
with sorted keys: every setting, with each condition's schedules written
out as segments and each point's label. Comments, spacing, the order of
keys in a mapping, 1 against 1.0, and a shape against the same segments
written out give the same hash. A grid's keys are the exception to key
order: their order sets the order of the points, so it is a setting. Any
setting that changes the runs, the batch size and max_gpu_hours change
the hash, and so does the order of seeds, conditions and grid values. The
prompt set's own hash, of the file's bytes, comes with the expansion as
`prompts_sha256`, and that of `units/known.csv` as `known_sha256`.

**Order and IDs.** One run per prompt, seed, condition point and control,
nested in that order: every run of the first prompt first, its first seed
first, and within a seed the conditions in the file's order, each point's
own run followed by its controls. IDs count up from `<name>-000001` in
that order. An experiment of more than 999,999 runs, which six digits
cannot number, is refused once the prompt set is read, before any
schedule is checked or control drawn.

**Controls.** A condition with `controls: n` and `control_seed: s` draws n
sets of neurons with `random_controls(units, s, avoid)`: units are the
condition's units over all its points, and avoid is `units/known.csv`,
every unit any condition of the experiment pushes, and the condition's
earlier controls. So control k is each unit's k-th distinct draw, and it
pushes the same neurons for every prompt, seed and grid value. More
controls than a layer has neurons left are refused before any is drawn.
A control run's schedule is its point's schedule with each unit swapped,
at the same doses on the same passes; every stored segment carries
`control_for` (the unit replaced) and `control_seed`, both None for a
condition's own run. A control run's `seed` is the sampling seed of the
run it mirrors, and `control_of` names that run. This changes decision
0010, where `pushlab steer --control random` put the control seed in the
seed field: here the seed field is always the sampling seed.

A control runs in the same batch as the run it mirrors (Batches, below).
With greedy decoding they then differ only in the neurons pushed. With
sampling they do not yet: `generate` seeds each row's random numbers from
the batch's seed and the row's place in the batch (decision 0009), and a
control sits in another row of the batch, so it draws other numbers as
well. Every batch starts at row 0, so the same row of each prompt's
batches draws the same numbers, tying those prompts' samples together.
And where a run and its controls are split over batches, control k can
land in its run's row and share its numbers. Decision 0009 meant S2's
runs to seed their own rows; see Consequences. The controls also depend
on `units/known.csv`, so the expansion keeps its hash, `known_sha256`
(None when no condition has controls; refused if the file changes while
it is read), and `--summary` shows it.

**Validation.** Before any run is made, every point's schedule and every
control's pass `validate(schedule, REGISTRY[model], max_new_tokens)`, at
the default ceiling of 1.4 unless the condition sets
`allow_over_ceiling: true`. This is the check decision 0013 left to
whatever runs schedule files. The first failure stops expansion with
`condition '<label>': schedule '<name>' is refused:` and the rules
broken. Warnings come back in `Expansion.warnings`, each naming its
condition's point.

**Batches.** In run order, a batch holds runs of one prompt and one seed,
at most `batch_size`: one prompt, so no row is padded, and one seed,
since `generate` draws a batch's samples from one seed. A point's own run
and its controls stay in one batch, since a batch's shape changes how
bfloat16 rounds its arithmetic (decision 0009): when they do not fit in
what is left of a batch, they start the next one. When they are more
than `batch_size`, they are split, and `Expansion.warnings` says so for
their condition. `batch_key` is `b<batch>:<row>`, batches counted from
`b000001` over the experiment. The cost: a prompt and seed with fewer
runs than `batch_size` make a short batch, as does a set that does not
fit, so an experiment of one schedule without controls runs one row at a
time. `--summary` shows the batches' sizes.

**Planned runs.** `expand(spec)` returns an Expansion: `spec_sha256`,
`prompts_sha256`, `known_sha256`, `runs` and `warnings`. Each run is a
PlannedRun: `condition` (the point's label), `control` (0, or k for
control k) and `record`, a RunRecord with no reply: empty `output_text`
and `output_tokens`, and the status `planned`. Its `model` is the
registry name and revision, its `decoding` S1's four keys plus
`temperature` and `top_p` when sampling, and its `schedule` the stored
segments above. So the plan's interface, `expand(spec) -> [RunRecord]`,
becomes `expand(spec) -> Expansion`, whose runs hold the RunRecords.

**The command.** `pushlab expand SPEC` prints a line naming the columns,
then one line per run: ID, condition, control, prompt, seed, batch key
and control_of, separated by tabs (a field holding a tab, a line break or
a double quote is quoted, as CSV does). `--summary` prints the settings,
the hashes, the batches and a table of own runs, controls and totals per
condition point, ending with a row `(all)`, which no label can be.
Warnings go to stderr; a problem goes there too, with the file's name,
and exit status 1. It loads no model.

**The checks.** On the CPU: three pulse lengths, two prompts and two
seeds give 12 runs and 24 controls with the same IDs, records and hashes
on every expansion; the order is prompt, seed, condition, control;
controls keep their run's timing and doses bit for bit, name the unit
they replace, stay in its layers, are five different neurons per unit
for five controls, skip `units/known.csv` (whatever it lists, its hash
changing with it) and units other conditions push, and a seed's draws are
pinned; batches hold one prompt and seed and at most batch_size runs,
keep every control in its run's batch, and split a run and controls too
many for one batch with a warning; a layer out of range, another model's
unit and a push over the ceiling stop expansion naming the condition,
unless the condition allows the push; so do more runs than IDs can
number (before any schedule is checked), more controls than a layer has
left (in the singular for one), an unreadable prompt set, and known
units that cannot be read or change while read; a planned record is
refused by `write_runs` until its status is a result's; 54 refusals of
the text name their setting or condition; 20 changes of one setting each
change the hash, while comments, key order, 1 against 1.0 (for every
dose param in a grid too) and a shape against its segments do not, and a
grid's key order does; the example file's hash is pinned; and the
command's help, list, quoting, summary, warnings and refusals.

## Why this shape

- **RunRecords with their labels beside them,** in a PlannedRun, rather
  than a record type of its own: the prompt asked for RunRecords, and a
  planned record becomes the result record by filling in its reply. The
  point's label has no RunRecord field, and adding one would change the
  runs table's schema, so it comes beside the record.
- **The status `planned`,** rather than `ok` with no reply: `write_runs`
  keeps refusing it, so a planned run is never saved as a result.
- **A hash of the settings, not of the file's bytes:** it changes when a
  setting does and not when a comment does. The manifest can still keep
  the bytes' hash, as the baseline's `experiment_sha256` does.
- **Controls that avoid every unit the experiment pushes:** a control
  stands for an ordinary neuron, and a neuron under study is not one.
- **The same control neurons for every point:** a pulse of 4 tokens and
  one of 64 are then compared on the same random neuron, as their own
  runs are on the same real one.

## Consequences

- The runner (`pushlab run`, a later step) can run the expansion batch by
  batch: each batch has one prompt and one seed. It must store each
  run's condition label, which no RunRecord field holds, and hold the
  experiment to `max_gpu_hours`, which expand only records.
- For sampled experiments the runner should seed each row from its
  run's seed and prompt, not its place in the batch, as decision 0009
  planned: for example the seed mixed with the prompt's ID, as
  `row_seed` mixes in a place. Then a run and its controls, and every
  condition of one prompt, draw the same random numbers and differ only
  in what is pushed, different prompts stay independent, and a reply no
  longer depends on its row. Once rows seed themselves, a batch could
  also hold several prompts and seeds, which would fill the short
  batches above.
- The controls depend on `units/known.csv` as well as the spec: a unit
  added there can change which neurons later controls draw. The code
  commit pins that file, and `known_sha256` shows that it changed.
- `qwen3-8b` still has no pinned revision, so its planned runs record
  `qwen3-8b@main`; pin it before its first run.
- `pushlab steer` keeps decision 0010's records, the control seed in the
  seed field. CLAUDE.md lists `pushlab expand`.
