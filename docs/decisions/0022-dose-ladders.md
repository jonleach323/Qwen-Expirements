# 0022: Dose ladders find each persona's working and collapse doses, beside a capability set and coherence measures that need no judge

Date: 2026-11-13 · Sprint: S3 · Status: accepted

## Context

- The prompt for this step ("Sprint S3, prompt 4 of 8", pasted) asks to
  measure whether a pushed model still writes well and can still do simple
  tasks, and to find each persona's working dose and breaking point:
  `prompts/capability100.jsonl`, 100 short questions with exact answers,
  and an exact-match grader; coherence measures that need no judge
  (repeated n-gram rate, word variety, and the reply's perplexity under
  the unsteered model); `pushlab ladder --units <file> --doses
  0.25,0.5,0.75,1.0,1.25 --prompts <file> --out <experiment>`, which
  writes an experiment that runs every unit at every dose; and
  `src/pushlab/analysis/ladder.py`, which finds from scored ladder runs
  each unit's working dose (the lowest dose where speaks_as reaches the
  target rate) and collapse dose (the lowest where coherence falls below
  the threshold), with confidence intervals. Thresholds live in
  `configs/panel.yaml`, starting at a speaks_as rate of 0.8 and a
  coherence threshold of 6.
- The research plan's "Dosing rules": doses are in units of the layer's
  median residual size; each unit's lowest reliable dose comes from a
  ladder, with its collapse dose. Liu (2026) puts collapse at about 1.4 to
  1.7 times the residual size for Qwen. Its "Measures": capability is
  exact-match accuracy on about 100 short questions; coherence is
  perplexity under the unsteered model, a repetition rate and a judge for
  long texts. Its "Measurement pitfalls": a plain word-variety score reads
  long fluent text as collapsed past about 128 tokens.
- Experiment files push single neurons only: a NeuronSet's ID is refused
  in them until a later step (decision 0019, `schedules/yaml_io.py`).
- The Neuron Persona Library states no licence, so none of its data, its
  persona neurons' IDs included, goes into git (decision 0020).
- The judge (decision 0021) asks the same persona about every reply of an
  experiment; a ladder of many personas needs each unit's replies asked
  about its own persona.

## Decision

**Thresholds** live in `configs/panel.yaml`, version 1
(`pushlab.analysis.panel`, which checks every key and refuses unknown
ones): `working_rate` 0.8, `coherence_below` 6, intervals at
`confidence` 0.95 from `resamples` 2,000 drawn with `seed` 0, and the
judge-free measures' settings, `ngram` 4, `window` 50, and the flags
`repetitive_above` 0.3 and `varied_below` 0.4, which are provisional
until checked against the judge and the owner's labels.

**The capability set** (`prompts/capability100.jsonl`, version 1): 40
arithmetic questions with whole-number answers, 30 facts and 30 simple
reasoning questions, each ending with its form ("Answer with the number
only.", "one word only" or "the name only"), written new for this set.
Their answers are in `prompts/capability100-answers.jsonl` beside it, as a
prompt set holds only its prompts: each answer's kind and type, other
accepted spellings, a number's units, and a check (an arithmetic
expression, which a test evaluates without `eval`, or why a fact is
right). A second agent answered every question from the question alone
and compared with the key. The grader (`pushlab.scoring.capability`,
version 1) reads the part after the last "answer:" or "answer is", else
the first line; compares numbers by value, without thousands commas or
units, and texts without case, spaces, quotes, bold or a final full stop;
and marks a reply strict when it is the answer alone.

**Coherence without a judge** (`pushlab.scoring.text_measures`, version
1): a reply is read as its words, lowercase, or, when most of its letters
are Chinese, Japanese or Korean, its characters. Its repetition is the
share of its 4-grams it already said; its word variety the share of
distinct units in each window of 50, averaged over every window, which
does not fall as fluent text grows. A reply is flagged repetitive above
0.3 and of low variety below 0.4. Its perplexity under the unsteered model
(`pushlab.engine.score_text`) comes from one teacher-forced pass over the
prompt, rendered as `generate()` renders it, and the reply's tokens, with
nothing pushed, batching only sequences of one length, so nothing is
padded; for an unpushed reply it equals `generate()`'s own
log-probabilities, which the tests check on both tiny models.

**`pushlab measure`** saves these as rows of the scores table (decision
0021): scorers `repetition`, `word_variety` and, when the prompt set has
an answers file, `capability` (judges `pushlab:text` and
`pushlab:exact-match`); with `--perplexity`, a GPU job the owner runs,
`perplexity` (judge `model:<model>@<revision>`). A rerun measures only
what is new.

**`pushlab ladder`** (`pushlab.runs.ladder`) writes an experiment file:
one condition per unit of the units file (`pushlab.units.ladder_units`: a
CSV with a `unit` column and, optionally, `direction`, `persona`,
`description` and `source`; a candidates list is one), labelled `u01`,
`u02` and so on, each a constant push of its unit, prompt pass included
unless `--no-prompt-pass`, with a grid of signed doses (the unit's
direction times each dose). So it runs units × doses × prompts × seeds
runs, plus an unpushed baseline with `--baseline` and matched random
controls with `--controls`. One seed decodes greedily; several sample
each reply at temperature 1. `max_gpu_hours` is by default 1.5 times the
time the runs' reply tokens take at the model's speed, the one `pushlab
run --dry-run` estimates with, rounded up to a quarter hour. A dose above
the ceiling of 1.4 sets `allow_over_ceiling`, as a ladder that seeks the
collapse dose must go there. The file's text is read back and expanded
before it is put in place, so a file that would not run is never left. A
units file with a unit whose `source` is the library can only be written
outside the repository, such as `/workspace/experiments/`, which an
experiment file may be, its prompt set still named from the repository's
root. Sets of neurons are refused until experiment files take them.

**Each unit's replies are asked about its own persona.** `pushlab score
SPEC --rubric speaks_as --units FILE` asks the replies of each unit, its
own and its random controls', and the unpushed replies, about that unit's
persona: the same question at every dose of the unit, so the judge never
learns the dose, which is what a ladder measures. This narrows decision
0021's rule, one persona for every reply of an experiment, to one persona
for every reply of a unit's ladder. A unit's persona is its `persona`
column's name, described by `--personas` when given, else by its
`description` column; units of one persona and description share one
question, so an unpushed reply gets one row per question. A pushed run of
no unit in the file, or of several, is asked nothing, and the header
counts it. `--windows sentences` works the same way.

**`pushlab doses`** (`pushlab.analysis.ladder`, `doses_command`) reads a
scored ladder: for each unit, the speaks_as verdicts about its persona and
the coherence scores of one judge, at each dose of its own runs, the
unpushed runs at dose 0. Its working dose is the lowest dose at which at
least 0.8 of its verdicts are speaks_as; its collapse dose, the lowest at
which its replies' mean coherence is below 6. A dose's share has a Wilson
interval. Each found dose has a percentile interval from 2,000 bootstrap
resamples of the prompts, a prompt's replies at every dose kept together,
as the ladder is paired by prompt; a resample that reaches no dose counts
beyond the ladder's top, so an interval can be open. Random controls get
their own speaks_as rate per dose, and the judge-free measures, when
saved, their means per dose.

## Why this shape

- **The answers beside the set,** not in it: a prompt set holds its
  prompts alone, so a model never sees an answer, and the grader reads
  them by prompt ID.
- **A key checked twice:** a wrong or ambiguous answer marks right replies
  wrong, which no number of questions can average away.
- **Word variety in windows,** as the plan's pitfall warns, and
  perplexity from one forced pass of the unsteered model, which measures
  how surprising the pushed text is to the model it came from.
- **One persona per unit's ladder:** asking every reply about every
  persona would multiply the judge's calls by the panel's size, and asking
  a reply about its own unit's persona tells the judge nothing about the
  dose.
- **A bootstrap over prompts** gives a dose's interval directly on the
  ladder's own doses, without assuming a curve, and keeps each prompt's
  replies together across doses.

## Consequences

- Library ladders are written and run on the pod, outside the repository;
  their reports stay there too.
- A judge should pass the labels check (decision 0021) before it scores
  a ladder at scale.
- The ladder's resolution is its doses: a working dose of 0.75 means
  somewhere above 0.5 and at most 0.75.
- Sets of neurons, such as Mayner's persona sets, wait for experiment
  files that take them.
- `--perplexity` loads the model: the owner runs it on the pod.

## Results

On a cloud session's CPU, on the tiny test models and a fake judge only;
the pod has run no ladder yet.
