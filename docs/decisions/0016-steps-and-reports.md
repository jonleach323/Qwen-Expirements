# 0016: Every reply token is saved with the pushes behind it; reports mark where they switch

Date: 2026-11-08 · Sprint: S2 · Status: accepted

## Context

- The development plan's data model has a steps table, one row per run
  and token: run_id and step ("step 0 is the first reply token"), region,
  active_units, a float16 logprob, and later kl_vs_base and readouts. Its
  store module writes "versioned Parquet files with a manifest per
  experiment" and offers "a DuckDB view across all experiments".
- The S2 prompt pasted for this step ("prompt 5 of 7") asked for:
  - a steps table written by the runner: run_id, step, region, token_id,
    the chosen token's log-probability, active_units and active_amounts,
    stored compactly (small integers and 16-bit floats), so that 1,000
    runs stay well under 100 MB;
  - an experiment manifest: code commit, lockfile hash, model revision,
    GPU, calibration hash, spec hash, start and end times and run counts
    by status;
  - `src/pushlab/store/query.py`, a DuckDB connection with views across
    every experiment's runs and steps;
  - `pushlab report` on an experiment and a list of conditions, each reply
    with short markers where the active set changes ([A on], [A off, B
    on], [A/B x8]) and a one-line legend, readable on a phone;
  - `pushlab verify <spec>`: no duplicate or missing run IDs, every run
    has steps, and each run's recorded active units match its schedule.
- Decision 0008 says a record kept per reply token takes the pushes
  behind a token's choice from the pass before it: the prompt pass for
  token 0, and the pass of token k − 1 for token k. The engine already kept
  that record, `Generation.amounts` (decision 0013).
- It is not `docs/prompts/S2/07-run-store.md` ("prompt 7 of 9"), whose
  store and runner the runner of decision 0015 already is. Its plan's
  steps record (prompt 5 of 9) is the same as this one.

## Decision

**A step** (`pushlab.store.steps`). Step j is reply token j and the pass
that chose it: the prompt pass for step 0, the pass that read reply token
j − 1 for step j. So every column of a row is about one choice:
- `token_id`, the token chosen (32-bit);
- `logprob`, its log-probability on that pass, from the model's own
  next-token scores, before any temperature or top-p, worked in float32
  (`decoding.chosen_logprobs`) and saved as a 16-bit float; NaN only
  where the scores were not finite;
- `active_units` and `active_amounts`, the units pushed on that pass, in
  the order the run's schedule first names them, and the amount added to
  each one's neuron (dose times the unit's dose-1 amount, decision 0013),
  saved as 16-bit floats. A unit pushed at dose 0 is listed, with 0;
- `region`, "reply" for every step for now; thinking and answer come with
  thinking detection;
- `step` itself, 16-bit, so a reply may have up to 32,768 tokens.

A push on reply token k, the schedule's own numbering, first changes token
k + 1 and so shows at step k + 1; a prompt-region segment shows at step 0.
A handoff at token 40 shows A on steps 0 to 40 and B from step 41.

Unit IDs are text, which Parquet stores once per file and refers to by
small integers. 1,000 replies of 1,000 tokens, half of them pushed, take
4.9 MiB in one file, 8.7 MiB as the runner saves them in batches of 10, and
15.1 MiB in batches of 1, each file carrying its own small header: well
under 100 MB however they are batched.

**Saved by the runner** (`pushlab.runs.shards`). Each batch has two files
of one name: `shards/<name>`, its runs, and `steps/<name>`, its due runs'
steps (`batch.reply_steps` makes them from each reply's tokens, logprobs
and amounts). Both are written whole (a hidden temporary file, flushed and
renamed), runs first, and only then is the batch marked in the index, so a
batch counts with both files or neither. A batch whose runs all failed has
an empty steps file, and a failed run has no steps. A name taken by either
file is never used again. A run a new version keeps keeps its steps, in
the old batch's steps file. `read_result_steps(folder)` reads the steps
of the folder's current version, by run ID.

**The engine** records the log-probabilities next to the amounts it
already kept (`Generation.logprobs`). They are left out of a reply's `==`:
across batch layouts a token's log-probability differs in its last bits,
as its scores do (decision 0009), while the reply is the same.

**The manifest** (`pushlab.store.manifest`, written by
`pushlab.runs.manifest`). The runner's manifest.json is checked as an
experiment manifest: beyond every manifest's fields (code commit, lockfile
hash, model ID and revision, GPU, CUDA, torch, transformers, start and end
times, run count) it must hold the experiment and version, the experiment
file's SHA-256, the calibration's (None when nothing is pushed), the runs
ok, failed and pending, adding up to the run count, and the sessions.
Its top-level code and machine fields (code commit, lockfile, GPU, CUDA,
driver, torch, transformers, packages) are the last session's, and its
start time is its first session's, of any version: each session's own
entry says what that session ran, and verify's code-change warning says
when a version's runs were made by more than one code. When the runner
runs no session, as when a new version keeps every run or
a session was killed after its last batch, it restamps the manifest's
version, hashes and counts from experiment.json and the index, writing
only if something changed (`manifest.restamp_manifest`); a broken
manifest only gives a warning.

**verify** (`pushlab verify SPEC`, `pushlab.runs.verify` and
`verify_steps`). It expands the experiment file and reads the folder's
current version by its latest marks, loading no model and writing
nothing. It reads each counted shard's rows itself, so that a duplicate
shows, then its steps file, and checks that:
1. the version was made from this experiment file, prompt set and known
   units; if not, that is one problem, and nothing else is compared with
   the file's plan;
2. every planned run is saved, ok or failed, and nothing unplanned is;
3. no shard holds a run twice, and no steps file a step of a run twice;
4. each saved record is the run the plan means (`versions.same_run`, with
   the record's own model download), naming the fields that differ;
5. a run with a reply has steps 0 to n − 1, one per token, with its
   reply's token IDs and a known region; a failed run has none; a counted
   shard's missing or unreadable file is a problem, not a crash;
6. each step's units are those its run's schedule pushes on the pass that
   chose that token (`Schedule.doses_at`, in `Schedule.units()` order, as
   compile sums them), and its amounts follow the doses without the
   calibration: each amount has its dose's sign and is 0 at dose 0, and
   amount ÷ dose is one number per unit, its dose-1 amount, on every step
   of a run and across all runs, up to twice the 16-bit rounding (an
   interval check, so a run made with a calibration more than about 0.2%
   off shows; one closer than that passes);
7. manifest.json, if there is one, has experiment.json's version and
   hashes and the index's counts; this is skipped, with a note, while a
   runner holds the folder's lock, as a session rewrites the manifest when
   it ends.
Problems are grouped, one per run and kind with "N more steps disagree",
and at most 50 are printed, then a count of the rest and the summary line.
It exits 0 with no problem, 1 with problems or when it refuses (an
unreadable or unexpandable file, a runs dir or folder that is not this
experiment's, nothing saved yet, a broken experiment.json or index).
About 4 seconds per 1,000 runs of 600 tokens.

**The report** (`pushlab report SPEC [--conditions LABEL ...] --out FILE`,
`pushlab.analysis.experiment_report` and `markers`). It reads the
folder's version, its steps and experiment.json's condition labels, and
writes one Markdown file with no tables, so that it reads on a phone: a
title, where it came from, the legend, a line on what a marker means, any
notes (runs not saved yet, a prompt set changed since, a tokenizer of
another download), then a section per prompt with each chosen condition's
runs, the condition's own run first and then its controls, each a bold
label and its reply. Without --conditions it shows every label, in run
order. The S1 mode, `--runs DIR ...`, is as it was.
- **Letters.** Each unit gets A to Z, then AA, AB and so on, in the order
  the shown runs first name them. A control's random neuron takes the
  letter of the unit it stands for (its segments' control_for), and its
  label names it: "pulse length=4, control 1 (A is qwen3.5-4b:L14:N1234)".
  The legend is one line: "A = qwen3.5-4b:L14:N8949 · B = ...".
- **Markers.** A step's set is its active units' letters; before the reply
  nothing is active. A marker goes before each step whose set differs from
  the step before: units switched off first, then those switched on,
  several joined with +: [A on], [A off], [A off, B on], [A+B on]. So it
  sits before the first token chosen with that change: a pulse on reply
  tokens s to e − 1 shows [A on] before token s + 1 and [A off] before
  token e + 1, and a handoff at k shows [A off, B on] before token k + 1.
- **Alternations.** A stretch is a run of steps with one set. At least
  four stretches in a row that take turns between two sets collapse into
  one marker at the first, with their count: [A/B x8], or [A on/off x8]
  when one set holds the other (a unit that repeats). Units on in both
  are named first ("C on, A/B x8"). Steps with nothing pushed before the
  first push are not a turn. The next marker compares with the last
  stretch.
- **Places.** A marker before step j goes where the reply's text and the
  tokenizer's decoding of its first j tokens stop agreeing, so it never
  splits a character. This needs the model's tokenizer, as on the pod;
  without it the report refuses and says so.
- **Markdown.** The reply is HTML-escaped, Markdown's active characters
  get a backslash, and line breaks become <br>. A marker is written
  [**A on**], bold inside the brackets: CommonMark does not make
  **[A on]** bold next to a letter, and markers usually sit next to one.
- It refuses an unknown label (listing the labels it has), an --out inside
  the runner folder or one that exists without --overwrite, and a folder
  that is missing or not this experiment's.

**The DuckDB views** (`pushlab.store.query.connect(runs_dir=None)`). One
in-memory connection has three views over every runner folder in the runs
dir (`--runs-dir`'s default: `$PUSHLAB_RUNS_DIR`, else `/workspace/runs`):
- `runs`, one row per saved run of each folder's current version: the
  runs table's columns (schedule and decoding as JSON text) and version;
- `steps`, one row per step of those runs: the steps table's columns
  (logprob as FLOAT, amounts as FLOAT[]), experiment and version;
- `conditions`, each run's experiment, version, run ID and condition label.
The marks are read from each folder's experiment.json and index in Python
and joined to read_parquet's rows by file and run ID, so an old version's
runs, unmarked shards and earlier attempts never show, a run saved twice
shows as its latest mark says, and a kept run is read from the older shard
that holds it. Folders without experiment.json, such as S1's, are skipped;
the views are a snapshot at connect time. Paths are SQL-quoted, DuckDB's
glob characters are escaped and hive partitioning is off, so a folder
named like `step=3` is never read as a column. A runs dir that is missing,
a file or itself a run folder is refused, as is a shard an index names but
that is gone.

## Why this shape

- **Step j is the pass that chose token j,** rather than the pass that
  read it: then a row's logprob and its pushes are about the same choice,
  as the divergence and readouts to come will be, and it is the record
  decision 0008 asked for. The price is one place of offset against the
  schedule's numbering, which verify checks and the report's note states.
- **Amounts, not doses:** what was added to each neuron is what the hooks
  did; a dose is the schedule's, which the runs table already holds.
  verify checks them against each other without the calibration.
- **One steps file per batch,** beside its runs shard and counted by the
  same mark, rather than one file per experiment: saving a batch costs the
  batch, and nothing saved is rewritten (decision 0015).

## Consequences

- Where this differs from the plan's data model:
  - kl_vs_base and the readouts (proj_*) are not here; they come with the
    scoring and readout modules;
  - a step's region is "reply", the schedules' own word (decision 0012),
    where the plan says "thinking or answer": with thinking off, every
    reply is the answer, and "reply" will split into thinking and answer
    when thinking detection comes;
  - token_id and active_amounts are added to the plan's columns, as the
    prompt asked;
  - the manifest has no pre-registration fields yet, as no phase has a
    pre-registration file; runs_by_status counts ok, failed and pending,
    as the runner writes no collapsed runs yet.
- "Every run has steps" means every run with a reply: a failed run has no
  reply and no steps, and verify counts it in its last line.
- `pushlab steer` (S1's folders) saves no steps; only the runner does.
- Folders saved before this change have no steps files: verify says so,
  and the views leave their steps out.
- **The checks.** On the CPU with tiny-vl: the steps of a handoff, a pulse
  and a ramp match the schedule and the dose table on every pass; a failed
  batch's empty steps file, kept runs' steps, a crash before the mark and a
  steps file that cannot be saved; the steps schema, its refusals and its
  size; each token's log-probability against its pass's scores and a
  teacher-forced pass; the manifest's fields, refusals and restamping;
  verify clean, and catching a planted duplicate, a missing run, a record
  of another run, steps that disagree with the schedule or the doses, and
  a stale manifest; the markers of a pulse, a handoff and an alternation,
  in unit tests and placed in a tiny model's replies; and the views over
  two experiments, retries, new versions and an empty runs dir.
