CONTEXT
This is Sprint S2, prompt 7 of 9. Read CLAUDE.md, the decision notes of
prompts 1 to 6, and in docs/plan/development-plan.md the "6. store"
module and the manifest under "Data model".
- src/pushlab/store/records.py writes one runs.parquet per folder,
  atomically (a hidden .tmp file renamed into place), and refuses an
  existing one; read_runs reads only that file. A process killed while
  writing leaves only the hidden .tmp file.
- src/pushlab/store/folders.py refuses a folder that already holds runs.
- Prompt 5 added the steps record; prompt 6 expands an experiment into
  planned runs and batches.
- The docs put run folders in /workspace/runs; S1's reference runs went to
  /workspace/outputs.

GOAL
A store that saves an experiment one batch at a time, so that a crash
loses at most one batch, and reads all the saved batches back as one
experiment.

BUILD
- A runner folder: one manifest, and for every finished batch a runs part
  and a steps part. A batch counts as saved only when both parts are:
  either both go into a hidden temporary folder that is renamed into place
  in one step, or the steps part is written first and the runs part last,
  a steps part without its runs part being a leftover that the batch's
  redo replaces. Hidden leftovers of a killed write are ignored.
- Writing refuses malformed records (RunRecord's checks and the steps
  record's) and a part with another schema; reading refuses a part with
  another schema.
- The manifest has the plan's fields (code commit, lockfile, model
  revision, GPU, CUDA and driver, start and end times, runs by status),
  the experiment file's path, SHA-256 and text, and the calibration's
  SHA-256, with one entry per session that worked on the folder. The
  pre-registration fields stay empty until a phase has one:
  docs/prereg/ does not exist yet.
- Readers: every run and step of a runner folder, in the planned order;
  and S1's single runs.parquet, so S1's folders still read.
- `pushlab report` reads runner folders. A runner folder holds many runs
  of each prompt, so choose its layout (one column per condition with the
  controls left out, or a --condition option) and record it.
- A DuckDB view: a function that opens one connection with runs and steps
  views over every run folder under a root, the plan's "DuckDB view across
  all experiments". The root is /workspace/runs, where the docs put run
  folders; the note says that S1's reference runs stay in
  /workspace/outputs.
- A decision note, the next free number: the folder layout, how a batch
  counts as saved, the manifest, the readers, the report's layout and the
  view.

CONSTRAINTS
- store/records.py is 357 lines and report.py near 300: new code goes in
  new modules.
- No runner yet (prompt 8): tests write batches through the store's own
  functions.
- No dependency changes (duckdb and pyarrow are already there); files
  under about 400 lines. Reviews follow CLAUDE.md's review rule.

TESTS FIRST
On the CPU:
- Batches written one by one read back unchanged, runs and steps, in the
  planned order.
- A write stopped between a batch's two parts (a stand-in that fails after
  the first): the batch does not count as saved, and redoing it leaves its
  runs and steps exactly once.
- Hidden leftovers are ignored; malformed records, and parts with another
  schema, are refused when written and when read.
- The manifest gains one entry per session.
- A folder in S1's layout, one runs.parquet written by steer, still reads,
  and report still lays S1 folders side by side.
- report on a runner folder, in the chosen layout.
- The DuckDB view counts the runs of two runner folders and an S1 folder.

DONE WHEN
- `uv run pytest -q` and `uv run pre-commit run --all-files` pass, and the
  pull request's checks pass.
- One pull request with a description, merged before prompt 8 starts. No
  GPU step.

REPORT BACK
End with a plain-language summary for a beginner: what changed, why, how I
can check it myself, and anything I must do by hand. Explain why a batch
counts as saved only once both its parts are on disk.
