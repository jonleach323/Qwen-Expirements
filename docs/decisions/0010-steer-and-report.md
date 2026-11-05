# 0010: pushlab steer saves one condition per folder, and pushlab report lays folders side by side

Date: 2026-11-05 · Sprint: S1 · Status: accepted

## Context

- S1 needs commands that run pushed generations, save them as the plan's
  run records, and put the replies side by side for reading.
- The plan's runs table ("Data model") has a `schedule` of "segments: unit,
  sign, dose, start, end, region", and a `control_of` that, "for a random
  control", names "the run it mirrors". Its runs module draws "random
  control neurons ... from the same layer with a seeded generator,
  excluding known units".
- S1 has neither experiment files nor schedules yet; S2 brings both. So
  these commands take their settings from the command line.
- A dose's amount, α · r_L / ‖W_down[:, n]‖ (decision 0006), needs the
  layer's size from the calibration (decision 0007) and the neuron's
  weights. The Qwen3.5-4B checkpoint holds one more `down_proj` than its 32
  layers, for its multi-token prediction head (`mtp.layers.0`).

## Decision

- **`prompts/smoke.jsonl`**, version 1: five neutral prompts, a pancake
  recipe, a short poem about the sea, comforting a sad friend,
  photosynthesis explained simply and a name for a new cat.
- **`pushlab amount --unit ID --alpha X`** prints the amount the dose adds
  to the neuron, the layer's size, the length of the neuron's write
  direction, how far the push moves the residual stream, and the
  calibration file. It loads the model and uses the same function as
  `steer`, rather than reading one weight from the checkpoint by name.
- **`pushlab steer`** writes one condition into one folder:
  - every prompt of the set in one left-padded batch, greedy, thinking off,
    at most `--max-new-tokens` (200) tokens each;
  - the same pushes on every prompt: each `--unit` with its own `--alpha`,
    all in one window, `--start` to `--end` (end not included), and the
    prompt pass unless `--no-include-prompt`. The default, the whole reply
    and the prompt, is the evil-Qwen recipe's constant push;
  - one run per prompt, in the set's order: `steer-000001` and on,
    `batch_key` `b000001:<row>`, seed 0, status `ok`, and `decoding` as
    `pushlab baseline` records it;
  - each run's schedule holds one segment per push: `unit`, `alpha`,
    `start`, `end`, `include_prompt`, `amount` (the number added, from
    `units.dose.amount`) and `control_for` (null unless the push is a
    control). `alpha` carries the plan's sign and dose; regions come with
    S2's schedules.
- **Random controls** (`--control random --control-seed N`) replace each
  unit by a neuron of the same model and layer, at the same dose:
  - draw k for a unit is a SHA-256 hash of the seed, the unit and k, as a
    number modulo the layer's neuron count (`units.controls`). The first
    draw that is not in `units/known.csv`, not a unit given and not an
    earlier unit's control is its control;
  - the runs are numbered `steer-control-000001` and on. Each names in
    `control_of` the run it mirrors: the steer run of the same prompt. Its
    seed is the control seed;
  - the manifest's `control` maps each unit to its control.
- **The manifest** holds what `pushlab baseline` records, plus:
  - the prompt set's file, version and SHA-256, and the prompts' texts;
  - the schedule and the control;
  - the calibration file and its SHA-256, both null with no push.
- **Refusals**, all before the model loads:
  - a folder that already holds runs, unless `--overwrite`;
  - a unit in the wrong form, of another model, out of range or given
    twice (push it once, with the doses added);
  - unequal numbers of `--unit` and `--alpha`, a dose that is not finite,
    an `--end` that does not come after `--start`;
  - `--control` without a seed or without units, or a seed without
    `--control`;
  - a broken prompt set, or a model that is not downloaded;
  - with pushes, a missing calibration. The message gives the command to
    run: `uv run pushlab calibrate --model <name>`.
- **`pushlab report --runs DIR ... --out FILE.md`** writes:
  - a list of the folders: path, run count, model, decoding and the pushes
    in words;
  - one table with a row per prompt (its ID, and its text from the
    manifests) and a column per folder, named after the folder;
  - in each cell, the reply. Pipes and backslashes are escaped, line
    breaks become `<br>`, and `<`, `>` and `&` become entities, so HTML
    shows as text; other Markdown in a reply, such as `**bold**`, still
    shows as formatting. A reply that reached its token limit says so, and
    a prompt a folder has no run of shows `—`. Rendered by markdown-it's
    table parser, replies full of pipes, backslashes and line breaks keep
    one cell per column.

  It refuses a folder without runs, a folder with two runs of one prompt,
  and an existing file unless `--overwrite`.
- **Code:**
  - `pushlab.cli_parser` holds every command's options, so `cli.py` stays
    under 400 lines and `--help` never loads torch;
  - `pushlab.store.folders` holds the checks on `--out` that `baseline`
    and `steer` share;
  - `models.load.downloaded_revision` reads a download's commit without
    loading the model, so `steer` and `amount` find the calibration first.

## The commands checked

On the tiny Qwen3.5 (tiny-vl), in float32 on the CPU:

- `steer` at dose 0 gives the same replies as `steer` with no push and as
  `pushlab baseline` with each prompt alone.
- The schedule comes back from `runs.parquet` exactly, amounts included.
- Random controls repeat with their seed, stay in their layer and skip
  `units/known.csv`. The draws for two seeds are pinned in the tests.
- `report` builds one column per folder from `steer` folders.

On the real Qwen3.5-4B in bfloat16 on a cloud session's CPU, with the
residual sizes measured there (decision 0007), 40 reply tokens per prompt,
and each command in a process of its own, as on the pod:

- `pushlab amount --unit qwen3.5-4b:L14:N8949 --alpha 1` prints +23.4927,
  with layer 14's size 10.9175 and a write direction 0.4647 long: the
  amount decision 0008 measured.
- `steer` at dose 0 gives the replies of `steer` without `--unit`, token
  for token, for all five smoke prompts.
- At dose 1 every reply changes. The sea poem opens "The sea is not just
  water, But a hunger that never sleeps".
- The random control with seed 1 drew L14 N8707. It adds +25.7040, since
  its write direction has another length, and changes the replies in other
  ways.
- `report` lays the four folders side by side.

## Why this shape

- **One condition per folder.** The report's columns are folders, and
  conditions that must match exactly run in the same batch layout:
  dose 0 against no push, same prompts in the same order (decision 0009).
- **`steer` and `steer-control` as experiment names.** A run's ID says
  which kind it is, and `control_of` links a control to its mirrored run
  without the control needing the other folder.
- **Hash draws for controls.** A SHA-256 hash gives the same neurons for a
  seed whatever the Python, NumPy or torch version, as `row_seed` does for
  sampling (decision 0009).
- **A control never lands on a unit given.** A control equal to the unit
  would be no control.
- **The prompts' texts in the manifest.** The folder says what was asked
  even if the prompt file changes later, and the hash says which file.
- **Loading the model for `amount`.** A weight read by name would need the
  checkpoint's naming, which differs between models and holds the extra
  `mtp` layer. Loading costs seconds and gives exactly the amounts `steer`
  records.

## Consequences

- Compare dose 0 with `steer` without `--unit`, not with `pushlab
  baseline`, which runs one prompt alone: on the GPU a batch can tip a
  near tie the other way.
- Run IDs repeat across folders: each `steer` folder has a `steer-000001`.
  A run is known by its folder and ID, as S0's baseline runs are. S2's
  runner names runs after experiment files.
- `control_of` names a run by number, so it finds the mirrored run only in
  a folder made from the same prompt set.
- The schedule's keys are S1's. S2's schedule language adds what it needs,
  such as regions.
- `seed` is 0 for greedy runs and the control seed for controls.
