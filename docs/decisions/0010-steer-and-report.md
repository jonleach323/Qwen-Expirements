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
    all in one window, on the passes that read reply tokens `--start` to
    `--end` (end not included), and on the prompt pass unless
    `--no-include-prompt`. A push on reply token k first changes token
    k + 1, and only the prompt pass changes token 0 (decision 0008). The
    default, the prompt and the whole reply, is the evil-Qwen recipe's
    constant push. A `--start` above 0 with the prompt pass on is allowed,
    since the owner asked for it on by default, but `steer` points out that
    the replies then change from their first token;
  - one run per prompt, in the set's order: `steer-000001` and on,
    `batch_key` `b000001:<row>`, seed 0, status `ok`, and `decoding` as
    `pushlab baseline` records it;
  - each run's schedule holds one segment per push: `unit`, `alpha`,
    `start`, `end`, `include_prompt`, `amount` (the number added, from
    `units.dose.amount`), `control_for` (null unless the push is a
    control) and `touched`. `alpha` carries the plan's sign and dose;
    regions come with S2's schedules;
  - `touched` is that run's count from the engine (decision 0009): how many
    of its reply tokens the push was on. A run whose reply ended before
    the window shows 0, so a push that never came cannot pass for one that
    did nothing.
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
- **The manifest** holds what any pushlab manifest records (code,
  lockfile, GPU, CUDA and library versions, the model and its commit, the
  device, data type, times and run counts), and:
  - `command` and `experiment`;
  - the prompt set's file, version and SHA-256, and the prompts' texts;
  - the schedule, without the per-run `touched`, and the control;
  - the calibration file and its SHA-256, both null with no push. The file
    is hashed as it is read, and refused if it changes meanwhile;
  - `decoding`, and `stop_token_ids`, the tokens that end a reply.

  A file is named from the repository's root when it is inside it, and in
  full otherwise. Unlike the baseline's, the manifest has no experiment
  file: the options are the experiment.
- **Refusals**, all before the model loads:
  - a folder that already holds runs, unless `--overwrite`;
  - a unit in the wrong form, of another model, out of range or given
    twice (push it once, with the doses added);
  - unequal numbers of `--unit` and `--alpha`, a dose that is not finite,
    an `--end` that does not come after `--start`;
  - `--start`, `--end` or `--no-include-prompt` without a `--unit`, and a
    window no pass can reach: with the prompt pass off, a `--start` past
    reply token `--max-new-tokens` − 2, the last one a pass reads;
  - `--control` without a seed or without units, or a seed without
    `--control`;
  - a broken prompt set, or a model that is not downloaded;
  - with pushes, a missing calibration. The message gives the command to
    run: `uv run pushlab calibrate --model <name>`.

  One check needs the model's weights, so it comes after loading but
  before generating: an amount that is not finite, from a huge dose.
- **`pushlab report --runs DIR ... --out FILE.md`** writes:
  - a list of the folders: path, run count, model, prompt set, decoding
    and the pushes in words;
  - one table with a row per prompt (its ID, and its text from the
    manifests) and a column per folder, named after the folder;
  - in each cell, the reply. Pipes and backslashes are escaped, every
    kind of line break (`\n`, `\r`, U+2028 and the like) becomes `<br>`,
    and `<`, `>` and `&` become entities, so HTML shows as text; other
    Markdown in a reply, such as `**bold**`, still shows as formatting.
    Rendered by markdown-it's table parser, replies full of pipes,
    backslashes and line breaks keep one cell per column;
  - under a reply, a note when it was cut off at its token limit (a reply
    whose last token ends its turn was not, which the manifest's stop
    tokens tell; without them the note says only that the reply reached
    the limit), and when no push reached it. A prompt a folder has no run
    of shows `—`.

  It refuses a folder without runs, a folder with two runs of one prompt,
  folders that give one prompt ID different texts, an `--out` that is a
  folder or cannot be written, and an existing file unless `--overwrite`.
  `--runs` may be given more than once.
- **Doses are printed as given,** such as `+1.0000004`, so two folders
  never look alike when their doses differ.
- **Code:**
  - `pushlab.cli_parser` holds every command's options, so `cli.py` stays
    under 400 lines and `--help` never loads torch;
  - `pushlab.store.folders` holds the checks on `--out` that `baseline`
    and `steer` share;
  - `models.load.downloaded_revision` reads a download's commit without
    loading the model, so `steer` and `amount` find the calibration first.

## The commands checked

On the tiny Qwen3.5 (tiny-vl), in float32 on the CPU, with tiny-vl's own
residual sizes as its calibration (larger made-up sizes swamped the model,
so that every dose gave the same reply):

- `steer` at dose 0 gives the same replies as `steer` with no push and as
  `pushlab baseline` with each prompt alone.
- The engine gets exactly the pushes the options name, window, prompt pass,
  every `--unit` and the calibration's sizes included, and the replies are
  the ones `generate()` gives with them. Each run's `touched` is the
  engine's count.
- The schedule comes back from `runs.parquet` exactly, amounts included.
- Random controls repeat with their seed, another seed draws others, each
  stays in its own unit's layer, and they skip `units/known.csv`. The draws
  for two seeds are pinned in the tests.
- The defaults push the prompt and every reply token, up to 200 tokens.
- `report` builds one column per folder from `steer` folders.
- Each of 65 deliberate mistakes in the new code makes a test fail.
- An adversarial review (three reviewers, then a skeptic per serious
  finding trying to refute it) read the code, built awkward run folders and
  rendered reports with markdown-it and marked (JupyterLab's renderer).
  Every finding a skeptic confirmed is fixed and pinned by a test.

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
- `units/known.csv` may list only units of models in the registry, as a
  test checks: a published unit of another model is added together with
  its model's registry entry, or every random control would be refused.
