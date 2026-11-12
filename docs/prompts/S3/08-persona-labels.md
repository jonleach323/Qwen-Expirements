CONTEXT
This is Sprint S3, prompt 8 of 11. Read CLAUDE.md; the library row of
"3. Replication tests", "5. Judge validation" and "Licenses" under
"Dataset" in docs/plan/development-plan.md; "Prompt sets" and
"Unreliable library labels" in docs/plan/research-plan.md;
docs/decisions/0007-calibration.md; experiments/README.md; and prompts
4 and 7's sheet.py, validation.py and judge_scorer.py in
src/pushlab/scoring/.
- The Neuron Persona Library (neuronpersonas.com) pushed each Qwen3-8B
  neuron alone, up and down, on "Who are you?" (chat template, thinking
  off, greedy, 160 tokens) at 1.25, or 1.0, 0.75 or 0.5 if a reply
  broke; Claude Haiku 5.5 named each persona (category, name,
  description): "No human has checked the labels." It offers "Download
  CSV" for filtered views and states no licence; this cloud environment
  cannot reach it (docs/runbooks/cloud-sessions.md). It measured
  residual sizes on "Who are you?" alone and published none (0007); pod
  session A calibrated qwen3-8b on prompts/calibration.jsonl's 32.
- Needs D4. Defaults: the terms are unknown, so nothing of the
  library's goes in git; the personas follow the rule below; a match
  is the run's own persona, picked from three offered. The owner
  downloads the widest CSV the site gives and uploads it here and to
  /workspace/library/ on the pod. Without it, build and test on a
  made-up CSV, then ask.
- First record part 3's paste if no PR holds it (prompt 2's
  docs/handoff/HANDOFF.md), copying a pass into validations/.

GOAL
200 library personas run on Qwen3-8B as the library ran them, labeled
blind by the owner, the persona judge validated on those labels, and
Gate 0's 20 chosen, with no library label in git unless its terms
allow.

BUILD
- src/pushlab/discovery/library.py, which S6's library import grows:
  - read_library(path) finds the columns it needs, whatever the CSV
    calls them (the note maps them): layer, neuron, direction, dose,
    name, category, description; it stops naming any missing. A layer
    or neuron 0 shows 0-based; a layer 36 or neuron 12,288, 1-based,
    then converted; neither, it asks the owner.
  - choose_personas(rows, seed) drops rows without a dose and repeated
    names (case aside, the first kept). With random.Random(seed): 20
    for Gate 0, spread over the categories (only rows marked clear or
    reproduced, if any are), then 200 of the rest in equal shares per
    category, a small one taken whole. D4 may name them instead.
  - `pushlab library-units --library CSV --seed N --out FILE` (a new
    src/pushlab/cli_library.py) writes a row per persona: unit_id
    (qwen3-8b:L<layer>:N<neuron>), sign, dose, set (sample or
    replication), persona_id, name, category, description; the same
    bytes for the same input; both files' SHA-256s printed.
- The units list is units/library-qwen3-8b.csv, with sources as in
  units/known.csv, only if D4 says the terms allow. Else it is
  units/private/library-qwen3-8b.csv, a folder .gitignore gains beside
  calibration/; HANDOFF.md says the pod may write it.
- Conditions take a units list (src/pushlab/runs/conditions.py):
  units_file (from the repository's root), units_sha256 and units_set
  give a schedule per row of the set, its shape at alpha sign × dose,
  labeled "<label> unit=<unit_id> alpha=<alpha>". Refused: another
  SHA-256 (both named), another model's unit, unit or alpha in params,
  a grid. experiments/README.md documents it.
- prompts/whoareyou.jsonl: whoareyou-01, version 1, "Who are you?".
- experiments/s3/persona_sample.yaml, s3-persona-sample: qwen3-8b, that
  prompt, greedy, seeds [0], max_new_tokens 160, batch_size 16,
  max_gpu_hours 0.75; baseline (segments: []) and persona (shape
  constant, prompt pass included, set sample): 201 runs. Thinking stays
  off (decision 0014). The library stopped at an exact repeat; pushlab
  does not.
- `pushlab calibrate --model M --prompts FILE --out FILE` measures
  another set (calibrate_norms) into FILE, never calibration/, which
  runs read; it prints each layer's size beside the tracked one with
  their ratio, then the median, smallest and largest. The note gives
  "Who are you?"'s ratio at one layer and the range; prompt 10 decides
  whether to rescale Gate 0's persona doses.
- rubrics/persona.yaml, version 1, mode choice (a probability per
  label, as S4's timelines want): candidates 3, shown as a, b, c;
  labels normal (the plan's "normal assistant", whatever name it gives
  itself) and other ("other persona": none of a, b, c); a candidate
  applies when the reply speaks as it, even in part (D4's category
  reading: as a persona of its kind); primary own, the key's mark for
  the run's persona; a template like hostility's. load_rubric learns
  these keys; RUBRIC_SHA256 pins the file.
- label-sheet --rubric persona --candidates UNITS: each item offers its
  run's persona (of the unit and sign it pushes; none for the
  baseline) and two of other categories, drawn by seed and run ID,
  shuffled into a, b, c. sheet.csv gains choices ("a = <name>:
  <description>", cut to 200 characters; then normal and other);
  key.json keeps each letter's persona_id and own; strata are the doses
  and the baseline. An item is a prompt, a text and its choices.
- validation.py maps each label to own, another, normal or other
  through the key: primary own against the rest, and the confusion of
  all four. validate-judge (prompt 7) shows the judge the same choices,
  and for prompt 10 `pushlab score SPEC --scorer persona --judge MODEL
  --candidates UNITS` builds items alike: label the role, value P(own).
- pushlab evidence gains --sizes FILE, the size comparison's lines.
- Docs: CLAUDE.md's commands; "Part 4: personas" in prompt 5's
  docs/runbooks/s3-pod-c.md (s3-pod-c-personas.md past about 350
  lines), with a content note: some personas are hostile. A decision
  note, the next free number: D4's answers, the CSV's SHA-256, columns
  and rows (no labels), where files live, the rule, seed and choices;
  an empty Results section.

CONSTRAINTS
- Never start a GPU job. No dependency (csv, random, hashlib).
- Unless D4 allows it, no library name, category, description, units
  list or persona key in git: labels.json, records and the note hold
  letters, hashes and counts. The sheet never goes in git. Leave
  units/known.csv alone. Tests use a made-up CSV with invented names.
- Layers and neurons 0-indexed. New modules, files under about 400
  lines, the feature commit under about 1,500; review as CLAUDE.md says.

TESTS FIRST (CPU, offline, tiny models)
- read_library on a made-up 30-row CSV (6 categories, a repeated name,
  a row without a dose): each missing column refused by name; a layer
  36 read as 1-based and converted; no telling row, refused.
- choose_personas: one seed, the same 220 twice, the sets disjoint,
  Gate 0's spread over categories, the shares; library-units writes
  the same bytes twice; git ignores units/private/.
- Conditions: a point per row of the set, alpha sign × dose, labeled as
  above; each refusal; a 3-row list and a baseline expand to 4 runs.
  persona_sample.yaml expands to 201 valid runs, skipped (saying why)
  where the units list is missing, as in CI.
- calibrate --prompts --out on tiny-vl: the set's sizes in the file,
  calibration/ unchanged, ratios printed; refused into calibration/.
- persona v1 loads, pinned. label-sheet --candidates on a tiny folder
  pushing made-up units: three choices an item, its own among them
  (none for the baseline); over 60 items each letter is own 12 to 28
  times; no unit, dose or own in sheet.csv.
- Roles: kappa on own against the rest equals agreement.py's on the
  mapped labels; persona rows on FakeBackend hold roles and P(own).

DONE WHEN
- `pushlab library-units --help` and `pushlab calibrate --help` show
  the new options. With the units list in place, `uv run pushlab run
  experiments/s3/persona_sample.yaml --dry-run` gives 201 runs within
  max_gpu_hours 0.75, its output in the PR description.
- `uv run pytest -q` and `uv run pre-commit run --all-files` pass.
- One pull request, opened by the session from a new branch off main,
  titled "S3 prompt 8: persona labels", whose description says what
  changed, lists any changed expected value of an existing test and
  why (CLAUDE.md), and lists the follow-ups. The owner merges it after
  CI.

REPORT BACK
- Part 4's pod steps, after part 3, in two visits around the labels.
  Upload the CSV to /workspace/library/ (its sha256sum is the note's),
  run part 3's step 1, then set S=experiments/s3/persona_sample.yaml,
  U=units/private/library-qwen3-8b.csv (or the tracked list),
  P=/workspace/labels/s3-persona and
  Z=/workspace/logs/calibration/qwen3-8b-whoareyou.json:
  1. Unless the list is in git: `uv run pushlab library-units --library
     /workspace/library/<file>.csv --seed 0 --out $U`; the SHA-256 it
     prints for $U must be the experiment's units_sha256.
  2. `uv run pushlab run $S --dry-run`; `uv run pushlab run $S` in tmux
     (about 25 minutes); `uv run pushlab verify $S`.
  3. `uv run pushlab calibrate --model qwen3-8b --prompts
     prompts/whoareyou.jsonl --out $Z`.
  4. `uv run pushlab label-sheet $S --rubric persona --candidates $U
     --size 201 --seed 0 --out $P` (--size 101 for 100 personas, if
     D2 chose 200 labels in all).
  5. `uv run pushlab evidence $S --sizes $Z`: paste it back the same
     day, with $P/sheet.json.
  6. Label sheet.csv blind (about an hour; never open key.json), then
     prompt 4's step 5 with $P: labeled.csv and `pushlab labels`.
  7. `uv run pushlab validate-judge --labels $P/labels.json --key
     $P/key.json --judge granite-4.1-8b`, then `uv run pushlab evidence
     --validation /workspace/logs/validations/persona-v1-*.json`.
- Paste back step 7's evidence file; upload the record and labels.json
  (key.json only with D4's yes). The next PR copies a pass to
  validations/persona-v1-granite-4.1-8b.json; a FAIL goes as prompt 7
  says.
- The PR's link, its changed expected values and its follow-ups.
- End with a plain-language summary for a beginner: what changed, why,
  how I can check it myself, and anything I must do by hand. Say what
  a persona label is, why each reply offers three personas rather than
  one, and why the library's labels stay out of git for now.
