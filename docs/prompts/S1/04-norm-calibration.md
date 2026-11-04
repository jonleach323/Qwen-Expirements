This is Sprint S1, prompt 4 of 8.

CONTEXT
Read CLAUDE.md and these parts of the plans:
- docs/plan/development-plan.md: the "2. units" module, whose decisions
  say each layer's median residual size is "measured once per model by
  `calibrate_norms` and cached";
- docs/plan/research-plan.md: the dose rule under "Concepts", and "What
  the October 2026 wave established".

Prompt 3 built neuron units, whose amount(alpha) needs these residual
sizes. S0's load_model and chat_prompt_ids load a model and format a
prompt.

GOAL
Measure each layer's median residual size once per model, keep it in a
cache, and give it to the dose rule.

BUILD
- src/pushlab/units/norms.py: calibrate_norms runs each calibration prompt
  through the model once, without generating. Format the prompts as the
  baseline does: chat template, thinking off. For each layer, it takes the
  median, over all real tokens, of the length of the residual stream at
  that layer. Padding tokens never count.
- Which residual stream "at a layer" means: match the published recipe.
  Read how occcat's and Mr-Neutr0n's repositories, linked in the research
  plan, compute it; clone them if the session can reach them. If it can't,
  use the stream leaving each layer, and say so in the decision note.
- prompts/calibration-v1.yaml: a versioned prompt set of about 64 short,
  neutral, everyday requests, with an ID and a version. Prompts never go
  in code (CLAUDE.md). The owner reviews them in the pull request.
- A cache, one JSON file per model, model commit and data type, in a
  folder on the pod volume such as /workspace/calibration. An environment
  variable can point elsewhere, as PUSHLAB_MODELS_DIR does for models.
  - The file holds the medians, the prompt set's ID and SHA-256, the token
    count, the code commit and the time.
  - load_norms reads it, and refuses a file made for another model commit,
    data type or prompt set.
- `pushlab calibrate --model NAME [--prompts FILE] [--dry-run] [--overwrite]`.
  It is a GPU job: the owner runs it.
  - --dry-run prints the model, prompt count and output file without
    loading the model.
  - Keep cli.py thin by putting the command's logic in the module.
- docs/runbooks/pod.md: how to run it, and where the file goes.
- A decision note (the next free number) covering:
  - the definition of the median residual size;
  - the prompt set and the cache;
  - why the values are measured in bfloat16 on the GPU, the data type
    every run uses.

CONSTRAINTS
- The plan says "measured once": calibrating never happens silently
  inside another command. A missing cache is an error that names the
  command to run.
- Never overwrite a cache without --overwrite.
- The calibration file is a model property that later runs depend on, not
  a run output. Prompt 8 copies the Qwen3.5-4B values into
  docs/baselines/s1/.
- No dependency changes.

TESTS FIRST
- On tiny_model and tiny_vl_model, the medians equal a hand calculation
  from the model's hidden states on the same prompts. There is one
  positive value per layer.
- A left-padded batch gives the same medians as the prompts run one at a
  time.
- The cache round-trips. load_norms refuses another model commit, data
  type or prompt set, and an existing file is refused without --overwrite.
- A prompt set file without an ID or version, or with no prompts or
  duplicates, is refused.
- `pushlab calibrate --help` lists the options. --dry-run never loads a
  model: replace load_model with one that fails, as S0's command tests do.
- End to end on tiny-vl (the tiny_downloads fixture), the command writes a
  cache that load_norms reads and prompt 3's amount() uses.

DONE WHEN
- `uv run pytest -q` passes on CPU, and the pull request's checks pass.
- The pull request shows `pushlab calibrate --help` and a dry run's output.
- On the pod (the owner, in S1's first pod session),
  `uv run pushlab calibrate --model qwen3.5-4b` and
  `uv run pushlab calibrate --model qwen3-8b` each write their cache with
  one finite, positive value per layer.

REPORT BACK
End with a plain-language summary for a beginner: what changed, why, how I
can check it myself, and anything I must do by hand.

This is the last prompt before S1's first pod session, so give the whole
session as one numbered list:
1. Start the pod and, before pulling, run `bash scripts/setup_pod.sh`.
   Note its step times: they are the warm-start row for decision 0005, on a
   restarted pod with nothing new to download.
2. `git pull`, then run the setup script again. This run downloads
   Qwen3-8B, so its time doesn't count as warm.
3. Copy /workspace/runs/s0-baseline off the pod.
4. Prompt 2's model shape test.
5. Both calibrations.

Say exactly what to paste back: the warm run's step times, the shape test's
last line, Qwen3-8B's download commit, and both calibration files.
