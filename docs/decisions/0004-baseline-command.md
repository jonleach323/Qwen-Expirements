# 0004: The baseline command loads downloaded models and decodes with its own loop

Date: 2026-10-29 · Sprint: S0 · Status: accepted

## Context

- Sprint 0 ends with one unsteered Qwen3.5-4B reply, generated and stored in
  the plan's run format: a runs table with the fields under "Data model" in
  `docs/plan/development-plan.md`, and a manifest per experiment.
- Qwen3.5-4B ships as a vision-language model
  (`Qwen3_5ForConditionalGeneration`). Its model card loads it with
  `AutoModelForMultimodalLM` and `AutoProcessor`. Thinking is on by default
  and can only be switched off through the chat template's
  `enable_thinking`; the `/think` and `/nothink` commands do nothing.
- The model's repository has no `generation_config.json`. Its config names
  `<|endoftext|>` as the end token, but a chat turn ends with `<|im_end|>`,
  the tokenizer's end token.
- `scripts/setup_pod.sh` downloads each model into
  `/workspace/models/<registry name>` with a record of the commit it got
  (decision 0003).

## Decision

`pushlab baseline --model <name> --out <folder>` writes one run and a
manifest. Its parts:

- **Loading** (`pushlab.models.load.load_model(name, dtype)`). It loads only
  from `/workspace/models/<name>`, or the folder `PUSHLAB_MODELS_DIR` names,
  and only if the download record there matches the registry's ID and
  revision. It never downloads. Vision-language models load with
  `AutoModelForMultimodalLM`, as the model card does, and plain ones with
  `AutoModelForCausalLM`. The tokenizer, the text half of the card's
  processor, loads with `AutoTokenizer`. The loaded shape must match the
  registry. The run records the commit from the download record.
- **Precision.** bfloat16 on a GPU and float32 on a CPU. The data type is
  saved in every run's `decoding` and in the manifest.
- **Prompt.** The tokenizer's chat template renders one user turn with
  `enable_thinking=False`, which ends the prompt with an empty thinking
  block. The text is then tokenized without adding special tokens.
- **Decoding** (`pushlab.engine.generate.greedy_generate`). pushlab's own
  loop takes the most likely token at each step and reuses the model's
  cache. It stops after `--max-new-tokens` tokens (200 by default) or after
  `<|im_end|>` or the config's end token, which is kept as the last token.
  `output_text` is the reply decoded without special tokens.
- **runs.parquet** (`pushlab.store.records`). The columns are exactly the
  plan's fields, in its order. `schedule` and `decoding` are JSON text with
  sorted keys, which DuckDB reads with `->>`. `output_tokens` is a list of
  32-bit integers and `seed` a 64-bit one; nothing may be missing. A run ID
  is `<experiment>-<counter>`, counting from `000001`. `batch_key` is
  `b<batch>:<row>`, so the baseline's is `b000001:0`. Malformed records are
  refused before anything is written.
- **manifest.json** has the code commit, the SHA-256 of `uv.lock`, the
  model's ID and commit, the GPU, the CUDA, torch and transformers
  versions, the start and end times and the run count. It also records
  whether the code had uncommitted changes, the driver and Python versions,
  the experiment file and its hash, the device, the data type, the prompt
  and the runs by status. A field that does not apply, such as the GPU on a
  CPU, is `null`.
- **Files are replaced only when asked.** If the folder already has
  `runs.parquet` or `manifest.json`, the command stops before loading the
  model unless given `--overwrite`. Each file is written to a temporary
  name and then renamed, so a failed write leaves no half file.
- **Settings live in the experiment file.** `experiments/p0/s0_baseline.yaml`
  names the experiment, the seed and the default prompt. A prompt given
  with `--prompt` gets the ID `custom-` plus the first 12 characters of its
  SHA-256, and the manifest records its text.

## Why this shape

- **Not Hugging Face's `generate()`.** Without a `generation_config.json`,
  `generate()` would stop only at `<|endoftext|>` and run on past the end of
  the turn, and it merges in whatever settings the model's files carry.
  The plan's engine needs its own loop anyway, for the hooks and the step
  counter. Tests check that each token is the most likely one on tiny
  Qwen3 and Qwen3.5 models, including Qwen3.5's linear-attention layers.
- **Not `AutoProcessor`.** It also loads the image and video processors,
  which pushlab never uses; the tokenizer renders the same chat template.
- **No download when loading.** A command run in a cloud session, or with a
  typo, must not start a 9 GB download. The download record also gives the
  exact commit to store with the run.
- **JSON as text** rather than Parquet's JSON type, so pandas, pyarrow and
  DuckDB all read it the same way.
- **The data type inside `decoding`.** The runs table has no column for it,
  and the plan records the data type with every run.

## Consequences

- The pod's smoke tests (`pytest -m gpu -k smoke`, step 7 of the setup
  script) now include `tests/gpu/test_baseline_smoke.py`. It runs the
  command on Qwen3.5-4B, so step 7 also loads the model once and takes
  longer; prompt 8 should measure it against the 3-minute warm target.
- Run outputs belong outside the repository, such as `/workspace/runs/`.
- A run's `model` is the registry name and the commit, such as
  `qwen3.5-4b@851bf6e…`. The registry's revisions stay `main` until
  prompt 8 pins them.
- The command loads torch only when it runs, so `pushlab --help` answers
  at once.
