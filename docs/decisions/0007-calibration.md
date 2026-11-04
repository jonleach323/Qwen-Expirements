# 0007: Doses are scaled by each layer's median residual size, measured on 32 chat prompts

Date: 2026-11-04 · Sprint: S1 · Status: accepted

## Context

- The dose rule (decision 0006) is c = α · r̃_L / ‖W_down[:, n]‖. Here r̃_L,
  the median size of the residual stream leaving layer L, is measured once
  per model, and it sets the unit of every dose.
- The evil-Qwen recipe defines it as "median(‖resid‖ leaving layer L,
  tokens 1:)" but names no text to measure it on. Both public replications
  used long plain prose with no chat template: occcat 4 passages (578
  tokens), Mr-Neutr0n the reference text from the recipe author's own tool
  (119 tokens). Both leave out each text's first token. hedwigai mixed plain
  and chat-formatted texts.
- The S1 task asks for 32 short everyday prompts in the chat template with
  thinking off, the way pushlab shows the model text in its runs.
- Two traps in transformers 5.21, both checked on the tiny models:
  - the last entry of `output_hidden_states` is the final norm's output,
    not the last layer's;
  - a forward pass that asks for hidden states leaves transformers' own
    recording hooks on the decoder layers.

## Decision

- **What is measured:** the output of every decoder layer L, counting from
  0. That is not the embeddings, not a layer's input and not the final
  norm's output. A forward hook on each decoder layer captures it. The
  layers are found as the block around each MLP
  (`pushlab.models.discovery.find_decoder_layers`), so Qwen3.5's
  vision-language wrapper makes no difference. Only the text path runs.
- **Each prompt runs alone,** so no padding has to be left out and every
  position is exact.
- **A token's length** is its L2 norm, taken in float32 from the
  activations in their own type (bfloat16 on the pod).
- **Each prompt's first token is left out.** In real models its vector is
  often far longer than the rest (the attention sink). Its own vector never
  counts; the later tokens, which read it, do.
- **The median** pools every remaining token of every prompt, per layer.
  With an even count it is the mean of the middle two. Mr-Neutr0n's
  `torch.median` takes the lower one; over hundreds of tokens the
  difference is negligible.
- **The prompts** are `prompts/calibration.jsonl`: 32 neutral everyday
  questions and requests, version 1. Each is rendered with the chat
  template and thinking off, exactly as `pushlab baseline` renders its
  prompt. The file's SHA-256 is recorded, so any edit to it shows.
- **No push may be active.** Calibration refuses a model with any forward
  hook other than transformers' recording hooks, and it removes its own
  hooks again, even after an error.
- **The file** is `calibration/<model>-<revision>.json`, in a folder of the
  repository that git ignores. It holds:
  - the sizes, one per layer;
  - the prompt set's file, version, SHA-256 and token count;
  - the measure's definition, the data type and the device;
  - the time, and the code, GPU and library versions, as a run's manifest
    records them.

  `load_calibration` refuses a file made for another model or revision, the
  wrong number of layers, and sizes that are not finite and above 0. A
  measured revision is measured again only with `--overwrite`.

## The measurement, checked against the replications

Qwen3.5-4B at commit `851bf6e` in bfloat16, on the CPU of a cloud session.
The pod's GPU will differ slightly.

| Layer 14 | Published | pushlab's code, on the replication's own text |
| --- | --- | --- |
| Mr-Neutr0n's median | 10.322 | 10.337 |
| occcat's median | 10.488 | 10.476 |
| Column length of N8949 | 0.46472 | 0.46472 |

With the replications' texts, pushlab reproduces both within 0.15%. The
text this decision chooses gives other numbers, because the median depends
on the text:

| Text | Layer 13 | Layer 14 | Dose +1 on L14 N8949 adds |
| --- | --- | --- | --- |
| occcat's passages, plain | 9.763 | 10.476 | +22.54 |
| The 32 prompts in the chat template (this decision) | 9.636 | 10.917 | +23.49 |
| The same 32 prompts, plain | 10.919 | 11.978 | +25.77 |

- At layer 14, dose 1 here is 4% more than in occcat's units and 6% more
  than in Mr-Neutr0n's. At layer 13 it is 1% less.
- Layers 17 to 29 are 2 to 15% lower than with occcat's passages.
- Half of the 698 measured tokens are the template's own: `user`, the line
  breaks, the assistant's header and the empty thinking block. They pull
  the medians down, and short questions push them up compared with prose.

## Why this shape

- **The chat template,** as the S1 task asks. pushlab's runs show the model
  its prompts in the template, so these sizes describe the stream that the
  pushes act on. The cost is the table above: doses here are not exactly
  the replications' doses.
- **Forward hooks rather than `output_hidden_states`,** whose last entry is
  the final norm's output.
- **A git-ignored folder in the repository.** The task names
  `calibration/`, and the pod writes the file in its clone of the
  repository. Ignored, the file never blocks a `git pull`, and it never
  counts as uncommitted code. The pod's values reach git by being pasted
  back, as the S0 manifest was.
- **The hash of the file's exact bytes,** like the experiment file's hash
  in the S0 manifest: `sha256sum prompts/calibration.jsonl` gives the same.

## Consequences

- Every dose in pushlab is in these units. Comparing a dose with another
  project's means comparing the calibrations: layer 14 is about 10.9 here
  and 10.3 to 10.5 in the replications.
- Changing the prompt set (a new version) needs a new calibration with
  `--overwrite`. `load_calibration` does not compare the file's hash with
  the current prompt set; code that runs experiments should.
- The CPU numbers above are a check, not the reference. Runs use the pod's
  calibration, measured on the GPU in bfloat16.
- A calibration made on a CPU is in float32 and says so in the file, but it
  would replace the GPU one if run with `--overwrite`.
