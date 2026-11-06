# 0007: Doses are scaled by each layer's median residual size, measured on 32 chat prompts

Date: 2026-11-04 · Sprint: S1 · Status: accepted

## Context

- The dose rule (decision 0006) is c = α · r̃_L / ‖W_down[:, n]‖. Here r̃_L,
  the median size of the residual stream leaving layer L, is measured once
  per model, and it sets the unit of every dose.
- The evil-Qwen recipe defines it as "median(‖resid‖ leaving layer L,
  tokens 1:)". Its author, lyra, has since published these sizes for
  Qwen3.5-4B in `norms/Qwen__Qwen3.5-4B.json` of
  github.com/lyramakesmusic/neuronary. Her `scripts/norms.py` says her tools
  read that table "so a dose means the same thing everywhere". She measured
  them in bfloat16 on her neuron explorer's reference text: about 120
  tokens of prose and code, with no chat template.
- The two replications with sizes in bfloat16 measured on plain text too:
  Mr-Neutr0n on the same reference text, and occcat on 4 prose passages
  (578 tokens). hedwigai, in 8-bit, measured six plain sentences plus three
  of them in the chat template with thinking off, and got 11.02 at layer
  14. Every one of these leaves out each text's first token.
- The Neuron Persona Library (Qwen3-8B) measures the way this decision does,
  "over the prompt's tokens (first token excluded)", with the prompt "Who
  are you?" "in the chat template, thinking off" (neuronpersonas.com,
  "Method"). It uses that one prompt and publishes no sizes.
- So the two references measure on different text. The S1 task asks for 32
  short everyday prompts in the chat template with thinking off. That is
  the library's procedure on more prompts, and it is the way pushlab's runs
  show the model its prompts.
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
- **A prompt set's format:** a JSON Lines file in `prompts/`, read by
  `pushlab.prompts.read_prompt_set`. Each line holds a `prompt_id`, the
  set's `version` and the `prompt`. Later prompt sets can use the same
  format.
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

## The measurement, checked against the published sizes

Qwen3.5-4B at commit `851bf6e` in bfloat16, on the CPU of a cloud session.
The pod's GPU will differ slightly.

| Layer 14 | Published | pushlab's code, on the same text |
| --- | --- | --- |
| lyra's table | 10.353 | 10.337 |
| Mr-Neutr0n's median | 10.322 | 10.337 |
| occcat's median | 10.488 | 10.476 |
| Column length of N8949 | 0.46472 | 0.46472 |

- On lyra's reference text, pushlab's code matches her table within 0.4%
  at every one of the 32 layers.
- It matches both replications within 0.15% at layer 14.

The text this decision chooses gives other numbers, because the median
depends on the text:

| Text | Layer 13 | Layer 14 | Dose +1 on L14 N8949 adds |
| --- | --- | --- | --- |
| lyra's reference text, plain | 9.745 | 10.337 | +22.24 |
| occcat's passages, plain | 9.763 | 10.476 | +22.54 |
| The 32 prompts in the chat template (this decision) | 9.636 | 10.917 | +23.49 |
| The same 32 prompts, plain | 10.919 | 11.978 | +25.77 |

- Against lyra's table, 22 of the 32 layers are within 5%.
  - Layer 14 is 5.4% higher, so dose 1 there pushes 5% harder than in the
    recipe. Layer 13 is 1.1% lower.
  - Layers 0 to 10 are 3 to 12% higher, layer 26 is 6% higher, and layers
    27 to 31 are 4 to 8% lower.
- Half of the 698 measured tokens are the template's own: `user`, the line
  breaks, the assistant's header and the empty thinking block. They pull
  the medians down, and short questions push them up compared with prose.

## Why this shape

- **The chat template,** as the S1 task asks. It is the Persona Library's
  procedure. pushlab's runs also show the model its prompts in the
  template, so these sizes describe the stream that the pushes act on. The
  cost is the second table: doses here are not exactly the recipe's.
  Measuring on lyra's reference text instead would match her table within
  0.4%.
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
  and 10.35 in lyra's table.
- This narrows the last consequence of decision 0006. `load_calibration`
  refuses a file made for another model or model commit. With one file per
  model revision, the data type and the prompt set's SHA-256 are only
  recorded: the code that loads sizes for a run must compare them with its
  own. A new prompt set needs a new calibration with `--overwrite`.
- The CPU numbers above are a check, not the reference. Runs use the pod's
  calibration, measured on the GPU in bfloat16.
- A calibration made on a CPU is in float32 and says so in the file, but it
  would replace the GPU one if run with `--overwrite`.
