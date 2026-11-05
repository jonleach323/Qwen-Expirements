# 0009: One loop generates every reply, padded on the left as Hugging Face pads

Date: 2026-11-05 · Sprint: S1 · Status: accepted

## Context

- The plan's engine has "a custom decoding loop (greedy or sampled,
  left-padded batches, cache reused between steps)" that "keeps the step
  counter exact". It is done when "a dose of zero gives token-identical
  output to no hooks; each batch row matches the same run done alone; and
  the step counter stays correct with left padding" (development plan,
  "4. engine").
- S0's `greedy_generate` (decision 0004) decoded one prompt. Pushes
  (decision 0008) need a loop that says which pass is running.
- Qwen3.5-4B mixes two kinds of memory. Its 24 linear-attention layers keep
  a running state, and its 8 full-attention layers keep every earlier
  token's keys and values.
- The S1 task asks that greedy replies equal Hugging Face's `generate()`
  token for token on the tiny model when nothing is pushed. In
  transformers 5.21, read in its code:
  - `generate()` gives each row positions from its attention mask: the
    count of real tokens so far, minus one, and 0 on padding. Each step
    adds one to the last position and a 1 to the mask. A mask of only
    ones is dropped. Only the newest position is scored.
  - Qwen3.5's linear-attention layers multiply padded positions by 0
    before they mix anything (`apply_mask_to_padding_states`), so left
    padding never enters their running state. A step that reads one token
    needs no mask there.
  - Qwen3.5 keeps rope offsets from its last `generate()` call on the model
    (`rope_deltas`), and uses them when it is called without positions.

## Decision

- **`generate(model, tokenizer, prompts, pushes, decoding, norms)`**
  (`pushlab.engine.generate`) returns one `Generation` per prompt, in the
  prompts' order: the text, the token IDs, the stop reason, and how many
  reply tokens each push touched. `norms`, the calibration's sizes, are
  needed only with pushes. In S1 every prompt gets the same pushes.
- **`Decoding`** (`pushlab.engine.decoding`) holds the strategy, greedy or
  sample, and `max_new_tokens`. Sampling needs a temperature above 0 and a
  seed; `top_p` is 1, every token, unless given. Greedy decoding refuses a
  temperature, a seed and a `top_p` below 1.
- **Passes:** each prompt is rendered in the chat template with thinking
  off (`chat_prompt_ids`). One prompt pass reads the whole batch; then the
  pass that reads reply token k runs for k = 0, 1, 2 and so on, reusing the
  model's cache. Every pass scores only the newest position. Before each
  pass the loop sets the StepState: `at_prompt` with the batch's padding,
  then `at_reply_token(k)`.
- **Left padding, as in `generate()`:** shorter prompts get the pad token
  on the left and 0 in the attention mask there. Positions are the count
  of real tokens so far minus one, 0 on padding, and one more at each
  step, while the mask grows by a 1. A batch with no padding gets no mask.
  Positions are always given, so Qwen3.5's kept rope offsets never apply.
- **Stopping:** a row ends after a stop token, which it keeps:
  `<|im_end|>`, the end of a chat turn, or the model's own end token
  (`stop_token_ids`). Its stop reason is `end_of_turn`; a row that reaches
  `max_new_tokens` first gets `max_new_tokens`. A finished row stays in the
  batch until every row has finished: it reads the pad token, and the
  StepState marks it as padding, so no push touches it.
- **Touched tokens:** for each row and push, the reply tokens whose pass
  the push was active on while the row was still running. No pass reads a
  reply's last token, so no push touches it; the prompt pass is no reply
  token. A push from 2 to 4 on a reply of 8 tokens touches 2.
- **Greedy** takes the highest score in float32, as `generate()` does.
- **Sampling** divides the scores by the temperature and turns them into
  probabilities in float32. With `top_p` below 1, a token stays only if
  the tokens likelier than it hold less than `top_p`, so the likeliest
  always stays. Each row draws with a random generator of its own, seeded
  with the seed, on the model's device. torch's global random state is
  never used.
- **`group_by_length=True`** runs only prompts of the same token length
  together, in the order they first come, so nothing is padded.
- **`pushlab baseline`** now uses this loop, greedy and with no pushes.
  `greedy_generate` is gone.

## The loop checked

On the tiny models, in float32 on the CPU, for Qwen3 and Qwen3.5:

- Greedy replies equal `generate()`'s, alone and in a left-padded batch.
- Each row of a padded batch equals its prompt alone: greedy and sampled,
  with and without a push.
- Each pass gets the inputs above: the left-padded prompts, the mask and
  each row's positions, then the next token at the next position.
- A puppet head that makes the model choose scripted tokens checks the
  stops and that a finished row is never pushed. A probe on `down_proj`
  checks which passes each push lands on.

On the real Qwen3.5-4B at commit `851bf6e`, in bfloat16 on a cloud
session's CPU:

- **The S0 prompt alone** gives 90 tokens, the same as `greedy_generate`
  gave, with bit-identical scores on every pass. The reply's SHA-256 is the
  S0 reference's (`367b721f…`, decision 0005), on the CPU too.
- **Batched against alone**, as the GPU test does it: the calibration
  set's shortest, middle and longest prompts (19, 23 and 27 tokens), 32
  reply tokens each.
  - The first and last replies are identical.
  - The middle one parts at its first word: "Staying" alone, "Working" in
    the batch. Alone, those two scored exactly the same (19.625), so
    rounding decided.
  - Scores differ by at most 0.28.
- **Deliberate mistakes in the batch**, against the correct replies alone:

  | Mistake | Scores differ by, padded rows |
  | --- | --- |
  | No attention mask | 3.9 and 4.5 |
  | Linear-attention layers not zeroing padding | 3.6 and 5.1 |
  | Every reply token one position too early | 0.22 |
  | Positions counting the padding | 0.25 |

  So the GPU test allows 1.0, well above rounding and well below what
  letting padding in does. A position off by one hides in the rounding,
  and positions that count the padding change nothing in effect: rotary
  positions only see distances between tokens, and those stay the same.
  The unit tests check the positions themselves.

## Why this shape

- **Padding exactly as `generate()` does it.** It is the reference the S1
  task names, and the tests can then hold the loop to it token for token.
  Copying its positions also gives a padded row the same positions as its
  prompt alone, not just the same distances between tokens.
- **Positions always given.** Without them, Qwen3.5 would rebuild
  positions from offsets left on the model by an earlier `generate()`
  call, which can be another batch's.
- **Finished rows stay in the batch.** The batch keeps its shape to the
  end, as in `generate()`, so a row's numbers never depend on when the
  others finished.
- **A random generator per row.** A sampled reply then depends only on its
  prompt and its seed, not on the batch around it. S2's runs each carry a
  seed, and each row's generator can take its own.
- **Scores, not tokens, compared on the GPU.** In bfloat16 a batch changes
  the shape of the arithmetic, so its rounding differs from a prompt
  alone. Two replies can then part where two tokens score nearly the same.
  The GPU test allows that, but only within a bound on the scores that
  letting padding in would break.

## Consequences

- The pod's step 7 runs `tests/gpu/test_generate_smoke.py` too, so it ends
  with `6 passed`. It loads the model a second time, so the step takes
  longer than the 2m 25s of decision 0005.
- On the GPU, a batched reply can, rarely, part from the same prompt run
  alone. Runs that must match exactly, such as dose 0 against no hooks,
  must run in the same batch layout: the same prompts, in the same order
  and padding.
- The baseline's runs keep their fields. Run it once on the pod into a new
  folder: its reply should equal the S0 reference reply, whose SHA-256 is
  in decision 0005.
- A sampled reply repeats with the same seed on the same kind of device
  and library versions. A CPU and a GPU draw differently.
- S2's per-row schedules fill the hook's per-row amounts (decision 0008),
  and per-row seeds seed each row's generator.
- A batch keeps every row's cache until its longest reply ends, so S2's
  runner must choose batch sizes for the GPU's memory.
