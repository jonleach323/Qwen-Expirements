This is Sprint S1, prompt 6 of 8.

CONTEXT
Read CLAUDE.md, the "4. engine" module in docs/plan/development-plan.md,
and docs/decisions/0004 and 0005.
- S0's greedy_generate (src/pushlab/engine/generate.py) decodes one
  prompt, with the model's cache, scoring only the newest position.
- Prompt 5's hooks push neurons and keep a step counter.

The plan's engine needs a custom loop for batches: "greedy or sampled,
left-padded batches, cache reused between steps", which "keeps the step
counter exact".

GOAL
Generate replies for a batch of prompts at once, with pushes applied, so
that each row gives what it would give alone.

BUILD
- A batched greedy loop in src/pushlab/engine/. The plan's interface is
  generate(model, prompts, dose_table, decoding). In S1 the pushes take the
  place of the dose table, which S2's schedules bring.
  - Prompts of different lengths are padded on the left, with an
    attention mask.
  - The cache is reused between steps, and only the newest position is
    scored, as in greedy_generate.
  - Each row stops after its own stop token, which is kept as its last
    token, or at max_new_tokens. A finished row doesn't change the others.
- Positions with left padding: each row's positions must start at its
  first real token.
  - Pass position IDs explicitly, in the form each model family expects.
    Qwen3.5's rotary embedding for text, images and video takes position
    IDs in three parts; read its transformers code.
  - Check how its linear-attention layers treat padding.
- Each forward pass first sets the hooks' step counter, then tells them
  which positions are padding: the prompt pass, then reply steps 0, 1, 2
  and so on, the same for every row however much padding it has.
- greedy_generate stays as it is. `pushlab baseline` keeps using it unless
  this loop, with a batch of one, reproduces the S0 reference reply on the
  GPU token for token. Prompt 7's pod session checks that.

CONSTRAINTS
- Greedy only. Sampling, with seeds, comes in a later sprint.
- No schedules or dose tables (S2), and no re-scoring of text (score_text
  comes later).
- Files stay under about 400 lines. No dependency changes.

TESTS FIRST
All on tiny_model and tiny_vl_model, in float32:
- A batch of one gives exactly greedy_generate's tokens.
- Each row of a left-padded batch, with prompts of different lengths,
  gives exactly the tokens of that prompt run alone.
- Rows stop independently: each ends at its first stop token, or at
  max_new_tokens.
- A test hook sees the step counter go "prompt pass, 0, 1, …" for every
  row, whatever its padding.
- A zero-dose push gives tokens identical to no push, for the whole batch.
  A pushed row leaves an unpushed row in the same batch exactly as it is
  without pushes.
- An empty batch, an empty prompt or max_new_tokens below 1 is refused.

DONE WHEN
`uv run pytest -q` passes on CPU, and the pull request's checks pass. The
GPU comparison with the S0 reference reply runs in prompt 7's pod session.

REPORT BACK
End with a plain-language summary for a beginner: what changed, why, how I
can check it myself, and anything I must do by hand. Explain left padding
and why positions must not shift.
