# 0018: One small experiment tells a setup problem from how the evil neuron behaves

Date: 2026-11-11 · Sprint: S2 (follow-up) · Status: accepted

## Context

- In S2's smoke test of dynamics (decision 0017), the evil neuron E
  (qwen3.5-4b:L14:N8949 at +1) changed many replies very little. The owner
  saw this in the reports; the run's results are not in the repository.
- The prompt for this step asked for one small experiment to tell whether
  that is a setup problem or how this neuron really behaves, and for a
  report that flags thinking-mode text and counts replies that hit
  max_new_tokens, with no engine change. It named decisions
  "0003-s1-engine-check" and "0004-s2-runner-check"; this repository's are
  0011 (S1's engine check) and 0015 (the runner), which this note uses.
- What was known:
  - S1 (decision 0011, docs/sprints/S1-review.md): E at +1 on the prompt
    pass and every reply token, for 200 tokens, turned 1 of 5 smoke replies
    hostile, 0 to 3 on any reading. Its push adds 4.0% more than occcat's.
    The zero-dose check passed on the GPU.
  - The smoke test gave replies 96 tokens. Only constant-e and the handoffs
    pushed the prompt pass; the pulses, onsets, alternations and ramp did
    not. No condition pushed E and W together, as the evil recipe does.
  - occcat's RESULTS.md (github.com/occcat/qwen35-evil-neuron, read
    2026-11-11, SHA-256 `05cff68c1a1238207ead3ded748367d4475f2a53157fa5c66dd109d066904568`):
    - the push applies "on prefill **and** on every decode step";
    - at dose 1 the "persona flips to hostile/manipulative", at 1.5 "text
      starts to break", and at 2 the output is degenerate, "`<think>}{`";
    - E and W together are stronger than E alone (perplexity 48.52 against
      36.20);
    - "**L16 N8927 +1 -> 12.5/100w, ppl 34.9** <- strongest evil neuron
      found", "**L16 N8927 is 3x hotter** at the same normalized dose", and
      "N8927 +1 ... / -1 stays polite (0.0)";
    - its samples were drawn at temperature 0.7, and its sweep scored 3
      prompts of 64 tokens by hostile words per 100.

## Decision

**The experiment,** `experiments/s2/evil_diagnosis.yaml` (experiment
`s2-evil-diagnosis`): qwen3.5-4b, the smoke test's 32 neutral prompts,
greedy, 200 reply tokens, batches of 8, max_gpu_hours 1.

| Label | What it pushes | What it asks |
| --- | --- | --- |
| baseline | nothing | the reference |
| e-1 | E at +1 on the prompt pass and every reply token | S1's s1-evil and the smoke test's constant-e, at 200 tokens |
| e-1, control 1 | L14 N1929, at +1 in the same way | whether e-1's changes are E's or any push's |
| e-1-25 | E at +1.25, the same way | the dose |
| e-1-5 | E at +1.5, the same way, above the ceiling of 1.4 | the limit, where occcat's text starts to break |
| e-1-reply-only | E at +1 on every reply token, not the prompt pass | how much comes through the prompt pass, which most smoke-test shapes left out |
| e-w-recipe | E at +1 and W (L13 N7956) at −1, the same way | the recipe itself, which the smoke test never ran |
| l16-n8927 | L16 N8927 at +1, the same way | a neuron occcat found 3 times more hostile |

- **L16 N8927's sign is +1.** occcat's RESULTS.md was reachable and gives
  +1 for its hostility and −1 as polite, so the file pushes +1 only.
- **e-1-5 sets `allow_over_ceiling: true`,** as the prompt asked: finding
  the limit is the point. No other condition goes above 1.4 in a layer.
- **The control** uses control seed 0, so it is L14 N1929, as in S1 and
  the smoke test.
- **The labels** hold no dots: e-1-25 is E at dose 1.25.
- **Each prompt's 8 runs fill one batch,** in the file's order, so every
  comparison within a prompt is in one batch layout (decision 0009).
- **The count and the cost.** 7 conditions and 1 control, times 32
  prompts: 256 runs, at most 51,200 reply tokens, in 32 batches of 8. At
  the costs file's 44 reply tokens a second, about 0.32 GPU-hours, $0.11
  to $0.24.
- **No engine change.** `constant` with `include_prompt: false` leaves out
  the prompt pass; the recipe is four plain segments, E and W each on the
  prompt pass and on the reply.

**The report addition** (`pushlab.analysis.reply_checks`, used by
`pushlab report SPEC`):
- **Thinking-mode text** is a `<think>` or `</think>` tag in a reply's
  text, or the tag's token among its tokens. With thinking off, the chat
  template ends the prompt with an empty thinking block, so a reply should
  hold neither. A flagged reply shows **(thinking-mode text)** before its
  text.
- **Cut off** means the reply holds max_new_tokens tokens and its last is
  not one that ends a reply: the tokenizer's end token, or one the
  model's generation config names, read from the downloaded folder without
  loading weights, as the generator stops at them. A reply that ended its
  turn on its last allowed token is not cut off.
  - The note after a reply changed with it: "reached the N-token limit",
    on any reply of that length, became "cut off at the N-token limit",
    only on replies cut off. Prompt 5's follow-up list had this.
- **The counts** sit at the top of every experiment report: for each
  condition, each control apart, how many replies were cut off and which
  prompts' replies have thinking-mode text.
- The `--runs` mode for S1's folders is unchanged.

## How to read it

- **Thinking-mode text in baseline or e-1** would be a setup problem: the
  template's thinking-off block is not holding. At 1.5 it may only be the
  model breaking down, as occcat saw at dose 2.
- **e-1 against its control and baseline.** If e-1's replies differ from
  baseline's no more than its control's do, E at +1 does little beyond
  any push, on these prompts. Then:
  - **e-1-25 or e-1-5 openly hostile:** the dose. Our dose 1 already pushes
    4% harder than occcat's, so a higher need would point at the decoding
    or the prompts, not the push.
  - **e-w-recipe hostile where e-1 is not:** the pair matters, and the
    smoke test, which pushed E alone, could not show it.
  - **l16-n8927 hostile where e-1 is not:** this is how E behaves: weaker
    than occcat's strongest neuron, as their sweep found (4.3 against 12.5
    hostile words per 100).
  - **nothing hostile anywhere, not even at 1.5 or with L16 N8927:** the
    push lands (S1 checked it), so the difference with occcat lies
    elsewhere: greedy decoding against their sampling at 0.7, everyday
    prompts against hedwigai's prompts about values, or how hostility is
    judged.
- **e-1-reply-only much weaker than e-1:** the prompt pass carries much of
  E's effect, so the smoke test's reply-only shapes started with less.
- **Cut off.** A reply cut off at 200 tokens is a partial view of where the
  reply was going; a condition that cuts off far more replies than
  baseline writes longer, as hostile rambling or breakdown can.

## Why this shape

- **One pass over the smoke test's prompts,** so the diagnosis answers
  about the run it explains, and 200 tokens, as S1 had, so a persona that
  shows late still shows.
- **Each candidate cause in one condition,** against the same baseline and
  the same control, in one batch per prompt: dose, prompt pass, pair,
  neuron.
- **The checks in the report,** not in a new command: the owner reads the
  report anyway, and its top now says at once whether a setup problem
  shows.

## Consequences

- The run is the owner's, on the pod (docs/runbooks/s2-evil-diagnosis.md).
- The smoke test's reports gain the counts when written again.
- One test expectation changed: `tests/unit/test_experiment_report.py`'s
  `LIMIT` became "cut off at the 16-token limit", and that test now
  ignores the flag on a reply in which tiny-vl wrote a thinking tag by
  chance.
- Left for later: sampled decoding, and a hostility score that does not
  depend on reading by eye.
