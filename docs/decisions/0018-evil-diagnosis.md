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
  0011 (S1's engine check) and 0015 (the runner). This note draws on 0011
  and 0017.
- What was known:
  - S1 (decision 0011, docs/sprints/S1-review.md): E at +1 on the prompt
    pass and every reply token, for 200 tokens, turned 1 of 5 smoke replies
    hostile, 0 to 3 on any reading. The hostile one answered a request for
    advice about a sad friend. Of E's four replies to everyday tasks, two
    were odd, not hostile (a recipe and an explanation), and two were close
    calls, dark or crude (a sea poem and cat names). neutral32 holds no
    prompts about feelings, so S1 predicts few openly hostile replies here,
    and odd, dark or crude ones. S1 measured E's push on the GPU: 4.0% more
    than occcat's. The zero-dose check passed on the GPU.
  - S1's random control, L14 N1929 at +1 the same way, made 4 of 5 replies
    loop, and one looping reply ended each repeat with a stray
    end-of-thinking tag, although thinking was off (0011). N1929 is this
    run's control too.
  - The smoke test gave replies 96 tokens. Only constant-e and the two
    E-first handoffs pushed E on the prompt pass; constant-w and the W-first
    handoffs pushed W there, and the pulses, onsets, alternations and ramp
    pushed reply tokens only. No condition pushed E and W at once, as the
    evil recipe does.
  - occcat's RESULTS.md (github.com/occcat/qwen35-evil-neuron, read
    2026-11-11, SHA-256 `05cff68c1a1238207ead3ded748367d4475f2a53157fa5c66dd109d066904568`):
    - the push applies "on prefill **and** on every decode step";
    - its dose table pushes E and W together: at dose 1 the "persona flips
      to hostile/manipulative", at 1.5 "text starts to break", and at dose
      2, on the 4-bit model, the output is degenerate, "`<think>}{`". It
      reports E alone at no dose but 1 (and −1);
    - E and W together disrupt the text more than E alone (perplexity
      48.52 against 36.20), though E alone gives "most of the effect by
      itself";
    - "**L16 N8927 +1 -> 12.5/100w, ppl 34.9** <- strongest evil neuron
      found", "**L16 N8927 is 3x hotter** at the same normalized dose", and
      "N8927 +1 ... / -1 stays polite (0.0)". That is a rate of hostile
      words per 100, on 3 prompts of 64 tokens, against E's 4.3. occcat
      gives no residual size for layer 16, so how our dose 1 there compares
      with theirs is not known;
    - its samples were drawn at temperature 0.7.

## Decision

**The experiment,** `experiments/s2/evil_diagnosis.yaml` (experiment
`s2-evil-diagnosis`): qwen3.5-4b, the smoke test's 32 neutral prompts,
greedy, 200 reply tokens, batches of 8, max_gpu_hours 1.

| Label | What it pushes | What it asks |
| --- | --- | --- |
| baseline | nothing | the reference |
| e-1 | E at +1 on the prompt pass and every reply token | S1's s1-evil and the smoke test's constant-e, at 200 tokens |
| e-1, control 1 | L14 N1929, at +1 in the same way | whether e-1's kind of change is E's or any push's |
| e-1-25 | E at +1.25, the same way | the dose |
| e-1-5 | E at +1.5, the same way, above the ceiling of 1.4 | the limit; occcat's text broke at 1.5 with E and W together |
| e-1-reply-only | E at +1 on every reply token, not the prompt pass | how much comes through the prompt pass, which most smoke-test shapes left out |
| e-w-recipe | E at +1 and W (L13 N7956) at −1, the same way | the recipe itself, which the smoke test never ran |
| l16-n8927 | L16 N8927 at +1, the same way | a neuron occcat found 3 times more hostile |

- **L16 N8927's sign is +1.** occcat's RESULTS.md was reachable and gives
  +1 for its hostility and −1 as polite, so the file pushes +1 only.
- **e-1-5 sets `allow_over_ceiling: true`,** as the prompt asked: finding
  the limit is the point. No other condition goes above 1.4 in a layer.
- **The control** is e-1's only, as the prompt asked. It uses control
  seed 0, so it is L14 N1929, as in S1 and the smoke test.
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
  template ends the prompt with an empty thinking block, so a reply without
  a push should hold neither. A flagged reply shows **(thinking-mode
  text)** before its text. Reasoning written without a tag is not found.
- **Cut off** means the reply holds max_new_tokens tokens and its last is
  not one that ends a reply: the tokenizer's end token, or one the model's
  generation config names. The report reads them from the downloaded
  folder without loading weights, as the generator stops at them. Qwen3.5-4B's
  folder has no `generation_config.json`, so the config comes from
  `config.json`, as transformers builds it for the model: `<|endoftext|>`,
  beside the tokenizer's `<|im_end|>`. A reply that ended its turn on its
  last allowed token is not cut off.
  - The note after a reply changed with it: "reached the N-token limit",
    on any reply of that length, became "cut off at the N-token limit",
    only on replies cut off. Prompt 5's follow-up list had this.
- **The counts** sit at the top of every experiment report: for each
  condition, each control apart, how many replies were cut off and which
  prompts' replies have thinking-mode text (with their seed, when there
  are several). A group whose runs all failed says so.
- The `--runs` mode for S1's folders is unchanged.

## How to read it

- **The push lands.** S1 measured its size on the GPU, and this run uses
  the same calibration file and amount rule. Setup's step 7, which the
  runbook asks to end `7 passed`, checks on the GPU that each row of a
  mixed batch gets only its own pushes. `pushlab verify` checks every
  step's pushes against its run's schedule.
- **Thinking-mode text.** Every run of a prompt has the same template, so
  only baseline tests it: a flag in baseline means thinking is not off, a
  setup problem. A flag in a pushed condition, with baseline clean, means
  the push is breaking the reply's format, as S1's control did at dose 1.
  Expect loops in e-1's control, and perhaps a stray tag.
- **e-1 against its control and baseline.** Compare what kind of change
  each makes, hostile content against loops, repetition or breakdown, not
  how much: N1929 changed S1's replies a lot without any persona.
  - e-1 hostile where its control is not: E's own effect, at 200 tokens.
    If e-1 turns hostile only after its first 96 tokens, the smoke test's
    96 tokens may have hidden it; but its replies need not follow
    constant-e's that far, as a batch of 8 rounds differently from one of
    16 and replies can part at a near-tie (decision 0009).
  - e-1 changed but not hostile, odd or dark rather: E acts, but not as a
    persona at this dose on these prompts.
  - e-1 much like baseline: E at +1 does little on these prompts, the
    smoke test's symptom.
- **Then, whatever e-1 does:**
  - **e-1-25 or e-1-5 openly hostile:** the dose. Our dose 1 already
    pushes 4% harder than occcat's, so a higher need would point at the
    decoding or the prompts, not the push.
  - **e-w-recipe hostile where e-1 is not:** the pair matters, and the
    smoke test, which never pushed E with W, could not show it.
  - **l16-n8927 hostile where e-1 is not:** this is how E behaves: weaker
    than occcat's strongest neuron, as their sweep found.
  - **nothing hostile anywhere, not even at 1.5 or with L16 N8927:** the
    push lands, so the difference with occcat lies elsewhere: greedy
    decoding against their sampling at 0.7, everyday prompts against
    prompts about values like hedwigai's, or how hostility is judged.
    Telling these apart needs another experiment file, such as these
    conditions on prompts about values, or sampled: the owner's call.
- **Only e-1 has a control.** Hostility at 1.25, at 1.5, with the pair or
  with L16 N8927 cannot be told, by this run alone, from what any push as
  strong would do.
- **e-1-reply-only much weaker than e-1:** the prompt pass carries much of
  E's effect, so the smoke test's reply-only shapes started with less.
- **Cut off.** Expect many replies cut off in every condition: the prompts
  invite long replies, and S1's baseline reached 200 tokens on 4 of 5.
  Loops reach the limit too, so read a condition's count together with
  what its replies say.

## Why this shape

- **One pass over the smoke test's prompts,** so the diagnosis answers
  about the run it explains, and 200 tokens, as S1 had, so a persona that
  shows late still shows.
- **Each candidate cause in one condition,** against the same baseline, in
  one batch per prompt: dose, prompt pass, pair, neuron.
- **The checks in the report,** not in a new command: the owner reads the
  report anyway, and baseline's line at its top shows at once whether the
  template's thinking-off block holds, as far as tags can show it.

## Consequences

- The run is the owner's, on the pod (docs/runbooks/s2-evil-diagnosis.md).
  PR 17 put the experiment file on main, so the run can start from there;
  its spec did not change since. The reports and their reading need the
  review's fixes, which came after PR 17: a report written before they
  reached main should be written again, with --overwrite.
- The smoke test's reports gain the counts when written again.
- One test expectation changed: `tests/unit/test_experiment_report.py`'s
  `LIMIT` became "cut off at the 16-token limit", and that test now drops
  the flag before comparing a reply's text: tiny-vl writes a thinking tag
  in 3 of its 10 replies by chance.
- Left for later: sampled decoding, controls at the higher doses and for
  the pair, and a hostility score that does not depend on reading by eye.
