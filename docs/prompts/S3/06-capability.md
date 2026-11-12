CONTEXT
This is Sprint S3, prompt 6 of 11. Read CLAUDE.md; "Prompt sets" and
"Measures" in docs/plan/research-plan.md; src/pushlab/prompts.py;
experiments/README.md; and prompt 5's
src/pushlab/scoring/score_command.py and docs/runbooks/s3-pod-c.md.
- There is no capability set. A prompt set's lines hold exactly
  prompt_id, version and prompt, so the answers need a file of their own.
- Mr-Neutr0n's 32 questions (no licence found: do not copy them), at
  github.com/Mr-Neutr0n/qwen3.5-4b-neuron-steering, checked "both the
  answer and requested output form": 56.3% for Qwen3.5-4B unpushed, 0%
  with the evil pair at full dose. So report the form beside the answer.
- No owner decision. Part 2 also runs qwen3-8b, which prompt 1 pinned
  and pod session A downloaded. If part 1's evidence has come, record it
  first in prompt 5's note, as docs/handoff/HANDOFF.md (prompt 1) says,
  in a commit of its own.

GOAL
A fixed set of 100 short questions with exact answers, a grader that
needs no model, and each model's unpushed accuracy on it.

BUILD
- prompts/capability100.jsonl, version 1, IDs capability100-001 to
  capability100-100: 40 arithmetic (whole-number answers: sums,
  differences, products, exact quotients, a percentage, two steps), 30
  facts (one word or a name: chemical symbols, capitals, planets, units,
  authors of classic books) and 30 simple reasoning (orderings, days of
  the week, counting, short word problems). Each ends with its form:
  "Answer with the number only.", "Answer with one word only." or
  "Answer with the name only."
- You write all 100 in this session, new, from no benchmark, website or
  repository. Each answer must be the only right one: no fact that
  changes (officeholders, records, populations), is disputed or differs
  by country or convention; no trick; nothing harmful (the research
  plan's "Excluded") or about the model; no prompt of another set.
- prompts/capability100-answers.jsonl, a line per prompt_id: version,
  kind (arithmetic, fact or reasoning), type (number or text), answer,
  also (a few other accepted spellings), units (the unit's spellings, if
  any) and check (arithmetic: the expression; else why it is right).
- How the answers are checked:
  - a test evaluates each arithmetic check with a small hand-written
    evaluator over fractions (numbers, + - * / and brackets; never
    eval) and compares it with the answer;
  - you check each fact and reasoning answer twice, the second time from
    the question alone; the PR description lists every doubt first, then
    all 60 with their checks, for the owner to read before merging;
  - when part 2's evidence is back, the session that records it rereads
    each question both models miss: a wrong or ambiguous key becomes
    version 2 of both files, with a note; a hard question stays.
- src/pushlab/scoring/capability.py, CAPABILITY_VERSION = 1, no torch:
  - read_answers(path, prompt_set): one answer per prompt, the set's
    version, known kinds and types, or a refusal naming the line;
  - normalise: NFKC, lower case, spaces trimmed and collapsed, quotes,
    Markdown bold and a final full stop dropped; a number loses thousands
    commas and its unit, reads "−" as minus and compares by value ("12.0"
    is 12); a text drops a leading "the", "a" or "an";
  - extract: the text after the last "answer:" or "answer is", else the
    first non-empty line; a number answer takes its last number or, if
    that line has none, the reply's only number;
  - grade: correct when that equals an accepted answer (a text answer
    may also begin or end with it, as whole words); strict when the
    reply, trimmed and without one final full stop, is the answer or an
    also spelling, case aside; label correct, wrong or no_answer.
- `pushlab score SPEC --scorer capability`, in prompt 5's table: text
  only, no model or GPU; it refuses a set without <set>-answers.jsonl
  beside it. value 1 or 0; details the rule, strict, kind and the
  answers file's SHA-256, and a run counts as scored only for the
  current one; per condition "<k> of <n> correct (<accuracy>, 95%
  Wilson <low> to <high>); strict <s>; no answer <z>; cut off <c>;
  arithmetic a/40, fact f/30, reasoning r/30", by wilson_interval in
  prompt 4's src/pushlab/scoring/agreement.py; <c> counts the replies
  cut off at max_new_tokens, as pushlab report does (reply_checks.py:
  stops from the model's folder, no weights).
- experiments/s3/capability_baseline.yaml (s3-capability-baseline,
  qwen3.5-4b) and experiments/s3/capability_baseline_qwen3_8b.yaml
  (s3-capability-baseline-qwen3-8b, qwen3-8b): one condition, baseline,
  `segments: []`, the 100 prompts, greedy, seeds [0], max_new_tokens
  128 (an answer alone is a few tokens; 128 leaves room for one worked
  out first), batch_size 16, max_gpu_hours 0.25; both listed in
  experiments/README.md.
- Docs: CLAUDE.md's command; docs/runbooks/s3-pod-c.md, "Part 2:
  capability"; a decision note, the next free number: the set, who wrote
  and checked it, the grader's rules and version 1, and an empty Results
  section for both accuracies.

CONSTRAINTS
- Never start a GPU job: write both files and their --dry-run output.
- No dependency, no eval(). Questions and answers live in prompts/,
  never in code; tests pin the files' hashes, not the answers.
- Files under about 400 lines. Review as CLAUDE.md says.

TESTS FIRST (CPU, offline)
- The set reads as 100 prompts in ID order, 40, 30 and 30 by kind, each
  ending with one of the three forms and with one answer; both files'
  SHA-256s are pinned; no prompt is in another set of prompts/.
- Each arithmetic check evaluates to its answer; the evaluator refuses
  names, calls and attributes.
- grade on fixed replies: "391", "391." and "Answer: 391" with an
  explanation after it are correct, the first two strict; "**391**" and
  "17 × 23 = 391" correct, not strict; "1,234" is 1234, "−5" is -5,
  "12 cm" 12 when the unit is cm; "Canberra." strict; "The capital is
  Canberra." correct; "Not Canberra but Sydney" wrong; an empty reply,
  or no number for a number answer, no_answer.
- pushlab score --scorer capability on a tiny_spec folder whose prompts
  are three questions with answers beside them: a row per run while
  load_model raises; the summary line, its cut-off count right; a set
  without answers refused.
- Both experiment files expand to 100 runs, no controls, all valid.

DONE WHEN
- `uv run pushlab run experiments/s3/capability_baseline.yaml
  --dry-run`, and the same for the qwen3-8b file, say 100 of 100 runs
  within max_gpu_hours 0.25 (about 0.08 and 0.16 GPU-hours at the costs
  file's 44 and 22 reply tokens a second); both outputs are in the PR
  description.
- `uv run pytest -q` and `uv run pre-commit run --all-files` pass; say
  how long pytest took against the plan's two minutes.
- One pull request, opened by the session from a new branch off main,
  titled "S3 prompt 6: capability set and exact match", whose
  description says what changed, lists any changed expected value of an
  existing test and why (CLAUDE.md), and lists the follow-ups. The owner
  merges it after CI.

REPORT BACK
- Part 2's pod steps (docs/runbooks/s3-pod-c.md), after part 1, in the
  same visit or a later one:
  1. `cd /workspace/Qwen-Expirements && git pull && bash
     scripts/setup_pod.sh`, then `source /workspace/pushlab.env`.
  2. For FILE experiments/s3/capability_baseline.yaml, then
     experiments/s3/capability_baseline_qwen3_8b.yaml: `uv run pushlab
     run FILE --dry-run`; in tmux, `uv run pushlab run FILE` (minutes);
     then `uv run pushlab verify FILE`, which must find no problem.
  3. For each: `uv run pushlab score FILE --scorer capability` (seconds,
     no GPU), then `uv run pushlab evidence FILE`.
  4. Copy both folders off ("Copy runs off the pod" in
     docs/runbooks/experiments.md).
- Paste back both evidence files the same day (HANDOFF.md): each scores
  section holds its model's accuracy line.
- The PR's link, its changed expected values, its follow-ups and the
  doubts list.
- End with a plain-language summary for a beginner: what changed, why,
  how I can check it myself, and anything I must do by hand. Say what
  exact match and the strict form check each measure, and why a wrong
  answer key does more harm than a hard question.
