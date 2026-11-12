CONTEXT
This is Sprint S3, prompt 5 of 11. Read CLAUDE.md; "4. engine",
"7. scoring" and the scores table in docs/plan/development-plan.md;
"Measurement pitfalls to design around" in docs/plan/research-plan.md;
tests/unit/test_generate_logprobs.py; and prompt 3's
src/pushlab/store/scores.py and src/pushlab/scoring/windows.py.
- The steps table keeps each token's log-probability from the pushed
  pass that chose it, as a 16-bit float (src/pushlab/store/steps.py).
  Perplexity under the unsteered model needs a teacher-forced pass; only
  a test makes one.
- Liu (2026, arxiv.org/abs/2606.19831) calls text collapsed below a
  distinct-bigram ratio of 0.30 (Figure 10), and for unspaced scripts
  such as Chinese gates on character diversity, the share of distinct
  characters (Appendix E). Check both there if you can reach it.
- No owner decision. D5, before prompt 9, may change the collapse rule.
- The pod step needs pod session A's results (calibrations, a golden
  pass) on main. Pasted but in no PR: record them first, as
  docs/handoff/HANDOFF.md (prompt 1) says, in a commit of their own.
  Not pasted: build all of this, ask for them, and part 1 waits.

GOAL
Score how readable every reply is, perplexity under the unsteered model
included, and show that the teacher-forced pass gives generation's own
log-probabilities.

BUILD
- src/pushlab/engine/score_text.py: score_text(model, tokenizer,
  prompts, replies, *, schedules=None, batch_size=8) gives each reply
  token's log-probability under the unsteered model, in float32. Prompts
  render as generate() renders them; a reply is its token IDs, end token
  included, or a text (no special tokens). Left-padded batches take
  generate()'s attention mask and position IDs, keeping only the reply
  positions' logits. A schedule with a segment is refused, naming
  decision 0008: S4's pushed passes need a reply token per position.
- src/pushlab/scoring/coherence.py, COHERENCE_VERSION = 1, on a reply's
  units: lowercase words (runs of letters or digits) or, when most of
  its letters are Han, Hiragana or Katakana, its characters:
  - distinct_bigram_ratio: distinct unit bigrams over all bigrams; over
    characters, distinct characters over all (character diversity);
  - repeated_ngram_rate: the share of unit 4-grams seen before;
  - word_variety: distinct units per window of 128, averaged over the
    windows, so that long fluent text does not read as collapsed;
  - perplexity: exp of minus the mean of score_text's log-probabilities;
  - the rule: "collapsed" below a ratio of 0.30, or with under 10 units
    in 32 tokens or more (symbol soup); "short" under 10 units; else
    "coherent".
- configs/scoring.yaml holds these numbers as coherence version 1, and
  the check's tolerance. The scorer refuses a version other than
  COHERENCE_VERSION, and a test pins version 1 to its numbers.
- `pushlab score SPEC --scorer coherence [--dry-run] [--batch-size N]
  [--runs-dir DIR] [--logs-dir DIR]`, in
  src/pushlab/scoring/score_command.py with a table of scorers that
  prompts 6 and 7 extend; its parser in a new src/pushlab/cli_scoring.py
  (prompt 4's commands are in cli_labels.py). It:
  - refuses a folder a runner holds (folder.lock_held; write_scores
    takes the lock itself) or that does not match the file (as verify
    does), and loads the model at the revision its experiment.json
    names, in the experiment's data type;
  - writes a row for each ok run not yet scored at this version and
    output_sha256, through prompt 3's whole_reply and write_scores:
    value the ratio, label the rule's, details the measures, tokens and
    model;
  - checks the pass on each run with an empty schedule, its forced
    log-probabilities against its steps'. Pass: the median gap at most
    0.02 nats, each run's mean gap within ±0.05 (bfloat16 and 16-bit
    steps rule out equality; a one-token shift or a wrong template gives
    whole nats). It prints the count, median, 99th percentile and
    largest gap; a FAIL writes no row and exits 1;
  - writes /workspace/logs/scores/<experiment>-<scorer>-<time>.json
    (commit, code_dirty, counts, the check, speed, summary lines; no
    reply text), and prints a line per condition, controls apart: runs,
    collapsed, short, median ratio and perplexity;
  - with --dry-run loads and writes nothing: runs, tokens, time and cost
    at configs/costs.yaml's new scored_tokens_per_second (assumed 2,000
    for the 4B models, 1,000 for qwen3-8b).
- `pushlab evidence` (prompt 1) gains a scores section: for each scorer
  and version, its rows, summary lines and newest record.
- Docs: CLAUDE.md's commands; docs/runbooks/s3-pod-c.md, "Part 1:
  coherence"; a decision note, the next free number (score_text, the
  measures, rule version 1, the check; "collapsed" is a score's label,
  never a run's status), with an empty Results section.

CONSTRAINTS
- Never start a GPU job. No dependency. No reply text in git, records
  or evidence. Thresholds live in configs/scoring.yaml, not in code.
- New modules; split one past about 350 lines. Review as CLAUDE.md says.
- engine/score_text.py changes prompt 2's engine hash: say so in the PR.

TESTS FIRST (tiny models, CPU, offline)
- score_text on generate()'s greedy and sampled replies to
  test_generate_logprobs.py's PROMPTS, on tiny_model and tiny_vl_model,
  in one left-padded batch, equals each Generation.logprobs within 1e-4,
  as that file's teacher-forced test does for one row; batch sizes 1 and
  3 agree within 1e-5; a text scores as its tokens do. It refuses a
  schedule with a segment, counts that differ and an empty reply.
- Fixed texts: a fresh sentence's ratio 1.0, "a b a b a b" 0.4, one
  sentence said five times under 0.30, "你好你好你好" one third; the
  prompts of calibration.jsonl and neutral32.jsonl joined keep their
  word variety near one window's as their plain ratio falls; [-1, -1]
  gives perplexity e; 0.30 is coherent, 9 units short, 40 tokens of
  "}{" collapsed.
- pushlab score on tiny_spec's folder (tests/unit/conftest.py): a row
  per run with the whole-reply span, in the scores view; the check
  passes on its 5 unpushed runs, and a stand-in score_text 0.1 nats off
  fails it, writing no row; a second call scores nothing; --dry-run
  loads no model; a folder in use is refused; evidence shows the scores
  and never a planted reply text.

DONE WHEN
- `pushlab score --help` works, and the tests above show score_text
  giving generate()'s log-probabilities on both tiny models.
- `uv run pytest -q` and `uv run pre-commit run --all-files` pass.
- One pull request, opened by the session from a new branch off main,
  titled "S3 prompt 5: coherence and the teacher-forced pass", whose
  description says what changed, lists any changed expected value of an
  existing test and why (CLAUDE.md), and lists the follow-ups. The owner
  merges it after CI.

REPORT BACK
- Part 1's pod steps, once this PR is merged and pod session A's results
  are on main (or with parts 2 to 4 after prompt 8):
  1. `cd /workspace/Qwen-Expirements && git pull && bash
     scripts/setup_pod.sh`: step 7 ends `7 passed`; golden runs again
     for the new engine hash and must give session A's digest.
  2. `source /workspace/pushlab.env && uv run pushlab score
     experiments/s2/evil_diagnosis.yaml --scorer coherence --dry-run`:
     256 runs. Then without --dry-run, a few minutes: the check line
     ends `pass`. On a FAIL, once more with `--batch-size 1`, then stop.
  3. `uv run pushlab evidence experiments/s2/evil_diagnosis.yaml`.
- Paste back that evidence file the same day (HANDOFF.md). Expect no
  collapsed reply in baseline and many in e-1's control, whose replies
  decision 0018 read as mostly loops.
- The PR's link, its changed expected values and its follow-ups.
- End with a plain-language summary for a beginner: what changed, why,
  how I can check it myself, and anything I must do by hand. Say what
  teacher forcing and perplexity are, and why only unpushed runs can
  check the pass.
