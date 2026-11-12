CONTEXT
This is Sprint S3, prompt 3 of 11. Read CLAUDE.md; "6. store", "7.
scoring", "Data model" (sentences and scores) and S4 in "Sprint details"
in docs/plan/development-plan.md; "Measures" in
docs/plan/research-plan.md; docs/decisions/0016-steps-and-reports.md;
src/pushlab/store/; src/pushlab/runs/folder.py and shards.py; and
src/pushlab/analysis/markers.py.
- Nothing stores a score. The plan's scores table, "one row per run and
  scorer", has no span, so it cannot hold a verdict on part of a reply.
- Step j is reply token j (store/steps.py); markers.places() puts step
  j where the text and decode(tokens[:j]) stop agreeing.
- Prompts 5 to 7 write scores here; the judge (prompt 7) and S4's
  persona timelines score stretches of a reply, defined here once.
- No owner decision; CPU only; no pod step. If the owner pastes pod
  session A's results here, record them first as
  docs/handoff/HANDOFF.md (prompt 1) says, in a commit of their own.

GOAL
One table for every score, of a whole reply or of a window of its
sentences, and one window that the judge and S4's timelines share.

BUILD
- src/pushlab/store/scores.py: ScoreRecord, SCORES_SCHEMA,
  write_scores(folder, records), read_scores(folder). A row: run_id;
  scorer (lowercase letters and digits joined by hyphens);
  scorer_version (a whole number from 1); the span, never null:
  splitter_version, sentence_from, sentence_to, start_step, end_step
  (int32; "to" and "end" not included, as a segment's end), the whole
  reply being sentences 0 to its count and steps 0 to
  len(output_tokens); value (float64) and label (text), either null,
  not both; judge_model ("<registry name>@<commit>", or "" with no
  judge); details (JSON text of an object); text_sha256 (the scored
  text) and output_sha256 (the run's output_text when scored), of
  UTF-8; created_at (UTC).
- write_scores refuses, naming the problem: a malformed field, NaN or
  infinity; two records of one key (run, scorer, version, judge_model,
  splitter version, span); a run the folder's current version lacks or
  holds as not ok; a stale output_sha256; an end_step past the tokens.
  It holds folder.locked, never writing beside a runner. Each call
  writes one new scores/<scorer>-v<version>-<session>.parquet (session:
  UTC time and 8 random hex) by folder.write_whole, never replacing
  one; shards.remove_leftovers also sweeps scores/. read_scores refuses
  another schema, naming the differences, as read_steps_file does.
- src/pushlab/store/query.py: connect() gains a scores view,
  SCORES_SCHEMA's columns plus experiment and version, over each runner
  folder's scores/, joined to the runs view where sha256(output_text)
  equals output_sha256 (DuckDB's sha256 is hashlib's hex for UTF-8),
  so a reply a new version changed loses its scores; of each key only
  the newest row shows. CLAUDE.md's query line names the view.
- src/pushlab/scoring/sentences.py: SPLITTER_VERSION = 1,
  Sentence(index, start, end, text), split_sentences(text). Spans tile
  the text; the whitespace between two sentences starts the second, as
  Qwen's tokens carry a word's space; a blank text has none. Version 1:
  a line break (markers.LINE_BREAK) ends a sentence; so do ".", "!",
  "?" and "…", with closing quotes and brackets, before a space and a
  character that is not a lowercase letter, but not after a list
  number at a line's start, an initial or a listed abbreviation ("Dr.",
  "e.g."); so do "。", "！" and "？", with or without a space.
- sentence_steps(text, tokens, decode): each sentence's (start_step,
  end_step). A token belongs to the sentence its text starts in: a
  sentence starts at the first step whose place, by places()'s rule,
  is at or past its first character. The spans tile 0 to len(tokens).
- src/pushlab/scoring/windows.py: Window(run_id, prompt_id,
  splitter_version, sentence_from, sentence_to, start_step, end_step,
  context, text). text is its sentences as the reply has them; context
  is the reply before them, never after (causal, as S4 reads a reply in
  order), so len(context) is its first character.
  - whole_reply(run): the widest, every sentence and token, context "".
  - sentence_windows(run, size, stride, decode): windows of size
    sentences starting every stride sentences (1 ≤ stride ≤ size, so
    none is skipped), cut at the reply's end, none after the first that
    reaches it. Size 1, stride 1 gives S4's sentences table, its
    sentence_idx being sentence_from.
- A decision note, the next free number: scores become "one row per
  run, scorer, version, judge and span", the whole reply the full span,
  not a sibling table (one table, one view, one rule for stale rows);
  the splitter's versions; ends not included; S4's onset, offset and
  half-life read these spans. docs/plan/ stays as it is.

CONSTRAINTS
- CPU only. Nothing here loads a model: decode is passed in, as
  places() takes it. No reply text in git or fixtures.
- A splitter change that moves any span needs SPLITTER_VERSION 2.
- No dependency. New modules; files under about 400 lines, split at
  about 350. Review as CLAUDE.md says. Expected values:
  tests/unit/test_query.py's VIEWS and TYPES gain the scores view.

TESTS FIRST (CPU, offline, tiny models)
- Scores, on a runner folder from tiny_spec (tests/unit/conftest.py):
  write and read give the records back unchanged; each refusal above;
  a write while a runner holds the lock; a crash leaves no file.
- The view: after a new version changes one of two scored replies, its
  score goes and the other's stays; a key two sessions wrote shows
  once, the newer; S1's kind of folder adds no rows.
- The splitter's fixed examples, with exact spans that join back to
  the text: "Three ideas:\n1. Walk.\n2. Read a book.\n- Call a friend"
  (4, the last three starting "\n"); "It costs 3.5 dollars. Pay now."
  and "Ask Dr. Lee, e.g. at 5 p.m. today. She knows." (2 each);
  "Line one\nLine two\n\nLine three." and
  "你好。我是助手！你呢？" (3 each); 'He said "Stop!" Then he left.
  Wow!! Really?' (4); "J. R. R. Tolkien wrote it." and the cut-off
  "The sea is calm and the" (1 each); "" and "  " (none). A test pins
  SPLITTER_VERSION 1 to a SHA-256 of all their spans.
- sentence_steps: the tiny tokenizer (a token per byte) gives
  "你好。我" (0, 9) and (9, 12); a hand-made decode with the tokens
  "Walk", ".\n", "2", "." and " Read" gives "Walk.\n2. Read" (0, 2) and
  (2, 5).
- Windows: 4 sentences at size 2, stride 1 give [0, 2), [1, 3), [2, 4);
  5 at size 2, stride 2 give [0, 2), [2, 4), [4, 5); stride 0 or above
  size is refused; context + text is the reply up to the window's end;
  whole_reply of a tiny-model run spans all its tokens, its text the
  run's output_text.

DONE WHEN
- split_sentences("Hi. Bye.") gives "Hi." and " Bye.", and connect()
  on an empty runs dir has a scores view with its columns and no rows.
- `uv run pytest -q` and `uv run pre-commit run --all-files` pass; say
  how long pytest took against the plan's two minutes.
- One pull request, opened by the session from a new branch off main,
  titled "S3 prompt 3: scores and sentence windows", whose description
  says what changed, lists any changed expected value of an existing
  test and why (CLAUDE.md), and lists the follow-ups. The owner merges
  it after CI.

REPORT BACK
- No pod step. The PR's link, its changed expected values and its
  follow-ups.
- End with a plain-language summary for a beginner: what changed, why,
  how I can check it myself, and anything I must do by hand. Say what a
  sentence window is, and why its context holds only what comes before.
