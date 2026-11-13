# 0021: One judge, blind to conditions, two backends, verdicts in two tables, checked against the owner's labels

Date: 2026-11-13 · Sprint: S3 · Status: accepted

## Context

- The prompt for this step ("Sprint S3, prompt 3 of 8", pasted) asks for
  `src/pushlab/scoring/judge.py`, one Judge interface with two backends:
  `api`, Claude Haiku (`claude-haiku-5-5`) through the Anthropic API with
  `ANTHROPIC_API_KEY` from the environment, and `local`, an open model
  served by vLLM, of another family than the model under test. Three
  rubrics answer in structured JSON: `speaks_as` labels a text as
  speaks_as, about_theme, other_persona, assistant or incoherent for a
  given persona; `which_persona` picks one of a list of personas or
  "none"; `coherence` scores it from 0 to 10. Each works on a whole reply
  or a window of sentences, and verdicts are cached by backend, model,
  rubric version and text. `pushlab score <experiment> --rubric <name>
  [--windows sentences]` writes the scores table and, with windows, a
  sentences table; `--dry-run` prints the calls and their cost from
  `configs/costs.yaml`. `pushlab label export` writes a CSV of sampled
  replies for the owner to label, and `pushlab label check` reports each
  rubric's accuracy and Cohen's kappa against those labels. The judge
  never sees which condition produced a reply; failed calls are retried
  with backoff and recorded as failures, never guessed. A lesson from
  other replications: count only replies that speak AS a persona, not
  those that only talk ABOUT its theme.
- The development plan ("7. scoring") wants one judge interface for every
  rubric, served locally through vLLM, an optional API adapter, every
  verdict cached by judge, rubric version and text, and a validation
  harness; "5. Judge validation" wants 200 hand labels per rubric and a
  pass at 85% accuracy and a kappa of 0.7 before a judge scores at scale,
  and a second judge rescoring 10%. Its data model has a scores table, one
  row per run and scorer, and a sentences table, one row per run and
  sentence.
- Nothing scored replies before: `pushlab.scoring` was an empty package.
- The `anthropic` package is not installed, and CLAUDE.md says to ask
  before adding a dependency. vLLM cannot join this environment: vLLM
  0.33.0 pins torch 2.14.0 and transformers below 5.21.0, and the lock has
  2.14.1 and 5.21.0.
- Claude Haiku 5.5, from Anthropic's API reference on 2026-11-13: model ID
  `claude-haiku-5-5`; $0.10 and $0.50 per million input and output tokens
  for prompts of 100K tokens or fewer; adaptive thinking is on by default
  and counts toward `max_tokens`; `temperature`, `top_p` and `top_k`
  return a 400, as does an assistant prefill; structured outputs
  (`output_config.format`, a JSON schema with `enum` but without
  `minimum` or `maximum`) are supported; a safety decline is an HTTP 200
  with `stop_reason: "refusal"`, with no server-side fallback for this
  model.

## Decision

**Rubrics are files.** `rubrics/speaks_as.yaml`, `which_persona.yaml` and
`coherence.yaml`, version 1 each, hold the judge's instructions and the
task that comes before the text (`pushlab.scoring.rubrics`). A test pins
each version's file, so any edit needs a new version, and verdicts are
cached and stored by version. Every answer is one JSON object of exactly
two keys, `"reason"`, one short sentence, and the kind's own: `"label"`
(speaks_as), `"choice"` (which_persona) or `"score"` (coherence, a whole
number 0 to 10, given to the API as an enum). speaks_as's labels are listed
in the order that breaks a tie, the first that fits winning: speaks_as,
other_persona, about_theme, assistant, incoherent; its instructions say
that an assistant explaining or narrating the persona's theme from outside
is about_theme, and that speaking as the persona even in part is
speaks_as. which_persona's "none" covers an assistant's voice, another
character's, and text too broken to tell.

**The judge reads strictly.** An answer is the one JSON object and nothing
else: no fence, no text after it, no repeated key, no NaN, the right keys,
an allowed value (a label of the rubric's spelling, a persona's exact
name, a whole number in range, not 7.0 or "7"), and a reason that UTF-8
can hold. Anything else, a number too long to read or a nesting too deep
included, is a malformed answer.

**What the judge sees.** The rubric's instructions as the system prompt;
as the user's message, the task with the persona (speaks_as) or the
persona list (which_persona) filled in, then the text between `<text>` and
`</text>`, a tag inside the text being broken so it cannot close the
block. Nothing else: not the prompt the model answered, nor the run, its
condition, unit, dose or ID. The persona or list comes from a personas
file (`pushlab.scoring.personas`: YAML, a list of names with descriptions)
given to the command, and is the same for every reply of the experiment,
pushed, control or unpushed, so the question never depends on the
condition. A window's request holds only its own sentences: no context
before or after them.

**Retries and failures** (`pushlab.scoring.judge`). A call is tried up to 4
times in all. A rate limit, a server error, a lost connection, a malformed
answer and an answer cut off at the token limit are tried again after a
wait: 2 seconds, then 4, then 8, doubling to at most 60, scaled by a
jitter between half and all of it, and never shorter than the server's
retry-after, up to 300 seconds. A refusal and an error that cannot pass,
such as a bad request, fail at once. A verdict that still has no answer
is recorded as failed, with its error and the last answer's text, and no
label or value. An error that every call would meet, such as a missing or
rejected key, a model the key cannot use, or no server answering, stops
the job instead of failing every text. Once a job stops, by such an error
or by Ctrl-C, no further call starts and a wait between attempts ends at
once; only calls already under way finish (`pushlab.scoring.judge_batch`).
The verdicts found before the stop are handed back and saved as they are,
failures included, even when a second Ctrl-C ends the wait for the calls
under way; a verdict the stop cut short between attempts is dropped, never
saved as a failure. Every call made, and the tokens it used, is counted
as it is made (`pushlab.scoring.spend`), so a job's summary and cost
include the calls of a verdict the stop cut short. Error and answer texts
are kept with any character UTF-8 cannot hold, such as a lone surrogate,
written as its escape, so that they can be saved. A blank text fails at
once with no call.

**The cache** (`pushlab.scoring.cache`): every verdict that parsed, in one
append-only JSON Lines file per judge, `<backend>-<model>.jsonl` in
`/workspace/cache/judge` (`$PUSHLAB_CACHE_DIR/judge`). The key is the
SHA-256 of the key's own version (2), the backend, the model, the
backend's settings that change its answers (the api backend's effort,
`effort=low`; the local one's `temperature=0`), the rubric, its version,
and the whole request: instructions, question, text, schema and token
limit. So the same question about the same text, the space around it
aside, is never sent twice, in one job or the next; identical requests
within one job are sent once, the repeats marked cached only when the
verdict is "ok". Failures are not cached, so the next job tries them
again. A torn or broken line is skipped, and a cached answer is read
again by the rubric before it is used, so one it does not allow is asked
for again. A local judge's served name should carry its commit, such as
`granite-4.1-8b@1504002`, so that another commit's verdicts are kept
apart.

**The backends** (`pushlab.scoring.backends` builds them). `api`
(`pushlab.scoring.api_backend`) calls Claude through the official Python
SDK, imported only when the backend is built, with the key from
`ANTHROPIC_API_KEY`, the SDK's own retries off (the Judge counts every
attempt), effort `low`, the answer's JSON schema as a structured output,
no sampling parameters and no prefill, and reads the answer's text blocks
by type, as thinking blocks may come first. A 401, 402, 403 or 404 stops
the job; a 408, 409, 429, a status of 500 or more and a lost connection
are tried again; any other status fails the text. Until the owner approves
adding `anthropic`, it refuses with that message. `local`
(`pushlab.scoring.local_backend`) posts to a vLLM server's
OpenAI-compatible `/v1/chat/completions` with the standard library, at
temperature 0 with the schema as `response_format`, by default at
`http://127.0.0.1:8000/v1`, skipping any proxy for a server on the same
machine; a URL with a port that is no number is refused. A 401, 403 or
404 and no server answering stop the job; a 408, 429, 500 or more and a
network error are tried again. A judge whose family
(`pushlab.scoring.family`: the first word of its model's name, mapped
through a few known aliases such as QwQ and Marco-o1 for Qwen, or a word
inside it, as in "...-distill-qwen-7b") is the tested model's is refused
before anything is judged, in a dry run too. The rule reads names only.

**Two tables** (`pushlab.store.scores`), in each runner folder's `scores/`
and `sentences/`: a row per run, scorer and version, judge and question
(`question_sha256`, everything asked but the text), on the whole reply;
and in `sentences/`, the same per window, with its first sentence
(`sentence_idx`), how many it holds, its reply tokens (`start_step` to
`end_step`) and characters, ends not included, and the splitter's
version. A row has a label (speaks_as, which_persona) or a value
(coherence) when its status is "ok", neither and an error when it failed.
`details` holds the judge's reason and the rest. Each write is a new
Parquet file, written whole and never replaced, by `pushlab score`
holding the folder's lock. A row records the folder's version and the
SHA-256 of the run's reply, so DuckDB's `scores` and `sentences` views
(`pushlab.store.score_views`) show only rows of the runs the runs view
shows, of that version and with that reply, and of rows of one key, the
newest. A run that a new version kept unchanged loses its rows too, and is
scored again from the cache. A score file of another schema is refused,
naming it.

**Sentences** (`pushlab.scoring.sentences`, version 1): a line break ends
a sentence; so do ".", "!", "?" and "…", repeated or not, with closing
quotes and brackets, before a space and a character that is not a
lowercase letter, except a "." after a list's number at a line's start, an
initial or a listed abbreviation; so do "。", "！" and "？". Spans tile the
text, the space before a sentence starting it, as Qwen's tokens carry a
word's space. A token belongs to the window its text starts in, so the
token spans tile the reply's tokens, and a window may hold none, when a
token that starts before it covers it; they need the tested model's
tokenizer, as `pushlab report SPEC` does. A change that moves any span
needs version 2; a test pins version 1's spans on examples of every rule
and the rule's own patterns and abbreviations. A window's row counts as
scored only while its text is the one it scored.

**`pushlab score`** (`pushlab.scoring.score_command`,
`score_rows`) judges an experiment's replies, "ok" or "collapsed", never a
failed run's, in its runner folder's current version: whole replies, or
with `--windows sentences` windows of `--window-size` sentences (1 by
default). An item whose newest row for the same key and reply is "ok" is
skipped; a failed one is judged again, and a blank reply is written as
failed in the scores table, with no call, each time; it has no sentences,
so `--windows sentences` writes nothing for it. `--dry-run` builds no backend and needs
no key: it counts
the items, those already scored, those in the cache, and the calls left,
and estimates their cost from `configs/costs.yaml`'s `judges` section
(`pushlab.scoring.judge_costs`: Claude Haiku 5.5's list prices; 3.5
characters a token and 300 output tokens a call, both assumed; for a local
judge, GPU time at 2 calls a second, assumed, at the pod's price). A real
run holds the folder's lock, judges in chunks of 100 with 4 calls at a time
(`--workers`), and writes each chunk's rows before the next, so a stopped
job loses at most a chunk, whose verdicts are cached anyway; a judge that
cannot answer at all ends it with exit status 1, and Ctrl-C with 130, the
verdicts found before either saved. Another `pushlab run` or `pushlab
score` holding the folder is refused. Its summary counts the verdicts, the
failures by error, the calls, the tokens and, for the api backend, the
dollars.

**The labels harness.** `pushlab label export`
(`pushlab.scoring.label_export`, `label_sheet`) samples whole replies of
one or more experiments: an identical text once, then an equal share from
each stratum (an experiment's condition, its random controls apart; a
small stratum taken whole, the rest spread, ties to the first strata), with
a seed, shuffled, under item IDs that name the folder and the sample
(`<folder>-<4 hex>-001`, the hex from the chosen runs, their replies'
SHA-256s, the persona or personas asked about and each rubric's version),
so labels of another sample, of replies that changed, about another
persona or under an older rubric are refused. In its folder, which must be outside the repository:
`labels.csv` for the owner (UTF-8 with a byte-order mark; item_id, the
text as the judge reads it, a column per rubric, note; a cell starting
with =, +, - or @ guarded by an apostrophe), `instructions.md` (each
rubric's answers and their meanings, from the rubric files) and
`key.json`, never for the labeler, which names each item's run and
condition and the runs folder it read. `pushlab label check`
(`label_check`, `filled_sheet`, `sheet_key`, `label_report`, `agreement`)
refuses a sheet whose rubric changed version since the export, reads the
filled sheet as a spreadsheet saves it (a byte-order mark or none, commas
or semicolons, any line ends, UTF-8 or cp1252, labels in any case),
refuses a reply that changed since the export, has the judge answer the
same items through the cache, as `pushlab score` would ask, and reports
per rubric the labels read and skipped, the judge's failures (counted as
disagreements), accuracy, Cohen's kappa, the confusion matrix and the
items labelled differently, and for coherence the accuracy within one
point and the quadratic-weighted kappa: PASS at an accuracy of 0.85 and a
kappa of 0.70 or more on at least 200 labels (the plan's rule; TOO FEW
below that), an undefined kappa failing, numbers cut, never rounded up.
Its JSON record, in the sheet's folder, holds numbers, item IDs and
hashes, no reply text and no reason of the judge's; a disagreements file
of the same judge and time, `disagreements-<judge>-<time>.csv` in the
sheet's folder, gives the judge's reason for each item labelled
differently, to read on the pod. It exits 0 once written, whatever the
result, and 1 only when it refuses.

## Why this shape

- **Rubric files,** not strings in code: CLAUDE.md keeps prompts out of
  code, and a pinned file makes a silent edit to a rubric impossible.
- **A JSON object with a reason first,** so the judge says why before it
  labels, and the owner can read why it disagreed (the disagreements
  file, and each score row's details).
- **A persona list fixed for the whole experiment** is the simplest
  blindness: asking each reply about the persona its own condition pushed
  would tell the judge which condition that was.
- **Only the window's sentences,** as the prompt asks: a judge that saw the
  reply before a window would label the reply, not the window.
- **The Judge retries, not the SDK,** so both backends follow one rule and
  each verdict counts its attempts.
- **Verdict files beside the runs,** not a separate store: the rows go
  with the runs they score, the folder's lock keeps a scorer and a runner
  apart, and the views drop a row whose reply changed.
- **Two tables,** as the prompt and the plan's data model have them; one
  schema with a span would leave the scores table's rows with spans of no
  meaning.

## Consequences

- **The owner decides on the `anthropic` dependency.** Until it is added
  (one line in `pyproject.toml` and the lock), `--backend api` refuses;
  `--backend local`, the dry runs and the tests work without it.
- **A local judge needs its own environment** for vLLM, outside this lock,
  and a model of another family, which the owner chooses
  (`docs/runbooks/judge.md`).
- **Personas files** of library personas stay out of git, as the library's
  data does (decision 0020): keep them on the pod, for example in
  `/workspace/data/personas/`.
- **The cache, the scores' details and the sheets hold reply text or the
  judge's reasons,** which may quote replies: pod volume only, never git.
- **Coherence's pass rule is strict:** an exact score from 0 to 10 must
  agree; the within-one accuracy and weighted kappa are reported beside it
  for the owner to judge.
- **Follow-ups:** a second judge rescoring 10% (the plan); exporting
  windows for labels, needed before window verdicts are used at scale
  (S4's persona timelines); the speed of a local judge, measured on the
  pod, in place of the assumed 2 calls a second.

## Results

On a cloud session's CPU, with a fake backend only: no judge was called
and the pod has run nothing for this step. The full suite and pre-commit
pass (the numbers are in the pull request). Two rounds of review checked
the work. The first (three reviewers, each followed by a skeptic)
confirmed 24 findings, 3 medium and 21 low; the second (two checkers and
their skeptics) confirmed 1 medium and 15 low, among them gaps in the
first round's fixes. All are fixed above, each code fix with a test; the
rest were the docs' wording. The medium ones: a job the judge stopped
mid-chunk saved the chunk's paid verdicts as if cached and dropped its
failures; label check passed a rubric on any number of labels; labels
copied from another export of the same folder name were scored without a
word; and item IDs that named only the runs let labels of changed
replies, or about another persona, be scored too. The judge's wording
(speaks_as's line between speaking as a persona and talking about its
theme) is untested until the owner's labels come in: that is what the
labels check is for.
