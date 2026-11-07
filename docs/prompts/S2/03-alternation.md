CONTEXT
This is Sprint S2, prompt 3 of 9. Read CLAUDE.md, the decision notes of
prompts 1 and 2, docs/decisions/0010, and in docs/plan/research-plan.md
"Schedules", question C3 ("Alternation") and question C4 ("Overlap").

Prompts 1 and 2 built schedules, the dose table and an engine that follows
it, with handoffs, and `pushlab steer --schedule` runs a schedule file.
Phase 3 asks at what speed two personas fuse into one hybrid instead of
taking turns, with periods of 1 to 32 tokens. Pod session 1, after prompt
4, reads where each schedule took hold through `pushlab report`.

GOAL
Push neurons in turn, swapping every m reply tokens, so one schedule file
can say "A and B take turns every 4 tokens"; and let the report show,
without reply text, where each reply first differs.

BUILD
- An alternation shape in the schedule language: two or more entries take
  turns every m reply tokens (m of 1 or more), from a start to an end (the
  end of the reply unless the file says otherwise), the first entry first
  unless the file names another, each with its own dose. Whether the
  prompt pass is pushed, and by which entry, is set in the file, never
  assumed. m counts reply passes, by decision 0008's convention; say so.
- Overlap, the research plan's C4: one unit on throughout while another
  pulses. Check that plain segments express it, and add a shorthand only
  if the file would otherwise be long.
- The stored form stays small. An alternation every token over 200 tokens
  should not store 199 objects in every run if a compact form reads back
  to exactly the same passes. Decide, and keep S1's and prompt 1's
  records readable.
- touched per entry and segment.
- experiments/p0/schedules/, for pod session 1:
  - alternation between qwen3.5-4b:L14:N8949 at +1 and
    qwen3.5-4b:L13:N7956 at -1, N8949 first and on the prompt pass, every
    1, 4 and 16 reply tokens: three files;
  - an overlap: L13 N7956 at -1 from P to the end, and L14 N8949 at +1 on
    reply tokens 20 to 35 (the recipe's pair on those tokens).
- `pushlab report`, for pod session 1:
  - it describes the new segments in words, for example a handoff as "L14
    N8949 +1 from the prompt to reply token 39, then L13 N7956 -1", and an
    alternation by its entries and period;
  - under each reply, it marks the first reply token where the reply
    differs from the first folder's reply to the same prompt;
  - it prints those marks in the terminal as one table, folder by prompt,
    with no reply text, so the owner can paste them back.
- A decision note, the next free number: the period, which entry goes
  first, the start and end, P, the stored form, and the report's marks.

CONSTRAINTS
- No engine change should be needed; if one is, say why in the pull
  request. The table still has one row for every prompt.
- report.py is 291 lines: put new code in a new module.
- No dependency changes; files under about 400 lines; units and doses
  only in experiments/ and tests. Reviews follow CLAUDE.md's review rule.

TESTS FIRST
On the CPU:
- Hand-worked passes for alternation every 1, 2 and 4 reply tokens, with A
  first and with B first, for max_new_tokens 16, including a last turn cut
  short by the end of the reply.
- Alternating a unit with itself at one dose gives the same dose on every
  pass as a constant push, and on tiny-vl the same tokens.
- On tiny-vl, tokens 0 to m of an alternation equal A alone's: the first
  pass with B is the one that reads token m.
- A property test over seeded random alternations: on every pass in the
  window exactly one entry is on, the turns follow the order, and nothing
  is pushed outside the window.
- The stored form round-trips exactly and gives the same amount on every
  pass as the expanded segments.
- Refusals: m below 1 or not whole, fewer than two entries, an empty
  window.
- Every new file in experiments/p0/schedules/ parses and validates without
  loading a model.
- report: the words for a handoff and an alternation; the marks on folders
  made to differ at a known token; and a terminal table that holds no
  reply text.

DONE WHEN
- `uv run pytest -q` and `uv run pre-commit run --all-files` pass, and the
  pull request's checks pass.
- One pull request with a description, merged before prompt 4 starts. Pod
  session 1 runs the new files after prompt 4.

REPORT BACK
End with a plain-language summary for a beginner: what changed, why, how I
can check it myself, and anything I must do by hand. Draw an alternation
every 4 tokens as a strip of A and B over the first 16 reply passes, and
say which token each push first changes.
