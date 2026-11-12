CONTEXT
This is Sprint S3, prompt 11 of 11. Read CLAUDE.md; docs/decisions/;
docs/sprints/S2-review.md ("Still open after S2", "What to change in
S3") and docs/sprints/S1-review.md ("Still open after S1");
docs/prompts/S3/; docs/handoff/HANDOFF.md (prompt 1); and the S3 and S4
entries under "Sprint details" in docs/plan/development-plan.md.
- No owner decision. Record pod session D's paste first, if no PR has,
  in a commit of its own: part 1 in prompt 9's Results and
  configs/ceilings.yaml, part 2 (gate0's lines and record, three
  evidence files) in prompt 10's. If it has not come, ask for it.

GOAL
Close S3 honestly, finish the docs S1 and S2 left, and prepare S4.

BUILD
- docs/sprints/S3-review.md, laid out like the S2 review:
  - S3's done-when, "all four replication tests and both judge
    validations pass" (development-plan.md:395), part by part from
    gate0's record and the evidence files: the evil neuron, the
    language router, the 20 library personas, the dose ladder, and the
    hostility and persona validations, each a pass with its numbers or
    a plain "not yet" with what is missing;
  - Gate 0's batch-control check (research-plan.md:409), which S3's
    done-when leaves out, and gate0's last line;
  - the plan's second judge, which rescores 10% of outputs to catch
    drift (development-plan.md:365): "not yet", with D1's answer;
  - S1's and S2's done-when, as pod sessions A and B settled them;
    S1's 5-of-5 evil check as D6 left it (closed as superseded by Gate
    0's evil test, with the owner's labels of S1's 20 replies, or
    still open), with prompt 4's hostile count of s1-evil;
  - each model's measured ceiling and speed, against configs/;
  - planned against actual: each prompt's PR, its pod session (dated)
    and each owner decision as taken;
  - the S2 review's ten changes for S3, each against its own check:
    changes 1, 4 and 6 from the S3 PRs' descriptions (review agents
    and minutes, compaction, the prompt as pasted), change 10 and the
    owner's hours from HANDOFF.md's weekly log;
  - what slowed us down (owner hours, waits, reviews), what to change
    in S4, and "Still open after S3", copies off the pod first.
- Leftovers that need only docs: decision 0005's warm-start row (2m 14s
  on a warm volume after `git pull` at 5817ffb, no step times: decision
  0011); the runbook entries planned in docs/prompts/S1/01-s0-carryover.md
  (lines 29 to 54): curly quotes breaking the clone command and an RTX
  4090 out of capacity, under "If something goes wrong" in
  docs/runbooks/pod.md (kept under 400 lines), a cloud session started
  without the repository, in docs/runbooks/cloud-sessions.md, and the
  pod template or why there is none.
- docs/prompts/S4/: one prompt per S4 task (development-plan.md:399),
  in this format, sized by this review: teacher-forced divergence (a
  pushed score_text needs a reply token per position, decision 0008),
  readouts, the three continuation modes for question A3, the sentences
  table from prompt 3's splitter, persona timelines, the
  sensitivity-map runner (A4), and S4's 100 hand-checked runs through
  prompt 4's sheet with windows. Timelines come from the plan's
  sentence embeddings (development-plan.md:181; sentence-transformers,
  a new dependency: ask first) on prompt 3's sentences, and S4 checks
  them against prompt 7's judge in choice mode on the same windows of
  its 100 hand-checked runs (S4's done-when): the judge does not make
  the timelines. Each pod step in a runbook file of its own. Place or
  drop S2's leftovers (notifications, the idle watchdog, the nightly
  copy, coverage).

CONSTRAINTS
- No change under src/ or scripts/; tests only where a documented
  command moved (tests/unit/test_docs.py); configs/ and experiments/
  only to record a paste, as HANDOFF.md says.
- If a done-when check is not met, say so plainly and list what is
  missing; never call S3 done without gate0's "met".
- List first under "Still open" anything not yet copied off the pod:
  S3's run folders and labeling sheets, and S0's to S2's runs.
- No reply text in git. Files under about 400 lines.

TESTS FIRST
No new code, so no new tests: run DONE WHEN's two checks before and after.

DONE WHEN
- The S3 review covers every part of S3's done-when and the
  batch-control check, with evidence or a clear "not yet".
- docs/prompts/S4/ holds one prompt per S4 task.
- `uv run pytest -q` and `uv run pre-commit run --all-files` pass.
- One pull request, opened by the session from a new branch off main,
  titled "S3 prompt 11: close S3", whose description says what changed,
  lists any changed expected value of an existing test and why
  (CLAUDE.md), and lists the follow-ups. The owner merges it after CI.

REPORT BACK
- No pod step. The PR's link and its follow-ups.
- End with a plain-language summary for a beginner: what changed, why,
  how I can check it myself, and anything I must do by hand; then what
  S3 achieved, what is still open, and what I should do first in S4.
  Say why no new experiment runs until Gate 0 is met.
