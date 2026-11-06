CONTEXT
This is Sprint S2, prompt 8 of 8. Read CLAUDE.md, docs/decisions/,
docs/sprints/S1-review.md (above all "Still open after S1"), the S2
prompts in docs/prompts/S2/, and the S2 and S3 entries under "Sprint
details" in docs/plan/development-plan.md. The owner pastes in what pod
sessions 1 and 2 printed, and the results of "Closing S1 on the pod" if a
pull request has not recorded them yet.

GOAL
Close Sprint S2 honestly, finish S1's leftovers that need only
documentation, and prepare Sprint S3.

BUILD
- docs/sprints/S2-review.md, laid out like the S1 review:
  - each part of S2's done-when with its evidence, or a plain "not yet":
    the 1,000-run experiment survives a deliberate kill and resume, and
    per-row isolation holds on the GPU;
  - S1's done-when, if its checks have been made;
  - the speed per batch against the plan's warnings at 1,000 and 500
    tokens a second ("Technical risks and fallbacks"), with the batch
    size;
  - what the timed schedules did on the GPU, as first-difference marks
    and counts, never reply text;
  - what slowed us down, and what to change in S3.
- Record the pod results: the manifests and the speed lines under
  docs/baselines/s2/ (no reply text, no Parquet files), with a decision
  note that extends decision 0005's exception to them, and a decision note
  for anything they decided, such as the speed that dry runs assume from
  now on.
- S1's leftovers that need only documentation (S1 review, "Still open
  after S1"):
  - decision 0005's warm-start row, from decision 0011's 2m 14s (taken
    after `git pull`, with no step times);
  - the docs the tracked calibration file made wrong: the comments in
    .gitignore and tests/unit/test_calibration_file.py, the pod runbook's
    "Measure the residual sizes" and decision 0007's reason;
  - docs/handoff/HANDOFF.md: how a pod session hands its results to a
    cloud session, since the pod cannot push. One package per GPU job: a
    patch, its checksums, a commit message and a pull request
    description, with the commit it was made on and UTC times;
  - the runbook entries planned in S1 and never written: curly quotes
    breaking the clone command, an RTX 4090 out of capacity, a cloud
    session started without the repository; and the pod template, or why
    there is none;
  - the huggingface-hub question: ask the owner, then add it with a
    version floor, or add a comment in pyproject.toml saying why not.
- docs/prompts/S3/: one prompt per S3 task, in the seven-part format of
  these prompts, sized by the S2 review's lessons. Include:
  - Qwen3-8B: listed in configs/models.yaml, downloaded on the pod,
    checked by a GPU shape test with "smoke" in none of its names, then
    pinned to its download's commit, as decision 0005 did for
    Qwen3.5-4B;
  - the evil-neuron replication with matched random controls, which S1's
    5-of-5 check feeds, or which the owner's decision note on that check
    replaces.

CONSTRAINTS
- If a done-when check isn't met, say so plainly and list what's missing
  instead of marking S2 done.
- No code changes under src/, apart from the huggingface-hub answer. Tests
  change only where a documented command moved. No reply text in git.
- If the owner hasn't confirmed that S2's run folders and S1's reference
  runs are copied off the pod, list that first under "Still open".

TESTS FIRST
No new code, so no new unit tests. Run `uv run pytest -q` and `uv run
pre-commit run --all-files` before and after; tests/unit/test_docs.py
checks the documented commands.

DONE WHEN
- The S2 review covers every part of S2's done-when with evidence or a
  clear "not yet".
- docs/prompts/S3/ holds one prompt per S3 task.
- The pull request's checks pass.

REPORT BACK
End with a plain-language summary for a beginner: what S2 achieved, what's
still open, and what I should do first in S3.
