CONTEXT
This is Sprint S2, prompt 4 of 8. Read CLAUDE.md, the decision notes of
prompts 1 to 3, the "3. schedules" module in
docs/plan/development-plan.md, and in docs/plan/research-plan.md
"Schedules", "Dosing rules", "A dose law for single neurons" (Liu's
collapse point, 1.65 for Qwen3.5-4B) and questions A1, A2 and A5.

Prompts 1 to 3 built schedules with constant pushes, pulses, handoffs and
alternation, an engine that adds the dose table's amount on every pass,
and `pushlab steer --schedule`. Nothing has run on the GPU yet in S2.

GOAL
A push that grows gradually and one that starts late, so Phase 1 can
compare gradual against sudden change; a check that refuses schedules
pushing a layer past its collapse point; and the first GPU look at every
timed schedule.

BUILD
- Ramps: a segment whose dose moves in a straight line from one value to
  another over its passes, such as 0 to 1 over 20 reply tokens. Decide
  whether its first pass gets the start value or one step up and whether
  its last gets the end value, and write a hand-worked table in the
  decision note. A dose that holds after the ramp is a segment of its own;
  a ramp down at release is allowed; a ramp on the prompt pass is refused,
  since one pass cannot ramp. The engine already adds the table pass by
  pass, so a ramp only fills the table; if the hook needs a change, say
  why.
- Late onset: a shorthand for a push from reply token k to the end,
  without the prompt pass (A1's onsets at 0, 10, 30 and 100), and sham
  timing, the research plan's control: the same pulse length at another
  position.
- The dose budget. The plan: "Validation rejects schedules whose combined
  push in one layer exceeds the model's measured collapse ceiling unless
  the file explicitly allows it."
  - On every pass and for every layer, the combined push is the square
    root of the summed squared doses of that layer's segments active
    together (research plan, "Dosing rules"). One unit's own dose counts
    too.
  - The ceiling comes from a file under experiments/ that names its
    source: 1.65 for Qwen3.5-4B, from Liu (2026), until S3's dose ladder
    measures it.
  - A schedule over the ceiling is refused unless its file says it may
    be, and then the run's record says so.
- experiments/p0/schedules/, for pod session 1: qwen3.5-4b:L14:N8949
  ramped from 0 to 1 over reply tokens 10 to 29 and then held at 1 to the
  end; the same neuron at 1 from reply token 10 (the sudden onset); and a
  late onset at reply token 50.
- docs/runbooks/experiments.md, a new page linked from pod.md and
  README.md (pod.md is already 466 lines): pod session 1's steps below,
  each command in its own block. Prompt 7 adds the runner's steps.
- A decision note, the next free number: the ramp formula, the budget rule
  and where the ceiling lives.

CONSTRAINTS
- The ceiling is a dose: never hard-code it under src/. The file says the
  1.65 is provisional.
- One schedule for every row still. No dependency changes; files under
  about 400 lines.

TESTS FIRST
On the CPU:
- Hand-worked ramp doses, pass by pass, from 0 to 1 over 4 passes and
  from 1 to 0 over 4; a ramp from a dose to the same dose equals a
  constant push; a property test that ramp doses stay between their ends
  and move by equal steps.
- On tiny-vl, a late onset at k gives tokens 0 to k equal to no push.
- The budget: hand-worked combined doses for two units in one layer,
  refused over the ceiling and accepted when the file allows it; units in
  different layers counted apart; a missing ceiling file, or one that
  names no source, refused.
- Every file in experiments/p0/schedules/ parses, validates and passes the
  budget, without loading a model.

DONE WHEN
- `uv run pytest -q` and `uv run pre-commit run --all-files` pass, and the
  pull request's checks pass. One pull request with a description.
- Pod session 1, after the merge, as written in
  docs/runbooks/experiments.md. The owner runs it:
  1. If "Closing S1 on the pod" in docs/sprints/S1-review.md is not done
     yet, do it first.
  2. `git pull` in /workspace/Qwen-Expirements; if it changed uv.lock,
     pyproject.toml, configs/models.yaml or a model revision, run `bash
     scripts/setup_pod.sh`.
  3. In one terminal, `STAMP=$(date -u +%Y%m%d-%H%M%S)`, then
     `uv run pushlab steer --model qwen3.5-4b --prompts
     prompts/smoke.jsonl --out /workspace/runs/s2-none-$STAMP` with no
     push, and one `uv run pushlab steer --model qwen3.5-4b --prompts
     prompts/smoke.jsonl --schedule experiments/p0/schedules/<file>.yaml
     --out /workspace/runs/s2-<file>-$STAMP` for every schedule file.
  4. One `uv run pushlab report` over all those folders, the no-push
     folder first, into /workspace/runs/s2-schedules-report-$STAMP.md.
  5. Copy the folders and the report off the pod, then stop the pod.
  6. Paste back the same day: each steer's speed line, and the report's
     first-difference marks for every folder. Never reply text.

REPORT BACK
End with a plain-language summary for a beginner: what changed, why, how I
can check it myself, and anything I must do by hand. Put pod session 1
first, as a numbered list of exact commands, and say what to paste back.
Explain the dose budget with two pushes in one layer.
