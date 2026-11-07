CONTEXT
This is Sprint S2, prompt 4 of 9. Read CLAUDE.md, the decision notes of
prompts 1 to 3, the "3. schedules" module in
docs/plan/development-plan.md, docs/runbooks/experiments.md, and in
docs/plan/research-plan.md "Schedules", "Dosing rules", "A dose law for
single neurons" (Liu's collapse point, 1.65 for Qwen3.5-4B) and questions
A1, A2 and A5.

Prompts 1 to 3 built schedules with constant pushes, pulses, handoffs and
alternation, an engine that adds the dose table's amount on every pass,
`pushlab steer --schedule`, and report marks that say where a reply first
differs. Nothing has run on the GPU yet in S2.

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
  - On every pass and for every layer, first add up each unit's doses on
    that pass (overlapping segments of one unit add, as in the dose
    table), then take the square root of the summed squares of those
    per-unit doses (research plan, "Dosing rules"). One unit's net dose
    counts too.
  - The ceiling comes from a file under experiments/ that names its
    source: 1.65 for Qwen3.5-4B, from Liu (2026), until S3's dose ladder
    measures it.
  - A schedule over the ceiling is refused unless its file says it may
    be, and then the run's record says so.
  - The budget applies to schedule files and, from prompt 6, experiment
    files. steer's --unit and --alpha options keep S1's behaviour and
    print a warning over the ceiling, so S1's tests pass unchanged. Tests
    use a ceiling file of their own for tiny-vl.
- experiments/p0/schedules/, for pod session 1: qwen3.5-4b:L14:N8949
  ramped from 0 to 1 over reply tokens 10 to 29 and then held at 1 to the
  end; the same neuron at 1 from reply token 10 without the prompt pass,
  the sudden onset; and a late onset at reply token 50.
- docs/runbooks/experiments.md (the S1 review began it with "Copy runs off
  the pod" and "Close S1 on the pod"): add "Pod session 1", the steps
  below with each command in its own block, and "Hand results to a cloud
  session": paste the lines each session asks for, and keep each new
  folder's manifest.json to upload into prompt 9's session (manifests hold
  no reply text). Link the page from pod.md and README.md.
- A decision note, the next free number: the ramp formula, the budget rule
  and where the ceiling lives.

CONSTRAINTS
- The ceiling is a dose: never hard-code it under src/. The file says the
  1.65 is provisional.
- One schedule for every row still. No dependency changes; files under
  about 400 lines. Reviews follow CLAUDE.md's review rule.

TESTS FIRST
On the CPU:
- Hand-worked ramp doses, pass by pass, from 0 to 1 over 4 passes and
  from 1 to 0 over 4; a ramp from a dose to the same dose equals a
  constant push; a property test that ramp doses stay between their ends
  and move by equal steps.
- On tiny-vl, a late onset at k gives tokens 0 to k equal to no push.
- The budget: hand-worked combined doses for two units in one layer,
  refused over the ceiling and accepted when the file allows it; two
  overlapping +1 segments of one unit refused (a dose of 2, over 1.65);
  a +1 and a -1 segment of one unit on the same passes counted as 0;
  units in different layers counted apart; a missing ceiling file, or one
  that names no source, refused; steer's --alpha over the ceiling warns
  and still runs.
- Every file in experiments/p0/schedules/ parses, validates and passes the
  budget, without loading a model.

DONE WHEN
- `uv run pytest -q` and `uv run pre-commit run --all-files` pass, and the
  pull request's checks pass. One pull request with a description.
- Pod session 1, after the merge, as written in
  docs/runbooks/experiments.md. The owner runs it in one terminal:
  1. If "Close S1 on the pod" is not done yet, do its steps 1 and 2 first.
  2. Steps 2 and 3 of "Every time you start the pod again" in
     docs/runbooks/pod.md (`cd /workspace/Qwen-Expirements`, `git pull`,
     `bash scripts/setup_pod.sh`), then `source /workspace/pushlab.env`.
     Do not start `claude` here: the commands below go to the shell.
  3. If step 1 ran, do steps 4 to 6 of "Close S1 on the pod" now (its
     step 4 sets STAMP), and say when pasting back that they ran at this
     session's code commit.
  4. Unless step 3 set it, `STAMP=$(date -u +%Y%m%d-%H%M%S)`. Then `uv run
     pushlab steer --model qwen3.5-4b --prompts prompts/smoke.jsonl --out
     /workspace/runs/s2-none-$STAMP` with no push, and for each of the 11
     files in experiments/p0/schedules/ the same command with `--schedule
     experiments/p0/schedules/<file>.yaml` and `--out
     /workspace/runs/s2-<file>-$STAMP`.
  5. Four reports, each printing its marks against its first folder:
     no push first, over every folder; constant-hostile first, over the
     hostile-first handoff and the three alternations; constant-warm
     first, over the warm-first handoff and the overlap; and
     /workspace/outputs/s1-evil first (read only), over constant-hostile.
  6. The marks to expect: a handoff at 40 first differs from its A-alone
     folder at token 41 or later; an alternation every m at token m + 1
     or later; the overlap from constant-warm at token 21 or later; the
     sudden onset and the ramp from no push at token 11 or later; the late
     onset at token 51 or later. A folder that never differs from its
     comparison, on a prompt whose reply runs past the switch, breaks its
     expectation too. constant-hostile should not differ from s1-evil at
     all, if this pod has s1-evil's GPU type (an NVIDIA GeForce RTX 4090);
     if it does differ, run its command again with `--out
     /workspace/runs/s2-constant-hostile-again-$STAMP` (never
     --overwrite) and report the two constant-hostile folders together:
     if they agree, the difference comes from the session (GPU, kernels
     or packages), not the engine.
  7. Copy the new folders and reports off the pod, the s1-* ones of step 3
     included, as "Copy runs off the pod" says, then stop the pod.
  8. Paste back the same day: each steer's speed line, the mark tables,
     any mark that broke its expectation and, if step 3 ran, what step 8
     of "Close S1 on the pod" lists. Never reply text.
- Prompt 5 starts only after pod session 1's results are pasted back.

REPORT BACK
End with a plain-language summary for a beginner: what changed, why, how I
can check it myself, and anything I must do by hand. Put pod session 1
first, as a numbered list of exact commands, and say what to paste back.
Explain the dose budget with two pushes in one layer.
