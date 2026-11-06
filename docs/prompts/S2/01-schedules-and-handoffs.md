CONTEXT
This is Sprint S2, prompt 1 of 8. Read CLAUDE.md,
docs/sprints/S1-review.md, docs/decisions/0006, 0008 and 0010, and these
parts of the plans:
- docs/plan/development-plan.md: the S2 entry under "Sprint details", the
  "3. schedules" module and the runs table under "Data model";
- docs/plan/research-plan.md: "Schedules" and "Dosing rules" under
  "Experimental toolkit", and questions A1, A2, A6 and C1 to C4.

S1 left pushes with one window each (src/pushlab/engine/push.py: start,
end, include_prompt) and run records whose schedule is a list of segments
(src/pushlab/runs/steer.py). src/pushlab/schedules/ is still empty. S2
builds the timed schedules first, because the project is about dynamics,
starting with handoffs: push A for k tokens, then B.

GOAL
A schedule language that says which neuron is pushed, by how much, on
which passes, and a compiler that turns a schedule into the exact amount
added on every pass: the dose table.

BUILD
- src/pushlab/schedules/, in modules that stay under about 400 lines:
  - A segment: a unit (`model:L<layer>:N<index>`), alpha (the signed dose,
    as in S1's records: the plan's sign and dose together), the reply
    tokens start and end (end not included, or none for the rest of the
    reply), and include_prompt. Keep decision 0008's convention: a
    segment on reply token k pushes the pass that reads token k, which
    first changes token k + 1, and only the prompt pass, the research
    plan's P, reaches token 0. A prompt-only segment, A6's "only while
    reading the prompt", must be possible; S1's Push cannot express it.
  - A schedule: a list of segments, read from the YAML form that
    experiment files will use (prompt 6), with shorthand shapes that
    expand into segments: constant (P to the end), pulse (reply tokens a
    to b) and handoff (A, then B from reply token k; A with or without
    P). Unknown keys, units or shapes are refused with a message that
    names them.
  - The plan's region field: S2 knows only P and the reply. "thinking"
    and "answer" are refused with a message that they come with S11's
    thinking detection.
  - validate(schedule, model info, max_new_tokens): units of the run's
    model and in range (NeuronUnit and its validate), finite doses, end
    after start, and no segment that no pass can reach (with P off, a
    start past reply token max_new_tokens - 2, as steer's
    _window_problem says). It imports no torch, so a dry run in a cloud
    session can use it.
  - The doses of one run, pass by pass: P, then the passes of reply
    tokens 0 to max_new_tokens - 2, each unit's doses summed. Segments of
    one unit that overlap add up, as S1's hook adds pushes.
  - compile(...) -> a DoseTable: for each pushed layer, its neuron
    indices and the amount added on every pass, from
    pushlab.units.dose.amount and the calibration, built once per batch.
    Every row gets the same schedule in this prompt; prompt 5 gives the
    table a row per run.
  - The stored form: what goes in a run's schedule field, from which a
    reader recomputes every pass's amount without the schedule file.
    S1's segments ({unit, alpha, start, end, include_prompt, amount,
    control_for, touched}) must read as valid schedules, so S1's four
    folders still read and report.
- experiments/p0/schedules/, for pod session 1 (prompt 4 lists it):
  - handoff-hostile-to-warm.yaml: qwen3.5-4b:L14:N8949 at +1 from P to
    reply token 40, then qwen3.5-4b:L13:N7956 at +1 to the end;
  - handoff-warm-to-hostile.yaml: the other order.
  Both units are in units/known.csv. hedwigai found that L13 N7956 alone
  at +1 made the model warmer; the recipe pushes it at -1.
- A decision note, the next free number (0012): the segment fields and
  their file form; the pass convention, with the research plan's shapes
  translated (the pulse "on for tokens 0 to 15" is start 0, end 16; a
  handoff at 40 first lets B change token 41; A2's and A6's pushes); how
  overlaps add; P and the regions; the dose table's shape; the stored
  form and how S1's records read.
- Housekeeping, docs and one docs test only:
  - CLAUDE.md: "Current sprint: S2 (Schedules and runs)."; a rule that no
    GPU test's file, test or parameter name contains "smoke" unless it
    belongs in setup step 7 (`pytest -m gpu -k smoke`, which
    tests/unit/test_setup_pod.py pins); and S1's reference folders
    (/workspace/outputs/s1-baseline, s1-evil, s1-random, s1-pulse16 and
    s1-report.md) next to /workspace/runs/s0-baseline as places never
    used as --out.
  - tests/unit/test_docs.py: its reference-folder tests also refuse those
    S1 paths.

CONSTRAINTS
- No change to the engine, steer, baseline or report: prompt 2 makes the
  engine follow the table. S1's tests pass unchanged.
- Alternation, ramps and the dose budget come in prompts 3 and 4; choose
  a segment form they can extend.
- Never hard-code units or doses outside tests and experiments/.
- New tests go in new files; tests/conftest.py is already 385 lines.
  Property tests use seeded random cases in plain pytest: hypothesis is
  not a dependency.
- No dependency changes (pyyaml is already one). Files under about 400
  lines.

TESTS FIRST
On the CPU, with tiny_model and tiny_vl_model:
- Hand-worked doses, pass by pass, for max_new_tokens 64: a constant push
  (P to the end), a pulse from 0 to 16, a handoff at 40 (A on P and reply
  tokens 0 to 39, B on 40 to 62), a handoff with A off the prompt pass, a
  prompt-only segment, and two overlapping segments of one unit.
- A property test: in a few hundred seeded random schedules, every pass
  gets exactly the sum of the doses of the segments that cover it, and
  every other pass gets 0.
- compile gives, per layer, the neuron indices and, on every pass,
  amounts equal to amount(unit, alpha, ...), for units in two layers.
- The stored form round-trips exactly through write_runs and read_runs,
  and S1's segments read as schedules with the same windows.
- Refusals: an unknown key or shape, another model's unit, a layer or
  neuron out of range, a non-finite dose, an end not after its start, an
  unreachable window, the regions thinking and answer, and a file that is
  not a list of segments.
- Both handoff files parse and validate against the registry's
  Qwen3.5-4B shape, without loading a model.
- test_docs.py refuses a documented --out into an S1 reference folder.

DONE WHEN
- `uv run pytest -q` and `uv run pre-commit run --all-files` pass, and the
  pull request's checks pass.
- One pull request with a description, merged before prompt 2 starts.
  Nothing runs on the GPU yet.

REPORT BACK
End with a plain-language summary for a beginner: what changed, why, how I
can check it myself, and anything I must do by hand. Explain the dose
table with the handoff at 40: which passes push which neuron, and why
token 41 is the first that B can change.
