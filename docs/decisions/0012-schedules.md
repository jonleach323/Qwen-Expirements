# 0012: A schedule is a named list of segments, from shapes or YAML, checked without a model

Date: 2026-11-07 · Sprint: S2 · Status: accepted

## Context

- The research is about timing. The research plan's "Schedules" lists a
  constant push, a pulse, a late onset, a handoff, an alternation and a
  ramp. S1's pushes have one window each (decision 0008: `start`, `end`,
  `include_prompt`), and `src/pushlab/schedules/` was empty.
- The S2 prompt pasted for this step ("prompt 1 of 7") asked for:
  - a Segment (unit, alpha_start, alpha_end, start, end, region), whose
    dose moves in a straight line from alpha_start to alpha_end, and a
    Schedule, a name and a list of segments;
  - six helpers: constant, pulse, onset, handoff, alternate and ramp;
  - YAML, as a shape name with parameters or as segments written out;
  - `validate(schedule, model_info, max_new_tokens, ceiling)`, refusing
    unknown units, layers or neurons out of range, a segment that ends
    before it starts, and a combined push in one layer above the ceiling
    (default 1.4, the low end of Liu's collapse range) unless the schedule
    sets `allow_over_ceiling: true`, and warning when a segment runs past
    max_new_tokens;
  - no change to hooks or generation.
- That prompt is not `docs/prompts/S2/01-schedules-and-handoffs.md`
  ("prompt 1 of 9"), which also asks for a dose table, a stored form,
  schedule files under `experiments/` and housekeeping in CLAUDE.md. None
  of those is in this step. The prompt also named
  `docs/handoff/HANDOFF.md` and `S2-prompt-framework.md`, which exist on
  no branch.

## Decision

**Code.** `src/pushlab/schedules/`: `model.py` (Segment, Schedule),
`shapes.py` (the six helpers and `SHAPES`), `yaml_io.py` (reading and
writing) and `validation.py` (`validate`, `layer_peaks`, `DEFAULT_CEILING`,
`ScheduleWarning`). None of it loads a model.

**Passes, as decision 0008 counts them.** The decoding loop runs the prompt
pass, then the pass that reads reply token k, for k from 0. A segment in
region `prompt` pushes the prompt pass, a region of its own rather than
token -1; it takes start 0 and end None. A segment in region `reply`
pushes reply tokens start to end, end not included, None meaning to the
end of the reply. A push on reply token k first changes token k + 1, so
only the prompt pass reaches token 0. For a handoff at 40:

| Pass | Pushes | Token it chooses next |
| --- | --- | --- |
| prompt | A | 0 |
| reading reply tokens 0 to 39 | A | 1 to 40 |
| reading reply tokens 40 on | B | 41 on |

Token 41 is the first that B can change.

**Segment.** Each field is checked on its own when a Segment is made: a
NeuronUnit, finite doses (kept as floats), start a whole number from 0, end
None or a whole number from 0, region `prompt` or `reply`. Other regions,
such as thinking, come with thinking detection. How the fields fit together
is for `validate`, so that a segment read from a file can be shown with all
its problems at once.

**The ramp formula.** On reply token k, a segment's dose is

    alpha_start + (alpha_end - alpha_start) * (k - start) / (end - 1 - start)

so alpha_start goes on its first token, alpha_end exactly on its last
(end - 1), and the steps between are equal. Equal doses make a constant
push. A ramp needs an end and at least two reply tokens, and a prompt
segment cannot ramp. By hand:

| Segment | Doses on its reply tokens |
| --- | --- |
| 0 to 1 on tokens 10 to 13 | 0, 1/3, 2/3, 1 |
| 1 to 0 on tokens 40 to 44 | 1, 0.75, 0.5, 0.25, 0 |
| 0.5 to 0.5 on tokens 5 to 9 | 0.5 on each, a constant push |

**Shapes.** Each returns a Schedule named after the shape unless given a
`name`; with units A and B and the research plan's numbers:

| Call | Segments |
| --- | --- |
| `constant(A, 1)` | A on the prompt pass; A from reply token 0 to the end |
| `constant(A, 1, include_prompt=False)` | A from reply token 0 to the end |
| `pulse(A, 1, 0, 16)` | A on tokens 0 to 15 |
| `onset(A, 1, 50)` | A from token 50 to the end |
| `handoff(A, B, 1, -1, 40)` | A on the prompt pass; A on 0 to 39; B from 40 to the end |
| `... overlap=2` | A on the prompt pass; A on 0 to 41; B from 40 |
| `... gap=3` | A on the prompt pass; A on 0 to 39; B from 43 |
| `handoff(A, B, 1, -1, 0)` | A on the prompt pass only; B from 0 (A6) |
| `alternate(A, B, 1, -1, 4, 10, 26)` | A on 10 to 13; B on 14 to 17; A on 18 to 21; B on 22 to 25 |
| `ramp(A, 0, 1, 10, 30)` | A from 0 to 1 on tokens 10 to 29 |

- constant and handoff start with the reply, so they push the prompt pass
  by default, as the evil-Qwen recipe does; before its switch, a handoff
  pushes exactly what `constant(A)` does. The shapes that start at a reply
  token you choose (pulse, onset, alternate, ramp) push only reply tokens;
  `alternate(..., include_prompt=True)` gives the prompt pass to A.
- A handoff overlaps or leaves a gap, not both. One where A would push
  nothing (no prompt pass, switch at 0) is refused.
- `alternate` lists every turn as a segment, the last cut short at end, so
  it needs an end: to alternate to the end of the reply, give the reply's
  max_new_tokens. Its period counts reply tokens.

**YAML.** A schedule is a mapping: `name`, optional `allow_over_ceiling`,
and either `segments` or `shape` with `params`. A segment takes `unit`,
`alpha` (a constant push) or `alpha_start` and `alpha_end` (a ramp),
`start` (default 0), `end` (default null) and `region` (default `reply`).
Unknown keys, params, shapes and units, a key given twice, and a number
YAML read as text (`1e-1`, or a quoted number) are refused with a message
that names them. `to_yaml` always writes the segments out, every field
given, so `from_yaml(to_yaml(s)) == s` with every dose bit for bit, and a
file in either form reads back to the same Schedule the helpers build.

**validate(schedule, model_info, max_new_tokens, ceiling=1.4).**
- model_info is the model's registry entry (a ModelSpec): a unit of
  another model is unknown. A ModelInfo, which has no name, checks only
  that each unit's model has this shape and that the units name one model.
- It refuses, every problem listed in one ValueError: unknown units;
  layers and neurons out of range; a reply segment whose end is not after
  its start; a prompt segment with a start or end, or a ramp; a ramp with
  no end, or of one reply token; and a combined push above the ceiling.
- **The combined push** in a layer on a pass is the square root of the
  summed squares of its units' doses there, each unit's dose being the sum
  of its segments on that pass, as the hook adds pushes (decision 0008).
  So two pushes of 1 in one layer give 1.41 and are refused; in two layers
  they pass; +1 and -1 of one unit on the same tokens count as 0, and two
  +1 segments of one unit as 2. A push exactly at the ceiling passes.
- **Every pass is checked:** the prompt pass and every reply token. Between
  two tokens where a segment starts or ends, each dose moves in a straight
  line, so the combined push is highest at one end of the stretch, and
  checking those ends checks every token, however long the reply.
  `layer_peaks` gives each layer's highest push and the first pass with it;
  a test compares it with a scan of every pass in 400 random schedules.
- The verdict does not depend on max_new_tokens. Only the warnings do: a
  reply segment whose end is past max_new_tokens, or whose start is past
  the last pass (max_new_tokens - 2), is accepted with a ScheduleWarning.
  validate both issues each warning and returns them as a list.

## Why this shape

- **alpha_end on the last token,** rather than the dose the line would
  reach at end: every dose written in a schedule is a dose some pass gets,
  so the ceiling check reads as the file does. A 0-to-1 ramp then a hold
  at 1 is `ramp(A, 0, 1, 10, 30)` and `onset(A, 1, 30)`.
- **The prompt pass counts for the ceiling,** although the prompt asked
  for reply tokens: the prompt pass is pushed like any other, and Liu's
  collapse point is about the size of a push, not when it comes.
- **1.4 lives in code,** against CLAUDE.md's rule that doses belong in
  `experiments/`, because the prompt asked for it as validate's default. It
  is a guard, not a dose that an experiment chooses, and validate takes any
  other ceiling. Liu measured 1.65 for Qwen3.5-4B; each model's own ceiling
  should replace it once the dose ladder measures one.
- **A ModelSpec rather than only a ModelInfo,** because a shape cannot tell
  models apart: L14 N8949 is in range for Qwen3-8B too (decision 0006).
- **Warn and return.** A script that ignores the result still shows the
  warnings; a command can print the list in its own words and ignore
  ScheduleWarning.
- **Problems all at once:** a schedule file is fixed in one pass.

## Consequences

- The engine does not follow schedules yet. The next step can build its
  per-pass amounts from `Schedule.doses_at`, which lists a unit pushed at
  dose 0 apart from one not pushed, and should call `validate` first.
- An alternation every token over 600 tokens is 600 segments. A compact
  stored form for runs, and S1's run records read as schedules, are left
  for the steps that store schedules.
- Every model gets the same ceiling until one is measured, and a schedule
  over it says so only through `allow_over_ceiling`; no run records it yet.
- Schedule names follow experiment names (lowercase letters and digits
  joined by hyphens), so they can go into run IDs and folder names.
