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
  - six helpers, among them `alternate(a, b, alpha_a, alpha_b, period,
    start=0, end=None)`;
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

**Code.** `src/pushlab/schedules/`: `model.py` (Segment, Schedule,
`describe_pass`, `dose_text`), `shapes.py` (the six helpers and `SHAPES`),
`yaml_io.py` (`from_yaml`, `to_yaml`, `from_mapping`, `to_mapping`,
`read_schedule`), `yaml_text.py` (`load_yaml`, `dump_yaml`) and
`validation.py` (`validate`, `layer_peaks`, `DEFAULT_CEILING`,
`MOST_PASSES`, `ScheduleWarning`, `TooManyPasses`). None of it loads a
model.

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

**Segment.** The prompt's six fields, then two optional ones, `on_for` and
`off_for`: a reply segment that repeats is on for on_for tokens from its
start, off for off_for, on again, until its end. Each field is checked on
its own when a Segment is made: a NeuronUnit, finite doses (kept as
floats), start a whole number from 0, end None or a whole number from 0,
region `prompt` or `reply`, and on_for and off_for both None or both
whole numbers from 1. Other regions, such as thinking, come with thinking
detection. How the fields fit together is for `validate`.

**The ramp formula.** On reply token k, with t = (k - start) / (end - 1 -
start), a segment's dose is alpha_start + (alpha_end - alpha_start) * t,
worked as alpha_start * (1 - t) + alpha_end * t and kept between the two
ends, so that alpha_start falls on its first token and alpha_end on its
last (end - 1) exactly, with equal steps between and no dose past an end. Equal doses make a constant push. A ramp needs an end and
at least two reply tokens, and a segment that repeats, or a prompt
segment, cannot ramp. By hand:

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
| `handoff(A, B, 1, -1, 0)` | A on the prompt pass only; B from 0 |
| `alternate(A, B, 1, -1, 4, 10)` | A from 10, 4 on then 4 off; B from 14, 4 on then 4 off |
| `alternate(A, B, 1, -1, 4, 10, 26)` | the same, ending before 26: A on 10 to 13 and 18 to 21, B on 14 to 17 and 22 to 25 |
| `ramp(A, 0, 1, 10, 30)` | A from 0 to 1 on tokens 10 to 29 |

- constant and handoff start with the reply, so they push the prompt pass
  by default, as the evil-Qwen recipe does; before its switch, a handoff
  pushes exactly what `constant(A)` does. The shapes that start at a reply
  token you choose (pulse, onset, alternate, ramp) push only reply tokens;
  `alternate(..., include_prompt=True)` gives the prompt pass to A.
- A handoff overlaps or leaves a gap, not both. One where A would push
  nothing (no prompt pass, switch at 0) is refused, and so is an
  alternation that ends before B's first turn.
- An alternation is one repeating segment per unit, so `end=None`, the
  default, runs it to the end of a reply of any length, and a record holds
  two segments however short the period. Its period counts reply tokens.
- Parameters that would push nothing, or not what the shape's name says,
  are refused when the shape is built: a pulse of no tokens, a ramp of one
  token, a period of 0, a start or end that is not a reply token.

**YAML.** A schedule is a mapping: `name`, optional `allow_over_ceiling`,
and either `segments` or `shape` with `params`. A segment takes `unit`,
`alpha` (a constant push) or `alpha_start` and `alpha_end` (a ramp),
`start` (default 0), `end` (default null), `region` (default `reply`) and,
when it repeats, `on_for` and `off_for`. Unknown keys, params, shapes and
units, and a number given as text, are refused with a message that names
them, by the key the file used. `yaml_text.load_yaml` reads the text: a
key given twice in one mapping is refused, as each mapping is read in, so
one only merged in (`<<`) is checked too, while a key may still override
one that a merge brings in; and numbers and true/false follow YAML 1.2, so
`010` is not 8, `1:20` is not 80, `1e-1` is a number, `yes` and dates are
text, and an explicit tag such as `!!int 010` is refused. `to_yaml` always
writes the segments out, through `dump_yaml`, which quotes text either
YAML would read as something else (a name `1e-3`), so
`from_yaml(to_yaml(s)) == s` with every dose bit for bit, the sign of a
zero included, and a file in either form reads back to the same Schedule
the helpers build.

**validate(schedule, model_info, max_new_tokens, ceiling=1.4).**
- model_info is the model's registry entry (a ModelSpec): a unit of
  another model is unknown. A ModelInfo, which has no name, checks only
  that each unit's model has this shape and that the units name one model.
- It refuses, every problem listed in one ValueError: unknown units;
  layers and neurons out of range; a reply segment whose end is not after
  its start; a prompt segment with a start or end, or that ramps or
  repeats; a ramp with no end, of one reply token, or that repeats; and a
  combined push above the ceiling. allow_over_ceiling lifts that last rule
  and no other.
- **The combined push** in a layer on a pass is the square root of the
  summed squares of its units' doses there, each unit's dose being the sum
  of its segments on that pass, as the hook adds pushes (decision 0008).
  So two pushes of 1 in one layer give 1.41 and are refused; in two layers
  they pass; +1 and -1 of one unit on the same tokens count as 0, and two
  +1 segments of one unit as 2. A push exactly at the ceiling passes; one
  too large for a float is refused.
- **Every pass is checked:** the prompt pass and every reply token, each
  layer on its own. The tokens where a segment starts or ends cut the
  reply into stretches with the same segments present; those that repeat
  line up again every lcm of their cycles, L tokens, and along tokens L
  apart every dose is constant or moves in a straight line, so the push is
  highest at the first or the last of them. The first and last L tokens of
  each stretch are therefore enough, and past the last start or end the
  first L are. `layer_peaks` gives each layer's highest push and the first
  pass with it, exact up to rounding in the last digit where ramps add up;
  tests compare it with a scan of every pass in 400 random schedules. A
  layer whose repeating segments would need more than `MOST_PASSES`
  (1,000,000) passes is refused with `TooManyPasses`: it takes several
  alternations of large periods with no common factor in one layer.
- The verdict does not depend on max_new_tokens. Only the warnings do: a
  reply segment that starts past the last pass (max_new_tokens - 2), that
  would push a token past max_new_tokens (for one that repeats, an on-turn
  there), or whose ramp ends on the reply's last token, which no pass
  reads, is accepted with a ScheduleWarning. validate both issues each
  warning and returns them as a list.

## Why this shape

- **Repeating segments.** The prompt's `alternate(..., end=None)` means to
  the end of the reply, and a list of turns with fixed ends cannot say that
  without the reply's length. Two optional fields keep the prompt's six,
  let `end=None` mean what it says, and keep an alternation every token
  over 600 tokens to two segments in every record.
- **alpha_end on the last token,** rather than the dose the line would
  reach at end: the doses written in a schedule are the doses passes get,
  as long as the reply is long enough (validate warns when it is not), so
  the ceiling check reads as the file does. A 0-to-1 ramp then a hold at 1
  is `ramp(A, 0, 1, 10, 30)` and `onset(A, 1, 30)`.
- **The prompt pass counts for the ceiling,** although the prompt asked
  for reply tokens: the prompt pass is pushed like any other, and Liu's
  collapse point is about the size of a push, not when it comes.
- **1.4 lives in code,** against CLAUDE.md's rule that doses belong in
  `experiments/`, because the prompt asked for it as validate's default. It
  is a guard, not a dose that an experiment chooses, and validate takes any
  other ceiling. Liu measured 1.65 for Qwen3.5-4B; each model's own ceiling
  should replace it once the dose ladder measures one.
- **A ModelSpec as well as a ModelInfo:** a ModelSpec names the model, so a
  unit of another model is unknown even when the two have the same shape
  (as `tiny` and `tiny-lm` do; decision 0006).
- **YAML 1.2's numbers:** in a project about timing, `start: 010` read as 8
  would move a window without a word. Dates go too: no schedule field is a
  date, and a date-shaped name or number is clearer refused as text.
- **Warn and return.** A script that ignores the result still shows the
  warnings; a command can print the list in its own words and ignore
  ScheduleWarning.
- **Problems all at once:** validate lists every problem of a schedule it
  can read. Reading stops at the first problem in the file itself.

## Consequences

- The engine does not follow schedules yet. The next step can build its
  per-pass amounts from `Schedule.doses_at`, which lists a unit pushed at
  dose 0 apart from one not pushed, and should call `validate` first.
- An experiment file that holds a schedule should be read with
  `yaml_text.load_yaml`, so its keys and numbers are read as a schedule
  file's are; it does not belong to schedules alone, and can move when the
  experiment files come.
- A run's stored schedule is left for the steps that store schedules. The
  segments written here are not S1's records: `pushlab report`'s
  `describe_push` reads `include_prompt` and `alpha`, so it must learn
  `region`, `alpha_start`, `alpha_end`, `on_for` and `off_for`, or refuse
  them, before these are stored as they are.
- Every model gets the same ceiling until one is measured, and a schedule
  over it says so only through `allow_over_ceiling`; no run records it yet.
- Schedule names follow experiment names (lowercase letters and digits
  joined by hyphens), so they can go into run IDs and folder names.
