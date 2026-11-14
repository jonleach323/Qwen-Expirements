# 0023: A panel of up to 24 personas, chosen stage by stage from experiment files that name their units by rule

Date: 2026-11-13 · Sprint: S3 · Status: accepted

## Context

- The prompt for this step ("Sprint S3, prompt 5 of 8", pasted) asks for
  experiment files and selection rules that turn hundreds of candidates
  (decision 0020) into a panel of 24 strong, distinct, reliable personas:
  `prompts/heldout32.jsonl`, 32 everyday prompts never used before;
  `experiments/s3/library_repro.yaml`, 50 library personas sampled across
  layers on "Who are you?" with the library's exact settings and per-prompt
  calibration; `experiments/s3/screen_a.yaml`, every candidate at its
  reference dose on 8 held-out prompts; `pushlab panel stage b`, `ladder`
  and `preview`, each writing the next experiment from the last one's
  scores; `pushlab panel select`, which keeps up to 24 personas that speak
  as themselves on at least 80% of the 32 prompts at their working dose,
  meet the coherence threshold, are told apart by the which_persona rubric
  at least 90% of the time and spread across categories, and writes
  `units/panel_qwen3-8b.yaml` and `docs/panel/qwen3-8b.md`; and golden
  tests for Qwen3-8B. Fewer than 24 that qualify are all kept, if at least
  16, with advice on what would help.
- The Neuron Persona Library states no licence, so none of its data, its
  neurons' IDs and personas' names included, goes into git (decision
  0020). Its candidates live in `units/candidates_qwen3-8b.csv`, made on
  the pod and ignored by git.
- Sets of neurons work in Python but not yet in schedule and experiment
  files, the runner, `verify` and `report` (decision 0019). Mayner's
  candidates are sets.
- The per_prompt calibration mode works in `generate` but not yet in the
  runner (decision 0019); the library measures each prompt's own residual
  sizes.
- Candidates carry no category: the library's rows are all "persona" and
  Mayner's "role" (decision 0020).

## Decision

**Sets in schedule files and run records** (`pushlab.schedules.yaml_io`).
A segment that pushes a set gives the set's ID as its unit and its
definition beside it:

```yaml
- unit: qwen3-8b:set:pirate-top50
  set: {entries: [[17, 1234, 0.52], [18, 99, -0.31]], size: 96.04, source: mayner role pirate}
  alpha: 1.0
```

`to_mapping` writes every set this way and `from_mapping` reads it back,
so a schedule with sets round-trips, every number bit for bit; a set ID
without its definition, or a definition beside a neuron, is refused. A
shape's params cannot hold a definition, so a schedule that pushes a set
is written as segments. Run records keep their segments the same way, so
the runner, `verify` and `report` rebuild each set from its record alone.
A set's ID names one set in the whole experiment: two definitions of one
ID are refused. Random controls for sets are still refused, and a
neuron's random control never lands on a neuron a set of the experiment
pushes.

**Experiment files** (`pushlab.runs.spec`, `conditions`, `expand`) take
three optional settings more:

- `calibration: per_prompt` measures each prompt's own residual sizes, as
  the library does; `fixed`, the default, is decision 0007's. The runner
  measures each prompt once a session, with no push, before the first
  batch that needs it, and `verify` checks each unit's amounts per prompt;
  a set with its own size is scaled by that one number either way.
- `prompt_ids: [...]`: which prompts of the set to run, in that order,
  each once.
- `candidates:` names a candidates list (`file`, from the repository's
  root or absolute) and which of its rows to use: `source` (library or
  mayner) and `kind` (neuron or set) filter it; `keys` lists exactly these
  candidates, in order, each a key or `{key: K, doses: [...]}`; `sample:
  {count: N, across: layer, seed: S}` draws N neurons spread across
  layers. A condition with `each_candidate:` becomes one condition per
  candidate, labelled `<label>-0001` on, each a constant push of its unit
  in its direction at `doses`, multiples of its reference dose (default
  `[1.0]`; a key's own `doses` win), prompt pass included unless
  `include_prompt: false`, with `allow_over_ceiling` as a condition takes
  it. Several doses make a grid, labelled by alpha, as `pushlab ladder`'s.

So a file in the repository names its units by rule and holds none of the
library's data; its runs are the same wherever the list is the same, and
the spec's hash covers every unit it resolves to. On a machine without
the list, `expand` says where it is made. The new settings enter the
spec's hash only when they differ from their defaults, so every earlier
experiment file keeps its hash.

**The repository's files.** `experiments/s3/library_repro.yaml`: 50
library neurons sampled across layers (seed 0), each at the library's
strength in its direction, on `prompts/whoareyou.jsonl` ("Who are
you?"), greedy, 160 tokens, thinking off, per_prompt.
`experiments/s3/screen_a.yaml`: every candidate at its reference dose
(the library's strength for a neuron, Mayner's for a set, which is dose
1) on the held-out set's first 8 prompts, greedy, 160 tokens, thinking
off, fixed calibration as every later stage. Generation always runs with
thinking off (decision 0009).

**The stages** (`pushlab panel stage`), each reading the last stage's
experiment, run and scored with `pushlab score SPEC --rubric speaks_as`,
which asks each reply about its own candidate's persona from the
experiment's candidates list (as `--units <candidates list>` does; a
candidate whose persona is named "none", which_persona's answer for no
persona, is left out, so no stage keeps it), and with coherence where a
stage needs it, and
writing the next experiment outside the repository, as it names the
library's units:

1. `b`: screen_a's 120 candidates with the highest speaks_as rates, ties
   to the more verdicts, then the list's order, at their reference dose
   on all 32 held-out prompts.
2. `ladder`: stage b's 40 best, at 0.5, 0.75, 1, 1.25 and 1.5 times their
   reference dose, on the first 16 prompts.
3. `confirm`: each ladder candidate with a working dose below any collapse
   dose (decision 0022's analysis), at its working dose, on all 32
   prompts. The prompt's rule asks for 32 prompts at the working dose,
   which the ladder's 16 cannot show, so this stage is added.
4. `select` (below), then `preview`: 6 varied pairs of the panel, each in
   four shapes at the personas' working doses, on the last 16 prompts: a
   handoff at reply token 20, alternation every 8 tokens, a 16-token pulse
   of the first, and the second from reply token 30.

Sizes, doses, prompt counts and timings are configs/panel.yaml's
`selection` section (`pushlab.panel.settings`), version 2.

**Selection** (`pushlab panel select CONFIRM --ladder LADDER`). It first
refuses the two files given the wrong way round: one experiment for both,
a confirm that pushes a candidate at several doses, or a ladder that
pushes each at one (the settings ask for two or more ladder doses). Each
candidate's working and collapse doses come from the ladder; a candidate
fails the first rule it breaks: no working dose below any collapse dose;
no reply in stage confirm at that working dose; a reliability, its
speaks_as share over the 32 prompts, a prompt without a verdict counting
against it, below 0.8; a mean coherence below 6. Of finalists whose
personas have one name, case and spacing aside, only the strongest stays,
since which_persona tells personas apart by name. With fewer than 16
finalists, select stops there, writing nothing, and says what would help.
Each finalist's category comes next, from a categories file the owner
fills in, `key,category`, one of the five; select writes it with the
categories blank when it is missing or unfinished, and stops. The
which_persona rubric then sees a sample of up to 8 of each finalist's
replies that speak as it, drawn with a seed and the finalist's key (so
one finalist's sample does not depend on who else is a finalist), with
every finalist as an option; a finalist named for fewer than 90% of them
is dropped, so of two personas the judge confuses, the one whose replies
are taken for the other goes. Without `--yes`, select prints the calls
this takes, how many are already scored or in the cache, and their cost,
and stops, having called and written nothing; with it, it judges, saves
the verdicts in the confirm folder's scores table, and goes on. Of those
left, select takes up to 24, round by round across the categories, the
strongest first (reliability, then coherence, then distinctness), within
a category the kind the panel has fewer of. Fewer than 24 are all kept if
at least 16, with advice from the rule most of the others failed (such as
a higher or lower dose, larger sets, or more candidates) and the
categories left thin; fewer than 16 write nothing. It writes the panel
file (`pushlab.panel.panel_file`) and a page with two example lines per
persona, both ignored by git, as they name the library's personas.

**Golden tests** (`tests/golden/qwen3-8b.yaml`,
`tests/gpu/test_golden_qwen3_8b.py`, `pushlab golden record`): 18 cases,
3 prompts each unpushed and greedy, pushed by a made-up neuron under
fixed and per_prompt calibration, by a made-up set, and sampled at two
seeds, 32 tokens in bfloat16; each reply's fingerprint is the SHA-256 of
its token IDs, stored with the environment that made it. They are
recorded on the pod, outside the repository, and copied into
`tests/golden/qwen3-8b.json` through a pull request; until then the GPU
test skips, saying so.

## Why this shape

- **Units by rule,** not by name: the repository keeps the library's data
  out of git and still holds the experiments, which reproduce from the
  list they name.
- **Sets beside their IDs** in segments: a run record rebuilds its set
  without the list, and the spec's hash covers every coefficient.
- **A confirm stage:** a working dose chosen on 16 prompts is checked on
  all 32, as the selection rule asks.
- **A categories file:** the sources have no categories, and 40 finalists
  are quicker to sort by hand than a judge is to check.

## Consequences

- Stage files, the panel and its page live on the pod, outside git.
- A ladder of sets is written by `pushlab panel stage ladder`;
  `pushlab ladder` still takes single neurons.
- The golden test passes only once its fingerprints are recorded.

## Results

- Built in nine commits on `claude/focused-heisenberg-faly2w`; the CPU
  suite passes (`uv run pytest -q`: about 4,000 tests, the GPU tests
  skipped).
- `pushlab expand --summary` on a made-up list of 850 candidates (575
  neurons over 36 layers and 275 sets, deleted afterwards, as the real
  list is made on the pod) gave library_repro 50 runs in 4 batches and
  screen_a 6,800 runs (850 candidates times 8 prompts) in 432 batches.
- A review of the finished code found five faults, fixed before the last
  commit: stage ladder's hint pointed to `pushlab doses` with the whole
  candidates list, which that command refuses (stage confirm prints the
  doses instead); `--out`'s help said stage files go outside the
  repository only sometimes; `stage preview` offered a `--judge` it then
  refused; select took a ladder and a confirm given the wrong way round;
  and this record left out the name rule.
- Not yet run on the pod: the candidates list, Qwen3-8B's calibration,
  the stages, the panel and the golden fingerprints are the owner's to
  make (docs/runbooks/panel.md).
