# Climb a dose ladder on the pod

Steps to find each persona's working dose, where it reliably speaks as
the persona, and its collapse dose, where its text breaks (decision
0022). Pushing a neuron harder makes a persona surer, then breaks the
model's writing; a ladder runs every unit at a few doses to find both
points. Every step here reads or writes run folders on the pod; the GPU
steps are the owner's to start.

A judge should pass the labels check first (docs/runbooks/judge.md):
the ladder trusts its speaks_as and coherence verdicts.

## 1. Choose the units

A units file is a CSV with a `unit` column (a neuron's ID) and,
optionally, `direction` (+1 or -1), `persona`, `description` and
`source`. A list of persona candidates is one already, so copy the rows
you want from it into a file of their own, on the pod:

```bash
mkdir -p /workspace/units /workspace/experiments /workspace/reports
```

```bash
head -1 units/candidates_qwen3-8b.csv > /workspace/units/panel-1.csv
```

then add the rows you choose (any editor, or `grep` for the persona
names). Library personas stay off git (decision 0020), so keep the file
under `/workspace`. `pushlab ladder` takes single neurons: a set's row is
refused, and `pushlab panel stage ladder` ladders sets (decision 0023,
docs/runbooks/panel.md).

## 2. Write the ladder

```bash
uv run pushlab ladder --units /workspace/units/panel-1.csv --doses 0.25,0.5,0.75,1.0,1.25 --prompts prompts/neutral32.jsonl --out /workspace/experiments/panel-1-ladder.yaml --baseline
```

It writes one condition per unit, `u01`, `u02` and so on, each pushing
its unit at every dose in its direction, and prints how many runs that
is: units × doses × prompts, plus the unpushed baseline. A ladder of
library units must be written outside the repository. Any prompt set
works: `prompts/neutral32.jsonl` gives each dose 32 replies, enough for a
rate within about ±15 points, and `prompts/capability100.jsonl` measures
too whether the model can still answer simple questions at each dose.

## 3. Run it

```bash
uv run pushlab run /workspace/experiments/panel-1-ladder.yaml --dry-run
```

shows the runs, GPU-hours and cost. Then run it in tmux as
docs/runbooks/experiments.md says, and check it with `pushlab verify`.

## 4. Score it

Each unit's replies are asked about its own persona, the same question at
every dose, so the judge never sees the dose:

```bash
uv run pushlab score /workspace/experiments/panel-1-ladder.yaml --rubric speaks_as --units /workspace/units/panel-1.csv --dry-run
```

then the same without `--dry-run` (add `--personas FILE` to give the
personas' descriptions). Then coherence, and the measures that need no
judge:

```bash
uv run pushlab score /workspace/experiments/panel-1-ladder.yaml --rubric coherence
```

```bash
uv run pushlab measure /workspace/experiments/panel-1-ladder.yaml
```

`pushlab measure --perplexity` also scores each reply's perplexity under
the unpushed model; it loads the model, so run it only while no other job
uses the GPU.

## 5. Read the doses

```bash
uv run pushlab doses /workspace/experiments/panel-1-ladder.yaml --units /workspace/units/panel-1.csv --out /workspace/reports/panel-1-doses.json
```

For each unit it prints, at every dose, the share of replies that speak
as the persona and the mean coherence, each with its interval; then the
working dose, the lowest dose where at least 80% speak as the persona,
and the collapse dose, the lowest where mean coherence falls below 6
(`configs/panel.yaml`), each with an interval from resampling the
prompts. An interval that ends "beyond the ladder" means the ladder did
not reach that dose often enough: climb further. Paste the printed
report into the cloud session; the JSON stays on the pod.
