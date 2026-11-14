# Choose the persona panel on the pod

Steps from the candidates list to a panel of up to 24 strong, distinct,
reliable personas for Qwen3-8B (decision 0023). Each stage runs the
candidates left by the stage before on held-out prompts
(`prompts/heldout32.jsonl`), and `pushlab panel` writes the next stage's
experiment file from its scores. The sizes, doses and prompt counts are
`configs/panel.yaml`'s `selection` section. Every run is a GPU job, the
owner's to start, in tmux (docs/runbooks/experiments.md); every judging
step calls the judge, so look at its `--dry-run` cost first.

The stage files, the panel and its page name the Neuron Persona
Library's personas, which stay out of git (decision 0020): the stage
files go under `/workspace/experiments/`, and the panel
(`units/panel_qwen3-8b.yaml`) and its page (`docs/panel/qwen3-8b.md`) are
ignored by git.

## 0. Before you start

- The candidates list, `units/candidates_qwen3-8b.csv`: `uv run pushlab
  fetch library`, `uv run pushlab fetch mayner`, then `uv run pushlab
  candidates --model qwen3-8b --out units/candidates_qwen3-8b.csv`
  (decision 0020).
- Qwen3-8B's calibration, which every stage after the library's check
  uses: `uv run pushlab calibrate --model qwen3-8b`.
- A judge that passed the labels check for speaks_as, coherence and
  which_persona (docs/runbooks/judge.md).
- A folder for the stage files:

```bash
mkdir -p /workspace/experiments
```

## 1. Check the library is reproduced (optional)

`experiments/s3/library_repro.yaml` pushes 50 of the library's neurons,
sampled across layers, as the library did: on "Who are you?", greedy,
160 tokens, at the library's strengths, each prompt's own residual sizes
(per_prompt calibration). It needs no calibration file.

```bash
uv run pushlab expand experiments/s3/library_repro.yaml --summary
```

```bash
uv run pushlab run experiments/s3/library_repro.yaml
```

```bash
uv run pushlab score experiments/s3/library_repro.yaml --rubric speaks_as --dry-run
```

Each reply is judged about its own candidate's persona, which the
experiment's candidates list gives (as `--units
units/candidates_qwen3-8b.csv` would). Leave out `--dry-run` to judge.

## 2. Screen every candidate

`experiments/s3/screen_a.yaml` pushes every candidate at its reference
dose (the library's strength, or Mayner's dose 1 for a set) on the first
8 held-out prompts. `expand --summary` prints its run count, the list's
candidates times 8, and `run --dry-run` its hours and cost; raise its
`max_gpu_hours` through a pull request if the list is longer than about
850.

```bash
uv run pushlab expand experiments/s3/screen_a.yaml --summary
```

```bash
uv run pushlab run experiments/s3/screen_a.yaml
```

```bash
uv run pushlab score experiments/s3/screen_a.yaml --rubric speaks_as
```

## 3. Stage b: the 120 best, on every prompt

```bash
uv run pushlab panel stage b experiments/s3/screen_a.yaml --out /workspace/experiments/s3-stage-b.yaml
```

It prints the candidates it kept with their speaks_as rates, and what to
run next: the file, then its speaks_as scores, as in step 2.

## 4. Stage ladder: the 40 best, at five doses

```bash
uv run pushlab panel stage ladder /workspace/experiments/s3-stage-b.yaml --out /workspace/experiments/s3-stage-ladder.yaml
```

Each candidate runs at 0.5, 0.75, 1, 1.25 and 1.5 times its reference
dose on the first 16 prompts. Run it, then judge speaks_as, as in step 2,
and coherence:

```bash
uv run pushlab score /workspace/experiments/s3-stage-ladder.yaml --rubric coherence
```

Stage confirm, next, prints each candidate's working and collapse
doses.

## 5. Stage confirm: each that works, at its working dose

```bash
uv run pushlab panel stage confirm /workspace/experiments/s3-stage-ladder.yaml --out /workspace/experiments/s3-stage-confirm.yaml
```

It keeps each candidate with a working dose below any collapse dose, at
its working dose on all 32 prompts. Run it, then judge speaks_as and
coherence, as in step 4.

## 6. Select the panel

```bash
uv run pushlab panel select /workspace/experiments/s3-stage-confirm.yaml --ladder /workspace/experiments/s3-stage-ladder.yaml --categories /workspace/experiments/categories.csv
```

The first time, it writes the categories file, one line per finalist with
its category blank: fill in each one with people, animals, fictional,
objects or mystical, and run it again. It then prints how many
which_persona calls its distinctness check takes, how many the cache
holds, and their cost, and stops; add `--yes` to make them. It writes the
panel and its page, or, if fewer than 16 personas pass every rule,
nothing, with what would help.

## 7. Preview the panel's pairs

```bash
uv run pushlab panel stage preview units/panel_qwen3-8b.yaml --out /workspace/experiments/s3-stage-preview.yaml
```

Six varied pairs, each pushed four ways (a handoff at reply token 20,
alternation every 8 tokens, a 16-token pulse of the first, and the second
from token 30) on the last 16 prompts. Run it, then read its replies with
`pushlab report`, as the command prints.

## 8. Golden tests

Before a long run, and after any change to the engine or the pod's
libraries, the golden tests check that Qwen3-8B still gives the same
replies (docs/runbooks/pod.md, "Golden tests").
