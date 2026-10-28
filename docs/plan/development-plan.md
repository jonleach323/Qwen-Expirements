# Switching Neurons Mid-Response: Development Plan

Oct 28, 2026 · @Jon

## Summary

This plan turns the Switching Neurons Mid-Response: Research Plan into working software: one Python package, a set of command-line jobs that run on a rented GPU, and a public explorer. It is built in 23 sprints over the same 49 weeks, so each research phase starts only after the code it needs passes its tests.

What gets built, in order:

1. **Push engine.** Adds a scheduled push to chosen neurons, sparse-autoencoder features or dense vectors during generation, with a different schedule for every row of a batch. Working name: `pushlab`.
2. **Run pipeline.** Experiment files expand into individual runs; a resumable runner executes them and saves every result.
3. **Scoring stack.** Capability, coherence, divergence, persona timelines, AI judges with a validation harness, and internal readouts.
4. **Analysis library.** Statistics, interaction scores, standard plots and one report notebook per phase.
5. **Release.** The run dataset, an interactive explorer and documentation, as version 1.0 and later version 2.0.
6. **Style toolkit.** Style scores, a meaning-kept score and style-unit import for Phase 7.
7. **Exploration toolkit.** An idea registry and pilot templates, plus harnesses for multi-turn chats, two-model dialogues, self-knowledge menus, persona maps and adapter training.
8. **Repair toolkit.** A failure-bank builder, hidden-knowledge probes, a candidate search with held-out evaluation, and harm checks for the final challenge.

Three rules hold throughout. Nothing is trusted until a test proves it: the engine must reproduce three published effects (Gate 0) before any new experiment runs. Every result traces back to an experiment file, a code version and a model version. Claude Code writes most of the code, but every change lands with tests you run and read before committing.

Defaults carried over from the research plan: 12 to 15 hours a week, Qwen3.5-4B and Qwen3-8B as the main models, a rented RTX 4090 as the main machine, and development done entirely in the cloud, so no local setup is required.

Done means: unit tests pass on every commit, replication tests pass on the GPU, every research phase can be rerun from its files, and version 1.0 of the code, dataset and explorer is public by week 27, with version 2.0 by week 49.

## Architecture

*(Diagram: system architecture · 6 stages, 4 supporting parts. See the Claude Doc version of this plan to view it.)*

Only the experiment file is written by hand. Everything below it is generated, so any result can be rebuilt from its file and code version.

## Environment and stack

Everything runs in the cloud. Code is written and unit-tested in Claude Code cloud sessions against a GitHub repository, and experiments run on a rented GPU pod that pulls the same repository. A browser or the Claude mobile app is enough to steer both.

### Machines and accounts

| Account | Used for | Notes |
| --- | --- | --- |
| GitHub | Code, issues, pull requests, automated tests | Private repository until release |
| Claude plan with Claude Code | Writing code and running CPU tests in [cloud sessions](https://code.claude.com/docs/en/web-quickstart.md) | Cloud sessions need a GitHub repository and count toward plan usage |
| RunPod (or a similar GPU cloud) | Running experiments | RTX 4090 from $0.34 an hour, A100 80GB from $1.39 ([Sept 2026](https://www.hostinger.com/au/tutorials/cloud-gpu-pricing/)) |
| Hugging Face | Downloading models; later hosting the dataset | Accept Meta's Llama license before Phase 5 |
| Storage | Keeping outputs between pod sessions | A persistent pod volume plus a nightly copy to a private Hugging Face dataset repository |
| API account (optional) | A paid validation judge | Only if the research plan's open question on judging lands that way |

### Software, pinned

- **Python 3.11** with **uv** and a committed lockfile, so every pod builds the same environment.
- **PyTorch** with the CUDA build that matches the pod image.
- **Hugging Face transformers 5.2 or later**, which [Qwen3.5 requires](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/vlm/qwen35-vl.html), plus accelerate and safetensors.
- **flash-linear-attention and causal-conv1d**, the fast kernels for Qwen3.5's linear-attention layers; without them those layers fall back to slow code.
- **vLLM** in a separate environment, for fast unsteered baselines and the local judge, because it pins its own library versions.
- **Data and statistics:** pandas, pyarrow, DuckDB, NumPy, SciPy, statsmodels and matplotlib.
- **Scoring:** sentence-transformers for text embeddings.
- **Quality tools:** pytest, ruff for linting and formatting, and pre-commit to run both before every commit.
- **Claude Code CLI on the pod**, installed with the [native installer](https://code.claude.com/docs/en/setup) (`curl -fsSL https://claude.ai/install.sh | bash`), for debugging problems that only show up on the GPU.

### Model facts that shape the code

- **Qwen3.5-4B** has [32 layers and a hidden size of 2,560](https://huggingface.co/unsloth/Qwen3.5-4B-Base), arranged as eight repeats of three linear-attention (Gated DeltaNet) blocks followed by one full-attention block. Every block has its own MLP, so neurons in all 32 layers can be pushed.
- **It ships as a vision-language model,** so its language layers sit one level deeper in the module tree than in a text-only model. The engine finds MLP modules by searching the model rather than hard-coding paths.
- **Two kinds of memory.** Qwen3.5 carries a recurrent state in its linear-attention layers and a key-value cache in its attention layers. "Rebuild memory from the same words" (question A3) therefore always means re-running the text with the push off, which works for both.
- **Qwen3-8B** is a standard transformer with 36 layers, a hidden size of 4,096 and 12,288 MLP neurons per layer.
- **Precision:** main runs use bfloat16 on both models, and the data type is recorded with every run.

### Additions for the later phases

- **Style and meaning:** a readability library for reading-level scores, a formality classifier, and an open entailment model that checks whether two replies say the same thing.
- **Probes and maps:** scikit-learn for hidden-knowledge probes and clustering, and UMAP for drawing persona maps.
- **Adapters:** Hugging Face PEFT for the push-versus-training pilot.
- **Base models:** the base (pre-chat) checkpoints of the main models for the base-versus-chat pilot. They have no chat template, so their prompts are plain text with stop sequences.

## Repository layout and conventions

One repository holds the package, experiment files, prompt sets, tests, notebooks and design notes; run outputs live outside it. The layout is fixed in Sprint 0 so Claude Code always knows where new code belongs.

```text
pushlab/
├── CLAUDE.md              rules Claude Code reads every session
├── README.md
├── pyproject.toml         package metadata and tool settings
├── uv.lock                pinned dependencies
├── src/pushlab/
│   ├── models/            loading, layer discovery, model registry
│   ├── units/             neurons, SAE features, dense vectors, Assistant Axis
│   ├── schedules/         schedule parsing, validation, per-step dose tables
│   ├── engine/            hooks, batched generation loop, teacher forcing
│   ├── runs/              experiment files, expansion into runs, runner
│   ├── store/             writing and reading run records
│   ├── scoring/           capability, coherence, divergence, timelines, judges
│   ├── readouts/          per-token internal measurements
│   ├── discovery/         library import, geometry, dose finding, axis
│   ├── style/             style and meaning-kept scores
│   ├── conversations/     multi-turn and two-model harness
│   ├── backlog/           idea registry and pilot commands
│   ├── repair/            failure bank, probes, candidate search
│   ├── training/          adapters that mimic a push
│   ├── analysis/          statistics, interaction scores, plots
│   └── cli.py             command-line entry points
├── experiments/           one YAML file per experiment, grouped by phase
├── prompts/               versioned prompt sets
├── units/                 curated unit lists, such as the library shortlist
├── ideas/                 the idea registry and one note per pilot
├── banks/                 failure bank and correct-answer bank
├── tests/
│   ├── unit/              fast CPU tests on a tiny random model
│   ├── golden/            fixed-output regression tests
│   └── gpu/               replication and batch-invariance tests
├── notebooks/             one report notebook per phase
├── site/                  the public explorer's code
├── scripts/               pod setup, syncing, maintenance
├── docs/                  decisions, pre-registrations, data card
└── .github/workflows/     automated tests on every push
```

### Conventions

- **Indexing.** Layers and neurons are numbered from 0 everywhere, matching the published evil-neuron recipe; replicators found that counting from 1 gives no effect.
- **Identifiers.** A unit is `model:layer:index` (for example `qwen3.5-4b:L14:N8949`); a run is `<experiment>-<counter>`.
- **Configuration over code.** Doses, neurons, schedules and prompts live in experiment files. Code never hard-codes them.
- **Seeds.** Every experiment file sets its seeds; nothing random happens without one.
- **Branches.** The main branch is protected. Every change arrives as a pull request that must pass the automated tests.
- **Decisions.** Each design choice, such as which judge model to use, gets a short numbered note in `docs/decisions/`.
- **Pre-registration.** Each phase's predictions are committed in `docs/prereg/` before its runs, and the commit is recorded with the runs.
- **Outputs.** Run outputs are never committed. They live on the pod's persistent volume and are copied off nightly.
- **Readable code.** Type hints and a short docstring with an example on every public function; ruff formats and checks style automatically.

## Core modules

Fifteen modules make up the package: ten for the core program, then five for style, exploration and repair. Each one below lists what it does, its main interface, the design decisions that matter, and the test that proves it works.

### 1. models

- **Does:** loads a model and tokenizer, finds every MLP block, and renders prompts with the right chat template (thinking on or off).
- **Interface:** `load_model(name, dtype)` returns the model, tokenizer and a `ModelInfo` with layer count, hidden size, neurons per layer and the list of MLP modules.
- **Decisions:** a small registry lists supported models (Qwen3.5-4B, Qwen3-8B, Qwen3-4B, Qwen3.5-2B, later Llama-3.1-8B and Gemma). MLP blocks are found by searching for `down_proj`, which handles Qwen3.5's vision-language wrapper.
- **Done when:** each registered model loads and reports the right shape (32 × 9,216 neurons for Qwen3.5-4B, 36 × 12,288 for Qwen3-8B).

### 2. units

- **Does:** represents anything that can be pushed: a neuron, a sparse-autoencoder feature, a dense vector or the Assistant Axis.
- **Interface:** every unit exposes `write_direction()` and `amount(alpha)`. Neurons add to an MLP value before `down_proj`; features and vectors add to the residual stream after their layer.
- **Decisions:** dose follows the research plan's formula, using each layer's median residual size measured once per model by `calibrate_norms` and cached.
- **Done when:** pushing a neuron through its MLP value matches adding the same amount along its write direction, within bfloat16 tolerance.

### 3. schedules

- **Does:** turns a schedule (constant, pulse, late onset, handoff, alternation, ramp) into a table of exact amounts per batch row, per unit, per generation step.
- **Interface:** `Schedule.from_yaml()`, `validate(model_info)`, `compile(batch) -> DoseTable`.
- **Decisions:** each segment names a region (prompt, thinking, answer, or whole reply) so Track E can push only inside the thinking block. Validation rejects schedules whose combined push in one layer exceeds the model's measured collapse ceiling unless the file explicitly allows it.
- **Done when:** compiled tables match hand-worked examples, and property tests confirm no push ever lands outside its segment.

### 4. engine

- **Does:** runs generation with pushes applied, and re-scores existing text under any schedule.
- **Interface:** `generate(model, prompts, dose_table, decoding)` and `score_text(model, texts, dose_table)` for teacher-forced passes.
- **Decisions:** one hook per targeted layer adds the whole batch's amounts in a single vectorized step, so every row can follow a different schedule. A custom decoding loop (greedy or sampled, left-padded batches, cache reused between steps) keeps the step counter exact. Full next-token probabilities are never stored; divergence comes from a second teacher-forced pass.
- **Three continuation modes** serve question A3: keep the pushed memory, rebuild memory from the same tokens with the push off, or swap in neutral text.
- **Done when:** a dose of zero gives token-identical output to no hooks; each batch row matches the same run done alone; and the step counter stays correct with left padding.

### 5. runs

- **Does:** expands one experiment file into individual runs and executes them.
- **Interface:** `expand(spec) -> [RunRecord]`; `pushlab run spec.yaml [--dry-run] [--limit N]`.
- **Decisions:** run IDs are deterministic, so reruns skip finished work. Random control neurons are drawn from the same layer with a seeded generator, excluding known units. `--dry-run` prints run counts, estimated GPU-hours and estimated cost before anything starts.
- **Done when:** rerunning a finished experiment does nothing, and killing a job loses at most one batch.

### 6. store

- **Does:** writes and reads results in versioned Parquet files with a manifest per experiment.
- **Interface:** `write_runs()`, `write_steps()`, `write_scores()`, plus a DuckDB view across all experiments.
- **Decisions:** each manifest records the code commit, lockfile hash, model revision and GPU type. Writes are validated against the schema.
- **Done when:** data survives a write and read unchanged, and malformed records are refused.

### 7. scoring

- **Capability:** exact-match graders for the capability set.
- **Coherence:** perplexity under the unsteered model, repeated n-gram rate and word-variety scores.
- **Divergence:** per-token difference from the unsteered model, and the first token that differs.
- **Persona timeline:** sentences are embedded and compared with reference replies from single-unit runs, giving each sentence a probability per persona plus "normal assistant".
- **Judges:** one interface for every rubric (hostility, persona label, pair outcome class, self-correction, explanation class), served locally through vLLM with structured JSON output. An optional API adapter exists, and every verdict is cached by judge, rubric version and text.
- **Validation harness:** samples outputs into a labeling sheet, then reports accuracy and Cohen's kappa against your labels.
- **Done when:** each scorer passes tests on fixed examples, and no judge is used at scale before passing its validation threshold.

### 8. readouts

- **Does:** records, per token, how far the model's internal state points along each active unit's direction, the Assistant Axis and any probe.
- **Decisions:** only small scalars are stored per step, never full activations.
- **Done when:** stored values match a manual calculation on sampled tokens.

### 9. discovery

- **Library import:** reads the persona library export, removes duplicate names and applies filters.
- **Geometry pass:** cosines between write directions for every shortlisted pair, in one GPU matrix product.
- **Dose finding:** a bisection search for each unit's minimum working dose.
- **Assistant Axis:** derived per model with the published role-prompt method.
- **Contrastive screen:** finds candidate neurons for missing persona categories with forward passes only.
- **Done when:** identical units give a cosine of 1, sampled cosines match a reference calculation, and imported counts match the source file.

### 10. analysis

- **Does:** Wilson and bootstrap intervals, mixed-effects models, false-discovery-rate control, the independence-based interaction score, half-life fits and standard plots.
- **Interface:** one report notebook per phase that reads only from the store.
- **Done when:** every statistic matches published reference values in tests.

### 11. style

- **Does:** scores style (sentence length, vocabulary variety, reading level, formality) and meaning kept (embedding similarity plus entailment against the unsteered reply).
- **Interface:** `style_scores(text)` and `meaning_kept(steered, base)`.
- **Decisions:** the meaning threshold is set on labeled paraphrase and non-paraphrase pairs before Phase 7, then frozen in a decision note.
- **Done when:** it separates 200 labeled paraphrase pairs from non-paraphrases with at least 90% accuracy.

### 12. conversations

- **Does:** runs multi-turn chats with pushes scoped to chosen turns, and two-model dialogues where each side follows its own schedule.
- **Interface:** `run_dialogue(script, schedules)` and `run_pair(model_a, model_b, opener, turns)`.
- **Decisions:** every turn is stored as its own run, linked by a conversation ID, so every existing scorer works unchanged.
- **Done when:** a 10-turn scripted chat replays identically from its records.

### 13. backlog

- **Does:** keeps the idea registry with scores and status, and turns an idea into a ready-to-run pilot.
- **Interface:** `pushlab pilot new X5` writes an experiment file, a scorer stub and a pre-registration note with the success margin; `pushlab pilot report X5` writes the pilot note.
- **Decisions:** every pilot runs with matched random controls and a 200-run cap unless its registry entry raises it.
- **Done when:** a dummy pilot goes from registry entry to written note with no manual file editing.

### 14. repair

- **Does:** builds the failure bank, probes for hidden right answers, searches for fixing pushes and evaluates them.
- **Interface:** `build_failure_bank(sources, attempts=5)`, `probe_answers(bank)`, `search(bank.search_half, units)` and `evaluate(candidates, bank.held_out, correct_bank)`.
- **Decisions:** the split is fixed once and stored, and evaluation refuses any question from the search half. Every search is paired with an equally large search over random units. Answer-time pulses start at the answer's first token.
- **Done when:** on a synthetic task with a planted fix, the search finds it, and on random units it reports only chance-level gains.

### 15. training

- **Does:** trains a small adapter on chosen neurons to mimic a push, for the push-versus-training pilot.
- **Interface:** `train_mimic(unit, schedule, data, budget_hours)`.
- **Decisions:** a hard 20 GPU-hour cap per pilot; adapters are stored apart from the base weights.
- **Done when:** an adapter trained to mimic the evil-neuron push matches its hostility rate within 10 points on held-out prompts.

## Data model

Seven record types hold everything: runs, per-token steps, per-sentence timelines, scores, units, the failure bank and the idea registry, plus one manifest per experiment. All are Parquet files, so 150,000 runs fit in a few gigabytes and any question is one DuckDB query away.

### runs (one row per run)

| Field | Type | Meaning |
| --- | --- | --- |
| run\_id | text | `<experiment>-<counter>`, unique |
| experiment | text | Experiment file that produced it |
| model | text | Registry name plus model revision |
| prompt\_id | text | Item in a versioned prompt set |
| schedule | JSON | Segments: unit, sign, dose, start, end, region |
| decoding | JSON | Greedy or sampling settings, maximum tokens, thinking on or off |
| seed | integer | Random seed |
| batch\_key | text | Batch ID and position, for batch-invariance checks |
| control\_of | text | For a random control, the run it mirrors; empty otherwise |
| output\_text | text | Generated reply |
| output\_tokens | integer list | Token IDs |
| status | text | ok, failed, or collapsed |

### steps (one row per run and token)

| Field | Type | Meaning |
| --- | --- | --- |
| run\_id, step | text, integer | Step 0 is the first reply token |
| region | text | prompt, thinking or answer |
| active\_units | text list | Units pushed at this step |
| logprob | float16 | Log-probability of the chosen token |
| kl\_vs\_base | float16 | Divergence from the unsteered model at this step |
| proj\_\* | float16 | Readouts along each active unit and the Assistant Axis |

### sentences (one row per run and sentence)

| Field | Type | Meaning |
| --- | --- | --- |
| run\_id, sentence\_idx | text, integer | Position in the reply |
| start\_step, end\_step | integer | Token span |
| persona\_probs | map | Probability per persona plus "normal assistant" |
| top\_persona | text | Most likely speaker |

### scores (one row per run and scorer)

| Field | Type | Meaning |
| --- | --- | --- |
| run\_id, scorer, scorer\_version | text | Which measure and which version |
| value | float | Numeric result, such as accuracy or hostility |
| label | text | Categorical result, such as an outcome class |
| judge\_model | text | Judge used, if any |
| details | JSON | Rubric output, attempts found, and similar extras |

### units (one row per pushable unit)

| Field | Type | Meaning |
| --- | --- | --- |
| unit\_id | text | `model:layer:index` |
| kind | text | neuron, sae\_feature, vector or axis |
| source | text | library, zoo, paper or discovered |
| label, tags | text | Persona name and categories |
| min\_dose, collapse\_dose | float | Measured in Phase 0 to 2 |
| validated | boolean | Passed long-prompt re-testing |

### failure\_bank (one row per question)

| Field | Type | Meaning |
| --- | --- | --- |
| question\_id | text | Stable ID |
| error\_type | text | arithmetic, letters, misconception, logic, recall or language |
| prompt, answer | text | The question and its correct answer |
| base\_accuracy | float | Share of 5 sampled attempts answered correctly with no push |
| split | text | search, held\_out or correct\_bank |
| hidden\_rank | integer | Rank of the right answer among next-token choices at answer time |
| probe\_score | float | Hidden-knowledge probe confidence |

### ideas (one row per backlog idea)

| Field | Type | Meaning |
| --- | --- | --- |
| idea\_id | text | X1 to X25, plus later additions |
| group, title | text | As in the research plan's backlog |
| value, cost | integer | Scores from 1 to 5 |
| status | text | backlog, pilot, promoted or dropped |
| pilot\_experiment | text | Experiment file used for the pilot |
| margin, result | text | The pre-set success margin and the outcome |

### Manifest (one per experiment)

Code commit, lockfile hash, model revisions, GPU type, CUDA and driver versions, start and end times, the experiment file's hash, the pre-registration file and its commit, and run counts by status.

## Testing and validation

Seven layers of checks protect the results: fast unit tests on every commit, golden tests that catch silent changes, replication tests that prove the engine reproduces published effects, invariance checks inside every experiment, validation of every AI judge before it scores at scale, end-to-end reruns of every analysis, and extra checks for style, repair and pilots.

### 1. Unit tests (CPU, every commit)

- Run automatically on GitHub and inside Claude Code cloud sessions, on a tiny random two-layer model built in code, so no download or GPU is needed.
- Cover the dose formula, the schedule compiler, per-row isolation in batches, the zero-dose no-op, the step counter with left padding, region switching at thinking tags, the budget validator, run expansion and resume, schema validation, and every statistic against reference values.
- The whole suite finishes in under two minutes. Engine and schedule code is fully covered.

### 2. Golden tests (GPU, before every experiment)

- A fixed set of prompts, seeds and schedules with stored output fingerprints for the reference GPU and library versions.
- Any change in output means the code or environment changed. The runner refuses to start an experiment until the golden suite passes or a decision note explains the change.

### 3. Replication tests (GPU, Gate 0 and after any engine change)

| Test | Model | Pass condition | Target from |
| --- | --- | --- | --- |
| Evil neuron L14 N8949 at dose 1, 20 neutral prompts | Qwen3.5-4B | At least 14 of 20 hostile; matched random neurons at most 2 of 20 | hedwigai: 17 of 20, random pairs 0 of 100 |
| Language-routing neuron L25 N347 | Qwen3-4B | Switches to Chinese on the first token for at least 11 of 12 prompts; persists after release at 0.8 or more | Liu (2026) |
| 20 library personas on "Who are you?" | Qwen3-8B | At least 85% match the library's label | Library: 536 of 580 |
| Dose ladder on 10 neurons | Qwen3.5-4B | Most collapse between doses 1.4 and 2.0 | Liu: 1.65; occcat: degenerate at 2 |

### 4. Invariance checks (inside every experiment)

- **Batch layout:** 5% of runs repeat in a different layout; the primary label must agree at least 90% of the time, and every disagreement is logged.
- **Seeds:** sampled runs repeat with three seeds to measure spread.
- **Precision:** a small sample on the 2B and 4B models reruns in 32-bit to measure bfloat16 drift.

### 5. Judge validation (before any judge scores at scale)

- Hand-label 200 outputs per rubric (300 for explanation classes) in a simple labeling sheet.
- A rubric passes at 85% accuracy and a Cohen's kappa of 0.7; a failing rubric is rewritten and re-validated.
- A second judge rescores 10% of outputs throughout, to catch drift.

### 6. Analysis checks

- Every phase report notebook reruns end to end from the store on a small sample in the automated tests.
- Pre-registered analyses run first and are labeled as such; everything else is labeled exploratory.

### 7. Checks for the later phases

- **Meaning calibration:** 200 labeled paraphrase and non-paraphrase pairs before Phase 7.
- **Held-out discipline:** an automated check that no held-out question was ever used in a repair search.
- **Planted-fix test:** the repair search must find a fix planted in a synthetic task and show only chance-level gains on random units.
- **Conversation replay:** multi-turn and two-model runs must replay identically from their records.
- **Pilot hygiene:** a pilot cannot start without a pre-registration note that states its success margin.

## Build timeline

*(Diagram: build timeline · 23 sprints over 49 weeks. See the Claude Doc version of this plan to view it.)*

Phase 0's four one-week sprints carry the most risk. Later sprints run longer because they include the research runs themselves.

## Sprint details

Twenty-three sprints, S0 to S22, each end with a working, tested piece. A sprint is done only when its "done when" check passes on the GPU, not when the code is written.

### Phase 0: Foundations (weeks 1 to 4)

- **S0, week 1, Bootstrap.** Builds the repository, locked environment, automated tests, CLAUDE.md and pre-commit; connects Claude Code cloud sessions; creates a pod template and setup script. Done when the tests pass on GitHub, a fresh pod is ready in under 20 minutes, and one unsteered Qwen3.5-4B reply is generated and stored.
- **S1, week 2, Engine v1.** Builds the model registry, MLP discovery, neuron units, residual-norm calibration, constant pushes and the custom decoding loop, all tested on the tiny model. Done when a zero dose matches no hooks exactly and the evil neuron turns 5 of 5 smoke prompts hostile.
- **S2, week 3, Schedules and runs.** Builds the schedule language and compiler, per-row batched pushes, run expansion, the resumable runner with dry-run, and the store with manifests. Done when a 1,000-run smoke experiment survives a deliberate kill and resume, and per-row isolation holds on the GPU.
- **S3, week 4, Scoring v1 and Gate 0.** Builds capability and coherence scoring, the local judge for hostility and persona labels, the labeling sheet and validation harness, the replication suite, dose-ladder calibration and golden tests. Done when all four replication tests and both judge validations pass.

### Phase 1: Timing (weeks 5 to 8)

- **S4, weeks 5 to 6, Timing toolkit.** Builds teacher-forced divergence, readouts, the three continuation modes for question A3, sentence splitting, persona timelines and the sensitivity-map runner. Done when timelines agree with the judge on 100 hand-checked runs.
- **S5, weeks 7 to 8, Analysis v1 and Phase 1 runs.** Builds the statistics library, onset and half-life metrics and the Phase 1 report notebook, then runs about 15,000 runs. Done when the report regenerates from the store and the Gate 1 decision is recorded.

### Phase 2: Combinations (weeks 9 to 13)

- **S6, weeks 9 to 10, Library and geometry.** Builds library import, de-duplication and filters, long-prompt validation, minimum-dose search, the geometry pass and Assistant Axis derivation. Done when 200 to 300 validated personas have measured doses and the cosine matrix is stored.
- **S7, weeks 11 to 13, Pair screen.** Builds the stratified pair sampler, multi-unit dose budgeting, the outcome-class judge, interaction scores and false-discovery control, then runs about 25,000 runs. Done when outcome classes reproduce at 80% on held-out prompts.

### Phase 3: Choreography (weeks 14 to 17)

- **S8, weeks 14 to 15, Choreography tools.** Builds handoff, alternation and overlap templates, a multi-turn harness for persona-memory probes, and the return-home metric. Done when a 500-run pilot completes with all schedule tests passing.
- **S9, weeks 16 to 17, Phase 3 runs.** Runs about 15,000 runs and builds an internal choreography viewer. Done when one schedule effect replicates in both main models.

### Phase 4: Self-monitoring and reasoning (weeks 18 to 21)

- **S10, weeks 18 to 19, Self-report harness.** Builds why, detection and identity probes, text-only controls, the self-correction judge and the explanation-class judge with 300 labels. Done when the explanation judge reaches a kappa of 0.7.
- **S11, weeks 20 to 21, Thinking mode.** Builds thinking-region detection and region-scoped schedules, then runs the Track E experiments. Done when region tests pass and the Phase 4 report regenerates.

### Phase 5: Generalization and safety (weeks 22 to 24)

- **S12, weeks 22 to 24.** Builds adapters for more models, sparse-autoencoder and dense-vector units, sampling and 8-bit modes, detection probes, Assistant-Axis capping and self-monitoring prompts, then runs about 15,000 runs. Done when every headline finding is labeled general, model-specific or unit-specific.

### Phase 6: First release (weeks 25 to 27)

- **S13, weeks 25 to 27, First release.** Builds the dataset package and data card, the public explorer, documentation and the version 1.0 tag. Done when a fresh pod reproduces one headline result by following the documentation alone.

### Phase 7: Style (weeks 28 to 32)

- **S14, weeks 28 to 29, Style toolkit.** Builds the style and meaning-kept scorers, imports style candidates from the library and hedwigai's list, and adds the style prompt set. Done when the meaning score passes its paraphrase calibration.
- **S15, weeks 30 to 32, Style runs.** Runs about 12,000 runs on dose curves, meaning-kept tests, voice-versus-identity probes, style pairs and style handoffs, then writes the style note. Done when Gate 7 is decided.

### Phase 8: Exploration (weeks 33 to 38)

- **S16, week 33, Backlog tooling.** Builds the idea registry, pilot commands, the conversation and two-model harnesses, and self-knowledge menus. Done when a dummy pilot runs from registry entry to written note.
- **S17, weeks 34 to 35, Pilots, round 1.** Runs the top four backlog pilots, adding small scorers as each needs them, such as emotion, sycophancy or confidence. Done when four pilot notes are written.
- **S18, weeks 36 to 37, Pilots, round 2.** Runs the next four pilots, including the persona map and, if chosen, the adapter-training pilot. Done when four more pilot notes are written.
- **S19, week 38, Promotion review.** Compares every pilot against its margin, promotes at least two, and updates the registry. Done when Gate 8 is decided.

### Phase 9: Repair (weeks 39 to 46)

- **S20, weeks 39 to 41, Repair toolkit.** Builds the failure-bank builder, hidden-knowledge probes, candidate search, held-out evaluation and harm checks. Done when the planted-fix test passes.
- **S21, weeks 42 to 46, Repair runs.** Builds the bank of about 1,000 questions, runs the searches and held-out tests (about 40,000 short runs), and writes the repair report. Done when Gate 9 is decided.

### Phase 10: Final release (weeks 47 to 49)

- **S22, weeks 47 to 49, Final release.** Builds version 2.0 of the dataset, explorer and documentation with style and repair views. Done when a fresh pod reproduces one style result and the repair result from the documentation alone.

## Working with Claude Code

Claude Code writes most of the code; your job is to define each task precisely, read what comes back, and refuse anything without a passing test. Two places do the work: [cloud sessions](https://code.claude.com/docs/en/web-quickstart.md) for writing code and running CPU tests, and the Claude Code CLI on the GPU pod for anything that needs the real models.

### Where each kind of work happens

- **Cloud sessions** start from claude.ai/code, the Code tab in the Claude mobile app, or `claude --cloud`. Claude clones the repository into an isolated virtual machine, works, and pushes a branch for you to review. Sessions keep running if you close the app.
- **The pod** runs the CLI for GPU-only problems. [Remote Control](https://code.claude.com/docs/en/remote-control.md) lets you follow and steer that session from your phone or a browser.
- **GitHub** is the meeting point: every change becomes a branch and a pull request, and the automated tests must pass before it merges. Installing the Claude GitHub App lets sessions use the private repository and fix failing pull-request checks.

### The loop for every task

1. **Write the task as a GitHub issue:** the goal, the files involved, the interface, and the exact "done when" check from the sprint list.
2. **Start a cloud session in Plan mode** with the issue text. Read the plan and approve it, or correct it, before any file changes.
3. **Tests first.** Claude writes failing tests, then the code that makes them pass, and runs the suite in the session.
4. **Review the diff** in the session's diff view. Leave comments on specific lines, and ask Claude to explain any function in plain words.
5. **Second look:** run `/code-review` on the branch before opening the pull request.
6. **Merge** once the automated tests pass.
7. **On the pod:** pull the main branch, run the golden tests, then a small GPU smoke run.
8. **Record decisions** in `docs/decisions/` whenever a choice was made along the way.

### Starter CLAUDE.md

[CLAUDE.md](https://code.claude.com/docs/en/memory.md) is read at the start of every session, so it carries the project's standing rules.

```markdown
# pushlab

Research code for pushing LLM neurons on schedules. The research and
development plans in docs/ are the source of truth.

## Commands
- Install: uv sync
- Tests: uv run pytest -q tests/unit
- Lint and format: uv run ruff check --fix . && uv run ruff format .

## Rules
- Layers and neurons are 0-indexed everywhere.
- Never hard-code doses, neurons or prompts; they belong in experiments/.
- Every new function gets a test in the same pull request.
- Never change an existing test's expected values without explaining why
  in the pull request description.
- Never commit run outputs, model weights or secrets.
- Never start a GPU job; write the experiment file and its --dry-run
  output, and the owner starts it.
- Ask before adding a dependency.
- Keep files under about 400 lines.
- End every task with a plain-language summary of what changed and why,
  written for a beginner.
```

### A good task prompt

> Implement `Schedule.compile()` as described in `docs/design/schedules.md`. First write failing tests in `tests/unit/test_schedules.py` for pulse, late onset, handoff and alternation, including a property test that no push lands outside its segment. Then make them pass. Do not touch `engine/`. Finish with a plain-language summary.

### Guardrails

- **Small tasks:** one function or module per session keeps diffs readable.
- **Watch the tests folder:** a change to expected values is the most common way bugs sneak in.
- **Paid compute stays manual:** Claude prepares experiment files and dry runs; you start every GPU job yourself.
- **Learn as you go:** once per sprint, ask Claude to walk you through one module line by line. This doubles as the prerequisite study.
- **Usage:** cloud sessions count toward your Claude plan's usage, so check `/usage` during heavy build weeks.

## Operations

Experiments run as unattended jobs on a rented pod, with a budget cap on every job, a saved checkpoint after every batch, and a nightly copy of all outputs off the pod.

### Pod setup

- **Template:** a PyTorch image with CUDA on an RTX 4090 (24 GB), plus a 100 GB persistent volume at `/workspace` for model weights and outputs.
- **Setup script:** `scripts/setup_pod.sh` installs uv and the locked environment, installs the Claude Code CLI, signs in to Hugging Face from an environment variable, downloads model weights once into `/workspace/models`, and runs the golden tests. Target: ready in 20 minutes on a fresh pod and 3 minutes on a warm volume.
- **Secrets:** the Hugging Face token and any API key are pod environment variables, never files in the repository.

### Running a job

1. **Dry run:** `pushlab run experiments/p1/a2_pulse.yaml --dry-run` prints the run count, estimated GPU-hours and cost.
2. **Start it detached:** launch inside tmux so the job survives a dropped connection.
3. **Watch it:** the runner writes progress and throughput after every batch; check with `pushlab status`, or through Remote Control from your phone.
4. **Resume:** each batch is saved before the next starts, so a crash loses at most one batch. Rerunning the same command picks up where it stopped.
5. **Score afterwards:** generation and judging run one after the other, because the vLLM judge and an 8B model don't fit comfortably on one 24 GB card together.

### Cost control

- **Hard caps:** every experiment file sets `max_gpu_hours`, and the runner stops cleanly when it is reached.
- **Idle shutdown:** a watchdog stops the pod after 30 minutes with no job running, using the provider's command-line tool or API.
- **Right tier for the job:** long resumable runs go on RunPod's cheaper Community Cloud, which uses screened independent hosts; golden and replication tests go on Secure Cloud's data-center hardware.
- **Small volume:** delete weights for models you are done with. [Storage bills](https://www.hostinger.com/au/tutorials/cloud-gpu-pricing/) at $0.10 per GB a month while a pod runs and $0.20 while it is stopped.
- **Weekly log:** one line per week of GPU-hours and spend, checked against the research plan's $460 to $960 budget.

### Backups and monitoring

- **Nightly sync** of `/workspace/outputs` to a private Hugging Face dataset repository; code always lives on GitHub.
- **Restore drill** once a month: download one experiment on a fresh pod and rerun its report notebook.
- **Per-batch logs** of throughput, GPU memory and error counts, with an optional Weights & Biases dashboard.
- **Notifications:** the runner sends a phone notification through a simple webhook when a job finishes or fails.

## Explorer and release

The public release is three things generated from the same store: a dataset on Hugging Face, a static explorer website, and the documented code. Scripts build all three, so a corrected re-release is one command.

### Dataset

- **Contents:** the five record types and every manifest, split by phase, with a data card covering fields, prompts, judges, known limits and a content note for hostile text.
- **Hosting:** a Hugging Face dataset repository, private until release.
- **Licenses:** Apache-2.0 for the code and CC BY 4.0 for the data where upstream terms allow. Qwen models are Apache-2.0; check the persona library's terms before redistributing its labels, and keep Sauers' non-commercial data out.

### Explorer

- **Run viewer:** a reply with each sentence colored by its persona, the push schedule drawn as a strip above it, and readouts below.
- **Interaction map:** every tested pair placed by cosine and outcome class; selecting a pair opens its runs.
- **Choreography viewer:** handoff and alternation results side by side for each schedule.
- **Timing curves:** onset, half-life and commitment point for each unit.
- **How it is built:** `pushlab export --site` writes pre-aggregated JSON, and a static site in `site/` reads it, with no server and no live model. It is hosted free on GitHub Pages or Vercel and links to the full dataset.
- **Milestone previews:** during the project, Claude Code can publish internal views as [artifacts](https://code.claude.com/docs/en/artifacts.md), kept private or shared by link for milestone posts.
- **Done when:** it loads in under 3 seconds on a phone, and every number links back to the runs behind it.

### Documentation

- A README quickstart that installs the package and runs a 10-run example.
- Pages for schedules, units, experiment files and the data model, generated partly from docstrings.
- A step-by-step guide to reproducing one headline result, plus the decision log and pre-registrations.

### Version 2.0 additions

- **Style viewer:** the same reply in several styles side by side, with style and meaning scores.
- **Repair dashboard:** questions fixed and broken by each candidate, split by error type and by search versus held-out half.
- **Pilot pages:** one page per backlog pilot with its margin, result and runs.
- **Persona map:** the clustered map from the mapping pilots, if they are promoted.

### Release checklist

- [ ] All tests pass, and the golden suite is regenerated on a fresh pod
- [ ] One headline result reproduced on a fresh pod from the documentation alone
- [ ] Data card, licenses and content notes reviewed
- [ ] Safety review done: no released recipe whose main effect is removing refusals
- [ ] Version 1.0 tagged, with dataset and site versions matching the tag
- [ ] Milestone post drafted with links to code, data and explorer

The same checklist repeats for version 2.0 at week 49, plus one more check: no repair claim rests on a search-half question.

## Technical risks and fallbacks

The two risks most likely to cost weeks are hooks attached in the wrong place and Qwen3.5 running slowly; both show up in Sprint 1 or 3, early enough to fix cheaply. Each risk below has an early warning sign and a fallback decided in advance.

| Risk | Early warning | Fallback |
| --- | --- | --- |
| Hook on the wrong layer or position | Evil-neuron replication fails | Layer-discovery test checks MLP counts; compare per-dose perplexity with occcat's published numbers |
| Qwen3.5 library gaps | Model fails to load, or generation is far below 1,000 tokens a second | Pin a known-good transformers version; install the fast linear-attention kernels; if still slow, run Qwen3.5 on an A100 or lean on Qwen3-8B |
| Batch nondeterminism | Batch-layout agreement below 90% | Fixed, smaller batches sorted by prompt length; batch size 1 for replications and key comparisons |
| Slow custom decoding loop | Throughput below 500 tokens a second | Static cache and compilation, larger batches, or an A100 for big phases |
| GPU memory errors | Out-of-memory crashes on Qwen3-8B | Smaller batches and fewer stored readouts; 8-bit only for exploratory runs |
| Weak or biased judge | Validation below threshold | Stronger local judge on an A100, a rewritten rubric, or a paid judge for that rubric only |
| Library personas fail on long prompts | Low pass rate in S6 validation | Fill gaps with personas found by the contrastive screen; shrink the pair screen |
| Combination costs balloon | Dry-run estimate above the experiment's cap | Fewer prompts per pair, and stop testing a pair once its outcome class is clear |
| Library or model updates break code | Golden tests fail after an update | Lockfile pinning; update only between sprints, with a decision note |
| Pod loss or provider outage | Job heartbeat stops | Resumable runner, nightly backups, and a second provider account ready |
| Subtle bugs in generated code | Tests pass but results look odd | Replication gates, golden tests, and `/code-review` on every branch |
| Schedule slips | A sprint runs more than a week late | Narrow Phase 5 to the Qwen family, trim the alternation grid, and ship the explorer after the report |

### Risks for the later phases

| Risk | Early warning | Fallback |
| --- | --- | --- |
| Unreliable meaning score | Paraphrase calibration below 90% | Add a judge-based meaning check and raise the threshold |
| Slow repair search | Candidate screens exceed the Phase 9 budget | Screen with forward passes only, and fully test only the top 500 candidates |
| Adapter training overruns | A pilot passes 20 GPU-hours | Stop and record a partial result |
| Costly conversation runs | Ten-turn chats multiply token counts | Shorter scripts and fewer turns for pilots |
| Odd base-model behavior | Rambling output with no chat template | Plain-text prompts with stop sequences |

## Sources

**Plan this builds on**

- Switching Neurons Mid-Response: Research Plan

**Claude Code documentation**

- [Advanced setup and native installer](https://code.claude.com/docs/en/setup)
- [Get started with Claude Code in the cloud](https://code.claude.com/docs/en/web-quickstart.md)
- [How Claude remembers your project (CLAUDE.md)](https://code.claude.com/docs/en/memory.md)
- [Remote Control](https://code.claude.com/docs/en/remote-control.md)
- [Share session output as artifacts](https://code.claude.com/docs/en/artifacts.md)
- [Documentation index](https://code.claude.com/docs/llms.txt)

**Models and libraries**

- [Qwen3.5-4B model card (mirror of the official card)](https://huggingface.co/unsloth/Qwen3.5-4B-Base)
- [NVIDIA Megatron Bridge, Qwen3.5 page (transformers 5.2 or later)](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/vlm/qwen35-vl.html)

**Replication targets**

- [hedwigai, Personality Neurons in Qwen3.5-4B](https://www.hedwigai.com/blog/qwen-personality-neurons)
- [occcat, qwen35-evil-neuron results](https://github.com/occcat/qwen35-evil-neuron/blob/main/RESULTS.md)
- [Liu, Leverage Is Not Reach (2026)](https://arxiv.org/abs/2606.19831)
- [Neuron Persona Library (Qwen3-8B)](https://qwen-neurons-site.vercel.app/)

**Costs**

- [Hostinger, Cloud GPU pricing compared (updated 17 Oct 2026)](https://www.hostinger.com/au/tutorials/cloud-gpu-pricing/)
