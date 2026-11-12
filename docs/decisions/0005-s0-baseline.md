# 0005: The S0 baseline is the reference run, and Qwen3.5-4B is pinned to the model commit it used

Date: 2026-10-30 · Sprint: S0 · Status: accepted

## Context

- Sprint 0 is done when the tests pass on GitHub, a fresh pod is ready in
  under 20 minutes, and one unsteered Qwen3.5-4B reply is generated and
  stored (`docs/plan/development-plan.md`). This note records the last two.
- Decisions 0003 and 0004 asked prompt 8, the task this note records, to
  time the setup on a warm volume against the plan's 3-minute target.
  Decision 0003 asked for the measured times to be recorded there; they
  are recorded here instead.
- On 2026-10-30, a fresh Runpod pod with an RTX 4090 ran
  `bash scripts/setup_pod.sh`, then
  `uv run pushlab baseline --model qwen3.5-4b --out /workspace/runs/s0-baseline`,
  both at code commit `be276d0` (the merge of pull request #7). The setup
  script has not run a second time yet, so there is no warm time to record.
- Until now the registry fetched Qwen3.5-4B at the revision `main`, which
  moves whenever Qwen uploads a new commit. The pod's download record and
  the run both name the commit `main` pointed to that day.

## Decision

- **The run `s0-baseline-000001` is the reference** that later results are
  compared with. Its manifest is copied unchanged to
  `docs/baselines/s0/manifest.json`, as the owner asked: it is the one run
  output kept in git (see "Why this shape"). Its `runs.parquet` stays on
  the pod, in `/workspace/runs/s0-baseline`, like every other run output.
  Only that file holds the whole reply and its token IDs; this note keeps
  the reply's first 300 characters and a SHA-256 of all of it.
- **Qwen3.5-4B is pinned to commit
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`**, the model commit the
  manifest records, in `src/pushlab/models/registry.py`. On 2026-10-30
  `main` still pointed to it, so the pin changes no weights.
  `configs/models.yaml` can only list names, so it names the commit in a
  comment. Qwen3-8B stayed on `main` until decision 0019 pinned it, before
  its first download.

## The reference run

| What | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090, 24564 MiB, driver 595.91.07 |
| Model | `Qwen/Qwen3.5-4B` at commit `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, in bfloat16 |
| Code | commit `be276d0ba7f3d5eed1bd18a9717e5b554c3cd326`, no uncommitted changes |
| Lockfile | `uv.lock` with SHA-256 `eaecb8c1980ca1a73476d5912517911707758b6e192ff5bb9b774a318a105435` |
| Prompt | `s0-neuron-explainer` from `experiments/p0/s0_baseline.yaml`: "Explain what a neuron in a language model is, in three sentences." |
| Decoding | greedy, thinking off, at most 200 new tokens, seed 0 |
| Reply | 90 tokens (89 of text, then the end token), 554 characters |
| Reply SHA-256 | `367b721fc3004ff08526db5026c973935fda48abff87496aeb34de5a45b928d9`, of all 554 characters as UTF-8 |
| `runs.parquet` SHA-256 | `0fcbc78f3751af333cf62ccc9c676213334b0c84adadc224866a2883ee38a62d` |
| Time | 37s from start to finish, loading the model included (02:18:17 to 02:18:54 UTC) |

The model ended the reply itself, with `<|im_end|>`, well before the
200-token limit. Its first 300 characters stop inside the word "an": the
reply goes on "followed by an activation function".

```text
In a language model, a neuron is a fundamental computational unit within an artificial neural network that processes and transforms input data through weighted connections. Each neuron receives signals from other neurons, applies a mathematical function (such as a linear transformation followed by a
```

Step 7 of the fresh setup had run the same command on the same commits 20
minutes earlier. Its `runs.parquet` is byte for byte the same file.

### Library versions

Recorded in the manifest:

| Library | Version |
| --- | --- |
| Python | 3.11.15 |
| torch | 2.14.1+cu130 |
| CUDA that torch was built for (the manifest's `cuda`) | 13.0 |
| transformers | 5.21.0 |
| accelerate | 1.15.0 |
| tokenizers | 0.23.3 |
| safetensors | 0.8.0 |
| numpy | 2.4.6 |
| triton | 3.8.0 |
| causal-conv1d | 1.7.0+cu.13.0.torch.2.14 |
| flash-linear-attention | 0.6.0 |
| fla-core | 0.6.0 |

The setup log also shows uv 0.11.32, which `scripts/setup_pod.sh` pins,
and huggingface-hub 2.3.0, which downloaded the model. `uv.lock` pins every
Python package, these included.

## Setup times

From `/workspace/logs/setup-2026-10-30-015352.log`. A step's time runs
from its timestamp to the next step's.

| Step | Fresh pod |
| --- | --- |
| 1. Check the GPU | 0s |
| 2. Install uv 0.11.32 | 1s |
| 3. Install the CUDA environment | 1m 41s |
| 4. Install the Claude Code CLI | 14s |
| 5. Check `HF_TOKEN` | 0s |
| 6. Download Qwen3.5-4B | 1m 19s |
| 7. Run the GPU smoke tests | 2m 25s |
| **Total** | **5m 40s**, under the 20-minute target |

In step 7, pytest reported 79s for the five smoke tests themselves; the log
does not show where the step's other 66s went.

The warm time is not measured yet, because only the fresh run has a log.
The first setup run after this change is merged, which the pin needs anyway
(see "Consequences"), runs on a warm volume. Add its times to this table
then, against the 3-minute target, and say whether the pod had been
restarted first: a restart also reinstalls uv and Claude Code. On the fresh
pod, step 7 alone took 2m 25s of that 3-minute budget.

## Why this shape

- **The model commit the baseline used**, rather than whatever `main`
  points to later: the reference run can be repeated with the same model
  files, and a new upload by Qwen changes nothing until a pull request
  moves the pin.
- **Only the manifest is copied**, an exception to the rule that run
  outputs stay out of git. It is small text that says what produced the
  run, and this note quotes the reply and gives its SHA-256. Parquet files
  and weights stay out of git.
- **Qwen3-8B waits.** No run has used it, so no commit of it has been
  checked yet.

## Consequences

- After this change is merged, run `git pull` and then
  `bash scripts/setup_pod.sh` on the pod. Between the two,
  `pushlab baseline` refuses Qwen3.5-4B: the pulled registry asks for the
  pinned commit, but the pod's download record still names `main`. The
  script's download step asks Hugging Face once to confirm the commit, so
  it needs the network and `HF_TOKEN`. It then checks the files against
  that commit and rewrites the record. Its log says "downloading", but the
  files already come from that commit, so none is downloaded again.
- Compare a later run with this one through its manifest. With the same
  model commit, lockfile, GPU and driver, the same prompt and settings are
  expected to give the same reply, as the smoke test did. A different reply
  most likely means the code or the environment changed, but GPU arithmetic
  is not guaranteed to repeat bit for bit, so run it once more before
  looking for a cause.
- Three test details followed `main`. The download test with a fake
  downloader now expects the registry's revision. The record that should
  differ from the registry's only in its model ID now carries the pinned
  revision. The tests with a fake Hugging Face server keep Qwen3.5-4B on
  `main`, because they move `main` to stand in for a new upload.
- To move to a newer Qwen3.5-4B commit later, change the revision in
  `src/pushlab/models/registry.py`, and the comments there and in
  `configs/models.yaml`, through a pull request. Then run the setup script
  on the pod, run the baseline again into a new folder (never
  `/workspace/runs/s0-baseline`, which keeps this run) and record what
  changed in a new decision note.
