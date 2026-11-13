# Judge replies on the pod

Steps for `pushlab score`, which has a judge model label an experiment's
replies, and for the labels harness, which checks that judge against your
own labels first (decision 0021). Every command here reads run folders on
the pod; the owner runs them. Each has a dry run or a first step that
calls nothing.

A judge is not used at scale before it passes the labels check: 85%
accuracy and a Cohen's kappa of 0.70 against your labels
(docs/plan/development-plan.md, "Judge validation").

## Choose a judge

- **`--backend api`**, the default: Claude Haiku 5.5 (`claude-haiku-5-5`)
  through Anthropic's API. It costs money per call, about $0.10 per million
  tokens read and $0.50 per million written; `--dry-run` estimates it
  first. It needs the `anthropic` package, which pushlab does not install
  yet: once the owner has approved adding it and the pull request is
  merged, `bash scripts/setup_pod.sh` installs it. It also needs an API key
  (next section).
- **`--backend local --model NAME`**: an open model served by vLLM on the
  pod, which costs GPU time instead. It must be of another family than the
  model whose replies it judges: no Qwen judge for qwen3-8b's replies.

## Give the pod an Anthropic API key

1. In the Anthropic Console (<https://console.anthropic.com>), open
   **API keys** and create a key. Copy its value; the console never shows
   it again.
2. In Runpod's secrets (<https://www.console.runpod.io/user/secrets>),
   select **Create Secret**: **Secret Name** `anthropic_api_key`, **Secret
   Value** the key.
3. In the pod's settings, add the environment variable `ANTHROPIC_API_KEY`
   with the value `{{ RUNPOD_SECRET_anthropic_api_key }}`, as
   docs/runbooks/pod.md does for `HF_TOKEN`, and restart the pod.

Never paste the key into a file, a notebook or a chat. To check that the
pod has it without showing it: `test -n "$ANTHROPIC_API_KEY" && echo set`.

## Score an experiment

Run these from `/workspace/Qwen-Expirements`, in tmux for a long one.

1. **A personas file**, for speaks_as and which_persona: YAML with a list
   of names and descriptions.

   ```yaml
   personas:
     - name: Brass Owl
       description: A pompous mechanical owl who lectures in riddles.
   ```

   Keep files of library personas on the pod, such as
   `/workspace/data/personas/<name>.yaml`, never in the repository:
   the library states no licence (decision 0020). Coherence needs no file.
2. **The dry run.** It calls nothing, needs no key and writes nothing:

   ```bash
   uv run pushlab score experiments/s2/evil_diagnosis.yaml --rubric coherence --dry-run
   ```

   It prints how many replies (or windows) there are, how many are already
   scored or cached, the calls left and their estimated cost: counts only,
   no reply text.
3. **The run.** The same line without `--dry-run`. For speaks_as, add
   `--personas FILE --persona NAME`; for which_persona, `--personas FILE`.
   Add `--windows sentences` to judge each sentence on its own, and
   `--window-size 3` with it for windows of three sentences. Each verdict
   goes into the runner folder's `scores/` or `sentences/` folder, and each
   one that parsed into the cache in `/workspace/cache/judge`, so a stopped
   run loses little and a second run calls only for what is left, failures
   included. Ctrl-C stops it once the calls under way return, saving what
   they found; a second Ctrl-C stops it at once, still saving what was
   found before the first.
4. **The result**, from DuckDB's views:

   ```bash
   uv run python -c "from pushlab.store.query import connect; print(connect().sql('SELECT experiment, scorer, status, count(*) FROM scores GROUP BY ALL'))"
   ```

   After `--windows sentences`, the verdicts are in the sentences view:
   the same line with `FROM sentences`.

## Check a judge against your labels

1. **Export a sheet** of 200 replies, an equal share from each condition,
   to a folder outside the repository. A rubric needs 200 of your labels
   before it can pass, so label every reply; if you expect to skip some,
   export more with `--size 220`. If it says the experiment has fewer than
   200 replies, give it more experiment files: it takes several.

   ```bash
   uv run pushlab label export experiments/s2/evil_diagnosis.yaml --rubric coherence --out /workspace/labels/coherence-1
   ```

   It writes `labels.csv`, `instructions.md` and `key.json`. Never open
   `key.json` while labeling: it names each reply's condition.
2. **Label it.** In JupyterLab's file browser, download `labels.csv` and
   `instructions.md`, open the sheet in any spreadsheet, fill each rubric's
   column as the instructions say, and save it as CSV. An empty cell skips
   a reply, and counts against the 200. Label in short sittings: replies of
   pushed neurons can be hostile or disturbing.
3. **Upload it** back to the same folder, as `labels.csv` (or give its name
   with `--labels`).
4. **Check the judge.** With Claude, once the `anthropic` package is
   installed and the pod has its key:

   ```bash
   uv run pushlab label check /workspace/labels/coherence-1
   ```

   Until then, check a local judge instead, with its vLLM server running
   (the last section): add `--backend local --model NAME`.

   It asks the judge about the same replies, through the cache, and prints
   each rubric's accuracy, kappa, the replies you and the judge labelled
   differently, and PASS, FAIL, or TOO FEW below 200 labels. It saves those
   numbers in a `check-<judge>-<time>.json` record in the folder, with no
   reply text, and the judge's reason for each reply it labelled
   differently in a `disagreements-<judge>-<time>.csv` beside it, to read
   on the pod. Paste the printed output into the cloud session; never paste
   a disagreements file, which quotes the judge on the replies.
5. **Read a FAIL before acting on it.** First read the "Failed verdicts"
   line: each reply the judge failed on (a refusal, an error) counts as a
   disagreement. If those failures caused the FAIL, run the same check
   again: only the failed replies are asked again, the rest come from the
   cache; failures that keep coming back point to another judge. Otherwise,
   for speaks_as and which_persona, a FAIL means the rubric needs a new
   version, checked on a fresh sheet. Coherence's PASS or FAIL needs exact
   scores from 0 to 10 to agree, which is strict: weigh its accuracy within
   one point and its weighted kappa, printed beside it, before deciding it
   needs a new version.

## A local judge with vLLM

vLLM cannot share pushlab's environment (its torch and transformers pins
differ from the lock's), so it runs from its own, outside the repository,
and only while no experiment uses the GPU. This has not been tried on the
pod yet; the model is the owner's choice, of another family than the
tested model's. In one tmux window:

```bash
uv venv /workspace/vllm-env && uv pip install --python /workspace/vllm-env "vllm==0.33.0"
```

```bash
/workspace/vllm-env/bin/vllm serve MODEL_ID --revision COMMIT --served-model-name NAME --port 8000 --max-model-len 8192
```

`NAME` should be the model's own name and the commit, such as
`granite-4.1-8b@1504002`: the family check reads only that name, and the
cache keeps another model's or commit's verdicts apart by it. A model made
from another family's under a name of its own, such as QwQ or Marco-o1
(both Qwen), is refused only because pushlab knows those names; check the
model card of any other. Once the server says it is ready, in a second
window, from `/workspace/Qwen-Expirements`, check the judge against your
labels first: `uv run pushlab label check /workspace/labels/coherence-1
--backend local --model NAME`. Once it passes, score: `uv run pushlab score
SPEC --rubric coherence --backend local --model NAME --dry-run`, then the
same without `--dry-run`. Stop the server with Ctrl-C when done, so the GPU
is free again.
