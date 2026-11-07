# Set up a GPU pod on Runpod

`scripts/setup_pod.sh` turns a fresh Runpod GPU pod into a machine ready for
pushlab research. It runs eight steps:

1. It checks that a GPU is visible, and prints its name and memory.
2. It installs uv 0.11.32, the version CI uses, if the pod has no uv or
   another version. Runpod's image brings its own uv, usually a newer one.
3. It installs the locked CUDA environment: the CUDA build of torch, the GPU
   kernels for Qwen3.5 and the dev tools.
4. It installs the Claude Code CLI if it is missing.
5. It checks that your Hugging Face token is set.
6. It downloads the models in `configs/models.yaml` to `/workspace/models`.
7. It runs the GPU smoke tests: checks that torch can use the GPU and that
   the fast kernels for Qwen3.5 load, one reply from Qwen3.5-4B with
   `pushlab baseline`, three replies from the generation loop, each alone
   and then as one batch, and four replies with four schedules in one
   batch, each against its prompt alone. Three of the tests load the 9 GB
   model, so step 7 can show nothing for a few minutes.
8. It prints the total time and whether it beat the target.

The target is under 20 minutes on a fresh pod and under 3 minutes on a warm
volume, where an earlier run has finished. Expect the first run to take 5 to 10
minutes; most of it is downloading about 3 GB of CUDA packages and the 9 GB
Qwen3.5-4B model. Each step is logged with a timestamp to
`/workspace/logs/setup-<date and time>.log`. To see the steps without running
them, run `bash scripts/setup_pod.sh --dry-run`.

## Before you start

- **Merge the pull request that adds `scripts/setup_pod.sh` first.** The pod
  clones `main`.
- You need a Runpod account with some credit, a Hugging Face account and your
  GitHub account. Use a computer: the steps involve copying tokens between
  websites.

## 1. Create the two tokens

The pod needs two tokens. Each one is a password for one purpose, which you
can cancel at any time without changing your real password. Both are shown
only once, so keep the browser tab open until step 2 is done.

**Hugging Face token (`HF_TOKEN`)**, for downloading models:

1. Open <https://huggingface.co/settings/tokens> and select **New token**.
2. Choose the **Read** role and name it `pushlab-pod`.
3. Create it and copy the value. It starts with `hf_`.

**GitHub token (`GITHUB_TOKEN`)**, for cloning and pulling this repository:

1. Open this link. It fills in the name, a 90-day expiry and read-only access
   to repository contents:
   <https://github.com/settings/personal-access-tokens/new?name=pushlab-pod&description=Clone+and+pull+Qwen-Expirements+on+the+Runpod+pod&target_name=jonleach323&expires_in=90&contents=read>
2. Under **Repository access**, choose **Only select repositories** and pick
   `jonleach323/Qwen-Expirements`. Without this, the token can't read the
   repository, because it is private.
3. Under **Permissions**, check that **Contents** says **Read-only**.
   **Metadata: Read-only** is added automatically.
4. Select **Generate token** and copy the value. It starts with
   `github_pat_`.

## 2. Store your tokens in Runpod

Runpod keeps the tokens as encrypted secrets. Once the pod's settings name
them (step 3 of the next section), Runpod hands them to the pod as
environment variables each time it starts, so you never put them in a file
yourself. `scripts/pod.env.example` lists the variable names the pod expects.

1. Open <https://www.console.runpod.io/user/secrets> and select
   **Create Secret**.
2. **Secret Name:** `hf_token`. **Secret Value:** paste the Hugging Face token.
   Select **Create Secret**.
3. Do the same with **Secret Name** `github_token` for the GitHub token.

Runpod never shows a secret's value again. To change one later, use
**Manage**, then **Edit Secret Value**.

Runpod's start script also copies every pod variable, tokens included, into
the file `/etc/rp_environment` on the pod. That file is wiped when the pod
stops. Never copy it to `/workspace` or into the repository.

## 3. Deploy the pod

1. In the [Runpod console](https://www.console.runpod.io/pods), select
   **+ New**, then **Pod** (or **Deploy** on the Pods page).
2. **Template:** search for `pytorch` and pick Runpod's official PyTorch
   template. In its **CUDA version** dropdown, choose **13.0**. The torch
   version doesn't matter, because the script installs its own, but
   pushlab's torch needs a machine whose NVIDIA driver supports CUDA 13.0
   (driver 580 or newer). In the older deploy page, use
   **Additional filters → CUDA Versions** instead and select 13.0, 13.1,
   13.2, 13.3 and 13.4: it matches versions exactly. Step 1 of the script
   checks the driver. Keep **Start Jupyter notebook** on.
3. **Environment variables:** select **Set overrides** (in the older deploy
   page, **Edit Template**) and add three variables:

   | Key | Value |
   | --- | --- |
   | `HF_TOKEN` | `{{ RUNPOD_SECRET_hf_token }}` |
   | `GITHUB_TOKEN` | `{{ RUNPOD_SECRET_github_token }}` |
   | `UV_NO_SYNC` | `1` |

   The key icon next to a value lists your secrets and writes the
   `{{ RUNPOD_SECRET_... }}` reference for you. `UV_NO_SYNC=1` stops
   `uv run` from replacing the CUDA build of torch with the CPU-only one, in
   every terminal and notebook from the moment the pod starts. The script
   still installs the environment, because `uv sync` ignores it.
4. **GPU:** choose **RTX 4090**. Open **Filter**. **Cloud type** should
   already say **Secure**: leave it, because choosing a cloud type, even the
   one already shown, resets the other filters. (If it says **Community**,
   choose **Secure**.) **CUDA versions** should say **5 items selected**:
   13.0 to 13.4, set by the template. If it says **Any**, open it and tick
   13.0, 13.1, 13.2, 13.3 and 13.4. Then select **Apply filters**.
5. **Storage:** leave the container disk at its default. Under persistent
   storage, choose **Volume disk**, set it to 100 GB and keep the mount path
   `/workspace`.
6. Check the price in the summary, then select **Deploy Pod** (or
   **Deploy On-Demand**).

What survives what:

- **The volume disk (`/workspace`)** keeps the repository, the Python
  environment, the models and the logs when the pod stops or restarts. It is
  deleted when you terminate the pod. It costs $0.10 per GB a month while the
  pod runs and $0.20 while it is stopped: $10 or $20 a month for 100 GB.
- **The container disk (everything else)** is wiped whenever the pod stops.
  That includes the uv and Claude Code the script installs and your Claude
  sign-in, which the script and `claude` set up again.
- **If your Runpod balance reaches $0**, Runpod terminates a pod that has no
  network volume, and `/workspace` is lost with it. A stopped pod still costs
  about $20 a month, so keep enough credit, or turn on low-balance
  notifications or auto-pay on the
  [billing page](https://www.console.runpod.io/user/billing).

## 4. Open a terminal on the pod

On the Pods page, select your pod. Under **HTTP Services**, select
**Jupyter Lab**, then **File → New → Terminal**. A JupyterLab terminal keeps
running if you close the browser tab; reopen it from the **Running Terminals
and Kernels** panel on the left.

## 5. Clone the repository

Paste these lines into the terminal:

```bash
cd /workspace
git -c credential.helper='!f() { echo username=jonleach323; echo "password=$GITHUB_TOKEN"; }; f' \
  clone https://github.com/jonleach323/Qwen-Expirements.git
cd Qwen-Expirements
git config credential.helper '!f() { echo username=jonleach323; echo "password=$GITHUB_TOKEN"; }; f'
```

The `credential.helper` setting tells git to read the password from
`$GITHUB_TOKEN` each time it needs one. So git never saves the token, and
`git pull` keeps working after a restart, because the setting is saved in
the repository on the volume.

## 6. Run the setup script

```bash
bash scripts/setup_pod.sh
```

It ends like this:

```text
[2026-10-29 14:37:52] [8/8] Checking the total time against the target
      Finished in 6m 40s: beat the 20-minute target for a fresh pod.

The pod is ready. New terminals pick up its settings by themselves;
in this one, run first:  source /workspace/pushlab.env
Then sign in to Claude Code with:  claude
```

Run the two commands it shows. `claude` prints a link: open it, sign in and
paste the code back into the terminal.

## Check that it worked

- Step 1 shows `NVIDIA GeForce RTX 4090` and about 24,000 MiB of memory.
- Step 7 ends with `7 passed`.
- Step 8 says `beat the 20-minute target`.
- The output never says `the pod does not set UV_NO_SYNC=1 itself`.
- `ls /workspace/models/qwen3.5-4b` lists `config.json` and the
  `.safetensors` weight files.
- `grep UV_NO_SYNC /etc/rp_environment` prints `export UV_NO_SYNC="1"`.
  That file holds the pod's own variables; `echo $UV_NO_SYNC` would also
  show the copy in `pushlab.env`.
- This prints `2.14.1+cu130 NVIDIA GeForce RTX 4090`:

  ```bash
  uv run python -c "import torch; print(torch.__version__, torch.cuda.get_device_name())"
  ```

## Every time you start the pod again

1. Open a JupyterLab terminal (step 4 above).
2. Run:

   ```bash
   cd /workspace/Qwen-Expirements
   git pull
   bash scripts/setup_pod.sh
   ```

   It reinstalls uv and Claude Code, checks everything else and skips the
   finished work. Step 7 loads the model three times for its tests, so it
   is the slowest step on a warm volume.
3. Run `source /workspace/pushlab.env` and `claude` again, as the script
   says.

Stop the pod whenever you are not using it, because the GPU is billed for
every second the pod runs. Terminate it only when you are done for good: that
deletes `/workspace`.

## Generate the baseline reply

`pushlab baseline` loads a downloaded model, asks it one question with
thinking off, always picks the most likely next token, and saves the reply
as a run. It is a GPU job, so you start it yourself:

```bash
cd /workspace/Qwen-Expirements
uv run pushlab baseline --model qwen3.5-4b --out /workspace/runs/baseline-$(date -u +%Y%m%d-%H%M%S)
```

Its first line should say `Loading qwen3.5-4b in bfloat16 on the GPU...`.
If it says `float32 on the CPU`, torch cannot use the GPU: press Ctrl-C and
run `bash scripts/setup_pod.sh`. `--model` works only for a model that
`configs/models.yaml` lists, because the setup script downloads only those.

It prints the reply and writes two files into the `--out` folder:

- `runs.parquet`, the run `s0-baseline-000001`: the reply's text and token
  IDs, the model and its commit, and the decoding settings.
- `manifest.json`, what produced it: the code commit, the lockfile's hash,
  the model's ID and commit, the GPU, the CUDA and driver versions, the
  versions of torch, transformers and the GPU kernels, and the start and
  end times.

The question comes from `experiments/p0/s0_baseline.yaml`; `--prompt "..."`
asks another one, and `--max-new-tokens` caps the reply (200 by default).
The command above gives every run a new folder, named after the date and
time. It refuses a folder that already holds a run unless you add
`--overwrite`, which replaces that run for good. Never use
`/workspace/runs/s0-baseline` as `--out`: it holds the S0 reference run
(`docs/decisions/0005-s0-baseline.md`), and its `runs.parquet` is the only
copy of that reply's token IDs. Keep outputs under `/workspace/runs`,
outside the repository: run outputs never go in git.

To look at a run, start Python with `uv run python` and type this. It reads
the S0 reference run; give your own `--out` folder to read another.

```python
from pathlib import Path
from pushlab.store.records import read_runs

print(read_runs(Path("/workspace/runs/s0-baseline"))[0].output_text)
```

## Measure the residual sizes

A dose is measured in each layer's typical residual size, so each model
needs its sizes measured once before any neuron is pushed. `pushlab
calibrate` runs the 32 prompts in `prompts/calibration.jsonl` through the
model and saves the median size leaving each layer. It is a GPU job, so you
start it yourself:

```bash
cd /workspace/Qwen-Expirements
uv run pushlab calibrate --model qwen3.5-4b
```

Its first line should say `Loading qwen3.5-4b in bfloat16 on the GPU...`,
as for the baseline. It prints one size per layer and saves them in
`calibration/qwen3.5-4b-<commit>.json`, a folder that git ignores, so the
file never counts as a change to the repository. A model is measured once:
running it again stops unless you add `--overwrite`. Decision
`docs/decisions/0007-calibration.md` says what the numbers mean.

## Push a neuron and compare the replies

Pushes are measured against the residual sizes, so measure them first (the
section above). Then, from `/workspace/Qwen-Expirements`:

1. See what a dose adds to a neuron. The command loads the model, because
   the amount depends on the neuron's weights:

   ```bash
   uv run pushlab amount --unit qwen3.5-4b:L14:N8949 --alpha 1
   ```

2. Reply to the five smoke prompts with no push, at dose 0, at dose 1, and
   with a random neuron of the same layer at dose 1. Each command writes
   one folder, and refuses a folder that already holds runs unless you add
   `--overwrite`:

   ```bash
   uv run pushlab steer --model qwen3.5-4b --prompts prompts/smoke.jsonl --out /workspace/runs/s1-smoke-none
   ```

   ```bash
   uv run pushlab steer --model qwen3.5-4b --prompts prompts/smoke.jsonl --out /workspace/runs/s1-smoke-dose0 --unit qwen3.5-4b:L14:N8949 --alpha 0
   ```

   ```bash
   uv run pushlab steer --model qwen3.5-4b --prompts prompts/smoke.jsonl --out /workspace/runs/s1-smoke-dose1 --unit qwen3.5-4b:L14:N8949 --alpha 1
   ```

   ```bash
   uv run pushlab steer --model qwen3.5-4b --prompts prompts/smoke.jsonl --out /workspace/runs/s1-smoke-control --unit qwen3.5-4b:L14:N8949 --alpha 1 --control random --control-seed 1
   ```

   By default a push starts on the prompt and lasts the whole reply, as
   in the evil-Qwen recipe. `--start`, `--end` and `--no-include-prompt`
   narrow it: a push on reply token k first changes token k + 1, so for a
   push that starts late, add `--no-include-prompt`. Replies at dose 1 can be hostile; like every run output,
   they stay on the pod.

3. Lay the folders side by side in one Markdown table:

   ```bash
   uv run pushlab report --runs /workspace/runs/s1-smoke-none /workspace/runs/s1-smoke-dose0 /workspace/runs/s1-smoke-dose1 /workspace/runs/s1-smoke-control --out /workspace/runs/s1-smoke-report.md
   ```

   To read it, right-click the file in JupyterLab's file browser and choose
   **Open With > Markdown Preview**; a double-click shows its raw text.

Compare dose 0 with the folder made without `--unit`: both run the same
prompts in the same batch, so their replies must be identical. `pushlab
baseline` runs one prompt alone, and on the GPU a batch can tip a near tie
the other way (decision 0009). Decision `docs/decisions/0010-steer-and-report.md`
says what each command records.

## Rules on the pod

- **Never run plain `uv sync` on the pod.** It would replace the CUDA build
  of torch with the CPU-only one. After a `git pull` that changes `uv.lock`,
  `pyproject.toml`, `configs/models.yaml` or a model's revision in
  `src/pushlab/models/registry.py`, run `bash scripts/setup_pod.sh` again: it
  updates the environment, installs the `pushlab` command and downloads new
  models and revisions.
- Don't edit files of the repository on the pod, such as
  `configs/models.yaml`. Change them through a pull request from a cloud
  session, then `git pull`. An edit made on the pod makes every manifest
  record uncommitted code (`code_dirty: true`) and can stop a later
  `git pull`.
- `uv run ...` is safe wherever `UV_NO_SYNC=1` is set, and the pod variable
  from step 3 of "Deploy the pod" sets it in every terminal and notebook.
  Without it, a plain `uv run` replaces the CUDA build of torch with the
  CPU-only one.
- Never type a token into a command or save it in a file. The pod already has
  both tokens as environment variables.
- Keep your own files out of `/workspace/models`. The script manages those
  folders: when it downloads again, it replaces or deletes any file Hugging
  Face put there, including files you edited or fetched yourself with
  `hf download`. It leaves other files alone, but they don't belong there.

## If something goes wrong

The last step number on screen shows where the script stopped, and the log
file in `/workspace/logs` has everything. Fix the cause and run the script
again; it skips the work already finished. If a fix restarts the pod,
continue with "Every time you start the pod again" once it is running. If
no entry below matches, or the fix is not something you can do on the pod,
start a Claude Code cloud session and give it the message and the log.

- **"No GPU is visible".** The pod has no GPU. This happens when a stopped
  pod is started again after someone else rented its machine's GPUs:
  Runpod then offers to start it with CPUs only. Stop the pod and either
  start it again later, or accept Runpod's offer to migrate the pod's data
  when it says no GPU is free. Migrating makes a new pod, with a new ID, on a
  machine with a free GPU and copies `/workspace` to it. If the old pod is
  still listed afterwards, it keeps its own copy of `/workspace` and its
  storage bill: terminate it once the new pod has run the script. To avoid
  this,
  consider a **Network volume** instead of a volume disk next time: it is not
  tied to one machine (Secure Cloud only, created at
  <https://www.console.runpod.io/user/storage>).
- **"The NVIDIA driver is ..., but pushlab's CUDA 13.0 build of torch needs
  driver 580 or newer".** The machine's driver is too old. On a fresh pod,
  terminate it so it stops costing money (nothing on it is needed yet). On
  a warm volume disk, for example after a migration, terminating deletes
  `/workspace`, so copy off what you need first (a network volume is kept).
  Then deploy a new pod with CUDA 13.0 chosen as in steps 2 and 4 of
  "Deploy the pod".
- **The pod never finishes starting: its page says "Minimum CUDA version
  requirement not met (13.0)", or its system log says
  `unsatisfied condition: cuda>=13.0`.** The pod landed on a machine whose
  driver is too old for the CUDA 13.0 image, so the script cannot even run.
  Terminate it and deploy again with **CUDA versions** set as in step 4 of
  "Deploy the pod".
- **"HF_TOKEN is not set".** The pod has no `HF_TOKEN` variable. On the Pods
  page, open the pod's **⋮** menu, select **Edit Pod**, add the variable as
  in step 3 of "Deploy the pod" and select **Save**. The pod restarts, which
  keeps `/workspace`.
- **"warning: the pod does not set UV_NO_SYNC=1 itself".** Add the
  `UV_NO_SYNC` variable with the value `1` the same way, with **Edit Pod**.
  Until then, a plain `uv run` in a notebook, or in a terminal opened before
  the script ran, can swap in the CPU-only torch.
- **"Hugging Face refused the download: Client error '401 Unauthorized'
  ..."** The token is wrong or was deleted. Create a new one, update the
  `hf_token` secret with **Edit Secret Value**, and restart the pod.
- **"Hugging Face refused the download: Server error '5..'" or "429 Too
  Many Requests".** The trouble is on Hugging Face's side. Wait a few
  minutes and run the script again; the files that finished are kept.
- **"The download stopped: ..."** If the message mentions a connection, a
  timeout or missing files, the connection to Hugging Face failed: run the
  script again; the files that finished are kept. Anything else, such as
  "No space left on device", fails again until it is fixed, so ask Claude
  about the message and the log in `/workspace/logs` first.
- **"The download stopped: ValueError: commit ... has no config.json or no
  .safetensors weights".** The revision in `src/pushlab/models/registry.py`
  points at the wrong commit, so running the script again will not help.
  Ask Claude to fix it in a cloud session, merge it, then `git pull` on the
  pod and run the script again.
- **`git clone` or `git pull` fails with "Authentication failed" or
  "Repository not found".** The `GITHUB_TOKEN` variable is missing, the
  token has expired (after 90 days) or it can't see this repository.
  `echo ${GITHUB_TOKEN:+set}` prints `set` if the pod has the variable,
  without showing it. If it prints an empty line, add `GITHUB_TOKEN` with
  the value `{{ RUNPOD_SECRET_github_token }}` through **Edit Pod**, as for
  `HF_TOKEN`. If it prints `set`, create a new token as in step 1, update
  the `github_token` secret with **Edit Secret Value** and restart the pod.
  If it was the first `git clone` that failed, nothing was cloned: once the
  pod is running again, repeat "5. Clone the repository" instead.
- **"needs to be updated, but `--locked` was provided".** Someone changed
  `pyproject.toml` without `uv.lock`. Fix it in a cloud session with
  `uv lock`, merge it, then `git pull` on the pod.
- **Step 7 fails, with `FAILED` lines from pytest.** Running the script
  again loads the model again and fails the same way, so don't. Give
  Claude, in a cloud session, the `FAILED` lines and the log. A failed
  `test_the_qwen3_5_kernels_import` means causal-conv1d or
  flash-linear-attention is installed but does not load; step 3 has just
  checked them, so the fix needs Claude. "the reply ran to
  --max-new-tokens" means the model's reply did not end within 200 tokens;
  the log shows the reply. A failed
  `test_each_row_of_a_padded_batch_matches_its_prompt_alone` means a reply
  in a batch strayed further from the same prompt run alone than rounding
  explains: give Claude its `row ...` lines too. A failed
  `test_each_row_with_its_own_schedule_matches_its_prompt_alone` means a
  row's pushes were not exactly its own, or its scores strayed: give Claude
  the newest `/workspace/logs/isolation-*.json` too.
- **Step 7 passes with `1 warning` about a batched row's scores.** One of
  the two batch checks passed, but closer to its limit than expected.
  Nothing is wrong yet; paste the warning to Claude in a cloud session, so
  the limit in `tests/gpu/test_generate_smoke.py` can be judged against it.
- **`git pull` says "Your local changes to the following files would be
  overwritten by merge".** A file of the repository was edited on the pod.
  The change belongs in a pull request instead. If the edit may be worth
  keeping, first save it outside the repository with
  `git diff HEAD > /workspace/pod-edit.patch`, and give that file's text to
  Claude in a cloud session to make the change through a pull request. Then
  throw the local copy away with `git checkout HEAD -- <the file it names>`,
  and run `git pull` and `bash scripts/setup_pod.sh` again.
- **"error: Failed to spawn: `pushlab`".** First run
  `cd /workspace/Qwen-Expirements`: uv finds the command only inside the
  repository. If it still fails, the command is not installed yet, because
  the setup script has not run since a `git pull` added it. Run
  `bash scripts/setup_pod.sh`, then the command again.
- **"pushlab baseline: ... is not downloaded into /workspace/models/...".**
  If `configs/models.yaml` lists the model, run `bash scripts/setup_pod.sh`;
  it downloads the model, or its new revision after a pull that changed
  one. If the file does not list it, add it through a pull request from a
  cloud session, `git pull` on the pod, and then run the script.
- **You pressed Ctrl-C.** Run the script again.
- **The browser lost its connection.** The script keeps running on the pod.
  Reopen its terminal from **Running Terminals and Kernels** and let it
  finish. Run it again only if it has stopped.
