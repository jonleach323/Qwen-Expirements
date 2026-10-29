# Set up a GPU pod on Runpod

`scripts/setup_pod.sh` turns a fresh Runpod GPU pod into a machine ready for
pushlab research. It runs eight steps:

1. It checks that a GPU is visible, and prints its name and memory.
2. It installs uv if it is missing.
3. It installs the locked CUDA environment: the CUDA build of torch, the GPU
   kernels for Qwen3.5 and the dev tools.
4. It installs the Claude Code CLI if it is missing.
5. It checks that your Hugging Face token is set.
6. It downloads the models in `configs/models.yaml` to `/workspace/models`.
7. It runs the GPU smoke tests.
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

Runpod keeps the tokens as encrypted secrets and hands them to the pod as
environment variables when it starts. They never go in a file.
`scripts/pod.env.example` lists the variable names the pod expects.

1. Open <https://www.console.runpod.io/user/secrets> and select
   **Create Secret**.
2. **Secret Name:** `hf_token`. **Secret Value:** paste the Hugging Face token.
   Select **Create Secret**.
3. Do the same with **Secret Name** `github_token` for the GitHub token.

Runpod never shows a secret's value again. To change one later, use
**Manage**, then **Edit Secret Value**.

## 3. Deploy the pod

1. In the [Runpod console](https://www.console.runpod.io/pods), select
   **+ New**, then **Pod** (or **Deploy** on the Pods page).
2. **Template:** search for `pytorch` and pick Runpod's official PyTorch
   template. Any recent version works, because the script installs its own
   torch. Keep **Start Jupyter notebook** on.
3. **Environment variables:** select **Set overrides** (in the older deploy
   page, **Edit Template**) and add two variables:

   | Key | Value |
   | --- | --- |
   | `HF_TOKEN` | `{{ RUNPOD_SECRET_hf_token }}` |
   | `GITHUB_TOKEN` | `{{ RUNPOD_SECRET_github_token }}` |

   The key icon next to a value lists your secrets and writes the
   `{{ RUNPOD_SECRET_... }}` reference for you.
4. **GPU:** choose **RTX 4090**. Under **Filter**, set **Cloud type** to
   **Secure**. If the filters offer **CUDA Versions**, select 13.0 and newer:
   pushlab's torch needs an NVIDIA driver from the CUDA 13.0 generation
   (version 580 or newer). Step 1 of the script checks this.
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
  That includes uv, Claude Code and your Claude sign-in, which the script and
  `claude` set up again.

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
`$GITHUB_TOKEN` each time it needs one. So the token is never written to
disk, and `git pull` keeps working after a restart, because the setting is
saved in the repository on the volume.

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
- Step 7 ends with `2 passed`.
- Step 8 says `beat the 20-minute target`.
- `ls /workspace/models/qwen3.5-4b` lists `config.json` and the
  `.safetensors` weight files.
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
   finished work. It should beat the 3-minute target for a warm volume.
3. Run `source /workspace/pushlab.env` and `claude` again, as the script
   says.

Stop the pod whenever you are not using it, because the GPU is billed by the
hour. Terminate it only when you are done for good: that deletes `/workspace`.

## Rules on the pod

- **Never run plain `uv sync` on the pod.** It would replace the CUDA build
  of torch with the CPU-only one. To update the environment after
  `uv.lock` changes, run `bash scripts/setup_pod.sh` again.
- `uv run ...` is safe in any terminal that has loaded `pushlab.env`, which
  new terminals do by themselves. It sets `UV_NO_SYNC=1`, so uv uses the
  environment as installed.
- Never type a token into a command or save it in a file. The pod already has
  both tokens as environment variables.

## If something goes wrong

The last step number on screen shows where the script stopped, and the log
file in `/workspace/logs` has everything. Fix the cause and run the script
again; it skips the work already finished.

- **"No GPU is visible".** The pod has no GPU. This happens when a stopped
  pod is started again after someone else rented its machine's GPUs:
  Runpod then offers to start it with CPUs only. Stop the pod and either
  start it again later, or accept Runpod's offer to migrate the pod's data
  when it says no GPU is free. Migrating makes a new pod, with a new ID, on a
  machine with a free GPU and copies `/workspace` to it. To avoid this,
  consider a **Network volume** instead of a volume disk next time: it is not
  tied to one machine (Secure Cloud only, created at
  <https://www.console.runpod.io/user/storage>).
- **"The NVIDIA driver is ..., but pushlab's CUDA 13.0 build of torch needs
  driver 580 or newer".** The machine's driver is too old. Deploy a new pod
  with CUDA 13.0 or newer selected under **CUDA Versions**.
- **"HF_TOKEN is not set".** The pod has no `HF_TOKEN` variable. On the Pods
  page, open the pod's **⋮** menu, select **Edit Pod**, add the variable as
  in step 3 and select **Save**. The pod restarts, which keeps `/workspace`.
  Then run the script again.
- **"Hugging Face refused the download: 401".** The token is wrong or was
  deleted. Create a new one, update the `hf_token` secret with
  **Edit Secret Value**, and restart the pod.
- **`git clone` or `git pull` asks for a username or password.** The
  `GITHUB_TOKEN` variable is missing, the token has expired (after 90 days)
  or it can't see this repository. Create a new token as in step 1, update
  the `github_token` secret and restart the pod. Press Ctrl-C to get out of
  the password prompt.
- **"needs to be updated, but `--locked` was provided".** Someone changed
  `pyproject.toml` without `uv.lock`. Fix it in a cloud session with
  `uv lock`, merge it, then `git pull` on the pod.
- **You pressed Ctrl-C, or the connection dropped.** Run the script again.
