# Set up Claude Code cloud sessions

`scripts/cloud_setup.sh` gets a new cloud session ready to run the unit
tests:

1. It checks that the machine has no GPU.
2. It makes sure uv is installed.
3. It installs Python 3.11 and the packages locked in `uv.lock`: the
   CPU-only build of torch, plus the dev tools.
4. It prints the Python, torch and transformers versions.

It never downloads model weights and is safe to run more than once. On a
fresh machine it takes under a minute; on a machine that is already set up,
a few seconds.

Set it up once as described below, and every new pushlab session starts
ready.

## Before you start

- **Merge the pull request that adds `scripts/cloud_setup.sh` first.** The
  links below point at `main`, and new sessions start from `main`.
- **Use a computer.** The Claude Desktop app or <https://claude.ai/code> in a
  browser both work; the steps need hovering and copying a file. Sessions you
  later start from your phone use the same settings.

## Create a pushlab environment

Use a separate cloud environment for pushlab instead of changing your
existing one. Your other projects share that environment, and two settings
below would affect them: the network allowlist, and the snapshot that the
setup script builds.

1. Open
   <https://github.com/jonleach323/Qwen-Expirements/blob/main/scripts/cloud_setup.sh>
   and click the copy button at the top right of the file. It shows two
   overlapping squares and is labelled "Copy raw file" or "Copy raw content".
2. At <https://claude.ai/code> (or in the Desktop app), on the new-session
   screen, select the first control in the row under the message box. That
   is the environment selector.
3. Select **New environment** at the bottom of the menu.
4. Name it `pushlab`.
5. Select the **Network access** row and choose **Custom**:
   - In **Allowed domains**, enter one per line:

     ```text
     download-r2.pytorch.org
     download.pytorch.org
     wheels.astral.sh
     releases.astral.sh
     ```

   - Tick **Also include default list of common package managers**. That
     list includes PyPI, where most packages come from.
6. Select the **Setup script** row and paste the script into the box. You
   can also download the file and use **Choose file**.
7. Select **Add environment**.
8. From now on, pick **pushlab** in the environment selector when you start
   a session for this repository.

To change it later, open the selector, find **pushlab** under **Cloud**,
hover over its row and select the settings icon on its right. If **Cloud**
is a single row, select it first; if pushlab isn't listed, select **More**.
Inside a running session, select the **⌄** after the session title, then
**Edit cloud environment**. Save with **Save changes**.

### Why Custom network access

- **Trusted**, the default, allows PyPI and GitHub but not PyTorch's
  download site, which hosts the CPU-only torch build. The setup fails.
- **Full** works, but it lets sessions reach any website. This project
  doesn't need that.
- **None** blocks every download, so the setup fails.

What each extra domain is for:

| Domain | Needed for |
| --- | --- |
| `download-r2.pytorch.org` | Downloading the CPU-only torch build |
| `download.pytorch.org` | Updating `uv.lock`, e.g. after adding a dependency |
| `wheels.astral.sh` | Updating `uv.lock` (it lists the GPU kernel builds) |
| `releases.astral.sh` | Installing Python 3.11 if a machine lacks it |

GitHub keeps working at any level, because it goes through its own proxy.
Hugging Face is deliberately not listed: unit tests never download models.

### When the script runs

The cloud runs the script for the first session in the environment. If it
finishes within about five minutes, which it does, the cloud saves a
snapshot of the prepared machine. Later sessions start from that snapshot
and skip the script.

The script runs again when you change the script or the allowed domains,
and about once a week. If `scripts/cloud_setup.sh` changes in the
repository, paste the new version. A change to `uv.lock` alone needs
nothing: the first `uv run` in a session installs whatever is new.

## Check that it worked

1. Start a new session in the **pushlab** environment.
2. Ask Claude: "Run `bash scripts/cloud_setup.sh`, then
   `uv run pytest -q`, and show me the output."
3. You should see four numbered steps.
   - Step 3 should finish within a second or two, saying "Checked N
     packages" or "Installed N packages", with no "Downloading torch" line.
     That means the session started ready.
   - The versions should include `torch 2.14.1+cpu`.
   - The tests should end with a line like "94 passed, 3 skipped in 2.0s".
   - A warning about `UV_NATIVE_TLS` at the top of the test output is
     harmless.

## If something goes wrong

- **A session for pushlab doesn't start, or says the setup script
  failed.** You may not have a session to ask Claude in, so:
  1. Open the pushlab environment's settings, clear the **Setup script**
     box and select **Save changes**.
  2. Start a session and ask Claude to run `bash scripts/cloud_setup.sh` and
     show the output. The last numbered step shows where it stopped.
  3. Fix the cause (see below), paste the script back in and save.
- **An error names a domain**, for example "Failed to download" followed by
  a `download-r2.pytorch.org` address. Network access is blocking it. Add
  the domain as described above.
- **"invalid peer certificate: UnknownIssuer".** The pasted script is an old
  copy. Paste the current version from `main`.
- **"This machine has a GPU."** The script is for cloud sessions only. GPU
  pods use `scripts/setup_pod.sh`; see `docs/runbooks/pod.md`.
- **"The pushlab repository is not on this machine"**. The session was for
  another repository, so there was nothing to install. This shouldn't happen
  in a dedicated pushlab environment.
- **"needs to be updated, but `--locked` was provided".** `pyproject.toml`
  changed without `uv.lock`. Run `uv lock` and commit both files together.
- **Step 2 shows a `curl` error and then installs uv anyway.** This is
  expected: uv's own installer isn't on the allowlist, so the script installs
  uv from PyPI instead. Cloud machines normally come with uv already.
