# Set up Claude Code cloud sessions

`scripts/cloud_setup.sh` gets a new cloud session ready to run the unit
tests:

1. It checks that the machine has no GPU.
2. It makes sure uv is installed.
3. It installs Python 3.11 and the packages locked in `uv.lock`: the
   CPU-only build of torch, plus the dev tools.
4. It prints the Python, torch and transformers versions.

It never downloads model weights and is safe to run more than once. On a
fresh machine it takes about 40 seconds; on a machine that is already set up,
a few seconds.

Paste the script into your cloud environment's settings once, and every new
session starts ready.

## Paste the script into the environment's settings

1. Open
   <https://github.com/jonleach323/Qwen-Expirements/blob/main/scripts/cloud_setup.sh>
   and click the **Copy raw file** button (two overlapping squares) at the
   top right of the file.
2. Open your cloud environment's settings, in either of two ways:
   - In a running cloud session: select the **⌄** after the session title
     at the top, then **Edit cloud environment**.
   - At <https://claude.ai/code>, on the new-session screen: select the first
     control in the row under the message box. Under **Cloud**, hover over
     your environment's row and select the settings icon on its right.
3. Select the **Setup script** row. Delete anything already in the box and
   paste the script. You can also download the file and use **Choose file**.
4. Select **Save changes**.

The cloud runs the script once, then saves a snapshot of the prepared
machine. Later sessions start from that snapshot and skip the script. It
runs again when you change the script or the network settings, and about
once a week. If `scripts/cloud_setup.sh` changes in the repository, paste
the new version.

## Choose the network access level

Use **Custom**, with the default package managers plus four extra domains:

1. In the same settings dialog, select the **Network access** row and
   choose **Custom**.
2. In **Allowed domains**, enter one per line:

   ```text
   download-r2.pytorch.org
   download.pytorch.org
   wheels.astral.sh
   releases.astral.sh
   ```

3. Tick **Also include default list of common package managers**. That list
   includes PyPI, where most packages come from.
4. Select **Save changes**.

Why not the other levels:

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

## Check that it worked

1. Start a new cloud session in this environment.
2. Ask Claude: "Run `bash scripts/cloud_setup.sh`, then
   `uv run pytest -q`, and show me the output."
3. You should see four numbered steps. Step 3 should say "Checked 55
   packages" rather than downloading torch, which means the session started
   ready. The versions should include `torch 2.14.1+cpu`, and the tests
   should end with "passed, 1 skipped".

## If something goes wrong

- **The session says the setup script failed.** Ask Claude to run
  `bash scripts/cloud_setup.sh` and show the output. The last numbered step
  shows where it stopped.
- **An error names a domain** such as `download-r2.pytorch.org`. Network
  access is blocking it. Add the domain as described above.
- **"This machine has a GPU."** The script is for cloud sessions only. GPU
  pods get their own setup script later in Sprint 0.
- **"The pushlab repository is not on this machine yet."** This is
  harmless: the cloud ran the script before cloning the repository. The
  first `uv run` or `uv sync` in the session installs the packages, which
  takes about a minute.
- **"The lockfile needs to be updated, but --locked was provided."**
  `pyproject.toml` changed without `uv.lock`. Run `uv lock` and commit both
  files together.
