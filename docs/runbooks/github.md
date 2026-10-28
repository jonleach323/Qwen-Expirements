# Protect the main branch on GitHub

When you finish this runbook, GitHub will refuse to merge a pull request
into `main` until the **Lint and test** check has passed. Nobody, you
included, will be able to push straight to `main`, rewrite its history or
delete it. It takes about five minutes, all in the browser.

## Before you start

1. **The check must have run at least once.** GitHub only offers a check
   once it has seen it. Open the repository, click the **Actions** tab, and
   click **Tests** in the left sidebar. You should see at least one run. Every
   push and pull request starts one.
2. **Your GitHub plan must cover it.** For a *private* repository, GitHub
   enforces these rules only on paid plans (GitHub Pro, Team or
   Enterprise). On the free plan they work for public repositories only.
   This repository is private. You can see your plan at
   <https://github.com/settings/billing>. If you are on the free plan, read
   "If your plan does not include it" at the end first.

## Steps

1. Open <https://github.com/jonleach323/Qwen-Expirements>.
2. Click **Settings** in the row of tabs under the repository name. If you
   don't see it, click the **⋯** menu at the end of that row, then
   **Settings**.
3. In the left sidebar, under "Code, planning, and automation", click
   **Rules** (it may already be open), then **Rulesets**.
4. Click **New ruleset**, then **New branch ruleset**.
5. Under "Ruleset name", type `Protect main`.
6. Set "Enforcement status" to **Active**.
7. Leave the "Bypass list" empty, so the rules apply to you too.
8. Under "Target branches", click **Add a target**, then **Include default
   branch**. That is `main`.
9. In the list of rules below, make sure **Restrict deletions** and **Block
   force pushes** are ticked. They usually are already.
10. Tick **Require a pull request before merging**. Set "Required
    approvals" to **0**. GitHub never lets you approve your own pull
    request, so with 1 required approval you could not merge your own work.
11. Tick **Require status checks to pass** (the full label may end in
    "before merging"). Click **Add checks** and type `Lint and test`. Pick
    it from the suggestions, or click the **+** button next to what you
    typed. It must then appear in the list of required checks.
12. Scroll to the bottom and click **Create**.

## Check that it works

- Open any pull request. Near the bottom, the merge box lists
  **Lint and test** as *Required*. The merge button stays unavailable until
  the check passes.
- On the **Code** tab, open any file and click the pencil icon to edit it.
  When you commit, GitHub no longer offers to commit directly to `main`.
  It only offers to create a new branch and start a pull request.

## What changes for you

- Uploading files on the GitHub website, as you did with the plans, now
  goes through a pull request. When committing, choose **Create a new
  branch for this commit and start a pull request**, wait for the green
  check, then click **Merge pull request**.
- Claude Code sessions already work on branches and open pull requests, so
  nothing changes there.

## If your plan does not include it

GitHub may say the ruleset won't be enforced, or ask you to upgrade.
You have three choices:

- **Upgrade** your personal account to GitHub Pro, then follow the steps
  above.
- **Make the repository public.** The plan keeps it private until the first
  release, so this is only for later.
- **Go without enforcement.** The check still runs on every pull request
  and shows a green tick or a red cross next to it. Only click **Merge
  pull request** when it is green.

## Undo

Settings → **Rules** → **Rulesets** → **Protect main**. Set
"Enforcement status" to **Disabled**, or delete the ruleset at the bottom
of the page.
