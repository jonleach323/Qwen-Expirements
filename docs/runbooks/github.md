# Protect the main branch on GitHub

When you finish this runbook, GitHub will refuse to merge a pull request
into `main` until the **Lint and test** check has passed. Nobody, you
included, will be able to push straight to `main`, rewrite its history or
delete it. It takes about five minutes, all in the browser.

## Before you start

1. **Merge the pull request that adds `.github/workflows/tests.yml` first.**
   Until `main` contains that file, no other pull request would ever get
   the check, and so none of them could merge.
2. **The check should have passed recently.** GitHub suggests a check by
   name only if it passed in this repository in the last seven days. Open
   the repository, click the **Actions** tab, then **Tests** in the left
   sidebar, and look for a recent run with a green tick. If there is none,
   you can still type the name by hand in step 11.
3. **Your GitHub plan must cover it.** For a *private* repository, GitHub
   enforces these rules only on paid plans (GitHub Pro, Team or
   Enterprise). On the free plan they work for public repositories only.
   This repository is private. To see your plan, open
   <https://github.com/settings/billing>, click **Licensing** in the left
   sidebar and look at "Current plan". If it is the free plan, read "If
   your plan does not include it" at the end first.

## Steps

1. Open <https://github.com/jonleach323/Qwen-Expirements>.
2. Click **Settings** in the row of tabs under the repository name. If you
   don't see it, click the **⋯** menu at the end of that row, then
   **Settings**.
3. In the left sidebar, under "Code, planning, and automation", click
   **Rules** (it may already be open), then **Rulesets**.
4. Click **New ruleset**, then **New branch ruleset**.
5. Under "Ruleset name", type `Protect main`.
6. "Enforcement status" starts as **Disabled**, which would do nothing.
   Click it and choose **Active**.
7. Leave the "Bypass list" empty, so the rules apply to you too. As the
   repository's owner you can always edit or remove the ruleset later.
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
    typed. It must then appear in the list of required checks. Add it
    once, by that exact name, not as "Tests / Lint and test". Leave "Require
    branches to be up to date before merging" unticked.
12. Scroll to the bottom and click **Create**.

## Check that it works

- Open <https://github.com/jonleach323/Qwen-Expirements/rules>. **Protect
  main** should be listed as active.
- On your next pull request, the box near the bottom lists **Lint and
  test** twice, once for the push and once for the pull request. Both rows
  run the same checks and both count. The **Merge pull request** button
  stays unavailable until they are green.
- On the **Code** tab, open any file and click the pencil icon to edit it.
  When you commit, GitHub no longer offers to commit directly to `main`.
  It only offers to create a new branch and start a pull request.

## What changes for you

- Uploading or editing files on the GitHub website, as you did with the
  plans, now goes through a pull request:
  1. When committing, choose **Create a new branch for this commit and
     start a pull request**, then click **Propose changes**.
  2. On the next page, click **Create pull request**.
  3. Wait until the checks show a green tick.
  4. Click **Merge pull request**, then **Confirm merge**.
- Claude Code sessions already work on branches and open pull requests, so
  nothing changes there.

## If a pull request says "Expected — Waiting for status to be reported"

The required check never ran on that pull request. Common reasons: the
pull request's branch was created before `tests.yml` reached `main`, the
job's name in `tests.yml` was changed, or GitHub Actions is switched off
in Settings → Actions. Bring the branch up to date with `main` (the **Update
branch** button on the pull request), or fix the cause. If you are stuck,
pause the ruleset as described in "Undo".

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

Settings → **Rules** → **Rulesets** → click **Protect main**.

- To pause it, change "Enforcement status" to **Disabled**, then scroll to
  the bottom and click **Save changes**.
- To remove it for good, click the **⋯** button next to the ruleset's name
  at the top of the page, then **Delete ruleset**.
