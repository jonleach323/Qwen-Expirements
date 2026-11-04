This is Sprint S1, prompt 1 of 8.

CONTEXT
Read CLAUDE.md, docs/sprints/S0-review.md ("Still open after S0" and "What
to change in S1"), docs/runbooks/pod.md, docs/runbooks/cloud-sessions.md
and docs/decisions/0003-pod-setup.md and 0005-s0-baseline.md.

GOAL
Close the S0 items that need only documentation or a decision, so the
rest of S1 starts from a clean base.

BUILD
- CLAUDE.md: change the current sprint line to "S1 (Engine v1)".
- docs/runbooks/experiments.md, a new page for running and copying runs.
  docs/runbooks/pod.md is already 384 lines, so S1's later prompts add to
  this page, not to pod.md.
  - It starts with "Copy a run off the pod": how to download a run
    folder's files from JupyterLab's file browser to the owner's computer.
    Its first example is the S0 reference run in /workspace/runs/s0-baseline,
    which has no other copy. It needs only a pod start, without
    `git pull`.
  - Link it from pod.md and README.md.
  - Keep the baseline section in pod.md: tests/unit/test_docs.py checks
    that pod.md shows the baseline command.
- docs/runbooks/pod.md, under "If something goes wrong":
  - An entry for `Syntax error: "(" unexpected` (or bash's `syntax error
    near unexpected token '('`) when cloning. The straight quotes became
    curly ones, which phones and some apps do. Say how to paste the command
    safely, and how to turn off Smart Punctuation on an iPhone or iPad.
  - An entry for the RTX 4090 being out of capacity:
    - Waiting and trying again later often works.
    - First choices: an L40S or RTX 6000 Ada (48 GB, the 4090's
      generation).
    - Next: an RTX A6000 or A40 (48 GB, one generation older, cheaper and
      slower). An A100 80GB works but costs about twice as much.
    - Never a card under 24 GB, and never one older than the A6000/A40
      generation (they lack bfloat16).
    - Always keep Secure Cloud and the CUDA 13.0 to 13.4 filter.
- docs/runbooks/pod.md, "Deploy the pod": how to save the deploy settings
  as the owner's own Runpod template, with the GPU filter, the three
  environment variables and the 100 GB volume. Then a new pod needs fewer
  steps. This is the "pod template" the plan's S0 entry asks for. Follow
  Runpod's documentation; if you can't confirm the steps, say so instead
  of guessing.
- docs/runbooks/cloud-sessions.md, under "If something goes wrong": a
  session whose tests collect nothing, or whose /home/user is empty, was
  started without the repository. Start a new session with
  jonleach323/Qwen-Expirements and the pushlab environment both selected.
- docs/decisions/: a note (the next free number) proposing which cards
  stand in for the RTX 4090.
  - Results from different GPU types are compared through their
    manifests, which record the GPU.
  - A reference run is only repeated on its own GPU type.
  - Mark it "proposed"; the owner accepts it in the pull request.
- Ask the owner whether huggingface-hub should become a dependency of its
  own, since src/pushlab/models/download.py imports it directly.
  - If yes, add it with uv add and a version floor, and commit uv.lock
    with pyproject.toml.
  - If no, add a comment in pyproject.toml saying it comes in through
    transformers.

CONSTRAINTS
- Change no Python code under src/ or tests/, and no scripts.
- Every command in a doc must be one you have run, or one the runbooks
  already use.
- Short entries. pod.md must stay under about 400 lines (CLAUDE.md).
- Pod steps stay the owner's (CLAUDE.md).

TESTS FIRST
No new code, so no new unit tests. Run `uv run pytest -q` and
`uv run pre-commit run --all-files` before and after the change. Both must
pass, and tests/unit/test_docs.py also checks the runbooks' commands.

DONE WHEN
- The new page, the runbook entries, the template steps (or the stated
  reason there are none), the decision note and the CLAUDE.md line are in
  a pull request whose checks pass.
- The huggingface-hub question has the owner's answer, acted on.

REPORT BACK
End with a plain-language summary for a beginner: what changed, why, how I
can check it myself, and anything I must do by hand. Put these first:
- As soon as this is merged, start the pod and copy
  /workspace/runs/s0-baseline off it, following the new page. Don't pull,
  and don't run anything else.
- Until S1's first pod session, don't run `pushlab baseline` on the pod.
  Its checkout is still S0's, whose docs write into the reference folder.
- Check branch protection under Settings → Rules → Rulesets.
