This is Sprint S1, prompt 8 of 8.

CONTEXT
Read CLAUDE.md, docs/decisions/, docs/sprints/S0-review.md, the S1
prompts in docs/prompts/S1/, and the S1 and S2 entries under "Sprint
details" in docs/plan/development-plan.md. The owner pastes in what the
two S1 pod sessions printed:
- the warm setup times;
- the model shape test's result and Qwen3-8B's download commit;
- both calibration files;
- the s1-evil-smoke command's summary, its labels and its manifest;
- the gpu tests' results.

GOAL
Close Sprint S1 honestly and prepare Sprint S2.

BUILD
- docs/sprints/S1-review.md, laid out like the S0 review:
  - each part of S1's done-when with its evidence, or a plain "not yet":
    - a zero dose matches no hooks exactly, on the GPU;
    - the evil neuron turns 5 of 5 smoke prompts hostile, with the owner's
      labels;
    - the passing workflow run on main;
  - the models module's GPU shape check;
  - what slowed us down, and what to change in S2.
- Record the pod results:
  - decision 0005's warm-start row;
  - docs/baselines/s1/: a copy of the s1-evil-smoke manifest, the
    Qwen3.5-4B calibration file, and the labels with each reply's SHA-256
    (no reply text);
  - a decision note for anything the results decided. One example: whether
    `pushlab baseline` switches to the batched loop, after prompt 7's gpu
    test.
- Pin Qwen3-8B to the commit in its download record, in
  src/pushlab/models/registry.py and the comment in configs/models.yaml, as
  decision 0005 did for Qwen3.5-4B.
- docs/prompts/S2/: one prompt per S2 task, each in the seven-part format
  used in S0 and S1 (CONTEXT, GOAL, BUILD, CONSTRAINTS, TESTS FIRST, DONE
  WHEN, REPORT BACK). Size them by the S1 review's lessons.

CONSTRAINTS
- If a done-when check isn't met, say so plainly and list what's missing
  instead of marking S1 done.
- The Qwen3-8B pin is the only code change. A test that expected "main"
  for Qwen3-8B may change, with the reason in the pull request
  description.
- Reply text stays on the pod; only hashes, labels and manifests go in
  git.

TESTS FIRST
For the pin only: the registry test expects Qwen3-8B's pinned commit, and
the download tests still cover a revision change. Run `uv run pytest -q`
and `uv run pre-commit run --all-files`.

DONE WHEN
- The S1 review covers every part of S1's done-when with evidence or a
  clear "not yet".
- docs/prompts/S2/ holds one prompt per S2 task.
- The pull request's checks pass.

REPORT BACK
End with a plain-language summary for a beginner: what S1 achieved, what's
still open, and what I should do first in S2.
