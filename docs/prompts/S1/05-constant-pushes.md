This is Sprint S1, prompt 5 of 8.

CONTEXT
Read CLAUDE.md and these parts of the plans:
- docs/plan/development-plan.md: the "2. units" and "4. engine" modules.
  The engine's decisions say one hook per targeted layer adds the whole
  batch's amounts in one vectorized step, so every row can follow its own
  schedule.
- docs/plan/research-plan.md: "PyTorch hooks" under "Skills".

Prompt 2 gave load_model's result the MLP modules, prompt 3 built neuron
units with amount(alpha), and prompt 4 measured the residual sizes that
amount needs.

GOAL
Push chosen neurons by fixed amounts on every token, so that a zero dose
changes nothing at all.

BUILD
- src/pushlab/engine/hooks.py: a context manager that, for a list of
  pushes, registers one forward pre-hook on each targeted layer's
  down_proj. Leaving the context removes every hook, even after an error.
- A push is a bound neuron (prompt 3) with one amount per batch row.
  Neurons in the same layer share that layer's hook.
- The hook adds each row's amount to the neuron's MLP value at every real
  token: all of the prompt pass and each generated token. Padding
  positions are never pushed. The caller says which positions are padding;
  prompt 6's loop will.
- The hook state has a step counter: the prompt pass, then 0, 1, 2 and so
  on for reply tokens. Constant pushes ignore it, prompt 6's loop advances
  it, and S2's schedules will use it.
- tests/gpu/test_push_equivalence.py (marked gpu and slow) uses L14 N8949 on
  Qwen3.5-4B in bfloat16. The owner runs it in S1's second pod session.
  It checks:
  - pushing the neuron's MLP value matches adding the same amount along
    its write direction to the MLP output, within bfloat16 tolerance (the
    units module's done-when in the plan);
  - a zero dose gives exactly the output with no hooks.
  Its name must not contain "smoke".
- A decision note (the next free number) covering:
  - where the push enters (the down_proj input);
  - how padding and the batch rows are handled;
  - why the amount is added in the activations' own data type.

CONSTRAINTS
- Constant pushes only. Schedules, regions and per-step amounts come in
  S2.
- A row's push never touches another row.
- No change to S0's `pushlab baseline` or greedy_generate.
- No hard-coded units or doses outside tests. No dependency changes.

TESTS FIRST
All on tiny_model and tiny_vl_model, in float32:
- With a zero dose, the logits are bit-for-bit equal to running with no
  hooks, for the prompt pass and for cached one-token steps.
- Pushing neuron n by c through its MLP value gives the same layer output
  as adding c · W_down[:, n] to that layer's MLP output, within float32
  tolerance.
- In a batch where row 0 is pushed and row 1 has amount 0, row 1's logits
  equal the unpushed run's.
- Layers before the pushed one are unchanged, and padding positions'
  MLP values are untouched.
- Two neurons in one layer use one hook and add independently.
- No hook remains after the context, including after an exception inside
  it.
- A push for another model, or amounts for the wrong batch size, are
  refused.

DONE WHEN
- `uv run pytest -q` passes on CPU, and the pull request's checks pass.
- The GPU test is skipped on CPU and listed for S1's second pod session
  (prompt 7).

REPORT BACK
End with a plain-language summary for a beginner: what changed, why, how I
can check it myself, and anything I must do by hand. Explain in plain words
what a hook is and why a zero dose must change nothing.
