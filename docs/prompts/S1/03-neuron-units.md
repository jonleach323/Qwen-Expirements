This is Sprint S1, prompt 3 of 8.

CONTEXT
Read CLAUDE.md and these parts of the plans:
- docs/plan/development-plan.md: the "2. units" module and "Identifiers"
  under "Conventions";
- docs/plan/research-plan.md: "MLP neurons" and "Normalization" under
  "Concepts", and the dose rule after them.

Prompt 2 made load_model return the MLP modules in layer order. A neuron
is one of an MLP's intermediate values. It writes into the residual stream
through its column of down_proj, its "write direction".

GOAL
Represent a neuron that can be pushed, with its write direction and the
amount a dose adds.

BUILD
- src/pushlab/units/: a Neuron unit, named by model, layer and index. Its
  ID has the plan's form, such as `qwen3.5-4b:L14:N8949`, and parses back
  to the same unit. Layer and index must be in range for the registry's
  shape.
- Binding a neuron to a loaded model and its per-layer residual sizes
  gives the plan's interface:
  - `write_direction()` is the neuron's down_proj column (hidden size
    long), as float32.
  - `amount(alpha)` follows the research plan's dose rule,
    c = alpha · r̃_L / ‖W_down[:, n]‖. Here r̃_L is the median residual size
    at the neuron's layer, and the column's length is computed in float32.
    Prompt 4 measures the residual sizes; until then, tests pass them in.
- A decision note in docs/decisions/ (the next free number):
  - the ID format;
  - why lengths are computed in float32;
  - that a negative dose pushes the other way.

CONSTRAINTS
- Neurons only. Sparse-autoencoder features, dense vectors and the
  Assistant Axis come in later sprints.
- No hooks (prompt 5), no file input or output, no command-line changes.
- Layers and neurons are 0-indexed everywhere (CLAUDE.md). The published
  recipe numbers them from 0 too: L14 N8949 is layer 14, index 8949.
- No hard-coded units or doses outside tests.
- No dependency changes.

TESTS FIRST
- IDs round-trip, and malformed ones are refused: a missing "L" or "N",
  negative or non-numeric numbers, or a model not in the registry.
- Range checks use the registry's shape. For qwen3.5-4b, layer 31 and
  index 9215 are accepted; layer 32 and index 9216 are refused.
- On tiny_model and tiny_vl_model, write_direction() equals that layer's
  down_proj weight column.
- amount() matches hand-worked numbers. With r̃ = 10 and a column of
  length 2: amount(1) = 5, amount(0) = 0, amount(-0.5) = -2.5.
- On the tiny models' real columns, the push the amount adds,
  amount(alpha) · write_direction(), has length |alpha| · r̃.
- Residual sizes of the wrong length, or not positive, are refused.

DONE WHEN
`uv run pytest -q` passes on CPU and the pull request's checks pass. This
prompt has no GPU step.

REPORT BACK
End with a plain-language summary for a beginner: what changed, why, how I
can check it myself, and anything I must do by hand. Explain in two or
three sentences what "dose 1" means for L14 N8949.
