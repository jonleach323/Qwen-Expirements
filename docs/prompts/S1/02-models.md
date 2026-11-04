This is Sprint S1, prompt 2 of 8.

CONTEXT
Read CLAUDE.md, the "1. models" module and the S1 entry under "Sprint
details" in docs/plan/development-plan.md, docs/decisions/0004 and 0005,
src/pushlab/models/ (registry.py, discovery.py, load.py) and
configs/models.yaml. S0 built the registry, MLP discovery and
load_model(name, dtype), but only Qwen3.5-4B has ever been downloaded and
loaded.

GOAL
Finish the models module as the plan describes it, and show on the GPU
that both main models load with the right shape.

BUILD
- load_model's result also carries the list of MLP modules in layer
  order, as the plan's interface asks. Find them with find_mlp_modules;
  later prompts hook into them.
- configs/models.yaml lists qwen3-8b as well as qwen3.5-4b, so the pod
  downloads it. It is about 16 GB more on the 100 GB volume.
- tests/gpu/test_model_shapes.py (marked gpu and slow) loads every model
  that configs/models.yaml lists, on the GPU in bfloat16. It checks the
  registry's shape: 32 layers × 9,216 neurons for Qwen3.5-4B, 36 × 12,288
  for Qwen3-8B. It also checks that the MLP list has one module per layer.
  Its name must not contain "smoke": setup step 7 selects tests with
  `-k smoke` and should stay fast.
- docs/runbooks/pod.md: mention the larger download, and show how to run
  this test.

CONSTRAINTS
- Don't register Qwen3-4B or Qwen3.5-2B yet. S3's replication tests bring
  them in.
- Qwen3-8B stays on revision "main" until it has been downloaded. Prompt 8
  pins it to the commit in its download record.
- tests/unit/test_model_download.py::test_the_pod_downloads_qwen3_5_4b
  expects ["qwen3.5-4b"]. Changing that expected value must be explained
  in the pull request description (CLAUDE.md).
- Layers stay 0-indexed. No dependency changes.

TESTS FIRST
- On the CPU, load_model on tiny-vl and tiny-lm (the tiny_downloads
  fixture) returns one MLP module per layer, in layer order. Each module's
  down_proj takes the registry's MLP size and gives the hidden size.
- configs/models.yaml lists both models, and every name in it is in the
  registry.
- The new GPU test is skipped without -m gpu, and `-m gpu -k smoke` does
  not select it (see test_setup_pod.py's selection test).

DONE WHEN
- `uv run pytest -q` passes on CPU, and the pull request's checks pass.
- On the pod (the owner, in S1's first pod session, together with prompt
  4), `bash scripts/setup_pod.sh` downloads qwen3-8b, and
  `uv run pytest -q -m gpu tests/gpu/test_model_shapes.py` ends with
  `2 passed`.

REPORT BACK
End with a plain-language summary for a beginner: what changed, why, how I
can check it myself, and anything I must do by hand. List the exact pod
commands. Say what to paste back: the test's last line, and the "commit"
in /workspace/models/qwen3-8b/.pushlab-download.json, which prompt 8 pins.
