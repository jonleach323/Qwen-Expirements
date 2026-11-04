This is Sprint S1, prompt 4 of 8.

CONTEXT
Read CLAUDE.md and these parts of the plans:
- docs/plan/development-plan.md: the "2. units" module, whose decisions
  say each layer's median residual size is "measured once per model by
  `calibrate_norms` and cached";
- docs/plan/research-plan.md: the dose rule under "Concepts", "What the
  October 2026 wave established", and "Dosing rules", which want doses in
  "the same rule as the evil recipe".

Prompt 3 built neuron units, whose amount(alpha) needs these residual
sizes. Two public replications of the recipe show how they measured them,
and the recipe author's own code measures them differently. Clone them if
the session can reach them:
- github.com/occcat/qwen35-evil-neuron (see steer_qwen35.py and
  RESULTS.md);
- github.com/Mr-Neutr0n/qwen3.5-4b-neuron-steering (see lab.py);
- github.com/lyramakesmusic/neuron-explorer, the author's code (see
  explorer/runtime.py and sweep/neuron_stats.py). Its median comes from
  statistics over web text and chat-formatted text in 1,024-token
  windows; only when those are missing does it measure a short reference
  text, the one Mr-Neutr0n used.

GOAL
Measure each layer's median residual size once per model, in the
recipe's units, keep it in a cache, and give it to the dose rule.

BUILD
- src/pushlab/units/norms.py: calibrate_norms, using the recipe's rule:
  "median(‖resid‖ leaving layer L, tokens 1:)". In practice:
  - Capture the stream leaving each decoder layer with a forward hook on
    that layer, as Mr-Neutr0n does. (occcat, in MLX, records the same
    stream as the input of the next layer's input_layernorm.) Don't use
    the model's output_hidden_states: in transformers 5.21 its last entry
    is the output of the final norm, not of the last layer.
  - Take each token's length in float32, from the activations in their own
    data type (bfloat16 on the pod).
  - Drop the first token of every passage, and never count padding.
  - Take the median over all remaining tokens, per layer.
- prompts/calibration-v1.yaml: a versioned set of neutral prose passages,
  a few hundred tokens each and a few thousand in all, with an ID and a
  version. Use plain text with no chat template, as both replications do;
  occcat used 578 tokens from 4 neutral passages. Text never goes in code
  (CLAUDE.md). The owner reviews the passages in the pull request.
- Any choice that differs from the replications changes the dose's units.
  The decision note must say so.
- A cache, one JSON file per model, model commit, data type and text set,
  in a folder on the pod volume such as /workspace/calibration. An environment
  variable can point elsewhere, as PUSHLAB_MODELS_DIR does for models.
  - The file holds the medians, the text set's ID and SHA-256, the token
    count, the code commit and the time.
  - load_norms reads it, and refuses a file made for another model commit,
    data type or text set.
- `pushlab calibrate --model NAME [--texts FILE] [--dtype bfloat16]
  [--dry-run] [--overwrite]`. It is a GPU job: the owner runs it.
  - --dtype defaults to bfloat16, as on the pod.
  - --dry-run prints the model, the text set, the token count and the
    output file, without loading the model. It must also work in a cloud
    session, which has no downloaded models. When the model is downloaded,
    take the model commit from its download record and count tokens with
    its tokenizer; otherwise print that the commit (and so the file name)
    and the token count are known only after the download.
  - Keep cli.py thin by putting the command's logic in the module.
- docs/runbooks/experiments.md (prompt 1 starts it): how to run it, and
  where the file goes.
- A decision note (the next free number) covering:
  - the definition: which of the three sources above it follows, and how
    it differs from the others;
  - the text set and the cache;
  - why the values are measured in bfloat16 on the GPU.

CONSTRAINTS
- The plan says "measured once": calibrating never happens silently
  inside another command. A missing cache is an error that names the
  command to run.
- Never overwrite a cache without --overwrite.
- The calibration file is a model property that later runs depend on, not
  a run output. Prompt 8 copies the Qwen3.5-4B values into
  docs/baselines/s1/.
- No dependency changes.

TESTS FIRST
- On tiny_model and tiny_vl_model, the medians equal a hand calculation
  from forward hooks on each layer, with the first token of each passage
  dropped. There is one positive value per layer.
- The last layer's median is the stream leaving that layer, not the final
  norm's output.
- Dropping the first token changes the result when that token's length
  is unusual. Write a test where it is.
- A batch with left padding gives the same medians, within float32
  tolerance, as the passages run one at a time.
- The cache round-trips. load_norms refuses another model commit, data
  type or text set, and an existing file is refused without --overwrite.
- A text set file without an ID or version, or with no passages or
  duplicates, is refused.
- `pushlab calibrate --help` lists the options. --dry-run never loads a
  model (replace load_model with one that fails, as S0's command tests
  do), and works when the model isn't downloaded.
- End to end on tiny-vl (the tiny_downloads fixture), the command writes a
  cache that load_norms reads and prompt 3's amount() uses.

DONE WHEN
- `uv run pytest -q` passes on CPU, and the pull request's checks pass.
- The pull request shows `pushlab calibrate --help`, and the dry run's
  output for qwen3.5-4b and qwen3-8b.
- On the pod (the owner, in S1's first pod session),
  `uv run pushlab calibrate --model qwen3.5-4b` and
  `uv run pushlab calibrate --model qwen3-8b` each write their cache, with
  one finite, positive value per layer.
- Qwen3.5-4B's layer-14 median and its amount(1) for L14 N8949 are close
  to the replications'. Both ran in bfloat16; Mr-Neutr0n recorded the
  pinned model commit, and occcat recorded none, though Hugging Face has
  served the pinned commit since 2026-03-02. Their values:
  - occcat: median 10.49, amount +22.57;
  - Mr-Neutr0n: median 10.32, amount +22.21;
  - both: a column length of 0.4647.

  A gap over about 10% is reported before pod session 2, because S1's
  done-when runs at dose 1.

REPORT BACK
End with a plain-language summary for a beginner: what changed, why, how I
can check it myself, and anything I must do by hand.

This is the last prompt before S1's first pod session, so give the whole
session as one numbered list:
1. Start the pod. If /workspace/runs/s0-baseline isn't copied off the pod
   yet, copy it first (docs/runbooks/experiments.md, read on GitHub).
2. Before pulling, run `bash scripts/setup_pod.sh`, and note its step
   times. They are the warm-start row for decision 0005, on a restarted
   pod with nothing new to download.
3. `git pull`, then run the setup script again. This run downloads
   Qwen3-8B, so its time doesn't count as warm. It is also the first GPU
   run of the code merged since S0's baseline.
4. Prompt 2's model shape test.
5. Both calibrations, then a one-line command that prints L14 N8949's
   column length and amount(1) for Qwen3.5-4B from its new cache. Write
   it, and try the same command on tiny-vl first.
6. Stop the pod.

Say exactly what to paste back, the same day, as a comment on this
prompt's GitHub issue (a layer-14 gap over about 10% goes there too):
- whether /workspace/runs/s0-baseline is copied off the pod;
- the warm run's step times, and step 7's last line from the post-pull
  run;
- the shape test's last line;
- Qwen3-8B's download commit;
- both calibration files, and the layer-14 comparison with the
  replications.
