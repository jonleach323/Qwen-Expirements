# 0020: Persona candidates come from the library and Mayner's data, one traceable row each, sensitive ones left out

Date: 2026-11-12 · Sprint: S3 · Status: accepted

## Context

- The prompt for this step ("Sprint S3, prompt 2 of 8", pasted) names two
  sources of persona candidates for Qwen3-8B and asks for one clean
  candidate list, sensitive personas excluded and every candidate traceable
  to its source: `pushlab fetch library` and `pushlab fetch mayner`, which
  download each source into `/workspace/data/` (Mayner's with his own
  download script), skip files already present and record the source URL
  and download date; `discovery/library_import.py` and
  `discovery/mayner_import.py`; and `pushlab candidates --model qwen3-8b
  --out units/candidates_qwen3-8b.csv`, which runs both, drops sensitive
  personas unless `--include-sensitive`, removes duplicate names within each
  source and prints counts by source. Large downloads happen on the pod;
  here the tests use small fixtures in each source's real format. Mayner's
  data is CC BY 4.0, to be credited in the README and in any output that
  uses it.
- **The Neuron Persona Library** (neuronpersonas.com) serves Qwen3-8B's
  files under `https://neuronpersonas.com/data/qwen/`. On 2026-11-12:
  `library.json` (23.9 MB), `combos.json` (2.0 MB), `sensitive.json`
  (7.4 KB), and one file of replies per layer under `layers/` (about 1 MB
  each), which nothing here needs.
  - `library.json` is a table: `fields` names the columns of each of its
    `rows` (`L`, `n`, `s`: layer, neuron and direction; `persona`, the
    name; `cat`, an index into its `categories`: default_assistant,
    altered_assistant, persona, incoherent, other; `w` and `c`, weirdness
    and coherence from 0 to 10; `tags`; `text`, an excerpt of the reply;
    `stop`; `a`, the strongest strength whose reply the judge did not call
    incoherent; `before`, the strengths tried first).
  - `combos.json` holds `pool`, the 167 personas the library verified by
    running them again (`pool_fields`: id, L, n, s, a, name, desc, w, c,
    text, repro, verified), and combinations of two of them, which are not
    candidates here.
  - `sensitive.json` lists the sensitive neurons as `"L:n:s"` (344 on
    2026-11-12, none in the pool) and sensitive combinations, by the rule
    its `rule` names ("site/sensitive.js and Detoxify").
  - The site states no licence; it asks to be cited.
- **Mayner's persona-neurons data** (huggingface.co/datasets/wmayner/
  persona-neurons, CC BY 4.0) is a 1.02 GB bundle of 637 files listed with
  their sha256 in a MANIFEST. His `download_data.py` (MIT, at commit
  `64c9d36`) copies it into `$PN_DATA_DIR`, checks every file against the
  MANIFEST, skips a file already there with the right checksum, and takes
  `--only` path prefixes. It needs only the standard library and his
  two-line `persona_neurons/config.py`, and parses as Python 3.11, so
  pushlab's own Python can run it. The importer needs about 110 MB of it:
  `split/top_neurons.parquet` (each of 275 roles' top 1,000 neurons),
  `vectors/without/` (the role vectors, for the exact |v_r| of decision
  0019), and two files of his scores, `role_steering_sizes/
  judged_replies.parquet` (his judge's scores of replies steered by each
  role's top 1, 2, 5, 10, 20, 50 ... neurons) and
  `split/concentration_neurons.parquet` (the share of |v_r| the top k
  neurons contribute).
- The development plan's discovery module "reads the persona library
  export, removes duplicate names and applies filters", and `units/` holds
  "curated unit lists, such as the library shortlist". The S2 review's
  decision D4 defaults to keeping nothing of the library in git.

## Decision

(Filled in below as the build lands.)
