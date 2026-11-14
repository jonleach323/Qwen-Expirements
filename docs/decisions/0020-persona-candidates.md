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
  (7.4 KB), and one file of replies per layer under `layers/` (1 to 2.5 MB
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
    its `rule` names ("site/sensitive.js and Detoxify"). Its `from` stamp
    gives the byte counts of the `library.json` and `combos.json` it was
    made from. The site also screens a persona's full reply with
    `sensitive.js` when it shows one, which the list does not cover.
  - The site states no licence; it asks to be cited.
- **Mayner's persona-neurons data** (huggingface.co/datasets/wmayner/
  persona-neurons, CC BY 4.0) is a 1.02 GB bundle of 638 files listed with
  their sha256 in a MANIFEST. His `download_data.py` (MIT, at commit
  `64c9d36`) copies it into `$PN_DATA_DIR`, checks every file against the
  MANIFEST, skips a file already there with the right checksum, and takes
  `--only` path prefixes. It needs only the standard library and his
  `persona_neurons/config.py`, which imports only `os` and `pathlib`, and
  parses as Python 3.11, so pushlab's own Python can run it. It saves each
  file in place before checking it, and opens its URLs with no timeout. The
  importer needs 279 of its files, about 105 MB:
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

**Where the data goes.** The data folder is `/workspace/data` on the pod,
or what `PUSHLAB_DATA_DIR` or `--data-dir` names
(`pushlab.discovery.sources`). The library's files go in `library/qwen/`,
Mayner's code in `mayner/code/` and his bundle's files in
`mayner/bundle/`, at their paths in the bundle. Each source's folder gets a
`FETCHED.json`: where its files came from, when, under which licence, and
with what credit.

**`pushlab fetch library`** (`discovery/fetch.py`, with the network side in
`discovery/fetch_http.py`) downloads `library.json`, `combos.json` and
`sensitive.json` from `https://neuronpersonas.com/data/qwen/`, about 26 MB.
A file already there is skipped unless `--refresh`. Every file it downloads
goes to `<name>.part` first and must be a JSON object with its own keys
(`fields` and `rows`; `pool_fields` and `pool`; `neurons`), so an HTML page
served in its place is refused, naming the URL and its Content-Type. Only
when all have arrived and passed are they renamed into place together, and
FETCHED.json is written straight after: a fetch that fails changes nothing,
so a newer `library.json` never lands beside an older `sensitive.json`, and
its message says to run it again with `--refresh` if that was given. A
skipped file keeps its record entry while its bytes and sha256 still match;
one that no longer matches is recorded with no download time, with a
warning that suggests `--refresh`. The record gives each file's URL, bytes,
sha256, download time (UTC) and the server's Last-Modified. The requests
say only "pushlab".

**`pushlab fetch mayner`** (`discovery/fetch.py`, `discovery/fetch_mayner.py`)
uses his own download script, as asked:
- it clones `github.com/wmayner/persona-neurons` into `mayner/code/`, moves
  it to the pinned commit `64c9d36f5bdde14bbd5b736cbd83eb6015e031eb`, and
  refuses a checkout with a changed tracked file, as his own code does for
  the code it pins;
- it saves the bundle's MANIFEST (about 63 KB), whose sha256 names the
  version fetched, since the bundle is served from the dataset's main
  branch;
- it runs his unchanged `download_data.py` with pushlab's Python in
  isolated mode, on the dataset's page, with `PN_DATA_DIR` set to
  `mayner/bundle/` and `--only` the four prefixes the importer needs (279
  files, about 105 MB; `--all` takes the whole 1 GB bundle). A one-line
  wrapper gives every connection a 60-second timeout, which his script has
  none of, and runs it with `runpy`. His script gets only `PN_DATA_DIR` and
  the variables a download needs (the path, home, locale, temporary folder,
  time zone, proxies and certificates), never the pod's `HF_TOKEN` or other
  secrets. His script checks every file against the MANIFEST and skips one
  already there with the right checksum;
- a failed run leaves no record; its message says that a file which failed
  his check may be left in place and is downloaded again on the next run,
  and to finish the fetch before `pushlab candidates`. The command warns if
  a prefix found no file, which his script does not.
The record gives the dataset's URL, the time, the prefixes, the script's
repository, commit and command line, the MANIFEST's sha256, the licence and
the credit.

**The library's candidates** (`discovery/library_import.py`) are every row
of `library.json` whose category is "persona", in the file's order, then
each verified pool persona that has no persona row, built from its pool row.
Values are read by the names the files give them (`fields`, `categories`,
`pool_fields`), never by position, and the reply text is never read into a
candidate. Each keeps the layer, neuron and direction as a `NeuronUnit`, the
library's strength (`a`), the name, tags, coherence and weirdness, whether
(L, n, s) is in the verified pool, the category, and the sensitive flag:
whether "L:n:s" is in `sensitive.json`, for that direction only. A row
repeated for the same neuron and direction counts once, the first; a pool
row whose `verified` is false is not in the pool; a row outside Qwen3-8B's
shape refuses the whole file, as another model's. The sensitive flag is
`sensitive.json`'s list only, not the site's wider screen of full replies.
`sensitive.json`'s `from` stamp must give the sizes of the `library.json`
and `combos.json` beside it (`discovery/library_files.py`): a list made from
another copy may miss newer sensitive personas, so it is refused, saying to
fetch again with `--refresh` (a missing stamp is not checked). A record
entry whose bytes or sha256 no longer match its file gives no download time.
A file that is not the library's JSON is refused with the same advice, and
a field or value not in its format asks whether the library changed it.

**Mayner's candidates** (`discovery/mayner_import.py`) are, for every role
in `split/top_neurons.parquet`, in name order, two NeuronSets: its top 10
and its top 50 neurons, made by `pushlab.units.mayner.top_neurons_set`, so
dose 1 is his strength (decision 0019). |v_r| comes from his role vectors
(`vector_norm_from_vectors`) when the role's and the default's files are
there, else from the split's shares, and the candidate says which
(`size_from`). His own scores are kept for reference, None where missing:
`mayner_scored` and `mayner_in_role`, the replies his judge scored for that
set and the share it scored 2 or 3 (his `in_role_summary`), from
`role_steering_sizes/judged_replies.parquet` (kind "neurons", draw 0, that
k); and `mayner_share`, the % of |v_r| the set contributes, from
`split/concentration_neurons.parquet`. Strength 1.0, direction +1, and
category "role", pushlab's label, since his data has none. His data flags
no role as sensitive, so none is. When `bundle/MANIFEST` is there, every
file the importer reads must have the size and sha256 it lists
(`discovery/mayner_manifest.py`), or the import is refused, naming up to ten
and saying to fetch again: his script saves a file before checking it, so a
failed fetch can leave a bad one in place.

**`pushlab candidates`** (`discovery/candidates_command.py`) runs both
importers, checks every unit against Qwen3-8B's shape, drops the sensitive
candidates unless `--include-sensitive`, then the duplicate names within
each source (`candidates.drop_duplicates`: the same name, case and spacing
aside, and as many neurons; the one kept is the verified one, then the most
coherent, the weirdest, the strongest, then by key), and writes the CSV
(`candidates.write_candidates`). Sensitive ones go first, so a name that a
sensitive persona shares with a safe one keeps the safe one. It prints, per
source, how many were read, dropped as sensitive, dropped as duplicates and
kept, and each source's licence and credit. It warns, on stderr, when a
source's candidates have no download time (its FETCHED.json is missing or
no longer matches its files, as after a failed fetch) and when Mayner's sets
are sized from the split's shares; the list is still written. It refuses an
existing `--out`
without `--overwrite` before reading anything, and a missing source names
the fetch to run; refusals exit 1, as every pushlab command's do.

**The CSV** has one row per candidate (`candidates.COLUMNS`): its key
(`library/L12N4410-`, `mayner/pirate-top10`), source, persona, unit and
kind, the number of neurons, direction, strength, the sensitive flag, the
category (the library's, or "role" for Mayner's), tags, coherence,
weirdness and pool flag, a set's size,
where the size came from and its entries, Mayner's scores, and where it
came from: the file in the data folder, the place in it, the file's URL,
the download time, and the source's licence and credit. The library's
credit is the citation its "Cite this" section asks for (Fonseca Rivera,
2026). Every row of Mayner's data carries his CC BY 4.0 credit with the
licence's address and says how pushlab adapted it (each role's top 10 and
top 50 neurons, sized from his vectors, with his judge's in-role share); the
record of his unchanged download carries the credit alone.
`read_candidates` reads the file back, units and all.

**Out of git.** No data of either source is in the repository: the tests
use made-up rows in each format (`tests/fixtures/library/qwen/`, and the
`mayner_data` fixture in `tests/unit/conftest.py`). Candidate lists,
`units/candidates_*.csv`, and a partly written `.csv.part`, are ignored by
git: the library states no licence,
and an ignored file never blocks a `git pull` on the pod. Committing one
waits for the owner's decision on the library's terms (D4 in the S2
review). README's "Data and credits" credits both sources.

## Why this shape

- **One record per source folder,** written after the files, so the
  importers can say for every candidate which URL it came from and when,
  without asking the network again.
- **His script, not ours, for his bundle,** as the prompt asks: it checks
  each file against his MANIFEST and skips good ones, and pinning his
  commit keeps the script from changing under us. pushlab's own Python runs
  it, since it needs only the standard library.
- **Only the files the importer needs,** about 105 MB of the 1 GB bundle,
  with `--all` for the rest.
- **Persona rows only.** The library's other categories (the default and
  altered assistant, incoherent replies, other) are not personas; the
  verified pool is added even where `library.json` has no persona row.
- **Sensitive first, then duplicates,** so the filter never hides a safe
  persona behind a sensitive one of the same name.
- **A wide CSV** rather than one file per source: the prompt asks for one
  list, and every row then carries its own provenance and credit.

## Consequences

- Mayner's roles have no sensitive flag. His 275 roles are the Assistant
  Axis's, mostly occupations and archetypes, but some, such as criminal,
  predator, saboteur or demon, may need one; that is the owner's call.
- The library's strength is in its own units, alpha times the median
  residual size of the steered prompt's own tokens: pushlab's per_prompt
  calibration (decision 0019) on "Who are you?". With the fixed calibration
  the same alpha pushes a little differently.
- The library's files change: `library.json` was last modified on
  2026-11-12. `--refresh` takes a new copy; the record keeps each copy's
  sha256 and time.
- Mayner's sets of 100 neurons have their scores in another of his files,
  which is not fetched; only 10 and 50 are built.
- The fetch commands need git, and network access to github.com,
  huggingface.co, Hugging Face's file servers (hosts under hf.co, such as
  us.aws.cdn.hf.co) and neuronpersonas.com, on the pod.

## Results

On a cloud session's CPU, with fixtures only; nothing was downloaded into
the repository and the pod has run nothing for this step. The full suite
passes: 2,516 tests in 102 s, 8 skipped (the GPU tests). On the fixtures,
`pushlab candidates` reads 8 library candidates (2 sensitive, 1 duplicate
name, 5 kept) and 6 of Mayner's (3 roles at 10 and 50 neurons, all kept,
with a warning that one role's two sets are sized from the shares, as the
fixture gives it no vectors). A sized review (three reviewers, three
skeptics) found one medium problem, a partly failed refresh that could mix
an old `sensitive.json` with a new `library.json`, and a dozen lows; all are
fixed above.
