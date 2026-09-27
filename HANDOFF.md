# Handoff — Business Entity Resolution (ML Challenge 2026)

This is the complete guide for taking over the project. It assumes you know Python and basic ML
but have never seen this code. Read it top to bottom once; afterwards `README.md` is the
short command reference.

Contents

1. Status and target
2. What is in the bundle
3. Set up a machine
4. Run the whole pipeline step by step
5. The full-data retrain (was running when this was written)
6. How each stage works and which knobs matter
7. Where the errors are
8. Ideas ranked by expected payoff
9. How to run an experiment without wasting hours
10. Submitting
11. Troubleshooting and known traps

---

## 1. Status and target

- A validated submission exists: `output/matching_results.tsv` and `output/candidate_pairs.tsv`
  (in the 573 MB `team_submission.zip`). It passes the official validator. **Submit it first**
  so we have a real leaderboard number.
- Validation macro F0.5 = **0.9747** (India 0.9722, US 0.9773), measured on 86 K held-out
  training entities. Blocking recall 99.24 %. Oracle F0.5 on the candidate set is 0.9974, so the
  remaining error is almost all classifier and threshold, not candidate generation.
- France exists only in the test set (259 K entities, no labels), so its quality is unmeasured.
  Its predicted singleton rate (4.8 %) and matches per entity look like India and US.
- Leaderboard on 26 Sep, 03:00 IST: rank 4–5 at **0.988**, ranks 6–12 at 0.987, 13–18 at 0.986,
  19–25 at 0.985. We are ≈ 1.1–1.3 points behind the top 25 on validation, possibly more on the
  real test set because of France.
- The write-up `Documentation_template.md` is complete except **team name and members**.

## 2. What is in the bundle

`code_and_model_for_team.zip` (17 MB), identical to the git repo:

```
business_entity_resolution/
  README.md              short command reference
  HANDOFF.md             this file
  requirements.txt       pinned Python packages
  src/
    translit.py          Indic-script -> Latin word dictionary learnt from the training labels
    normalize.py         name / address canonicalisation, state resolution, house numbers
    block.py             candidate generation (TF-IDF retrieval per country+state partition)
    features.py          49 pairwise / context features
    train.py             LightGBM + macro-F0.5 threshold selection
    scoring.py           the metric and the decision rule
    predict.py           test inference, per country, per stage, resumable
  scripts/
    run_train_sample.sh  block -> features -> train on the 14-state sample (≈ 20 min)
    run_full_retrain.sh  same on ALL of Source 1, then rescore the test set if better
    block_full.py        helper used by run_full_retrain.sh
    package.sh           validator + zip
work/
  model/model.txt        trained LightGBM (2,784 trees), config.json (threshold 0.68, one-to-one)
  translit.json          the transliteration dictionary
Documentation_template.md
validate_submission.py   the official validator (also in the challenge's utils/)
```

Not in the bundle, because they are big and regenerable: the dataset (2.3 GB, download from the
challenge portal), the normalised parquet files (1.2 GB, 15 min to rebuild), the candidate pairs
and scores (2 GB, hours to rebuild), and the outputs (1.3 GB, in the other zip).

## 3. Set up a machine

**Hardware.** The whole thing ran on a 16 GB Apple M4 laptop, but it swapped heavily and test
inference took 3 hours. With 32 GB it should be well under an hour; with 64 GB and 16 cores
(e.g. an AWS `r6i.4xlarge`, ≈ $1/h) expect ≈ 30 min end to end. Keep **25 GB of disk free**.
No GPU is used anywhere.

**Software.** Python 3.11 or newer (3.14 was used). Then:

```bash
mkdir student_resource && cd student_resource
# put the challenge's dataset/ and utils/ here, and the code folder under code/
python3 -m venv .venv
.venv/bin/pip install -r code/business_entity_resolution/requirements.txt
mkdir -p work output
```

Expected layout (every path is a command-line argument, so any layout works if you adjust them):

```
student_resource/
  dataset/train/train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
  dataset/test/test_source1.tsv    test_source2.tsv   test_source3.tsv
  utils/validate_submission.py
  code/business_entity_resolution/...
  work/      intermediates
  output/    final TSVs
```

All commands below are run from `student_resource/` with `SRC=code/business_entity_resolution/src`.

## 4. Run the whole pipeline step by step

Times are for the 16 GB laptop; halve them or better on a real machine.

### Step 1 — transliteration dictionary (2 min)

```bash
.venv/bin/python $SRC/translit.py --train-dir dataset/train --out work/translit.json
```

Aligns matched (Latin name, Indic-script name) pairs from the ground truth token by token and
keeps word translations seen ≥ 2 times with ≥ 50 % agreement. Output: 1,347 entries such as
`प्राइवेट -> private`. Prints coverage; expect ≈ 96 % of Indic tokens covered.

### Step 2 — normalise every record (12 min)

```bash
.venv/bin/python $SRC/normalize.py dataset/train/train_source1.tsv dataset/train/train_source2.tsv \
    dataset/train/train_source3.tsv dataset/test/test_source1.tsv dataset/test/test_source2.tsv \
    dataset/test/test_source3.tsv --out-dir work/norm --translit work/translit.json --jobs 8
```

Writes one parquet per input into `work/norm/`. Columns you will use:

| column | meaning |
| --- | --- |
| `name_norm` | lower-cased, accent-folded, punctuation-stripped name |
| `name_core` | `name_norm` minus legal forms (llc, pvt ltd, sarl…) and stop tokens |
| `name_compact` | `name_core` without spaces, so a domain-style name equals the spaced name |
| `addr_norm` | canonicalised address: abbreviations expanded, filler dropped, ordinals to digits |
| `addr_nums` | the digit-bearing tokens of the address |
| `house_no` | first number of the first street-like component |
| `state` | 2-letter code resolved from codes, full names, regional scripts, French regions / departments / cities; `""` if unresolved |
| `name_script`, `name_domain`, `addr_missing` | flags |

### Step 3 — choose the training entities

Training uses **complete (country, state) partitions** so that rank and competition features look
exactly as they will at test time. The current model used 8 US + 6 Indian states (431 K
entities, 20 % of Source 1):

```bash
.venv/bin/python - <<'PY'
import polars as pl
a = pl.read_parquet('work/norm/train_source1.parquet')
US = {'AZ','MD','MN','OK','WV','ME','NM','UT'}; IN = {'GJ','WB','RJ','KL','PB','BR'}
a.filter(((pl.col('country')=='US') & pl.col('state').is_in(list(US))) |
         ((pl.col('country')=='India') & pl.col('state').is_in(list(IN)))).write_parquet('work/train_a_sample.parquet')
PY
```

For the full-data version see section 5.

### Step 4 — blocking on the training entities (4 min)

```bash
.venv/bin/python $SRC/block.py --a work/train_a_sample.parquet \
    --b work/norm/train_source2.parquet work/norm/train_source3.parquet \
    --out work/train_pairs.parquet --gt dataset/train/train_ground_truth.tsv --threads 8
```

Prints, for the sample:

```
blocking recall: 1,481,865/1,493,240 = 0.9924; candidates/S1 = 56.3; total pairs = 24,280,033
   sim_name channel recall: 0.9095
   sim_addr channel recall: 0.9066
   S1 entities with full recall: 0.9748
```

If you change anything in `normalize.py` or `block.py`, this is the number to watch. Output
columns: `a_idx, b_idx` (row positions in `--a` and in the concatenation of the `--b` files, in
that order), `sim_name`, `sim_addr` (TF-IDF cosines, 0 when the pair came from the other channel).

### Step 5 — features (3 min)

```bash
.venv/bin/python $SRC/features.py --pairs work/train_pairs.parquet --a work/train_a_sample.parquet \
    --a-full work/norm/train_source1.parquet \
    --b work/norm/train_source2.parquet work/norm/train_source3.parquet \
    --out-dir work/train_feats --gt dataset/train/train_ground_truth.tsv --neg-frac 0.35 --jobs 8
```

Writes `work/train_feats/part_XXX.parquet`, one per group of partitions, labelled with `y`, keeping
every positive and 35 % of the negatives (9.5 M rows for the sample). `--a-full` is the whole
Source 1, used only to count how many entities share a name / address in the country.

### Step 6 — train and pick the threshold (10 min)

```bash
.venv/bin/python $SRC/train.py --features work/train_feats --gt dataset/train/train_ground_truth.tsv \
    --a work/train_a_sample.parquet --b work/norm/train_source2.parquet work/norm/train_source3.parquet \
    --out-dir work/model
```

Holds out 20 % of the entities, trains LightGBM with early stopping, then sweeps the decision
threshold for macro F0.5 with and without one-to-one assignment, and prints:

```
blocking ceiling (oracle on candidates) macro F0.5 = 0.9974
BEST threshold=0.68 one_to_one=True macro F0.5=0.9747
  India: macro F0.5 = 0.9722 over 43,499 entities
  US: macro F0.5 = 0.9773 over 42,656 entities
top features: combo, combo_gap, b_combo_gap, sim_addr, combo_rank, a_num_jacc, ...
```

Writes `work/model/model.txt`, `config.json` (threshold, one_to_one, val score, feature
importance) and `val_scored.parquet` (validation probabilities, useful for threshold experiments
without retraining).

`scripts/run_train_sample.sh` runs steps 4–6 in one go.

### Step 7 — test inference (3 h on 16 GB; far less with more RAM)

```bash
.venv/bin/python $SRC/predict.py --norm-dir work/norm --model-dir work/model \
    --out-dir output --work-dir work --jobs 8 --threads 8 --skip-existing
```

Runs three stages, once per country (France, India, US), each in its own process:

| stage | writes | time on the laptop |
| --- | --- | --- |
| block | `work/test_pairs_<country>.parquet` | France 23 min, India 93 min, US 4 min (mostly swapping) |
| score | `work/test_scored_<country>.parquet` (+ `_chunkNNN` checkpoints) | 11 + 30 + 29 min |
| output | `work/out_matches_<country>.tsv`, `work/out_cands_<country>.tsv` | 10 min |

then merges the per-country TSVs into `output/`. `--skip-existing` reuses any per-country file
already present, so a crash costs at most one chunk. To re-run only the scoring with a new model:
delete `work/test_scored_*.parquet` and `work/out_*` and run the command again with the new
`--model-dir`; the candidate pairs are reused.

### Step 8 — validate and package

```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
bash code/business_entity_resolution/scripts/package.sh <team_name>
```

Expected: `PASS — no blocking issues found.` and a zip with `output/`, `code/` and the
documentation. Note `package.sh` copies the 1.3 GB outputs before zipping; if disk is tight,
symlink them instead.

## 5. The full-data retrain

Only 20 % of Source 1 was used for the current model. Training on all of it is the most likely
single improvement. `scripts/run_full_retrain.sh` does it end to end:

1. For each of India and US: block the whole training Source 1 of that country against the
   Source 2/3 records of that country (`scripts/block_full.py`, prints recall), keeping row indices
   global, then compute features with `--country <c> --neg-frac 0.10` into `work/full_feats_<c>`.
2. `train.py --features work/full_feats_India work/full_feats_US --a work/norm/train_source1.parquet …
   --out-dir work/model_full`.
3. If the new validation macro F0.5 beats 0.9747 it backs up `work/out_matches_*.tsv` to
   `work/prev_out_matches_*.tsv`, rescores the test set with `work/model_full`, merges and validates.
   Otherwise it stops and leaves the existing submission untouched.

```bash
nohup bash code/business_entity_resolution/scripts/run_full_retrain.sh > work/run_full.log 2>&1 &
tail -f work/run_full.log
```

This run was started on the laptop on 26 Sep at 04:19 IST. If you receive the laptop's `work/`
folder, check `work/run_full.log` for a line `=== NEW VAL F0.5 = …`; if it is higher than 0.9747
and `=== DONE` follows, `output/` already holds the improved submission and `work/model_full/`
is the better model. Expected size: ≈ 120 M candidate pairs, ≈ 19 M labelled rows (7.5 M
positives), 30–40 min of LightGBM.

Memory notes for this run on 16 GB: `features.py --country` loads only that country's records;
`train.py` needs ≈ 2 × rows × 49 × 4 bytes, i.e. ≈ 7.5 GB for 19 M rows, which is why
`--neg-frac 0.10`. On a 64 GB machine use `--neg-frac 0.3` or higher.

## 6. How each stage works and which knobs matter

### Normalisation (`normalize.py`)
- Legal forms and stop words are stripped for `name_core` (`STOP`/`LEGAL` sets near the top of
  the file). French forms (sarl, sas, sasu, sci, eurl) were added from the data description and
  never tuned on labels.
- Address abbreviation tables (`ABBR`) cover US, Indian and French forms (rd/road, st/street,
  r./rue, bd/boulevard…). Null-ish tokens (`<null>`, `n/a`) are dropped.
- State resolution (`resolve_state`) is two-pass: codes and full names, then regional scripts,
  then for France region / department / city lookups. Telangana → AP and DC → WA are merged because
  the data mixes them. Records that end with `state == ""` are handled in blocking (below).
- Indic-script names are transliterated word by word with `translit.json`; words not in the
  dictionary fall back to `indic_transliteration` + `unidecode`. 14 % of Indic names are still only
  partly transliterated.

### Blocking (`block.py`)
- Partition = (country, state). Source 2/3 records with no state are appended to every partition
  of their country; Source 1 records with no state are compared to the whole country.
- Two channels per partition, each **forward** (top-k per Source 1 entity) **and reverse** (top-3
  Source 1 entities per Source 2/3 record):
  - name: TF-IDF over char 3-grams of `name_compact`, `top_name=20`, `thr_name=0.25`
  - address: TF-IDF over word tokens of `addr_norm + addr_nums`, `top_addr=15`, `thr_addr=0.20`
- `max_df=0.1` drops tokens present in > 10 % of a partition (e.g. "delhi" inside Delhi);
  without it the Delhi and Maharashtra partitions take an hour each. Recall on a 30 K sample with it:
  99.78 %.
- The reverse channels are what lift recall from 97.0 % to 99.2 %; capping the candidate list to
  the 30 best-scoring per entity costs 1.5 recall points, so nothing is capped.
- Knobs: `run_blocking(top_name, top_addr, thr_name, thr_addr)` and the `rev_name / rev_addr`
  constants (3). Retrieval is `sparse_dot_topn`, multithreaded.

### Features (`features.py`)
- `PAIR_COLS` (25): rapidfuzz `ratio / token_set_ratio / token_sort_ratio / partial_ratio` on
  `name_core`, `ratio` + Jaro-Winkler on `name_compact`, `ratio` on `name_norm`, token Jaccard,
  first-token equality, prefix relation, compact equality, token counts; the same ratio family on
  `addr_norm`, digit-token Jaccard and intersection, house-number relation (equal / one missing /
  conflict) and edit ratio, state relation.
- `CTX_COLS` (7): the two TF-IDF cosines, `amb_name` and `amb_addr` (how many Source 1 entities
  in the country share the name / address), candidate flags (non-Latin script, domain-style name,
  address missing).
- `GROUP_COLS` (17): per Source 1 entity — candidate count, max / rank / gap of name, address and
  combined similarity, count of near-perfect name and address candidates; per Source 2/3 record —
  degree, best combined similarity offered by any other entity, rank and gap among them.
- Pairs are processed in chunks that keep whole (country, state) partitions together
  (`iter_chunks`), so group features are exact. String work runs in a multiprocessing pool.
- Adding a feature: append to `_pair_feats` and `PAIR_COLS` (order must match), or add a Polars
  expression in `build_features` and its name to `CTX_COLS` / `GROUP_COLS`. `FEATURE_COLS` is
  what the model sees; retrain after any change.

### Model and decision (`train.py`, `scoring.py`)
- LightGBM binary, `num_leaves=127, learning_rate=0.05, min_child_samples=100,
  feature_fraction=0.8, bagging_fraction=0.8, lambda_l2=1.0`, early stopping on log-loss. These
  were never tuned.
- Decision: keep pairs with `p ≥ threshold`, then one-to-one: each Source 2/3 record goes to the
  Source 1 entity with the highest probability (greedy). Threshold chosen by sweeping macro F0.5 on
  the validation entities, singletons included (an entity with no true match and no prediction
  scores 1, with any prediction scores 0). The curve is flat from 0.60 to 0.75.
- `scoring.py` has `macro_f05`, `decide`, `decide_frame`, `sweep`; use them with
  `work/model/val_scored.parquet` for any decision-rule experiment.

## 7. Where the errors are (validation, 86,155 entities)

| group | entities | macro F0.5 |
| --- | --- | --- |
| singletons (0 true matches) | 4,852 | 0.964 (177 wrongly given a match) |
| 1 true match | 4,705 | 0.932 |
| 2 | 14,508 | 0.970 |
| 3 | 20,858 | 0.976 |
| 4+ | 41,232 | ≥ 0.981 |

- **False negatives: 14,167 pairs.** 2,241 never became candidates; 11,912 scored below 0.68; 14
  were lost to one-to-one. 41 % of the missed records have **no address** and a generic or
  truncated name ("Traders", "Big Seafood", "Wallas Auto Glass Service" for "Wallas American Auto
  Glass"). Most of the rest combine a changed house number with an edited name.
- **False positives: 3,097 pairs.** 88 % are distractors that are near-copies of a real entity:
  same name and street, house number changed (2246/b → 2250/b) or one name word swapped
  ("Wayanad Garments" → "Wayanad Sky"). Only 12 % of the wrongly matched records actually belong
  to another entity.
- The metric is a macro average over entities, so the 11 % of entities with 0 or 1 true matches
  carry a disproportionate share of the loss.

## 8. Ideas ranked by expected payoff

1. **Full-data retrain** (section 5). Most certain gain. Then tune: `num_leaves` 255, a few seeds
   averaged, `learning_rate` 0.03 with more rounds, or CatBoost on the same features.
2. **Abstain better on ambiguous single candidates.** Per-group thresholds (by `n_cands`,
   `amb_name`, or `b_addr_missing`) chosen on `val_scored.parquet`; or a second-stage model on
   entity-level features (best p, second-best p, gap) that decides whether to output anything.
3. **Address-missing candidates** (41 % of misses): add name-only evidence — acronym / initials
   match, phonetic key (jellyfish is installed), "number of entities in the state whose
   `name_compact` starts with this candidate's name", or a name-only TF-IDF cosine on `name_core`
   4-grams.
4. **Held-out-state validation.** Train on 10 of the 14 states, validate on the other 4, to see
   how much the score drops on unseen geography; that is the best available proxy for France.
   Then read 100 random French matches and 100 French singletons from `work/out_matches_France.tsv`
   next to the raw TSVs and fix whatever normalisation error you see.
5. **Blocking misses** (0.8 %): a third channel (phonetic or first-two-tokens key within the same
   city), or char 4-grams of `name_core`. Measure with `block.py --gt` on a 30 K sample first.
6. **Near-copy distractors:** a feature "how many other candidates of this entity share the exact
   `name_compact` and street but a different `house_no`", and "does a candidate with the identical
   house number also exist".
7. **Transliteration:** a character-level model or a larger dictionary built from all matched
   pairs (not only token-aligned ones). India trails the US by 0.5 pt.
8. **Global assignment** instead of greedy one-to-one: only 14 pairs at stake, low priority.

## 9. How to run an experiment without wasting hours

- **Blocking change** → `block.py --a <30 K sample> --gt …` (make the sample with
  `pl.read_parquet(...).sample(n=30000, seed=3)`), ≈ 5 min, compare recall and candidates/S1.
- **Feature change** → steps 5–6 on the existing `work/train_pairs.parquet`, ≈ 15 min, compare
  `BEST … macro F0.5`. Seeds are fixed, so runs are comparable.
- **Threshold / decision change** → no training: load `work/model/val_scored.parquet` and use
  `scoring.sweep` / `scoring.decide` with the truth from `block.gt_pairs_idx`.
- **Only then** rescore the test set (delete `work/test_scored_*` and `work/out_*`, rerun
  `predict.py` with the new `--model-dir`), validate, package, submit.
- Record every run: what changed, blocking recall, val macro F0.5, leaderboard score. Do not trust
  a validation gain under 0.001; the split has 86 K entities so noise is about that size.

## 10. Submitting

- Zip layout required by the challenge: `output/matching_results.tsv`, `output/candidate_pairs.tsv`,
  `code/…`, `Documentation_template.md`. `scripts/package.sh` builds it.
- Always run the validator before uploading. Fill team name and members in the documentation.
- Keep `work/prev_out_matches_*.tsv` (or the previous zip) so a worse submission can be rolled back.

## 11. Troubleshooting and known traps

- **Memory.** Never hold all 97 M test pairs with their strings in one process. Stages are per
  country and per chunk for that reason; `features.py` joins strings slice by slice. If a machine
  swaps, everything runs 2–3× slower; close browsers and IDE helpers.
- **Disk.** ≈ 25 GB free needed at peak. Big files: `work/norm` 1.2 GB, `work/test_pairs_*`
  1 GB, `work/test_scored_*` 0.8 GB, `output/` 1.3 GB, feature parts up to 2.5 GB.
- **Resume.** `predict.py --skip-existing` reuses per-country files; scoring also checkpoints per
  chunk. `run_full_retrain.sh` skips a country whose pairs / features already exist.
- **Empty match lists.** Polars `write_csv` quotes an empty string as `""`, which the validator
  rejects as an invalid ID. `predict.py` uses `quote_style="never"`. If you write TSVs any other
  way, check for `""`.
- **Row indices.** `a_idx` / `b_idx` are positions in exactly the frames passed as `--a` and
  `--b` (Source 2 then Source 3, concatenated in that order). `features.py --country` and
  `block_full.py` keep them global by adding the index before filtering.
- **Records without a state** are copied into every partition of their country, so their
  candidate counts are inflated on purpose.
- **Duplicate column names** in Polars selects raise `DuplicateError`; `A_COLS` already contains
  `state`, do not add it again.
- **Delhi / Maharashtra partitions** explode without `max_df`; if a partition takes more than
  10 min, that is the first thing to check.
- **The laptop run died once** when the terminal session ended; launch long runs with
  `nohup … &` (or `tmux`) and `caffeinate -i` on macOS.
