# Business Entity Resolution — pipeline

Reproduces `output/matching_results.tsv` and `output/candidate_pairs.tsv` from the
challenge data. Everything runs on a laptop (tested on an Apple M4, 16 GB RAM,
10 cores); no GPU, no external data or services.

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Expected layout (paths are arguments, so any layout works):

```
student_resource/
  dataset/train/…  dataset/test/…
  code/business_entity_resolution/src/
  work/            # intermediate files (created)
  output/          # final TSVs (created)
```

## Run end-to-end

Approximate wall-clock on a 16 GB M4 laptop: steps 1-2 ≈ 15 min, 4 ≈ 4 min, 5 ≈ 3 min,
6 ≈ 10 min, 7 ≈ 3 h (dominated by swapping; far less with 32 GB). Needs ≈ 25 GB free disk.

```bash
SRC=code/business_entity_resolution/src

# 1. learn the Indic-script -> Latin word dictionary from the training ground truth
.venv/bin/python $SRC/translit.py --train-dir dataset/train --out work/translit.json

# 2. normalise every record (names, addresses, states, house numbers)
.venv/bin/python $SRC/normalize.py dataset/train/train_source1.tsv dataset/train/train_source2.tsv \
    dataset/train/train_source3.tsv dataset/test/test_source1.tsv dataset/test/test_source2.tsv \
    dataset/test/test_source3.tsv --out-dir work/norm --translit work/translit.json --jobs 8

# 3. training sample = complete (country, state) partitions of the reference source
.venv/bin/python - <<'PY'
import polars as pl
a = pl.read_parquet('work/norm/train_source1.parquet')
US = {'AZ','MD','MN','OK','WV','ME','NM','UT'}; IN = {'GJ','WB','RJ','KL','PB','BR'}
a.filter(((pl.col('country')=='US') & pl.col('state').is_in(list(US))) |
         ((pl.col('country')=='India') & pl.col('state').is_in(list(IN)))).write_parquet('work/train_a_sample.parquet')
PY

# 4. blocking on the training sample (prints candidate recall against the ground truth)
.venv/bin/python $SRC/block.py --a work/train_a_sample.parquet \
    --b work/norm/train_source2.parquet work/norm/train_source3.parquet \
    --out work/train_pairs.parquet --gt dataset/train/train_ground_truth.tsv

# 5. pairwise features (labelled; keeps every positive and 35 % of the negatives)
.venv/bin/python $SRC/features.py --pairs work/train_pairs.parquet --a work/train_a_sample.parquet \
    --a-full work/norm/train_source1.parquet \
    --b work/norm/train_source2.parquet work/norm/train_source3.parquet \
    --out-dir work/train_feats --gt dataset/train/train_ground_truth.tsv --neg-frac 0.35 --jobs 8

# 6. LightGBM + macro-F0.5 threshold selection (20 % of the sample's entities held out)
.venv/bin/python $SRC/train.py --features work/train_feats --gt dataset/train/train_ground_truth.tsv \
    --a work/train_a_sample.parquet --b work/norm/train_source2.parquet work/norm/train_source3.parquet \
    --out-dir work/model

# 7. test inference: blocking -> features -> scoring -> assignment -> output/*.tsv
.venv/bin/python $SRC/predict.py --norm-dir work/norm --model-dir work/model --out-dir output --work-dir work --skip-existing
# runs block -> score -> output once per country, each in its own process;
# --skip-existing resumes from the per-country parquet parts already in work/ after an interruption

# 8. validate
python3 utils/validate_submission.py --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## Modules

| file | role |
| --- | --- |
| `translit.py` | word-level Indic-script → Latin dictionary learnt from matched training pairs |
| `normalize.py` | name / address canonicalisation, state resolution, house-number extraction |
| `block.py` | candidate generation: TF-IDF char-3-gram (name) and word (address) top-k retrieval inside (country, state) partitions, forward + reverse |
| `features.py` | pairwise string-similarity, ambiguity, rank and competition features |
| `train.py` | LightGBM binary matcher, macro-F0.5 threshold sweep, one-to-one assignment choice |
| `scoring.py` | challenge metric and decision rule |
| `predict.py` | end-to-end test inference and output writing |
