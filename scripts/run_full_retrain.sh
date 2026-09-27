set -e
cd /Users/sreenithin/Downloads/student_resource
SRC=code/business_entity_resolution/src
B="work/norm/train_source2.parquet work/norm/train_source3.parquet"
GT=dataset/train/train_ground_truth.tsv
for c in India US; do
  echo "=== BLOCK $c $(date +%H:%M:%S)"
  [ -f work/full_pairs_$c.parquet ] || .venv/bin/python code/business_entity_resolution/scripts/block_full.py $c 2>&1 | grep -v Warning
  echo "=== FEATURES $c $(date +%H:%M:%S)"
  [ -d work/full_feats_$c ] || .venv/bin/python $SRC/features.py --pairs work/full_pairs_$c.parquet --a work/norm/train_source1.parquet \
      --b $B --country $c --out-dir work/full_feats_$c --gt $GT --neg-frac 0.10 --jobs 8 2>&1 | grep -v Warning
  rm -f work/full_pairs_$c.parquet
done
echo "=== TRAIN $(date +%H:%M:%S)"
.venv/bin/python $SRC/train.py --features work/full_feats_India work/full_feats_US --gt $GT \
    --a work/norm/train_source1.parquet --b $B --out-dir work/model_full 2>&1 | grep -v Warning
NEW=$(python3 -c "import json;print(json.load(open('work/model_full/config.json'))['val_macro_f05'])")
echo "=== NEW VAL F0.5 = $NEW (old 0.9747) $(date +%H:%M:%S)"
if python3 -c "import sys; sys.exit(0 if $NEW > 0.9747 else 1)"; then
  for c in France India US; do cp -n work/out_matches_$c.tsv work/prev_out_matches_$c.tsv || true; done
  rm -rf work/full_feats_India work/full_feats_US
  echo "=== RESCORE TEST $(date +%H:%M:%S)"
  .venv/bin/python $SRC/predict.py --norm-dir work/norm --model-dir work/model_full --out-dir output --work-dir work --jobs 8 --threads 8 --skip-existing 2>&1 | grep -v Warning
  echo "=== VALIDATE $(date +%H:%M:%S)"
  python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test | tail -3
else
  echo "=== new model not better; keeping previous submission"
fi
echo "=== DONE $(date +%H:%M:%S)"
