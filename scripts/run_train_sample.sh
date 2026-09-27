set -e
cd /Users/sreenithin/Downloads/student_resource
SRC=code/business_entity_resolution/src
B="work/norm/train_source2.parquet work/norm/train_source3.parquet"
echo "=== BLOCK $(date)"
.venv/bin/python $SRC/block.py --a work/train_a_sample.parquet --b $B --out work/train_pairs.parquet --gt dataset/train/train_ground_truth.tsv --threads 8 2>&1 | grep -v Warning | tail -6
echo "=== FEATURES $(date)"
rm -rf work/train_feats; .venv/bin/python $SRC/features.py --pairs work/train_pairs.parquet --a work/train_a_sample.parquet --a-full work/norm/train_source1.parquet --b $B --out-dir work/train_feats --gt dataset/train/train_ground_truth.tsv --neg-frac 0.35 --jobs 8 2>&1 | grep -v Warning
echo "=== TRAIN $(date)"
.venv/bin/python $SRC/train.py --features work/train_feats --gt dataset/train/train_ground_truth.tsv --a work/train_a_sample.parquet --b $B --out-dir work/model 2>&1 | grep -v Warning
echo "=== DONE $(date)"
