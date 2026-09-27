set -e
cd /Users/sreenithin/Downloads/student_resource
TEAM=${1:-team}
echo "--- validator"
python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
echo "--- packaging"
rm -rf work/pkg && mkdir -p work/pkg/output work/pkg/code
cp output/matching_results.tsv output/candidate_pairs.tsv work/pkg/output/
cp -r code/business_entity_resolution work/pkg/code/
find work/pkg/code -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
cp work/Documentation_template.md work/pkg/Documentation_template.md
(cd work/pkg && rm -f "../${TEAM}_submission.zip" && zip -qr "../${TEAM}_submission.zip" output code Documentation_template.md)
ls -la work/${TEAM}_submission.zip && unzip -l work/${TEAM}_submission.zip | tail -15
