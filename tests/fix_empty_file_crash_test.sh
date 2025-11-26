# This script tests that the HTR processing does not crash on an empty Xournal++ file.

set -e

DATASET_DIR=/home/martin/.cache/huggingface/hub/datasets--PellelNitram--xournalpp_htr_examples/snapshots/d46812cf696a52322aba3fd5cfbd45f8b3352343/data
OUTPUT_DIR=/tmp/fix_empty_file_crash_test
mkdir -p ${OUTPUT_DIR}

python ../xournalpp_htr/run_htr.py -if ${DATASET_DIR}/empty.xopp -of ${OUTPUT_DIR}/test_empty_xopp.pdf
python ../xournalpp_htr/run_htr.py -if ${DATASET_DIR}/empty.xoj -of ${OUTPUT_DIR}/test_empty_xoj.pdf
python ../xournalpp_htr/run_htr.py -if ${DATASET_DIR}/first_upload.xoj -of ${OUTPUT_DIR}/test_non_empty.pdf