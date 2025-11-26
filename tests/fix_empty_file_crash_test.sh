# This script tests that the HTR processing does not crash on an empty Xournal++ file.

set -e

DATASET_DIR=/home/martin/.cache/huggingface/hub/datasets--PellelNitram--xournalpp_htr_examples/snapshots/ca4f8d43dee4bd626139822ba2d5f998dcf4bfa3/data
OUTPUT_DIR=/tmp/fix_empty_file_crash_test
mkdir -p ${OUTPUT_DIR}

python ../xournalpp_htr/run_htr.py -if ${DATASET_DIR}/empty.xopp -of ${OUTPUT_DIR}/test_empty_xopp.pdf
python ../xournalpp_htr/run_htr.py -if ${DATASET_DIR}/empty.xoj -of ${OUTPUT_DIR}/test_empty_xoj.pdf
python ../xournalpp_htr/run_htr.py -if ${DATASET_DIR}/empty_and_not_empty.xopp -of ${OUTPUT_DIR}/test_empty_and_not_empty.pdf
python ../xournalpp_htr/run_htr.py -if ${DATASET_DIR}/first_upload.xoj -of ${OUTPUT_DIR}/test_non_empty.pdf

echo "See ${OUTPUT_DIR} for output files."