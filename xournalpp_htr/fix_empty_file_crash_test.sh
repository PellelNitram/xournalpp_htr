set -e

SUFFIX=xopp

INPUT_FILE=/home/martin/.cache/huggingface/hub/datasets--PellelNitram--xournalpp_htr_examples/snapshots/d46812cf696a52322aba3fd5cfbd45f8b3352343/data/empty.${SUFFIX}
python run_htr.py -if ${INPUT_FILE} -of /tmp/test_empty.pdf

INPUT_FILE=/home/martin/.cache/huggingface/hub/datasets--PellelNitram--xournalpp_htr_examples/snapshots/d46812cf696a52322aba3fd5cfbd45f8b3352343/data/first_upload.xoj
python run_htr.py -if ${INPUT_FILE} -of /tmp/test_non_empty.pdf