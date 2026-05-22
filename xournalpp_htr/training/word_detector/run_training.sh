#!/usr/bin/env bash

# `pipefail` is required so a failed training run still aborts the script even
# though its output is piped into `tee` for logging.
set -euo pipefail # To catch all errors

# ========
# Settings
# ========

BASE_PATH=experiments

# ============
# Experiment 1
# ============

# Question: General hyperparameter tuning

EPOCH_MAX=200

for LEARNING_RATE in 0.0005 0.001 0.002
do
    for BATCH_SIZE in 16 32 64 128
    do

        echo "LR=${LEARNING_RATE}, BS=${BATCH_SIZE}"

        OUT="${BASE_PATH}/experiment1/lr${LEARNING_RATE}_bs${BATCH_SIZE}"
        mkdir -p "${OUT}"

        uv run python -m xournalpp_htr.training.word_detector.train \
            --learning_rate "${LEARNING_RATE}" \
            --batch_size "${BATCH_SIZE}" \
            --output_path "${OUT}" \
            --epoch_max "${EPOCH_MAX}" 2>&1 | tee "${OUT}/train.log"

    done
done

# ==================
# Future experiments
# ==================

# Other questions to answer by conducting additional experiments:
# - Do different model seeds change results?
# - Does batch size matter? -> already covered in general hyperparameter tuning
# - Longer training help w/ more patience?
# - Cheap as k-fold on data to get good estimate of true performance
