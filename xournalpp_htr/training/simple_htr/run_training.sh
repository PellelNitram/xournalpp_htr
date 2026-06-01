#!/usr/bin/env bash

# `pipefail` is required so a failed training run still aborts the script even
# though its output is piped into `tee` for logging.
set -euo pipefail

# ========
# Settings
# ========

BASE_PATH=experiments

# ============
# Experiment 1
# ============

# Question: General hyperparameter tuning (learning rate and batch size)

experiment1() {
    local EPOCH_MAX=100

    for LEARNING_RATE in 0.0005 0.001 0.002
    do
        for BATCH_SIZE in 32 64 128
        do

            echo "LR=${LEARNING_RATE}, BS=${BATCH_SIZE}"

            OUT="${BASE_PATH}/experiment1/lr${LEARNING_RATE}_bs${BATCH_SIZE}"
            mkdir -p "${OUT}"

            time uv run python -m xournalpp_htr.training.simple_htr.train \
                training.learning_rate="${LEARNING_RATE}" \
                training.batch_size="${BATCH_SIZE}" \
                output_path="${OUT}" \
                training.epoch_max="${EPOCH_MAX}" 2>&1 | tee "${OUT}/train.log"

        done
    done
}

# ==================
# Run experiments
# ==================

experiment1

# ==================
# Future experiments
# ==================

# Other questions to answer by conducting additional experiments:
# - Does augmentation help?
# - Longer training with more patience?
# - Effect of dropout?
