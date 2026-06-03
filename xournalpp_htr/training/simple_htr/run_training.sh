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

# ============
# Experiment 2
# ============

# Question: Does augmentation help?
# Uses best LR from experiment 1 (0.0005), sweeps batch size with augmentation
# on vs off. Longer training (200 epochs) to let augmented runs converge.

experiment2() {
    local EPOCH_MAX=200
    local LEARNING_RATE=0.0005

    for AUGMENTATION in false true
    do
        for BATCH_SIZE in 32 64 128
        do

            echo "AUG=${AUGMENTATION}, BS=${BATCH_SIZE}"

            OUT="${BASE_PATH}/experiment2/aug${AUGMENTATION}_bs${BATCH_SIZE}"
            mkdir -p "${OUT}"

            time uv run python -m xournalpp_htr.training.simple_htr.train \
                training.learning_rate="${LEARNING_RATE}" \
                training.batch_size="${BATCH_SIZE}" \
                augmentation.enabled="${AUGMENTATION}" \
                output_path="${OUT}" \
                training.epoch_max="${EPOCH_MAX}" 2>&1 | tee "${OUT}/train.log"

        done
    done
}

# ============
# Experiment 3
# ============

# Question: Does dropout help?
# Uses best LR from experiment 1 (0.0005) and best augmentation setting from
# experiment 2. Sweeps dropout rates.

experiment3() {
    local EPOCH_MAX=200
    local LEARNING_RATE=0.0005
    local BATCH_SIZE=64

    for DROPOUT in 0.0 0.2 0.5
    do
        for AUGMENTATION in false true
        do

            echo "DO=${DROPOUT}, AUG=${AUGMENTATION}"

            OUT="${BASE_PATH}/experiment3/do${DROPOUT}_aug${AUGMENTATION}"
            mkdir -p "${OUT}"

            time uv run python -m xournalpp_htr.training.simple_htr.train \
                training.learning_rate="${LEARNING_RATE}" \
                training.batch_size="${BATCH_SIZE}" \
                model.dropout="${DROPOUT}" \
                augmentation.enabled="${AUGMENTATION}" \
                output_path="${OUT}" \
                training.epoch_max="${EPOCH_MAX}" 2>&1 | tee "${OUT}/train.log"

        done
    done
}

# ==================
# Run experiments
# ==================

# time experiment1
time experiment2
time experiment3
