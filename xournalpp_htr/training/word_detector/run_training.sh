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

# Question: General hyperparameter tuning

experiment1() {
    local EPOCH_MAX=200

    for LEARNING_RATE in 0.0005 0.001 0.002
    do
        for BATCH_SIZE in 16 32 64 128
        do

            echo "LR=${LEARNING_RATE}, BS=${BATCH_SIZE}"

            OUT="${BASE_PATH}/experiment1/lr${LEARNING_RATE}_bs${BATCH_SIZE}"
            mkdir -p "${OUT}"

            uv run python -m xournalpp_htr.training.word_detector.train \
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

# Question: Does train-time augmentation improve model performance?

experiment2() {
    local EPOCH_MAX=100

    for AUGMENT in false true
    do
        for SEED_SPLIT in 42 43 44
        do

            echo "AUGMENT=${AUGMENT}, SEED=${SEED_SPLIT}"

            OUT="${BASE_PATH}/experiment2/aug${AUGMENT}_seed${SEED_SPLIT}"
            mkdir -p "${OUT}"

            uv run python -m xournalpp_htr.training.word_detector.train \
                augmentation.enabled="${AUGMENT}" \
                seed.split="${SEED_SPLIT}" \
                training.epoch_max="${EPOCH_MAX}" \
                output_path="${OUT}" 2>&1 | tee "${OUT}/train.log"

        done
    done
}

# ============
# Experiment 3
# ============

# Question: Does augmentation help under the original WordDetectorNN training
# regime (bs=10, unbounded epochs with early stopping)?

experiment3() {
    local EPOCH_MAX=10000
    local BATCH_SIZE=10

    for AUGMENT in false true
    do
        for SEED_SPLIT in 42 43 44
        do

            echo "AUGMENT=${AUGMENT}, SEED=${SEED_SPLIT}"

            OUT="${BASE_PATH}/experiment3/aug${AUGMENT}_seed${SEED_SPLIT}"
            mkdir -p "${OUT}"

            uv run python -m xournalpp_htr.training.word_detector.train \
                augmentation.enabled="${AUGMENT}" \
                seed.split="${SEED_SPLIT}" \
                training.epoch_max="${EPOCH_MAX}" \
                training.batch_size="${BATCH_SIZE}" \
                output_path="${OUT}" 2>&1 | tee "${OUT}/train.log"

        done
    done
}

# ==================
# Run experiments
# ==================

# time experiment1
# time experiment2
time experiment3

# ==================
# Future experiments
# ==================

# Other questions to answer by conducting additional experiments:
# - Longer training help w/ more patience?
# - Cheap as k-fold on data to get good estimate of true performance
