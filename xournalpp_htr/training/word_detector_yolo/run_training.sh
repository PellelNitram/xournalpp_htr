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

# Question: Baseline training with default hyperparameters

experiment1() {
    local EPOCHS=50

    echo "Baseline: default hyperparameters"

    OUT="${BASE_PATH}/experiment1/baseline"
    mkdir -p "${OUT}"

    uv run python -m xournalpp_htr.training.word_detector_yolo.train \
        training.epochs="${EPOCHS}" \
        output_path="${OUT}" \
        hydra.run.dir="${OUT}" 2>&1 | tee "${OUT}/train.log"
}

# ============
# Experiment 2
# ============

# Question: General hyperparameter tuning

experiment2() {
    local EPOCHS=50

    for BATCH in 8 16
    do
        for LR in 0.0005 0.001
        do

            echo "BS=${BATCH}, LR=${LR}"

            OUT="${BASE_PATH}/experiment2/bs${BATCH}_lr${LR}"
            mkdir -p "${OUT}"

            uv run python -m xournalpp_htr.training.word_detector_yolo.train \
                training.batch="${BATCH}" \
                training.lr0="${LR}" \
                training.epochs="${EPOCHS}" \
                output_path="${OUT}" \
                hydra.run.dir="${OUT}" 2>&1 | tee "${OUT}/train.log"

        done
    done
}

# ==================
# Run experiments
# ==================

time experiment1
time experiment2

# ==================
# Future experiments
# ==================

# Other questions to answer by conducting additional experiments:
# - Does a larger model (yolov8m) improve detection quality?
# - Does increasing imgsz (e.g. 1280) help with small words?
# - Augmentation tuning: mosaic strength, scale range
# - Confidence threshold sweep for precision/recall trade-off
