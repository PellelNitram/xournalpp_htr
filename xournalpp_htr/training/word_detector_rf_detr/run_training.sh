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

    uv run python -m xournalpp_htr.training.word_detector_rf_detr.train \
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

    for BATCH in 4 8
    do
        for LR in 5e-5 1e-4
        do

            echo "BS=${BATCH}, LR=${LR}"

            OUT="${BASE_PATH}/experiment2/bs${BATCH}_lr${LR}"
            mkdir -p "${OUT}"

            uv run python -m xournalpp_htr.training.word_detector_rf_detr.train \
                training.batch_size="${BATCH}" \
                training.lr="${LR}" \
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
# - Does a larger variant (model.variant=large) improve detection quality?
# - Does increasing model.resolution (e.g. 1288 = 56 * 23) help with small words?
# - Confidence threshold sweep for the precision/recall trade-off (recall is
#   the metric we optimise for).
# - How does RF-DETR's NMS-free decoding compare to YOLO on dense text lines?
