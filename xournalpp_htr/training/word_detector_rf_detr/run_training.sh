#!/usr/bin/env bash

# `pipefail` is required so a failed training run still aborts the script even
# though its output is piped into `tee` for logging.
set -euo pipefail

# Run from this script's directory so that `experiments/` and Hydra's run
# directory land next to the model code rather than wherever this was invoked.
cd "$(dirname "${BASH_SOURCE[0]}")"

# ========
# Settings
# ========

BASE_PATH=experiments

# ============
# Experiment 1
# ============

# Question: what does the baseline config achieve? A 1-epoch smoke run reached
# val recall 91.0% / precision 92.8%, so this establishes where a full run lands.

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

# Question: is the default learning rate right? RF-DETR ships lr=1e-4; sweep
# around it while holding everything else at the config.py defaults.

experiment2() {
    local EPOCHS=50

    for LR in 5e-5 1e-4 2e-4
    do

        echo "LR=${LR}"

        OUT="${BASE_PATH}/experiment2/lr${LR}"
        mkdir -p "${OUT}"

        uv run python -m xournalpp_htr.training.word_detector_rf_detr.train \
            training.lr="${LR}" \
            training.epochs="${EPOCHS}" \
            output_path="${OUT}" \
            hydra.run.dir="${OUT}" 2>&1 | tee "${OUT}/train.log"

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
# - Variant comparison (model.variant=small|medium|large) -- most directly
#   relevant to recall, which is what this detector is optimised for.
# - Does increasing model.resolution beyond 1024 help with small words?
#   Must stay a multiple of 32.
# - Confidence threshold sweep for the precision/recall trade-off.
# - How does RF-DETR's NMS-free decoding compare to YOLO on dense text lines?
