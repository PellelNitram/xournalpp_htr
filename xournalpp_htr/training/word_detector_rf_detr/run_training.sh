#!/usr/bin/env bash

# Training experiments for the RF-DETR word detector.
#
# Usage:
#   bash run_training.sh                  # experiment1 (baseline), the default
#   bash run_training.sh experiment1      # baseline with config.py defaults
#   bash run_training.sh experiment2      # learning-rate sweep
#   bash run_training.sh all              # every experiment, in order
#   EPOCHS=10 bash run_training.sh        # override the epoch count
#
# Run it under tmux: a baseline run is hours, not minutes.

# `pipefail` is required so a failed training run still aborts the script even
# though its output is piped into `tee` for logging.
set -euo pipefail

# Always run from this script's directory, so `experiments/` and Hydra's run
# directory land next to the model code no matter where this was invoked from.
cd "$(dirname "${BASH_SOURCE[0]}")"

# ========
# Settings
# ========

BASE_PATH=experiments
EPOCHS="${EPOCHS:-50}"

TRAIN_MODULE=xournalpp_htr.training.word_detector_rf_detr.train

# Run one training job. Args: output subdirectory, then Hydra overrides.
run_training() {
    local out="${BASE_PATH}/$1"; shift
    mkdir -p "${out}"

    echo "=== ${out} (epochs=${EPOCHS}) ==="
    uv run python -m "${TRAIN_MODULE}" \
        training.epochs="${EPOCHS}" \
        "$@" \
        output_path="${out}" \
        hydra.run.dir="${out}" 2>&1 | tee "${out}/train.log"
}

# ============
# Experiment 1
# ============

# Question: what does the baseline config achieve? A single 1-epoch smoke run
# reached val recall 91.0% / precision 92.8%, so this establishes where a full
# run lands.

experiment1() {
    run_training "experiment1/baseline"
}

# ============
# Experiment 2
# ============

# Question: is the default learning rate right? RF-DETR ships lr=1e-4; sweep
# around it while holding everything else at the config.py defaults.

experiment2() {
    for LR in 5e-5 1e-4 2e-4
    do
        run_training "experiment2/lr${LR}" training.lr="${LR}"
    done
}

# ==================
# Dispatch
# ==================

case "${1:-experiment1}" in
    experiment1) time experiment1 ;;
    experiment2) time experiment2 ;;
    all)         time experiment1; time experiment2 ;;
    -h|--help)   sed -n '3,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' ;;
    *)
        echo "Unknown experiment: $1" >&2
        echo "Valid: experiment1, experiment2, all" >&2
        exit 1
        ;;
esac

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
