#!/usr/bin/env bash

# `pipefail` is required so a failed training run still aborts the script even
# though its output is piped into `tee` for logging.
set -euo pipefail

# Step 2: train on the crops written by `run_pipeline.sh`.

BASE_PATH=experiments

# ============
# Experiment 1
# ============

# Question: Does a SimpleHTR trained on rendered IAM-OnDB crops beat the
# IAM-trained one in the benchmark pipeline (issue #150)? Defaults are the
# settings of SimpleHTR's best run (experiment 3: dropout 0.5, augmentation on,
# lr 0.0005, batch size 64, from scratch).

experiment1() {
    OUT="${BASE_PATH}/experiment1/baseline"
    mkdir -p "${OUT}"

    time uv run --extra training-simple-htr-ondb python -m xournalpp_htr.training.simple_htr_ondb.train \
        output_path="${OUT}" 2>&1 | tee "${OUT}/train.log"
}

# ==================
# Run experiments
# ==================

time experiment1
