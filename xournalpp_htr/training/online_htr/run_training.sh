#!/usr/bin/env bash

# `pipefail` is required so a failed training run still aborts the script even
# though its output is piped into `tee` for logging.
set -euo pipefail

BASE_PATH=experiments

# ============
# Experiment 1
# ============

# Question: How does the Carbune BiLSTM-CTC model of OnlineHTR perform without
# the time channel, i.e. on (dx, dy, n) only (issue #174)? Reference settings:
# 3 layers x 64 nodes, dropout 0.25; trained on IAM-OnDB lines and words.

experiment1() {
    OUT="${BASE_PATH}/experiment1/baseline"
    mkdir -p "${OUT}"

    time uv run --extra training-online-htr python -m xournalpp_htr.training.online_htr.train \
        output_path="${OUT}" 2>&1 | tee "${OUT}/train.log"
}

# ==================
# Run experiments
# ==================

time experiment1
