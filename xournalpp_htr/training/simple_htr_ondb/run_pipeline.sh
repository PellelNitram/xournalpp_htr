#!/usr/bin/env bash

set -euo pipefail

# Step 1: render IAM-OnDB into word crops under ./data (gitignored), next to
# this script. Inspect `data/preview_*.png` and `data/stats.json` before
# training.

uv run --extra training-simple-htr-ondb python -m xournalpp_htr.training.simple_htr_ondb.prepare_data
