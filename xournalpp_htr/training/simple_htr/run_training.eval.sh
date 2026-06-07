#!/usr/bin/env bash
set -euo pipefail

BASE_PATH=experiments

best_cer=999
best_dir=""

for json_file in "${BASE_PATH}"/experiment1/*/best_model.json; do
    [ -f "${json_file}" ] || continue
    cer=$(python3 -c "import json; print(json.load(open('${json_file}'))['cer'])")
    word_acc=$(python3 -c "import json; print(json.load(open('${json_file}'))['word_accuracy'])")
    dir=$(dirname "${json_file}")
    echo "${dir}: CER=${cer}, WordAcc=${word_acc}"
    if python3 -c "exit(0 if ${cer} < ${best_cer} else 1)"; then
        best_cer="${cer}"
        best_dir="${dir}"
    fi
done

echo
echo "Best model: ${best_dir}/best_model.pth (CER=${best_cer})"
