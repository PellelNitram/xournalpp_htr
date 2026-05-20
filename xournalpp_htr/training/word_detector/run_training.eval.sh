#!/usr/bin/env bash
set -euo pipefail

BASE_PATH=${HOME}/experiments

best_f1=0
best_dir=""

for json_file in "${BASE_PATH}"/experiment1/*/best_model.json; do
    [ -f "${json_file}" ] || continue
    f1=$(python3 -c "import json; print(json.load(open('${json_file}'))['f1'])")
    dir=$(dirname "${json_file}")
    echo "${dir}: F1=${f1}"
    if python3 -c "exit(0 if ${f1} > ${best_f1} else 1)"; then
        best_f1="${f1}"
        best_dir="${dir}"
    fi
done

echo
echo "Best model: ${best_dir}/best_model.pth (F1=${best_f1})"
