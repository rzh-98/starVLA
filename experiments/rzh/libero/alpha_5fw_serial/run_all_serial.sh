#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MATRIX_FILE="${MATRIX_FILE:-${SCRIPT_DIR}/framework_matrix.yaml}"

cd "${REPO_ROOT}"

mapfile -t FRAMEWORK_ROWS < <("${PYTHON_BIN}" - "${MATRIX_FILE}" <<'PY'
import sys
from omegaconf import OmegaConf

matrix = OmegaConf.load(sys.argv[1])
for item in matrix:
    print(f"{item.framework}\t{item.base_vlm}")
PY
)

if [[ "${#FRAMEWORK_ROWS[@]}" -eq 0 ]]; then
  echo "未在 ${MATRIX_FILE} 中找到任何框架配置。" >&2
  exit 1
fi

for row in "${FRAMEWORK_ROWS[@]}"; do
  IFS=$'\t' read -r framework base_vlm <<< "${row}"
  "${SCRIPT_DIR}/run_one.sh" "${framework}" "${base_vlm}"
done
