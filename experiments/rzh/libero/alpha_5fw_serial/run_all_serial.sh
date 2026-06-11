#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
MATRIX_FILE="${MATRIX_FILE:-${SCRIPT_DIR}/framework_matrix.yaml}"

cd "${REPO_ROOT}"

mapfile -t FRAMEWORK_ROWS < <("${PYTHON_BIN}" - "${MATRIX_FILE}" <<'PY'
import json
import sys

from omegaconf import OmegaConf

matrix = OmegaConf.load(sys.argv[1])
for item in matrix:
    overrides = item.get("overrides", {})
    overrides = OmegaConf.to_container(overrides, resolve=True) if overrides else {}
    row = {
        "framework": item.framework,
        "base_vlm": item.base_vlm,
        "per_device_batch_size": item.get("per_device_batch_size", 16),
        "freeze_modules": item.get("freeze_modules", "qwen_vl_interface"),
        "overrides": overrides,
    }
    print(json.dumps(row, ensure_ascii=False))
PY
)

if [[ "${#FRAMEWORK_ROWS[@]}" -eq 0 ]]; then
  echo "未在 ${MATRIX_FILE} 中找到任何框架配置。" >&2
  exit 1
fi

json_field() {
  local row_json="$1"
  local field="$2"
  "${PYTHON_BIN}" - "${row_json}" "${field}" <<'PY'
import json
import sys

row = json.loads(sys.argv[1])
value = row.get(sys.argv[2], "")
if value is None:
    print("")
elif isinstance(value, (dict, list)):
    print(json.dumps(value, ensure_ascii=False))
else:
    print(value)
PY
}

for row in "${FRAMEWORK_ROWS[@]}"; do
  framework="$(json_field "${row}" framework)"
  base_vlm="$(json_field "${row}" base_vlm)"
  per_device_batch_size="$(json_field "${row}" per_device_batch_size)"
  freeze_modules="$(json_field "${row}" freeze_modules)"
  framework_overrides_json="$(json_field "${row}" overrides)"

  PER_DEVICE_BATCH_SIZE="${per_device_batch_size}" \
  FREEZE_MODULES="${freeze_modules}" \
  FRAMEWORK_OVERRIDES_JSON="${framework_overrides_json}" \
    "${SCRIPT_DIR}/run_one.sh" "${framework}" "${base_vlm}"
done
