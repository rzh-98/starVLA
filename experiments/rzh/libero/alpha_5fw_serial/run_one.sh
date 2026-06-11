#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "用法：$0 <framework> <base_vlm>" >&2
  exit 2
fi

FRAMEWORK_NAME="$1"
BASE_VLM="$2"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ -z "${OMP_NUM_THREADS:-}" || "${OMP_NUM_THREADS}" == "0" ]]; then
  export OMP_NUM_THREADS=1
fi

cd "${REPO_ROOT}"

SEED="${SEED:-42}"
CONFIG_YAML="${CONFIG_YAML:-${SCRIPT_DIR}/config_base.yaml}"
LIBERO_DATA_ROOT="${LIBERO_DATA_ROOT:-playground/Datasets/LEROBOT_LIBERO_DATA}"
DATA_MIX="${DATA_MIX:-libero_all}"
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"
PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-16}"
if [[ "${FREEZE_MODULES+x}" != "x" ]]; then
  FREEZE_MODULES="qwen_vl_interface"
fi
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
DATALOADER_PIN_MEMORY="${DATALOADER_PIN_MEMORY:-false}"
DATALOADER_PERSISTENT_WORKERS="${DATALOADER_PERSISTENT_WORKERS:-false}"
DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-1}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-./playground/Checkpoints/starvla_alpha_libero_5fw}"
WANDB_PROJECT="${WANDB_PROJECT:-starVLA}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-80000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-10000}"
LOGGING_FREQUENCY="${LOGGING_FREQUENCY:-100}"
EVAL_INTERVAL="${EVAL_INTERVAL:-100}"
NUM_PROCESSES="${NUM_PROCESSES:-}"
RESUME="${RESUME:-0}"
OVERWRITE="${OVERWRITE:-0}"
FRAMEWORK_OVERRIDES_JSON="${FRAMEWORK_OVERRIDES_JSON:-}"
if [[ -z "${FRAMEWORK_OVERRIDES_JSON}" ]]; then
  FRAMEWORK_OVERRIDES_JSON="{}"
fi

export STARVLA_USE_SWANLAB="${STARVLA_USE_SWANLAB:-1}"
export SWANLAB_MODE="${SWANLAB_MODE:-online}"

SHORT_COMMIT="${SHORT_COMMIT:-$(git rev-parse --short HEAD)}"
RUN_ID="${RUN_ID:-alpha_libero_${FRAMEWORK_NAME}_${SHORT_COMMIT}_seed${SEED}}"
OUTPUT_DIR="${RUN_ROOT_DIR%/}/${RUN_ID}"

if [[ ! -e "${BASE_VLM}" ]]; then
  echo "base VLM 路径不存在：${BASE_VLM}" >&2
  echo "请先运行 ./experiments/rzh/libero/alpha_5fw_serial/prepare_assets.sh" >&2
  exit 1
fi

if [[ ! -d "${LIBERO_DATA_ROOT}" ]]; then
  echo "LIBERO 数据根目录不存在：${LIBERO_DATA_ROOT}" >&2
  echo "请先运行 ./experiments/rzh/libero/alpha_5fw_serial/prepare_assets.sh" >&2
  exit 1
fi

if [[ -z "${NUM_PROCESSES}" ]]; then
  NUM_PROCESSES="$("${PYTHON_BIN}" - <<'PY'
import torch

print(torch.cuda.device_count())
PY
)"
fi

if [[ "${NUM_PROCESSES}" -lt 1 ]]; then
  echo "torch.cuda.device_count() 没有发现可见 CUDA 设备。" >&2
  echo "请在启动训练前设置 CUDA_VISIBLE_DEVICES 或 NUM_PROCESSES。" >&2
  exit 1
fi

if [[ -e "${OUTPUT_DIR}" ]]; then
  if [[ "${OVERWRITE}" == "1" ]]; then
    rm -rf -- "${OUTPUT_DIR}"
  elif [[ "${RESUME}" == "1" ]]; then
    echo "复用已有输出目录继续训练：${OUTPUT_DIR}"
  else
    echo "输出目录已存在：${OUTPUT_DIR}" >&2
    echo "如需继续训练请设置 RESUME=1；如需覆盖请设置 OVERWRITE=1。" >&2
    exit 1
  fi
fi

mkdir -p "${OUTPUT_DIR}"

{
  echo "UTC时间：$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "仓库根目录：${REPO_ROOT}"
  echo "当前提交：$(git rev-parse HEAD)"
  echo "版本描述：$(git describe --tags --always --dirty)"
  echo "当前分支：$(git branch --show-current)"
  echo
  echo "[远端]"
  git remote -v
  echo
  echo "[状态]"
  git status --short --branch
} > "${OUTPUT_DIR}/git_info.txt"

WANDB_ARGS=(--wandb_project "${WANDB_PROJECT}")
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  WANDB_ARGS+=(--wandb_entity "${WANDB_ENTITY}")
fi

RESUME_ARGS=()
if [[ "${RESUME}" == "1" ]]; then
  RESUME_ARGS+=(--trainer.is_resume true)
fi

CONFIG_DOTLIST=(
  "framework.name=${FRAMEWORK_NAME}"
  "framework.qwenvl.base_vlm=${BASE_VLM}"
  "datasets.vla_data.data_root_dir=${LIBERO_DATA_ROOT}"
  "datasets.vla_data.data_mix=${DATA_MIX}"
  "datasets.vla_data.per_device_batch_size=${PER_DEVICE_BATCH_SIZE}"
  "datasets.vla_data.video_backend=${VIDEO_BACKEND}"
  "datasets.vla_data.num_workers=${DATALOADER_NUM_WORKERS}"
  "datasets.vla_data.pin_memory=${DATALOADER_PIN_MEMORY}"
  "datasets.vla_data.persistent_workers=${DATALOADER_PERSISTENT_WORKERS}"
  "datasets.vla_data.prefetch_factor=${DATALOADER_PREFETCH_FACTOR}"
  "trainer.max_train_steps=${MAX_TRAIN_STEPS}"
  "trainer.save_interval=${SAVE_INTERVAL}"
  "trainer.logging_frequency=${LOGGING_FREQUENCY}"
  "trainer.eval_interval=${EVAL_INTERVAL}"
  "trainer.freeze_modules=${FREEZE_MODULES}"
  "run_root_dir=${RUN_ROOT_DIR}"
  "run_id=${RUN_ID}"
  "seed=${SEED}"
  "wandb_project=${WANDB_PROJECT}"
)

if [[ -n "${WANDB_ENTITY:-}" ]]; then
  CONFIG_DOTLIST+=("wandb_entity=${WANDB_ENTITY}")
fi

if [[ "${RESUME}" == "1" ]]; then
  CONFIG_DOTLIST+=("trainer.is_resume=true")
fi

OVERRIDE_ROWS_TEXT="$("${PYTHON_BIN}" - "${FRAMEWORK_OVERRIDES_JSON}" <<'PY'
import json
import sys

overrides = json.loads(sys.argv[1])
for key, value in overrides.items():
    if isinstance(value, bool):
        value = "true" if value else "false"
    elif value is None:
        value = "null"
    else:
        value = str(value)
    print(f"{key}\t{value}")
PY
)"
OVERRIDE_ROWS=()
if [[ -n "${OVERRIDE_ROWS_TEXT}" ]]; then
  mapfile -t OVERRIDE_ROWS <<< "${OVERRIDE_ROWS_TEXT}"
fi

EXTRA_OVERRIDE_ARGS=()
for row in "${OVERRIDE_ROWS[@]}"; do
  IFS=$'\t' read -r key value <<< "${row}"
  CONFIG_DOTLIST+=("${key}=${value}")
  EXTRA_OVERRIDE_ARGS+=(--"${key}" "${value}")
done

COMMAND=(
  "${PYTHON_BIN}" -m accelerate.commands.launch
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml
  --num_processes "${NUM_PROCESSES}"
  starVLA/training/train_starvla.py
  --config_yaml "${CONFIG_YAML}"
  --framework.name "${FRAMEWORK_NAME}"
  --framework.qwenvl.base_vlm "${BASE_VLM}"
  --datasets.vla_data.data_root_dir "${LIBERO_DATA_ROOT}"
  --datasets.vla_data.data_mix "${DATA_MIX}"
  --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}"
  --datasets.vla_data.video_backend "${VIDEO_BACKEND}"
  --datasets.vla_data.num_workers "${DATALOADER_NUM_WORKERS}"
  --datasets.vla_data.pin_memory "${DATALOADER_PIN_MEMORY}"
  --datasets.vla_data.persistent_workers "${DATALOADER_PERSISTENT_WORKERS}"
  --datasets.vla_data.prefetch_factor "${DATALOADER_PREFETCH_FACTOR}"
  --trainer.max_train_steps "${MAX_TRAIN_STEPS}"
  --trainer.save_interval "${SAVE_INTERVAL}"
  --trainer.logging_frequency "${LOGGING_FREQUENCY}"
  --trainer.eval_interval "${EVAL_INTERVAL}"
  --trainer.freeze_modules "${FREEZE_MODULES}"
  --run_root_dir "${RUN_ROOT_DIR}"
  --run_id "${RUN_ID}"
  --seed "${SEED}"
  "${WANDB_ARGS[@]}"
  "${RESUME_ARGS[@]}"
  "${EXTRA_OVERRIDE_ARGS[@]}"
)

"${PYTHON_BIN}" - "${CONFIG_YAML}" "${OUTPUT_DIR}/config.full.yaml" "${CONFIG_DOTLIST[@]}" <<'PY'
import sys

from omegaconf import OmegaConf

from starVLA.model.framework.share_tools import apply_config_compat

config_yaml = sys.argv[1]
output_yaml = sys.argv[2]
dotlist = sys.argv[3:]

cfg = OmegaConf.load(config_yaml)
cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
cfg = apply_config_compat(cfg)
cfg.config_yaml = config_yaml
OmegaConf.save(cfg, output_yaml, resolve=True)
PY

{
  echo "#!/usr/bin/env bash"
  printf 'cd %q\n' "${REPO_ROOT}"
  printf 'export STARVLA_USE_SWANLAB=%q\n' "${STARVLA_USE_SWANLAB}"
  printf 'export SWANLAB_MODE=%q\n' "${SWANLAB_MODE}"
  printf 'export OMP_NUM_THREADS=%q\n' "${OMP_NUM_THREADS}"
  printf 'export CUDA_VISIBLE_DEVICES=%q\n' "${CUDA_VISIBLE_DEVICES:-}"
  printf 'export NUM_PROCESSES=%q\n' "${NUM_PROCESSES}"
  printf 'export VIDEO_BACKEND=%q\n' "${VIDEO_BACKEND}"
  printf 'export PER_DEVICE_BATCH_SIZE=%q\n' "${PER_DEVICE_BATCH_SIZE}"
  printf 'export FREEZE_MODULES=%q\n' "${FREEZE_MODULES}"
  printf 'export FRAMEWORK_OVERRIDES_JSON=%q\n' "${FRAMEWORK_OVERRIDES_JSON}"
  printf 'export DATALOADER_NUM_WORKERS=%q\n' "${DATALOADER_NUM_WORKERS}"
  printf 'export DATALOADER_PIN_MEMORY=%q\n' "${DATALOADER_PIN_MEMORY}"
  printf 'export DATALOADER_PERSISTENT_WORKERS=%q\n' "${DATALOADER_PERSISTENT_WORKERS}"
  printf 'export DATALOADER_PREFETCH_FACTOR=%q\n' "${DATALOADER_PREFETCH_FACTOR}"
  printf '%q ' "${COMMAND[@]}"
  echo
} > "${OUTPUT_DIR}/launch_command.sh"
chmod +x "${OUTPUT_DIR}/launch_command.sh"

TEE_ARGS=()
if [[ "${RESUME}" == "1" ]]; then
  TEE_ARGS=(-a)
fi

echo "启动框架 ${FRAMEWORK_NAME}，run_id=${RUN_ID}"
"${COMMAND[@]}" 2>&1 | tee "${TEE_ARGS[@]}" "${OUTPUT_DIR}/train.log"
