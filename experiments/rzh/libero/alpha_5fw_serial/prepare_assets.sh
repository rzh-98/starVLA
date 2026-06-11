#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

ASSET_ROOT="${ASSET_ROOT:-}"
if [[ -z "${ASSET_ROOT}" ]]; then
  echo "请先把 ASSET_ROOT 指向一块可写的大容量磁盘。" >&2
  echo "示例：ASSET_ROOT=/path/to/big_disk/starvla_assets" >&2
  exit 1
fi

MODEL_ROOT="${MODEL_ROOT:-${ASSET_ROOT%/}/models}"
DATA_ROOT="${DATA_ROOT:-${ASSET_ROOT%/}/datasets}"
INCLUDE_VLM_DATA="${INCLUDE_VLM_DATA:-0}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${DRY_RUN}" != "1" ]]; then
  mkdir -p "${MODEL_ROOT}" "${DATA_ROOT}"
fi

link_dir() {
  local source="$1"
  local target="$2"

  if [[ -e "${target}" && ! -L "${target}" ]]; then
    echo "目标已存在且不是符号链接：${target}" >&2
    echo "请先手动清理该路径，再重新运行。" >&2
    exit 1
  fi

  mkdir -p "$(dirname "${target}")"
  ln -sfn "${source}" "${target}"
}

"${PYTHON_BIN}" - "${REPO_ROOT}" "${MODEL_ROOT}" "${DATA_ROOT}" "${INCLUDE_VLM_DATA}" "${DRY_RUN}" <<'PY'
from pathlib import Path
import os
import shutil
import sys

from huggingface_hub import snapshot_download

repo_root = Path(sys.argv[1])
model_root = Path(sys.argv[2])
data_root = Path(sys.argv[3])
include_vlm_data = sys.argv[4] == "1"
dry_run = sys.argv[5] == "1"

modality_src = repo_root / "examples" / "LIBERO" / "train_files" / "modality.json"
libero_root = data_root / "LEROBOT_LIBERO_DATA"

tasks = [
    ("model", None, "Qwen/Qwen3-VL-4B-Instruct", model_root / "Qwen3-VL-4B-Instruct"),
    ("model", None, "StarVLA/Qwen3-VL-4B-Instruct-Action", model_root / "Qwen3-VL-4B-Instruct-Action"),
    ("model", None, "physical-intelligence/fast", model_root / "fast"),
    (
        "dataset",
        "dataset",
        "IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot",
        libero_root / "libero_spatial_no_noops_1.0.0_lerobot",
    ),
    (
        "dataset",
        "dataset",
        "IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot",
        libero_root / "libero_object_no_noops_1.0.0_lerobot",
    ),
    (
        "dataset",
        "dataset",
        "IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot",
        libero_root / "libero_goal_no_noops_1.0.0_lerobot",
    ),
    (
        "dataset",
        "dataset",
        "IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot",
        libero_root / "libero_10_no_noops_1.0.0_lerobot",
    ),
]

if include_vlm_data:
    tasks.append(("dataset", "dataset", "StarVLA/LLaVA-OneVision-COCO", data_root / "LLaVA-OneVision-COCO"))

print(f"资产根目录：{data_root.parent}")
print(f"模型目录：{model_root}")
print(f"数据目录：{data_root}")
print(f"下载端点：{os.environ.get('HF_ENDPOINT', 'https://huggingface.co')}")
print(f"是否包含可选 VLM 数据：{'是' if include_vlm_data else '否'}")

if dry_run:
    print("预览模式：不执行下载，仅打印计划。")
    for label, _repo_type, repo_id, target in tasks:
        print(f"  - {label}: {repo_id} -> {target}")
    print(f"  - 软链接: {model_root} -> {repo_root / 'playground' / 'Pretrained_models'}")
    print(f"  - 软链接: {data_root} -> {repo_root / 'playground' / 'Datasets'}")
    raise SystemExit(0)

for label, repo_type, repo_id, target in tasks:
    print(f"开始下载 {label}：{repo_id}")
    snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )
    if label == "dataset" and repo_id.startswith("IPEC-COMMUNITY/libero_"):
        meta_dir = target / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(modality_src, meta_dir / "modality.json")
        print(f"已写入 modality.json：{meta_dir / 'modality.json'}")

print("下载完成。")
PY

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

link_dir "${MODEL_ROOT}" "${REPO_ROOT}/playground/Pretrained_models"
link_dir "${DATA_ROOT}" "${REPO_ROOT}/playground/Datasets"

echo "资产准备完成。"
echo "模型已链接到：${REPO_ROOT}/playground/Pretrained_models"
echo "数据已链接到：${REPO_ROOT}/playground/Datasets"
