#!/usr/bin/env bash
set -u

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "用法：$0 <run_output_dir> [screen_name]" >&2
  exit 2
fi

RUN_DIR="$1"
SCREEN_NAME="${2:-}"
POLL_SECONDS="${POLL_SECONDS:-300}"
TRAIN_LOG="${RUN_DIR%/}/train.log"
FINAL_PT="${RUN_DIR%/}/final_model/pytorch_model.pt"
FINAL_SAFE="${RUN_DIR%/}/final_model/model.safetensors"

SHUTDOWN_BIN="${SHUTDOWN_BIN:-$(command -v shutdown || true)}"
if [[ -z "${SHUTDOWN_BIN}" ]]; then
  echo "未找到 shutdown 命令，无法设置自动关机。" >&2
  exit 1
fi

log() {
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"
}

shutdown_now() {
  log "执行关机命令：${SHUTDOWN_BIN} -h now"
  sync
  "${SHUTDOWN_BIN}" -h now
}

log "自动关机监控启动。run_dir=${RUN_DIR} screen=${SCREEN_NAME:-未设置} poll=${POLL_SECONDS}s"

while true; do
  if grep -q "Training complete. Final model saved" "${TRAIN_LOG}" 2>/dev/null; then
    log "检测到训练正常完成日志。"
    shutdown_now
    exit 0
  fi

  if [[ -n "${SCREEN_NAME}" ]] && ! screen -ls 2>/dev/null | grep -q "${SCREEN_NAME}"; then
    if [[ -s "${FINAL_PT}" || -s "${FINAL_SAFE}" ]]; then
      log "screen 已退出且 final_model 存在。"
      shutdown_now
      exit 0
    fi

    log "screen 已退出但未检测到 final_model，不自动关机，保留现场排错。"
    exit 1
  fi

  sleep "${POLL_SECONDS}"
done
