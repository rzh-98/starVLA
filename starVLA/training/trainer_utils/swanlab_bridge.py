import os
import logging

logger = logging.getLogger(__name__)

_TRUE_VALUES = {"1", "true", "yes", "on"}


def sync_swanlab_if_enabled() -> None:
    """Bridge W&B logs into SwanLab when STARVLA_USE_SWANLAB is enabled."""
    if os.getenv("STARVLA_USE_SWANLAB", "0").strip().lower() not in _TRUE_VALUES:
        return

    try:
        import swanlab
    except ImportError as exc:
        raise RuntimeError(
            "STARVLA_USE_SWANLAB=1 需要安装 swanlab。"
            "请先安装 experiments/rzh/libero/alpha_5fw_serial/requirements-extra.txt。"
        ) from exc

    mode = os.getenv("SWANLAB_MODE", "online")
    swanlab.sync_wandb(mode=mode, wandb_run=False)
    logger.info(f"SwanLab W&B bridge 已启用，mode={mode}。")
