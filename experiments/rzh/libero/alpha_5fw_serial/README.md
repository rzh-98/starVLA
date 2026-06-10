# StarVLA-alpha LIBERO 五框架串行实验

本目录用于记录和启动 LIBERO 上的 StarVLA-alpha 五框架对比实验。固定对比框架为：

- `QwenOFT`
- `QwenFast`
- `QwenPI`
- `QwenGR00T`
- `QwenAdapter`

启动器按顺序一次只运行一个框架。每个任务会使用 PyTorch 通过 `torch.cuda.device_count()` 看到的全部 GPU，输出写入 `./playground/Checkpoints/starvla_alpha_libero_5fw`。当 `STARVLA_USE_SWANLAB=1` 时，日志通过 W&B bridge 同步到 SwanLab 项目 `starVLA`。

## 文件说明

- `framework_matrix.yaml`：框架和 base VLM 的对应关系。
- `config_base.yaml`：LIBERO baseline 的共享配置。
- `run_one.sh`：启动单个框架。
- `run_all_serial.sh`：按 matrix 顺序串行启动全部框架。
- `requirements-extra.txt`：实验额外依赖版本。

## 冒烟测试

```bash
conda activate starVLA
cd /root/Code/starVLA
MAX_TRAIN_STEPS=1 SAVE_INTERVAL=1 EVAL_INTERVAL=1 ./experiments/rzh/libero/alpha_5fw_serial/run_all_serial.sh
```

## 正式串行训练

```bash
conda activate starVLA
cd /root/Code/starVLA
./experiments/rzh/libero/alpha_5fw_serial/run_all_serial.sh
```

## 常用覆盖项

- `CUDA_VISIBLE_DEVICES=0,1`：限制每个串行任务可见的 GPU。
- `NUM_PROCESSES=1`：手动覆盖 PyTorch 检测到的 CUDA 设备数量。
- `RESUME=1`：允许复用已有输出目录，并传入 `--trainer.is_resume true`。
- `OVERWRITE=1`：重新启动前删除已有输出目录。
- `RUN_ROOT_DIR=/path/to/checkpoints`：修改输出根目录。
- `WANDB_ENTITY=...`：如仍需使用 W&B entity，可显式传入。
- `STARVLA_USE_SWANLAB=0`：关闭 SwanLab bridge，保留官方 W&B 行为。

每个输出目录会记录 `train.log`、`config.full.yaml`、`git_info.txt` 和 `launch_command.sh`。
