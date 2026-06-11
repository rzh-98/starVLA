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
- `prepare_assets.sh`：下载并整理模型、LIBERO 数据，最后软链接回仓库。
- `run_one.sh`：启动单个框架。
- `run_all_serial.sh`：按 matrix 顺序串行启动全部框架。
- `requirements-extra.txt`：实验额外依赖版本。

## 矩阵覆盖项

`framework_matrix.yaml` 不只记录 framework 与 base VLM，也记录每个 framework 的关键训练覆盖项。当前策略是：

- `QwenOFT`、`QwenPI`、`QwenGR00T`、`QwenAdapter`：冻结 `qwen_vl_interface`，训练各自 action/head/adapter 相关参数，`per_device_batch_size=16`。
- `QwenFast`：不冻结 `qwen_vl_interface`，因为 FAST action tokenizer 本身没有可训练参数，训练信号落在 VLM 的 action token 预测上；为降低显存压力，`per_device_batch_size=2`。
- `QwenAdapter`：额外覆盖 `framework.action_model.hidden_dim=2560`，匹配当前 `Qwen3-VL-4B` 的 hidden size；同时覆盖 `num_actions_chunk=8` 和 `future_action_window_size=7`，匹配 LIBERO 的 action horizon。

如果后续某个 framework 需要额外配置，可以在对应矩阵项里添加：

```yaml
overrides:
  some.config.key: value
```

这些覆盖项会同时进入真实训练命令和输出目录中的 `config.full.yaml`。

## Baseline 复现 TODO

状态约定：

- `done`：已经完成并可复查。
- `doing`：当前正在跑或正在验证。
- `todo`：还未开始。
- `stopped`：主动停止，不作为当前主线结果。

当前主线从“轻量 frozen 对比”调整为“高显存 full fine-tune baseline 复现”。原因是 `QwenOFT`、`QwenPI`、`QwenGR00T`、`QwenAdapter` 在五框架矩阵里默认冻结 `qwen_vl_interface`，显存主要花在前向和 action/head 小模块上；96G 显卡更适合解冻 Qwen-VL 主干的 full fine-tune。

- `done`：Fork 工作流已整理，`origin` 指向个人 fork，`upstream` 指向官方仓库。
- `done`：SwanLab bridge 已接入，项目名固定为 `starVLA`。
- `done`：模型和数据已迁移到 `/root/data/Pretrained_models` 与 `/root/data/Datasets`，仓库内通过软链接访问。
- `done`：五个 framework 的 1-step smoke 已跑通，覆盖 `QwenOFT`、`QwenFast`、`QwenPI`、`QwenGR00T`、`QwenAdapter`。
- `stopped`：`alpha_libero_QwenOFT_e18341e_seed42` 已主动停止。该 run 是 `QwenOFT + freeze qwen_vl_interface`，显存占用低，不符合当前“优先利用 96G 显存跑大任务”的目标。
- `done`：已探测 `QwenPI full fine-tune`，即 `framework=QwenPI`、`freeze_modules=""`、`base_vlm=Qwen3-VL-4B-Instruct`。`per_device_batch_size=8/6/4` 都在第 1 步 optimizer state 初始化阶段 OOM；其中 batch 6/4 峰值约 `92.8G`，还需要额外分配约 `16.5G`。结论：当前单张 96G 卡不适合直接跑 `QwenPI` 全量 AdamW fine-tune，除非改用 CPU offload、8-bit optimizer、冻结部分 Qwen 层或多卡 ZeRO。
- `doing`：当前高显存正式 baseline 改为 `QwenFast full fine-tune bs8`，即 `framework=QwenFast`、`freeze_modules=""`、`base_vlm=Qwen3-VL-4B-Instruct-Action`、`per_device_batch_size=8`、`data_mix=libero_all`、`seed=42`。probe 已稳定跑过 50 step，显存约 `95.3G / 97.9G`，是当前单卡 96G 上能实际跑起来且最吃显存的配置。
- `todo`：如果 `QwenFast full fine-tune bs8` 能稳定跑过 100 step，保留正式 run 继续到 `80000` step，仅保存最终模型，避免磁盘被周期 checkpoint 填满。
- `todo`：在同一 high-vram 输出根目录下补跑 `QwenGR00T full fine-tune`，用于和 `QwenFast full fine-tune` 比较。
- `todo`：后续重新挑战 `QwenPI full fine-tune` 时，优先尝试 CPU optimizer offload 或 8-bit optimizer，而不是继续单纯降低 batch。
- `todo`：后续需要官方 cotrain baseline 时，再启用 `train_starvla_cotrain.py` 和 VLM 数据；当前本机只准备了 LIBERO VLA 数据，尚未下载 `StarVLA/LLaVA-OneVision-COCO`。

高显存实验输出根目录固定为：

```text
./playground/Checkpoints/starvla_alpha_libero_high_vram
```

当前 `doing` run：

```text
screen: starvla_qwenfast_fullft_bs8
run_id: alpha_libero_QwenFast_fullft_bs8_b17ac05_seed42
SwanLab: https://swanlab.cn/@antidote/starVLA/runs/0ru3o71p
output_dir: ./playground/Checkpoints/starvla_alpha_libero_high_vram/alpha_libero_QwenFast_fullft_bs8_b17ac05_seed42
```

本 run 为了贴近 96G 显存上限，使用：

```text
framework=QwenFast
freeze_modules=""
per_device_batch_size=8
max_train_steps=80000
save_interval=999999
eval_interval=999999
```

`eval_interval` 暂时拉大到超过总步数，是因为 `QwenFast full fine-tune bs8` 训练阶段已经占用约 `95.3G / 97.9G` 显存；如果训练中途执行生成式 eval，可能额外申请 KV cache 导致 OOM。正式评估建议在训练完成或阶段性停止后，用单独 eval 脚本和更小 batch 运行。

## 资产准备

这一步是计划的一部分，不是可选项。训练前请先把 `ASSET_ROOT` 指向一块可写的大容量磁盘，然后执行：

```bash
ASSET_ROOT=/path/to/big_disk/starvla_assets \
./experiments/rzh/libero/alpha_5fw_serial/prepare_assets.sh
```

如果当前机器无法直连 `huggingface.co`，可以临时指定镜像端点：

```bash
HF_ENDPOINT=https://hf-mirror.com \
ASSET_ROOT=/path/to/big_disk/starvla_assets \
./experiments/rzh/libero/alpha_5fw_serial/prepare_assets.sh
```

如果希望模型和数据分别落在固定目录，也可以显式指定：

```bash
HF_ENDPOINT=https://hf-mirror.com \
ASSET_ROOT=/root/data/starvla_assets \
MODEL_ROOT=/root/data/Pretrained_models \
DATA_ROOT=/root/data/Datasets \
./experiments/rzh/libero/alpha_5fw_serial/prepare_assets.sh
```

默认会准备这些资源：

- `Qwen/Qwen3-VL-4B-Instruct`
- `StarVLA/Qwen3-VL-4B-Instruct-Action`
- `physical-intelligence/fast`，供 `QwenFast` action tokenizer 使用
- `IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot`
- `IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot`
- `IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot`
- `IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot`

脚本会把模型链接到 `playground/Pretrained_models`，把数据链接到 `playground/Datasets`，这样训练脚本可以继续沿用仓库里的标准相对路径。

如果你只想先确认下载计划，可以用：

```bash
DRY_RUN=1 ASSET_ROOT=/path/to/big_disk/starvla_assets \
./experiments/rzh/libero/alpha_5fw_serial/prepare_assets.sh
```

如果你后面要切到 `train_starvla_cotrain.py`，可以额外加：

```bash
INCLUDE_VLM_DATA=1
```

这样会顺手下载 `StarVLA/LLaVA-OneVision-COCO`。

## 本机资产记录

当前机器的基础资产放在：

```text
/root/data/Pretrained_models
/root/data/Datasets
```

本次已完成下载：

- `Qwen/Qwen3-VL-4B-Instruct`
- `StarVLA/Qwen3-VL-4B-Instruct-Action`
- `physical-intelligence/fast`
- 四个 `IPEC-COMMUNITY` LIBERO 数据子集

落盘后总占用约 `19G`。仓库内已创建软链接：

```text
/root/Code/starVLA/playground/Pretrained_models -> /root/data/Pretrained_models
/root/Code/starVLA/playground/Datasets -> /root/data/Datasets
```

## 本机 GPU 环境记录

当前 GPU 为 Blackwell 架构，旧的 `torch==2.6.0+cu124` 会报 `no kernel image is available for execution on the device`。本机已升级为：

```text
torch==2.11.0+cu128
torchvision==0.26.0+cu128
```

新版本 `torchvision` 不再提供 `torchvision.io.VideoReader`，而这批 LIBERO mp4 需要 PyAV 软解 AV1，因此本实验默认使用 `pyav` 读取视频。

PyAV 解码 AV1 视频时，多进程 DataLoader 容易放大内存和解码器资源占用。当前实验默认使用保守设置：

```text
num_workers=0
pin_memory=false
persistent_workers=false
prefetch_factor=1
```

冒烟测试和首轮正式训练先以稳定跑通为主。后续如果 GPU 吞吐偏低，可以逐步试 `DATALOADER_NUM_WORKERS=1` 或 `DATALOADER_NUM_WORKERS=2`，每次只改一个变量并观察 `train.log`、显存、内存和 SwanLab 曲线。

## 冒烟测试

```bash
conda activate starVLA
cd /root/Code/starVLA
ASSET_ROOT=/path/to/big_disk/starvla_assets ./experiments/rzh/libero/alpha_5fw_serial/prepare_assets.sh
MAX_TRAIN_STEPS=1 SAVE_INTERVAL=1 EVAL_INTERVAL=1 ./experiments/rzh/libero/alpha_5fw_serial/run_all_serial.sh
```

## 正式串行训练

```bash
conda activate starVLA
cd /root/Code/starVLA
./experiments/rzh/libero/alpha_5fw_serial/run_all_serial.sh
```

## 存储注意

当前机器磁盘总量约 `227G`。1-step smoke 已显示单个 final model 大致占用：

- `QwenOFT`：约 `9G`
- `QwenFast`：约 `9G`
- `QwenPI`：约 `16G`
- `QwenGR00T`：约 `9G`
- `QwenAdapter`：约 `12G`

如果按默认 `max_train_steps=80000`、`save_interval=10000` 跑完五个 framework，周期 checkpoint 加 final model 会明显超过当前磁盘容量。当前机器上建议正式训练先用：

```bash
SAVE_INTERVAL=999999 ./experiments/rzh/libero/alpha_5fw_serial/run_all_serial.sh
```

这样每个 framework 只保留训练结束的 `final_model`，代价是中途断点恢复能力较弱。后续如果扩到 600G 以上，再恢复 `SAVE_INTERVAL=10000`。

## 常用覆盖项

- `CUDA_VISIBLE_DEVICES=0,1`：限制每个串行任务可见的 GPU。
- `NUM_PROCESSES=1`：手动覆盖 PyTorch 检测到的 CUDA 设备数量。
- `VIDEO_BACKEND=pyav`：视频读取后端，当前实验默认使用 `pyav`，避免新版本 `torchvision` 缺少 `VideoReader`，也避免 `decord` 无法读取这批 AV1 mp4。
- `DATALOADER_NUM_WORKERS=0`：DataLoader worker 数量，当前默认 `0`，优先保证 PyAV 读取稳定。
- `DATALOADER_PIN_MEMORY=false`：是否固定页内存，当前默认关闭。
- `DATALOADER_PERSISTENT_WORKERS=false`：是否保留 DataLoader worker，`num_workers=0` 时不会启用。
- `DATALOADER_PREFETCH_FACTOR=1`：每个 worker 预取 batch 数，只有 `num_workers>0` 时传给 PyTorch。
- `RESUME=1`：允许复用已有输出目录，并传入 `--trainer.is_resume true`。
- `OVERWRITE=1`：重新启动前删除已有输出目录。
- `RUN_ROOT_DIR=/path/to/checkpoints`：修改输出根目录。
- `WANDB_ENTITY=...`：如仍需使用 W&B entity，可显式传入。
- `STARVLA_USE_SWANLAB=0`：关闭 SwanLab bridge，保留官方 W&B 行为。

每个输出目录会记录 `train.log`、`config.full.yaml`、`git_info.txt` 和 `launch_command.sh`。
