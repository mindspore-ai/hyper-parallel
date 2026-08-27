# Tiny Qwen3-MoE 训练演示

这份示例使用仓库内的紧凑 Qwen3-MoE checkpoint 和示例目录内的确定性 `TinyCausalDataset`，不依赖 Hugging Face Hub，适合验证现有 Trainer/FSDP 能力和 7 种 CP wrapper。

## 1. 环境准备

示例默认使用 8 张 Ascend NPU。先确认设备和环境：

如果使用预构建或自行构建的完整 HP wheel，请先阅读[完整 wheel 安装指南](../../installation.md)，并在仓库目录之外完成安装验证。

```bash
npu-smi info
source /path/to/env.sh
```

进入仓库根目录，并选择 PyTorch backend：

```bash
cd /path/to/hyper-parallel
export HYPER_PARALLEL_PLATFORM=torch
```

训练数据由示例目录内的 `TinyCausalDataset` 生成，模型文件位于：

```text
examples/training_demo/tiny_qwen3_moe/
```

如果需要重新生成模型，执行：

```bash
python examples/training_demo/prepare_tiny_model.py
```

脚本使用固定随机种子 `42`。如果目标目录已经包含 `config.json` 和 `model.safetensors`，脚本会复用现有文件。

## 2. 一步 FSDP 冒烟

先用 1 个训练 step 验证环境、模型加载和 Trainer 初始化：

```bash
LABEL=tiny_fsdp_smoke \
HCCL_IF_BASE_PORT=11000 \
MASTER_PORT=11050 \
bash examples/training_demo/run.sh 8 \
  --training.train_iters=1 \
  --checkpoint.save_ckpt=false \
  --checkpoint.restore_from=null
```

`run.sh 8` 会启动 8 个进程，并默认使用 NPU `0,1,2,3,4,5,6,7`。如果这些设备正在被占用，应先选择空闲设备并调整运行脚本中的可见设备列表。

## 3. 完整 FSDP 训练

YAML 默认训练 25 steps，并从相同的初始 checkpoint 开始：

```bash
LABEL=tiny_fsdp \
HCCL_IF_BASE_PORT=11000 \
MASTER_PORT=11050 \
bash examples/training_demo/run.sh 8
```

日志写入：

```text
output/run_tiny_fsdp.log
```

checkpoint 写入：

```text
outputs/training_demo/tiny_qwen3_moe_checkpoints/global_step_25
```

## 4. FSDP + CP2 训练

CP2 通过 CLI override 打开，不需要复制或修改 YAML：

```bash
LABEL=tiny_fsdp_cp2 \
HCCL_IF_BASE_PORT=11020 \
MASTER_PORT=11070 \
bash examples/training_demo/run.sh 8 \
  --accelerator.cp_size=2 \
  --checkpoint.checkpoint_dir=./outputs/training_demo/tiny_qwen3_moe_cp2_checkpoints \
  --checkpoint.restore_from=null
```

日志写入：

```text
output/run_tiny_fsdp_cp2.log
```

CP2 的关键配置已经在 [train.yaml](../../../examples/training_demo/train.yaml) 中声明：

- `sequence_parallel: true` 保持 activation 的 CP 布局契约一致。
- `when: cp` 的 `qwen3_moe_flash_attention_cp_wrapper` 在 `cp_size > 1` 时安装。
- `create_attention_mask_in_dataloader: false`，由 wrapper 根据 CP rank 自动构造带 query offset 的 causal mask。
- `labels_are_shifted: true`，保证 CP 分片前已经保留跨分片的 next-token target。

## 5. 七种 CP wrapper 配置

专用配置位于 [cp_configs](../../../examples/training_demo/cp_configs/)。每个配置默认使用 8 个进程、训练 1 step，并关闭 checkpoint，适合作为独立冒烟用例。

| 配置 | wrapper | CP 拓扑 |
| --- | --- | --- |
| `sync_colossal.yaml` | `qwen3_moe_flash_attention_cp_wrapper` | CP2，同步 K/V all-gather |
| `sync_load_balance.yaml` | `sdpa_hf_load_balance_cp_wrapper` | CP2，同步 Head-Tail 负载均衡 |
| `sync_ulysses.yaml` | `sdpa_hf_ulysses_cp_wrapper` | CP2，同步 Pure Ulysses |
| `sync_hybrid.yaml` | `sdpa_hf_hybrid_cp_wrapper` | CP4，Ulysses degree 2 |
| `async_colossal.yaml` | `qwen3_moe_async_colossal_cp_wrapper` | CP2，异步 K/V all-gather |
| `async_ulysses.yaml` | `qwen3_moe_async_ulysses_cp_wrapper` | CP2，异步 Pure Ulysses |
| `async_hybrid.yaml` | `qwen3_moe_async_hybrid_cp_wrapper` | CP4，Ulysses degree 2 |

运行单个用例：

```bash
bash examples/training_demo/run_cp_wrappers.sh sync_ulysses
```

依次运行全部 7 个用例：

```bash
bash examples/training_demo/run_cp_wrappers.sh all
```

额外 CLI override 会透传到每个用例，例如：

```bash
bash examples/training_demo/run_cp_wrappers.sh async_colossal \
  --training.train_iters=2
```

Hybrid 用例要求 `cp_size=4` 且 `ulysses_degree=2`；tiny 模型的 4 个 Q head 和 2 个 K/V head 均满足整除约束。所有配置使用 `seq_len=16`，同时满足 CP 分片和 Head-Tail 的序列长度约束。

## 6. 日志与精度检查

训练过程中可以在日志中查看 step、loss 和 grad norm：

```bash
rg "step|loss|grad_norm|Training" output/run_tiny_fsdp.log
rg "step|loss|grad_norm|Training" output/run_tiny_fsdp_cp2.log
rg "step|loss|grad_norm|Training" output/run_cp_*.log
```

比较 FSDP 与 CP2 时，必须使用相同的初始 checkpoint、训练 steps、global batch size、数据种子和数据顺序。当前示例的默认值是：

```text
seed=42, num_samples=256, seq_len=16, vocab_size=256, train_iters=25
```

这个 tiny checkpoint 的 `max_position_embeddings` 是 `64`，因此 `seq_len` 不应超过 64。16K 长序列需要先同步扩大模型配置，并评估 CP 显式 causal mask 的显存开销。

## 7. 单元测试

在 CPU 上运行针对性回归测试时显式选择 PyTorch backend：

```bash
export HYPER_PARALLEL_PLATFORM=torch
python -m pytest tests/ut/trainer/test_training_demo.py -q
```

测试覆盖预移位 labels、7 个 YAML 的 wrapper Target 与 CP 拓扑。完整 8 卡 NPU 训练需要 Ascend runtime、`torch_npu`、HCCL 以及空闲 NPU 设备。
