# 使用 HyperParallel 训练 DeepSeek-V4.1-Flash

[English](README.md) | **简体中文**

DeepSeek-V4.1-Flash 是多模态 MoE 模型。建议从 4K 图文配方开始，在一台昇腾主机上
验证视觉编码器、语言模型和分布式优化器的训练链路。

## 1. 支持范围与配方验证状态

| 配方 | 模型、初始化与数据 | 昇腾资源与并行配置 | 验证状态 |
| --- | --- | --- | --- |
| [4K 图文预训练](../../../examples/training_demo/deepseek_v41/train_deepseek_v41_vlm_online.yaml) | 保留全部 40 层文本网络和 32 个视觉块，参数宽度缩至 1/4，16 个路由专家。随机初始化，在线打包图文 JSONL。 | Atlas 900 A3/A5 SuperPoD 单机，16 个逻辑 NPU die。TP1 / CP4 / EP8 / PP1，稠密 FSDP8、专家 FSDP1、编码器 CP4。 | 下文单步命令在此拓扑上暂无公开实测结果。 |
| [4K 纯文本预训练](../../../examples/training_demo/deepseek_v41/train_deepseek_v41_online.yaml) | 同样裁剪文本主干并随机初始化。在线编码本地文本或脚本生成的测试数据。 | 同为单机 16 个逻辑 die，关闭视觉网络和编码器并行。 | 此配方暂无公开短跑结果。 |
| [1M 图文性能基准](../../../examples/training_demo/deepseek_v41/train_deepseek_v41_vlm_1M%20online_128die.yaml) | 裁剪参数宽度，128 个路由专家，随机初始化，使用仅供性能测试的固定专家路由。 | 八台主机，每台 16 个逻辑 die。TP1 / CP128 / EP128 / PP1，稠密 FSDP128、专家 FSDP1。 | 仓库提供配置和启动脚本，此处暂无该配置的端到端指标。 |

表中的设备数按启动器使用的逻辑 die 计。已发布的 910B3 实测使用另一套 8 卡配置，
详见[验证结果](#5-验证结果)。

## 2. 环境与资产准备

- 准备 Linux 昇腾训练环境。16 die 参考环境使用 HDK 25.5.5、CANN 9.1.0、
  Python 3.11.14、PyTorch 2.12.0 CPU wheel、torch-npu 2.12.0 和 Transformers ≥ 5.11.0。
  见 [HyperParallel 安装指南](../../../docs/installation.zh.md)和[模型环境配置](../../../examples/training_demo/deepseek_v41/deepseek_v41_quick_start.md#1-准备环境)。
- 编译 Multicore，安装 Omni Ops 设备包和 Python 扩展。CSA2 融合路径还需要配套的
  ops-transformer 设备算子与 `cann_ops_transformer` wheel。设置 `OPS_TRANSFORMER_ROOT`
  为构建说明中的目录，启动前加载 CANN 和 Omni Ops 环境。[融合算子要求（英文）](../../components/functional/README.md#deepseek-v41-fused-operators)

从[官方模型仓库](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)准备配置、
分词器和视觉处理资产。随机初始化配方在本地读取这些文件，无需权重分片。
启动脚本在缺少裁剪版 Engram 资产时会自动生成。

```bash
export MODEL_DIR="$PWD/model_from_hf/DeepSeek-V4.1-Flash"
hf download deepseek-ai/DeepSeek-V4.1-Flash \
  config.json tokenizer.json tokenizer_config.json chat_template.jinja inference/config.json \
  --local-dir "${MODEL_DIR}"
```

准备本地图文 JSONL，例如从 CC12M 整理图文对。每行一条记录，相对图片路径以 JSONL
所在目录为基准。处理器按 `inference/config.json` 应用视觉限制，加载器在线将完整样本
打包进 4096 token 窗口。[图文数据指南](../../../docs/guide/data/deepseek_v41_vlm_online_data_guide.zh.md)

```json
{"image": "images/000001.jpg", "caption": "A cat sitting on a chair."}
```

## 3. 快速启动

在仓库根目录将 `DATA_PATH` 指向本地图文 JSONL，在 16 个逻辑 die 上执行一个优化器步。

```bash
export DATA_PATH=/path/to/image_caption/train.jsonl
RUN_NAME=quick_start NPROC_PER_NODE=16 \
  bash examples/training_demo/deepseek_v41/run_deepseek_v41_vlm_online.sh \
  "${MODEL_DIR}" "${DATA_PATH}" \
  --training.train_iters=1 --checkpoint.restore_from=null --debug.check_dataset=warn
```

启动器将 `MODEL_DIR` 同时传给 `model.config_path` 和 `dataset.model_assets.config_path`。
成功时退出码为 0，生成 `output/training_demo/deepseek_v41/quick_start.success`，并在
`output/training_demo/deepseek_v41/run_quick_start.log` 中记录已完成的步数、loss、
梯度范数和耗时。本次运行关闭 checkpoint 保存。

## 4. 完整运行与常用配置

去掉 `--training.train_iters=1` 可运行默认的 100 步，也可以指定目标步数。
[纯文本启动脚本](../../../examples/training_demo/deepseek_v41/run_deepseek_v41_online.sh)
接收模型目录及可选的 JSONL 路径，文本记录使用 `text` 字段。默认数据文件不存在时，
脚本会生成测试文本。预处理会丢弃不足窗口的尾部 token，以满足 CSA2 和 CP 对齐要求。

| 配置项 | 4K 配方默认值与调整方式 |
| --- | --- |
| 数据与训练目标 | 通过 `dataset.data_path` 指定自有数据。图文 SFT 使用以 assistant 回复结尾的对话，并设置 `dataset.training_mode=sft` 和 `dataset.data_transform.sample_format=messages`。只监督最后一条 assistant 回复。 |
| 序列长度 | 保持 `dataset.data_transform.max_seq_len` 和 `dataloader.collate_fn.pad_to_length` 一致。图文配方的 `dataloader.token_budget` 也设置为同一数值（4096）。 |
| Batch 与分片 | `training.global_batch_size=8`、`training.micro_batch_size=1` 在稠密 DP4 下累计两个 microbatch。按总进程数联动调整 `accelerator.*_size` 和 `fsdp_config.*_shard_size`。 |
| 精度与内存 | FP32 初始化、优化器主参数和梯度归约，BF16 前向参数。两份配方均使用 Host Engram 和 `activation_checkpoint.mode: full`。 |
| Checkpoint 保存与恢复 | 启用 `checkpoint.save_ckpt=true` 并设置 `checkpoint.save_steps`。续训时沿用模型、拓扑、数据和 checkpoint 目录，设置 `checkpoint.restore_from=LATEST`。配方配置了恢复优化器、调度器、训练进度、随机数状态和数据游标，前提是对应状态已保存。 |

两个启动器默认在 `output/training_demo/deepseek_v41/artifacts/` 下使用独立的
checkpoint 目录，图文为 `checkpoints`，纯文本为 `text_checkpoints`。加载官方 HF 权重
需要使用发布版参数维度、全部 384 个路由专家、发布版 Engram 资产及完整权重分片，
并按原始尺寸规划资源。[权重加载步骤](../../../examples/training_demo/deepseek_v41/deepseek_v41_quick_start.md#加载官方预训练权重)

## 5. 验证结果

| 验证项 | 配置与结果 |
| --- | --- |
| 历史实机短跑 | 单机 8 张昇腾 910B3 NPU（每张 64 GiB HBM），使用随机初始化的裁剪模型与合成数据。三个纯文本配置各完成 6 步，loss 和梯度范数均有限。图文实验完成 10 个优化器步并保存 checkpoint。[公开实测记录](../../../docs/deepseek_v41_article/README.zh.md#4-训练结果训练指标可视化与模型正确性验证) |
| 数值验证 | [权重映射测试](../../../tests/ut/auto_models/models/deepseek_v41/test_checkpoint_mapping.py)和 [Chunk Loss 测试](../../../tests/ut/auto_models/test_deepseek_v41_chunk_loss.py)覆盖组件。16 die 配方的端到端数值对齐在此处仍待验证。 |
| 收敛与续训 | 上述配方的长程收敛与断点续训仍待验证。 |
| 性能 | 公开记录提供 8 卡实验的吞吐与显存数据。16 die 和 128 die 配方暂无公开性能数据。 |

## 6. 限制与故障排查

- CSA2 融合路径要求 TP1，并关闭 candidate pool。遇到算子导入或设备符号缺失时，检查
  Python wheel 与设备包是否从同一份 ops-transformer 源码构建。测试非融合注意力时，
  使用文档中的回退配置。[CSA2 配置](../../../examples/training_demo/deepseek_v41/deepseek_v41_quick_start.md#关闭-csa2-融合算子)
- MegaMoe 配置使用 EP 大于 1 和 PP1。使用普通 EP 与 grouped GEMM 时，按
  [模型快速启动文档](../../../examples/training_demo/deepseek_v41/deepseek_v41_quick_start.md#关闭-megakernel-后使用普通-ep)调整替换规则。
- 如果样本在打包时被丢弃，检查记录格式和本地图片路径。运行 1M 性能配方时，将
  `DATA_PATH` 指向已有的图文 JSONL，并检查
  [多机启动器](../../../examples/training_demo/deepseek_v41/run_deepseek_v41_vlm_1M_online_128die.sh)
  中的运行时路径和 rank 设置。当前生成器无法识别启动器传入的 `--sample-format image_caption`。
  使用真实路由时将 MoE 前向规则中的
  `fix_router` 设为 `false`。

## 延伸阅读

- [DeepSeek-V4.1-Flash 架构与训练报告](../../../docs/deepseek_v41_article/README.zh.md)
- [Trainer YAML 配置](../../../docs/guide/trainer/yaml_config.zh.md)
- [分布式 Checkpoint](../../../docs/guide/distributed_checkpoint.zh.md)
