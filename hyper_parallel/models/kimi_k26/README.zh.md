# 使用 HyperParallel 训练 Kimi-K2.6

[English](README.md) | **简体中文**

Kimi-K2.6 是月之暗面的图文 MoE 模型。建议用 4 超节点配方在昇腾上验证
多模态 SFT 训练链路并测量吞吐。

## 1. 支持范围与配方验证状态

| 配方 | 模型、初始化与数据 | 昇腾资源与并行配置 | 验证状态 |
| --- | --- | --- | --- |
| [4 超节点多模态 SFT 性能基准](recipes/train.yaml) | 保留全部 61 层文本网络和 384 个路由专家，冻结视觉塔。随机初始化，使用合成图文对话和固定专家路由。 | 4 个 SuperPoD，512 个逻辑 NPU 设备（64 台主机，每台启动 8 个进程）。TP1 / CP1 / EP128 / PP1，稠密 FSDP128、专家 FSDP4。 | 配方记录了此规模下的历史性能结果，下文单步命令暂无公开实测结果。 |

性能基准设置 `fix_router: true`，为各目的 rank 分配固定份额的 token，忽略门控分数。
正确性和收敛验证使用真实路由，并按该路由的负载重新评估显存需求。

## 2. 环境与资产准备

- 使用 Linux 昇腾 PyTorch 和 torch-npu 环境，准备 CANN/HCCL 和 NPU grouped matmul 支持。
  各主机使用相同的 HyperParallel 版本、运行时与算子。[安装指南](../../../docs/installation.zh.md)
- 已有记录使用 Transformers 5.13.0，提供原生 `kimi_k25` 和 `deepseek_v3` 支持。
  原生图像处理器需要与 PyTorch 配套的 torchvision。生成演示图片需要 Pillow，
  读取模型仓库的 `tiktoken.model` 词表需要 `tiktoken`。

从 [moonshotai/Kimi-K2.6](https://huggingface.co/moonshotai/Kimi-K2.6) 准备本地资产。
性能配方使用模型配置、处理器配置、分词器文件和对话模板，权重随机初始化。
在各主机或共享存储上准备以下目录。

```bash
export MODEL_DIR=/path/to/Kimi-K2.6
hf download moonshotai/Kimi-K2.6 \
  config.json preprocessor_config.json tokenizer_config.json tiktoken.model chat_template.jinja \
  --local-dir "${MODEL_DIR}"
```

启动器将 `model.config_path`、`model.pretrained_model_name_or_path` 和
`dataset.model_assets.pretrained_model_name_or_path` 指向此目录，并为模型和处理器
启用本地加载。两处 `trust_remote_code` 均保持 `false`，使用 Transformers 原生实现。

[数据生成脚本](recipes/prepare_kimi_vlm_data.py)会输出 JSON 数组和合成 PNG 图片。
启动前先生成数据并移除重复的顶层 `images` 字段。当前加载器将该字段用于带
`<image>` 占位符的记录，与内嵌图片路径同时使用会报错。

```bash
export DATA_DIR="$PWD/output/kimi_k26/kimi_k26_demo_data"
python -m hyper_parallel.models.kimi_k26.recipes.prepare_kimi_vlm_data \
  --output-dir "${DATA_DIR}" --num-samples 1024
python - "${DATA_DIR}/data.json" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
records = json.loads(path.read_text(encoding="utf-8"))
for record in records:
    record.pop("images", None)
path.write_text(json.dumps(records, ensure_ascii=False) + "\n", encoding="utf-8")
PY
```

真实数据使用相同的图文对话格式，可以保存为 JSON 数组或 JSONL。相对媒体路径以数据文件
所在目录为基准，每条记录以 assistant 回复结尾。数据变换将单张图片面积限制在 262,144
像素内，并将文本与图像 token 截断或填充到 8192 token。

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "url": "images/sample_0000.png"},
        {"type": "text", "text": "Describe the following image."}
      ]
    },
    {
      "role": "assistant",
      "content": [{"type": "text", "text": "A colorful geometric pattern."}]
    }
  ]
}
```

## 3. 快速启动

每台主机都在仓库根目录执行命令。将 `NODE_RANK` 分别设为 0 到 63，将
`MASTER_ADDR` 改成 rank 0 主机的可达 IP。所有主机使用同一会合端口、模型资产和
已准备好的数据。使用共享存储时，准备一份演示数据，在各主机上设置同一 `DATA_DIR`。

```bash
export NODE_RANK=0
export MASTER_ADDR=192.0.2.10
NNODES=64 NPROC_PER_NODE=8 MASTER_PORT=6100 RUN_NAME=quick_start \
  bash hyper_parallel/models/kimi_k26/recipes/run_kimi_k26_4sn.sh \
  "${MODEL_DIR}" --training.train_iters=1
```

成功时完成一个优化器步，各主机均以状态码 0 退出。各主机默认将日志写入
`output/kimi_k26/quick_start_rank<node_rank>.log`。本次运行关闭 checkpoint 保存。
固定路由运行用于检查完成状态和耗时。比较 loss 和专家负载时使用真实路由。

## 4. 完整运行与常用配置

用同一启动器接入真实图文数据，并根据数据集设置训练步数。开始正确性或收敛验证前，
在 YAML 的 `plan_overrides` 中找到 `kimi_k26_ep_compute_fn`，将 `fix_router` 改为 `false`。

| 配置项 | 配方默认值与调整方式 |
| --- | --- |
| 数据与步数 | 通过 `--dataset.data_path=/path/to/train.jsonl` 指定准备好的数据。需要在构建数据集时过滤不可训练样本，可以设置 `dataset.data_config.filter_trainable=true`。用 `training.train_iters` 指定目标步数，性能配方默认为 6 步。 |
| 序列与图片 | 媒体 token 挤占回答空间时，联动调整 `dataset.data_transform.max_seq_len`（8192）和 `dataset.data_transform.image_max_pixels`（262144）。配方使用定长填充。 |
| Batch 与分片 | `training.global_batch_size=512`、`training.micro_batch_size=1` 在 DP512 下每个优化器步处理一个 microbatch。YAML 使用 `fsdp_config.dp_shard_size=128` 和 `fsdp_config.edp_shard_size=4`。 |
| 精度与激活内存 | 使用 BF16 模型参数和全重算，关闭激活交换。通过 `model.freeze_patterns` 冻结视觉塔。 |
| 预训练权重初始化 | 设置 `model.load_pretrained=true`，准备与发布版模型尺寸一致的反量化 BF16 权重。构建器严格检查权重键名和形状。上游发布的是 INT4 压缩权重，使用此路径前须转换。 |
| Checkpoint 保存与恢复 | 设置 `checkpoint.save_ckpt=true`、共享的 `checkpoint.checkpoint_dir` 和 `checkpoint.save_steps` 以保存分布式 checkpoint。续训时沿用模型、拓扑和数据，将 `checkpoint.restore_from` 设为 `LATEST` 或具体路径。Trainer 可恢复已保存的优化器、调度器、训练进度、随机数状态和数据游标。 |

## 5. 验证结果

| 验证项 | 配置与结果 |
| --- | --- |
| 本地数据检查 | Transformers 5.13.0 的原生处理器可离线加载上述资产。按前文步骤处理演示数据后，配方使用的 Omni 数据集成功编码了一条记录，并填充到 8192 token。 |
| 短跑 | 上述单步命令在 512 设备拓扑上暂无公开实测结果。 |
| 数值验证 | [Chunk Loss 测试](../../../tests/ut/auto_models/test_kimi_k26_chunk_loss.py)对比完整 logits 的 loss 和梯度，[CP 输入测试](../../../tests/ut/auto_models/distributed/test_kimi_k26_cp_input.py)覆盖输入切分与掩码。完整模型在真实路由下的数值对齐仍待验证。 |
| 收敛与续训 | 此配方的长程收敛与断点续训仍待验证。 |
| 历史性能 | [配方注释](recipes/train.yaml)记录了最短步时 18.2617 s、含 padding 吞吐 228,873 token/s、峰值已分配显存 39.19 GB。测量使用 512 个设备、序列长度 8192、全局批量 512、微批量 1、冻结视觉塔、合成数据和固定路由，关闭激活交换。预热规则及测量步范围暂无记录。 |

## 6. 限制与故障排查

- 使用配方中的 Chunk Loss 时保持 TP1，并保持 `activation_checkpoint.swap_inputs: false`。
  此损失路径下，输入交换曾在反向重算时失败。
- 缩小规模后遇到显存不足时，检查专家分片、序列长度、图片和批量设置。此前在
  2 超节点、EP128、序列长度 8192、全局批量 4096 的运行中，反向阶段出现显存不足。
  启动器默认要求 512 个设备。
- 处理器加载失败时，检查本地处理器和分词器文件，并确认当前 Python 环境提供原生
  `kimi_k25` 实现。严格加载权重失败时，检查 BF16 转换、层数、专家数和权重分片完整性。

## 延伸阅读

- [Kimi-K2.6 模型族适配](integration.zh.md)
- [Trainer YAML 配置](../../../docs/guide/trainer/yaml_config.zh.md)
- [分布式 Checkpoint](../../../docs/guide/distributed_checkpoint.zh.md)
