# Wan2.1 训练示例

本示例使用 HyperParallel 训练栈执行 Wan2.1 T2V 全参数微调训练。

训练入口和配置文件如下：

- `scripts/train_dit.py`
- `examples/wan/wan2_1_t2v_14b_full.yaml`

配置文件保持旧版内容不变。新版配置解析器会将 YAML 中的旧
`hyper_parallel.auto_models...` 目标路径映射到新版 `models`、`data`、
`components` 和 `trainer` 包，因此不需要改写配置即可启动。

## 支持范围

当前 Wan 适配支持 DP/FSDP2 全参数训练：

- TP、CP、PP、EP、sequence parallel 和 loss parallel 必须保持关闭。
- 默认 `global_batch_size=8`、`micro_batch_size=1`、`dp_shard_size=8`。
- 默认启用 `activation_checkpoint.mode=full`。

## 准备 Wan2.1 模型

准备完整的 Diffusers 格式 Wan2.1 T2V 模型。可以直接从 Hugging Face 下载：

```bash
huggingface-cli download Wan-AI/Wan2.1-T2V-14B-Diffusers \
  --local-dir /data/models/Wan2.1-T2V-14B-Diffusers
```

模型根目录应包含以下文件和子目录：

```text
Wan2.1-T2V-14B-Diffusers/
├── model_index.json
├── transformer/
│   ├── config.json
│   └── diffusion_pytorch_model*.safetensors
├── tokenizer/
├── text_encoder/
├── vae/
└── scheduler/
```

训练时传入模型根目录，不要传 `transformer/` 子目录。模型构建器会从
`transformer/` 加载可训练的 DiT 权重，并从根目录加载冻结的 tokenizer、text
encoder、VAE 和 scheduler。

`condition_model_name_or_path=auto` 表示条件模型路径自动使用
`transformer/` 的父目录。

## 准备视频数据

### Tom-and-Jerry 数据集

转换脚本要求原始数据具有如下结构：

```text
Tom-and-Jerry-VideoGeneration-Dataset/
├── captions.txt
├── videos.txt
└── videos/
    ├── sample-000.mp4
    └── ...
```

`captions.txt` 和 `videos.txt` 的非空行数必须相同。`videos.txt` 中的路径会
按原始数据集根目录解析。

将原始数据转换为 Parquet：

```bash
python examples/wan/convert_tom_and_jerry.py \
  --dataset_path /data/datasets/Tom-and-Jerry-VideoGeneration-Dataset \
  --output_dir /data/datasets/Tom-and-Jerry-VideoGeneration-Dataset-parquet \
  --num_shards 30 \
  --num_proc 32
```

转换结果包含 `prompt`、`video_bytes` 和数据源标记字段。默认 YAML 使用：

```text
./Tom-and-Jerry-VideoGeneration-Dataset-parquet
```

数据放在其他位置时，通过 `--dataset.data_path` 覆盖该路径。

## 拉起训练

在仓库根目录执行。启动命令如下：

```bash
torchrun --nproc_per_node=8 scripts/train_dit.py \
  examples/wan/wan2_1_t2v_14b_full.yaml \
  --training.backend=hccl \
  --fsdp_config.dp_shard_size=8
```

使用本地模型和本地 Parquet 数据时：

```bash
torchrun --nproc_per_node=8 scripts/train_dit.py \
  examples/wan/wan2_1_t2v_14b_full.yaml \
  --model.pretrained_model_name_or_path=/data/models/Wan2.1-T2V-14B-Diffusers \
  --model.local_files_only=true \
  --dataset.data_path=/data/datasets/Tom-and-Jerry-VideoGeneration-Dataset-parquet \
  --dataset.data_config.hf_dataset_name=null \
  --training.backend=hccl \
  --fsdp_config.dp_shard_size=8
```

## Checkpoint 与输出

默认每 250 个 optimizer step 保存一次 checkpoint，目录为：

```text
./outputs/wan2_1_t2v_1_3b_full/checkpoints
```

恢复训练时传入 checkpoint 目录：

```bash
torchrun --nproc_per_node=8 scripts/train_dit.py \
  examples/wan/wan2_1_t2v_14b_full.yaml \
  --checkpoint.restore_from=/data/output/wan2_1/checkpoints/<step> \
  --training.backend=hccl \
  --fsdp_config.dp_shard_size=8
```
