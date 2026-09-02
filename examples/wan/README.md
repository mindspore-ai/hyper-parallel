# Wan2.1 AutoModels DiT Training

本目录给出 Wan2.1 1.3B T2V/I2V 在 Hyper-Parallel AutoModels 训练栈中的全参微调示例。入口是
`scripts/train_dit.py`，配置分别是 `wan2_1_t2v_1_3b_full.yaml` 和
`wan2_1_i2v_1_3b_full.yaml`。

## 机制概览

AutoModels 的训练由 YAML 中的 `_target_` 组件拼装。`parse_training_args`
读取配置后生成 `TrainerConfig`，`DiTTrainer` 继承共享的 `BaseTrainer`，因此分布式初始化、device
mesh、FSDP2 包装、优化器、学习率调度和 checkpoint 都复用 Hyper-Parallel 的标准训练流程。

Wan2.1 的适配链路如下：

1. `build_wan_transformer_model` 构建可训练的 Wan transformer，并调用
   `apply_model_infrastructure` 接入 AutoModels 的加载、FSDP2、激活重计算和 checkpoint 机制。
2. `WanConditionModel` 加载并冻结 tokenizer、text encoder、VAE、scheduler；I2V 任务额外加载 image
   processor 和 CLIP image encoder。
3. `build_wan_video_dataset` 复用 AutoModels 现有的在线 dataset 加载器，支持本地
   JSON/JSONL/Parquet/CSV/Arrow 数据；配置了 `data_path` 时会优先读取本地数据，并忽略
   `hf_dataset_name`。
4. `WanVideoTransform` 把原始样本转换成 VeOmni Wan 训练流需要的 `inputs`、`videos`、`images`。
5. `DiTCollator` 保留视频和图片的 list/tensor 结构，避免默认 collate 破坏变长视频样本。
6. `DiTTrainer.forward_backward_step` 在线调用 frozen condition model 生成扩散训练条件，再把条件传给
   trainable Wan transformer 计算 MSE loss 并做全参反传。

这里迁移的是 VeOmni Wan2.1 LoRA 流程中的在线数据处理、条件编码和模型前向逻辑，但训练方式改成全参微调：
配置中 `checkpoint.is_peft: false`，Wan builder 会拒绝 LoRA/PEFT 配置，优化器只接收可训练的 Wan
transformer 参数，condition model 始终冻结。

示例配置里的 `model.condition_model_name_or_path: auto` 表示 condition model 会自动跟随
`model.pretrained_model_name_or_path`：如果模型路径是完整 Diffusers 目录就直接使用该目录；如果模型路径是
`.../transformer` 子目录就自动回退到父目录。这样把 1.3B 切到 14B 时只需要覆盖一个模型路径。

## 默认数据集

两个示例配置默认读取本地目录 `./Tom-and-Jerry-VideoGeneration-Dataset-parquet`，用于快速打通训练链路：

```yaml
dataset:
  data_path: ./Tom-and-Jerry-VideoGeneration-Dataset-parquet
  data_config:
    dataset_type: mapping
    hf_dataset_name: null

dataloader:
  dataloader_type: distributed
  sampler_shuffle: true
  sampler_drop_last: false
```

这个目录应由原始 `Wild-Heart/Tom-and-Jerry-VideoGeneration-Dataset` 转换得到。原始数据目录形态是：

```text
Tom-and-Jerry-VideoGeneration-Dataset/
├── captions.txt
├── videos.txt
└── videos/
```

转换命令：

```bash
python examples/wan/convert_tom_and_jerry.py \
  --dataset_path ./Tom-and-Jerry-VideoGeneration-Dataset \
  --output_dir ./Tom-and-Jerry-VideoGeneration-Dataset-parquet
```

转换后的 parquet 字段约定如下：

| 字段 | 用途 |
| --- | --- |
| `prompt` | 文本条件，转换为 batch 中的 `inputs` |
| `video_bytes` | 目标视频，转换为 batch 中的 `videos[0]` |
| `source` | 数据来源标记，值为 `Tom-and-Jerry-VideoGeneration-Dataset` |

T2V 使用文本和目标视频训练。I2V 示例配置开启 `use_first_video_frame_as_image: true`，
会从目标视频第一帧派生参考图，因此可以复用同一个视频数据集跑通流程；正式 I2V 训练建议提供显式参考图字段。

如果只把 `dataset.data_path` 改成本地 parquet 目录，但没有把 `dataset.data_config.hf_dataset_name` 置为
`null`，旧代码会继续走 HuggingFace Hub 分支并报 `DatasetNotFoundError`。当前 Wan dataset builder
已经改为本地 `data_path` 优先；仍建议在配置里显式写 `hf_dataset_name: null`，这样配置意图最清楚。

## 数据格式

如果使用本地数据，把 `dataset.data_path` 指向本地文件或目录，并覆盖
`dataset.data_config.hf_dataset_name=null`。底层 loader 可处理 AutoModels 已支持的本地
JSON/JSONL/Parquet/CSV/Arrow 等格式。

T2V JSONL 示例：

```jsonl
{"prompt": "a cartoon cat runs through a kitchen", "video": "/abs/path/sample.mp4"}
```

I2V JSONL 示例：

```jsonl
{"prompt": "a cartoon cat runs through a kitchen", "image": "/abs/path/first.jpg", "video": "/abs/path/sample.mp4"}
```

字段名兼容范围：

| 类型 | 默认候选字段 |
| --- | --- |
| 文本 prompt | `prompt`, `text`, `caption`, `inputs` |
| 目标视频 | `video`, `video_path`, `video_bytes`, `videos` |
| I2V 参考图 | `image`, `image_path`, `image_bytes`, `images`, `first_frame` |

媒体路径可以是绝对路径；如果样本里写相对路径，请设置 `dataset.data_transform.data_dir=/abs/path/media` 作为解析根目录。

## 数据处理流程

`WanVideoTransform` 在 dataloader worker 中执行在线处理：

1. 从样本中读取 prompt、目标视频，以及 I2V 所需的参考图。
2. 视频优先使用 `torchcodec` 解码；环境没有 `torchcodec` 时回退到 `imageio`。也支持已经解码好的
   tensor、ndarray、PIL frame list 或 frame bytes。
3. 视频按 `fps: 24` 和 `max_frames: 81` 做采样与截断。
4. 示例默认 `frame_factor: null`，与 VeOmni `configs/dit/wan_sft.yaml` 保持一致，不额外强制
   `4n + 1` 帧数对齐；如果你想使用 Wan VAE 的理论帧数约束，可以手动设置
   `frame_factor: 4`、`frame_factor_remainder: 1`。
5. 输出 batch 样本为 `{"inputs": prompt, "videos": [video_tensor], "images": [...]}`。

`WanConditionModel` 在训练 step 内在线生成 diffusion 条件：

| 任务 | 条件处理 |
| --- | --- |
| T2V | UMT5 编码 prompt，Wan VAE 编码目标视频，scheduler 采样 timestep/noise，构造训练输入和 `training_target` |
| I2V | 在 T2V 基础上增加 CLIP image context，并把参考图 latent 与 mask latent 拼到 noisy latent 通道上 |

Wan transformer 的输出与 `training_target` 做逐样本 MSE，`DiTTrainer` 再按 AutoModels 的梯度累计和分布式规约规则完成 optimizer step。

## 对齐 VeOmni 首 Loss

如果要和 VeOmni 的 `wan_sft.yaml` 对齐首 loss，请先确认下面几项都一致：

| 检查项 | Hyper-Parallel 配置 |
| --- | --- |
| 模型权重根目录 | `model.pretrained_model_name_or_path=<Wan Diffusers 根目录>`，`condition_model_name_or_path: auto` |
| attention | `model.attn_implementation: flash_attention_2` |
| DP/FSDP | `tp/cp/pp/ep=1`，命令行覆盖 `--fsdp_config.dp_shard_size=8` |
| FSDP 输入 cast | `fsdp_config.mix_precision.cast_forward_inputs: false`，保护 RoPE fp32 tuple |
| 数据顺序 | `dataloader_type: distributed`，`sampler_shuffle: true`，`sampler_drop_last: false` |
| 采帧 | `fps: 24`、`max_frames: 81`、`frame_factor: null` |
| 随机性 | `training.seed`、`condition_model.cfg_negative_prob`、`enable_full_determinism` 与 VeOmni 一致 |

VeOmni 在 `MODELING_BACKEND=veomni` 时会把配置里的 `flash_attention_2` 解析成
`veomni_flash_attention_2_with_sp`，再通过 Transformers 的 `ALL_ATTENTION_FUNCTIONS` 进入 VeOmni 的
SP-aware wrapper。Hyper-Parallel 的 Wan 迁移路径不依赖 VeOmni 全局 patch，但会保持
`config._attn_implementation=flash_attention_2` 并进入本地 `wan_flash_attention_forward`。在 Ascend 上，
如果运行时 transformers 是带 NPU FA2 分支的版本，这条路径最终应调用 `torch_npu.npu_fusion_attention`
varlen kernel；如果 msprobe 仍只看到 DiT 内部的 `Functional.scaled_dot_product_attention`，优先检查
运行日志中的 `model.config._attn_implementation` 是否被默认成了 `sdpa`，以及训练环境实际 import 的
`transformers.__file__` 是否是预期版本。

注意：`dataset.data_config.shuffle: false` 只关闭数据源侧 shuffle；VeOmni 的 map-style dataloader 仍会在 sampler
层固定 shuffle。Hyper-Parallel 的 `single` 是顺序切分，`cyclic` 使用 Python `random.shuffle`，都不能严格复现
VeOmni/PyTorch sampler 的首批样本。若你在 `get_condition` 前打印 `inputs` 仍不一致，先检查两边的 parquet
文件顺序、数据集长度、`training.seed` 和 `epoch` 是否一致。若 `inputs` 已对齐但 loss 仍不一致，可临时加环境变量
`HP_WAN_DEBUG_FIRST_BATCH=1`，Hyper-Parallel 会打印首个 Wan batch 的 `timestep`、`zero_pred_loss`、
`prediction/target/latents` 统计；如果 `loss` 接近 `zero_pred_loss`，再优先检查 transformer 权重加载和前向语义。

如果首个带权重算子的输入已经一致，但 `patch_embedding.weight/bias`、
`condition_embedder.time_embedder.linear_1.*` 或 `blocks.0.attn1.to_q.weight` 不一致，先直接比较两边实际使用的
transformer checkpoint：

```bash
python examples/wan/inspect_wan_checkpoint.py \
  /abs/path/hyper-used/Wan2.1-T2V-14B-Diffusers \
  /abs/path/veomni-used/Wan2.1-T2V-14B-Diffusers/transformer
```

脚本会自动把 Diffusers 根目录解析到 `transformer/`，并打印这些关键 tensor 的 dtype、shape、统计量和 md5。
如果原始 checkpoint 的 fingerprint 已经不同，先统一两边的本地模型目录；注意
`Wan2.1_T2V_14B_Diffusers` 和 `Wan2.1-T2V-14B-Diffusers` 是两个不同路径。如果原始 checkpoint
完全一致但 Hyper-Parallel forward dump 仍显示全 0 bias 或随机初始化形态，优先检查 deferred pretrained loading
是否在补初始化 non-persistent state 时覆盖了已加载参数。

如果第一处差异出现在 `apply_rotary_emb`，并且 Hyper-Parallel 的 `freqs_cos/freqs_sin` 是 bf16 全 0，而
VeOmni 是 fp32 正常 cos/sin 值，说明 RoPE 这类 checkpoint 不保存的 non-persistent buffer 在
`to_empty()` 或 `model_init_dtype` 转换中被破坏。当前 AutoModels materialize 与 dtype conversion 会保留这类
buffer 的原始 fp32 值；Wan 的可训练权重仍可按 `model_init_dtype: bfloat16` 加载以控制显存。

如果 Hyper-Parallel 的 `freqs_cos/freqs_sin` 已经非 0 但 dtype 仍是 bf16，通常是 FSDP block pre-forward
在 `cast_forward_inputs: true` 时递归 cast 了 `rotary_emb` tuple。Wan 示例配置默认关闭这个开关，保持
RoPE 乘法与 VeOmni 一样走 `bf16 * fp32 -> fp32`，随后 `apply_rotary_emb` 再 cast 回 query/key 的 bf16。

如果你已经把 VeOmni 源码里的 sampler 改成 `shuffle=False`，才应该把 Hyper-Parallel 改回
`dataloader.dataloader_type: single` 来对齐顺序数据读取。

## T2V 训练

单进程 smoke run：

```bash
python scripts/train_dit.py examples/wan/wan2_1_t2v_1_3b_full.yaml
```

CUDA 多卡全参 FSDP2：

```bash
torchrun --nproc_per_node=8 scripts/train_dit.py examples/wan/wan2_1_t2v_1_3b_full.yaml \
  --training.backend=nccl \
  --fsdp_config.dp_shard_size=8
```

Ascend/NPU 多卡全参 FSDP2：

```bash
torchrun --nproc_per_node=8 scripts/train_dit.py examples/wan/wan2_1_t2v_1_3b_full.yaml \
  --training.backend=hccl \
  --fsdp_config.dp_shard_size=8
```

使用本地模型和本地 T2V 数据：

```bash
torchrun --nproc_per_node=8 scripts/train_dit.py examples/wan/wan2_1_t2v_1_3b_full.yaml \
  --model.pretrained_model_name_or_path=/abs/path/Wan2.1-T2V-1.3B-Diffusers \
  --model.local_files_only=true \
  --dataset.data_config.hf_dataset_name=null \
  --dataset.data_path=/abs/path/t2v_train.jsonl \
  --dataset.data_transform.data_dir=/abs/path/media \
  --training.backend=nccl \
  --fsdp_config.dp_shard_size=8
```

## I2V 训练

单进程 smoke run：

```bash
python scripts/train_dit.py examples/wan/wan2_1_i2v_1_3b_full.yaml
```

CUDA 多卡全参 FSDP2：

```bash
torchrun --nproc_per_node=8 scripts/train_dit.py examples/wan/wan2_1_i2v_1_3b_full.yaml \
  --training.backend=nccl \
  --fsdp_config.dp_shard_size=8
```

Ascend/NPU 多卡全参 FSDP2：

```bash
torchrun --nproc_per_node=8 scripts/train_dit.py examples/wan/wan2_1_i2v_1_3b_full.yaml \
  --training.backend=hccl \
  --fsdp_config.dp_shard_size=8
```

使用显式参考图的本地 I2V 数据：

```bash
torchrun --nproc_per_node=8 scripts/train_dit.py examples/wan/wan2_1_i2v_1_3b_full.yaml \
  --model.pretrained_model_name_or_path=/abs/path/Wan2.1-I2V-1.3B-Diffusers \
  --model.local_files_only=true \
  --dataset.data_config.hf_dataset_name=null \
  --dataset.data_path=/abs/path/i2v_train.jsonl \
  --dataset.data_transform.data_dir=/abs/path/media \
  --dataset.data_transform.use_first_video_frame_as_image=false \
  --training.backend=nccl \
  --fsdp_config.dp_shard_size=8
```

## 常用覆盖项

| 目标 | 覆盖项 |
| --- | --- |
| 修改训练步数 | `--training.train_iters=1000` |
| 修改全局 batch | `--training.global_batch_size=64` |
| 修改 micro batch | `--training.micro_batch_size=1` |
| 修改最大帧数 | `--dataset.data_transform.max_frames=49` |
| 修改采样 fps | `--dataset.data_transform.fps=16` |
| 修改 checkpoint 间隔 | `--checkpoint.save_interval=500` |
| 从 checkpoint 恢复 | `--checkpoint.load=/abs/path/checkpoints` |
| 只使用本地模型文件 | `--model.local_files_only=true` |

## 依赖与限制

运行 Wan 训练需要 `torch`、`transformers`、`diffusers`、`torchvision`、`datasets`。视频解码推荐安装
`torchcodec`，否则会尝试使用 `imageio` 回退。

当前实现支持 DP/FSDP2 全参微调。TP/CP/PP/EP、sequence parallel 和 loss parallel 暂时会被显式拒绝，等
Wan transformer 的张量切分、序列切分和条件模型并行策略补齐后再开启。

如果用 14B T2V 路径和 VeOmni 对齐，请确认 VeOmni 侧实际生效的 `ulysses_size` 也是 1；如果 VeOmni
开启 Ulysses SP，而 Hyper-Parallel 仍是纯 DP/FSDP2，两边的每步数据并行语义、attention kernel 和显存峰值都不会严格一致。
