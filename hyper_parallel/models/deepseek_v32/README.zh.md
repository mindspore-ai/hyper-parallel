# DeepSeek-V3.2 模型族适配

[English](README.md) | **简体中文**

DeepSeek-V3.2 在 DeepSeek-V3 的 MLA + MoE 结构上加入 DeepSeek Sparse Attention（DSA）：每层由
lightning indexer 为每个 query 选出 `index_topk` 个 key（官方 config 为 2048），MLA 只在这些 key 上
计算 attention。HyperParallel 训练的是 Transformers 原生的 `DeepseekV32ForCausalLM`，本目录提供它的
模块替换、并行规则和训练 recipe。

## 注册身份

| 项目 | 值 |
|---|---|
| HyperParallel 族名（`models/<family>/` 目录名、注册身份） | `deepseek_v32` |
| HF `config.model_type` | `deepseek_v32` |
| HF `architectures` | `DeepseekV32ForCausalLM` |
| 实现来源 | Transformers 原生实现（`transformers.models.deepseek_v32`），`trust_remote_code: false` |

## 目录结构

```text
models/deepseek_v32/
├── __init__.py                      # 模型包入口，不构造模型
├── deepseek_v32_spec.py             # 公共模块族、模型身份及有限旧 YAML 名称声明
├── generated/                      # 可选 codegen 产物，不拥有公共算法
└── recipes/                        # 训练、数据与拓扑配置；模块族规则由 spec 声明
    ├── run_parallel_offline.sh      # 八卡启动脚本
    ├── train.yaml                   # 发布模型训练配置
    └── train_parallel_offline.yaml  # 离线八卡裁剪配置
```

## 功能

- **NPU 模块族**，由 [deepseek_v32_spec.py](deepseek_v32_spec.py) 选择：`*.input_layernorm`、`*.post_attention_layernorm`、
  `model.norm` 换成融合 `RMSNorm`；`*.self_attn` 换成 `DeepseekSparseAttention`，indexer 与稀疏 attention 走
  CANN 融合算子；`*.mlp.experts` 选择[路由专家模块族](../../components/modules/moe/grouped_experts/README.md)。
- **并行**：FSDP、TP（含 loss parallel）、CP、EP。不支持 PP 和 sequence parallel。
- **训练行为**：
  - router 用 `e_score_correction_bias` 选专家（sigmoid 分数、分组 top-k），bias 保持初始值；
  - 没有负载均衡 loss；
  - 官方 checkpoint 的 MTP 层（第 61 层）加载时被忽略，不训练；
  - indexer 支持 KL 辅助损失训练，默认关闭。
- **数据**：支持因果序列和带显式边界的 packed 文档；不支持任意 padding mask、dropout 和 KV cache。

## 环境要求

- Transformers 需要包含 `deepseek_v32`：`requirements.txt` 要求 `transformers>=5.11.0`，既有原生模型验证记录为
  5.13.1；本轮新专家族核对了 5.13.1 源码接口，尚未重跑硬件数值和多卡验收。
- 昇腾 NPU，DSA 算子依赖 CANN 与 `cann_ops_transformer`，见
  [functional README](../../components/functional/README.md#deepseek-v32-dsa)。
- 权重：`from_pretrained` 可以直接加载官方 FP8 checkpoint（读取时反量化），也可以加载 BF16 checkpoint。第 61 层
  MTP 的权重会被忽略并打印 warning。

## 使用

### 用 recipe 训练

```bash
torchrun --standalone --nproc_per_node=1 \
  scripts/train_lm.py hyper_parallel/models/deepseek_v32/recipes/train.yaml \
  --model.pretrained_model_name_or_path=/path/to/DeepSeek-V3.2 \
  --dataset.model_assets.tokenizer.pretrained_model_name_or_path=/path/to/DeepSeek-V3.2
```

末尾追加的参数会覆盖 YAML 字段，例如 `--training.train_iters=200`。

### 多卡

在 recipe 或命令行中配置并行拓扑。spec 为整个 attention 边界选择 `DeepseekSparseAttention`，
并从 `accelerator.cp_size` 自动推导 `kv_cp_size`；模块族在 `cp_size > 1` 时选择 CP 成员。
spec 使用 `allgather` 并开启通信重叠，参数含义见
[稀疏注意力说明](../../components/modules/attention/deepseek_sparse_attention/readme.md)。
`GroupedExperts` 同样根据作业 mesh 选择 local/EP 执行，源 MoE 保留路由和共享专家编排。
模块匹配、构建和权重转换由 spec 与模块族负责，recipe 无需提供 `plan_overrides`。

```yaml
accelerator:
  tp_size: 2
  cp_size: 2
  ep_size: 4
  loss_parallel: true
fsdp_config:
  dp_shard_size: 2
  edp_shard_size: 2
```

例如 8 卡、TP2 × CP2 × DP2、EP4：

```bash
torchrun --standalone --nproc_per_node=8 \
  scripts/train_lm.py <你的 recipe> \
  --accelerator.tp_size=2 --accelerator.cp_size=2 --accelerator.ep_size=4 \
  --accelerator.loss_parallel=true --training.global_batch_size=2 \
  --fsdp_config.dp_shard_size=2 --fsdp_config.edp_shard_size=2
```

专家矩阵保持完整（ETP=1），EP 仅切专家维。需要满足：`world_size = tp × cp × dp`；
`n_routed_experts` 能被 `ep_size` 整除；`global_batch_size` 能被 `micro_batch_size × dp` 整除。

### 8 层裁剪模型的 8 卡 recipe

`recipes/train_parallel_offline.yaml` 在 8 卡上训练 DeepSeek-V3.2 的 8 层裁剪模型（3 个 dense 层 + 5 个 MoE 层，16 个路由
专家，8.40B 参数）：TP2 × CP2 × DP2，EP4 × EDP2，loss parallel 加 FSDP，启用全部模块替换。裁剪模型只需要官方目录中的
`config.json` 和 tokenizer 文件，权重为随机初始化的 BF16：

```python
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

source, target = "/path/to/DeepSeek-V3.2", "/path/to/DeepSeek-V3.2-8layer-crop"
released = AutoConfig.from_pretrained(source)
config = AutoConfig.from_pretrained(
    source, num_hidden_layers=8, n_routed_experts=16,
    # 逐层类型从官方列表里截取：类型名在不同 Transformers 版本里不同。
    layer_types=released.layer_types[:8], mlp_layer_types=released.mlp_layer_types[:8],
)
del config.quantization_config
torch.manual_seed(1234)
AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16).save_pretrained(target)
AutoTokenizer.from_pretrained(source).save_pretrained(target)
```

数据为用 DeepSeek-V3.2 tokenizer 处理好的 Indexed Dataset（`.bin`/`.idx`），见
[离线数据准备指南](../../../docs/guide/data/offline_preparation_guide.zh.md)。传入 checkpoint 目录和不带扩展名的数据
前缀，末尾追加的参数会覆盖 YAML 字段：

```bash
bash hyper_parallel/models/deepseek_v32/recipes/run_parallel_offline.sh \
  /path/to/DeepSeek-V3.2-8layer-crop /path/to/dataset_text_document
```

日志、数据缓存和 checkpoint 写到 `output/deepseek_v32`，可用 `OUTPUT_DIR` 修改。
