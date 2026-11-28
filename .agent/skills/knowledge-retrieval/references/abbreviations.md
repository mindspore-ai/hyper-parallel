# Abbreviations and Search Vocabulary

Search both abbreviations and full terms, and map user terminology to the
identifiers used in code. The dedicated Chinese column below is a
multilingual search lexicon; the explanatory text remains English.

## Parallel and distributed execution

| Abbreviation | Full term | Chinese search term | Common identifiers |
|---|---|---|---|
| TP | Tensor parallelism | 张量并行 | `tensor_parallel`, `tp_size` |
| DP | Data parallelism | 数据并行 | `data_parallel`, `dp` |
| PP | Pipeline parallelism | 流水线并行 | `pipeline_parallel`, `pp_size` |
| EP | Expert parallelism | 专家并行 | `expert_parallel`, `ep` |
| CP | Context parallelism | 上下文并行 | `context_parallel`, `cp_size` |
| FSDP / HSDP | Fully sharded / hybrid sharded data parallelism | 全分片 / 混合分片数据并行 | `fully_shard`, `hsdp` |
| DTensor | Distributed tensor | 分布式张量 | `dtensor`, `placement`, `Shard`, `Replicate` |

## Activations and device memory

| Abbreviation | Full term | Chinese search term | Common identifiers |
|---|---|---|---|
| AC | Activation checkpointing | 激活重计算 | `activation_checkpoint`, `recompute` |
| SAC | Selective activation checkpointing | 选择性激活重计算 | `selective`, `make_selective_checkpoint_context_fn` |
| Swap | Activation offload/swap | 激活换出 | `activation_swap`, `offload` |
| OOM | Out of memory | 显存不足 | `OutOfMemoryError` |

## Operators and testing

| Abbreviation | Full term | Chinese search term | Common identifiers |
|---|---|---|---|
| FA | FlashAttention | 融合注意力 | `flash_attention`, `fused_attention` |
| MoE | Mixture of experts | 混合专家 | `moe`, `router`, `experts` |
| CSA | Compressed sparse attention | 压缩稀疏注意力 | `compressed_*`, `shared_compressed_dsa_attention` |
| MTP | Multi-token prediction | 多词元预测 | `mtp`, drafter-related code |
| UT / ST | Unit / system tests | 单元测试 / 系统测试 | `tests/ut`, `tests/.../st` |

## Additional query mappings

| Chinese search term | Code/search vocabulary |
|---|---|
| 重算 / 重计算 | `recompute`, `checkpoint` |
| 显存 | Device `memory`, distinct from host RAM |
| 精度 | `precision`, `dtype`, `accuracy` |
| 并行度 | `*_size`, such as `tp_size`, and `parallel` |
| 切分 / 分片 | `shard`, `split`, `Shard` |
| 通信域 | `group`, `process_group` |
| 开关 / 配置项 | Configuration fields; not automatically environment variables |
| 门禁 | Gate / CI; see `.agent/skills/gate-doctor` |
| 首错 | First causal error in distributed diagnosis |

## Usage

- Search natural-language terms in documentation and identifiers in
  `hyper_parallel/**`.
- Search both the abbreviation and expansion, such as `SAC` and
  `selective activation checkpointing`.
- Try snake_case, CamelCase, and hyphenated variants.
- Register new project abbreviations here as they are introduced.
