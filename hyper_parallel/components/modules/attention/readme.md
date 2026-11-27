# Attention 目录结构与 YAML 通用规则

本目录按注意力算法分目录组织：本地实现负责计算，上下文并行实现补充通信，
family 提供统一的公共入口。各算法的使用参数见对应文档。

`linear_attention_utils/` 提供 KDA、GDN 共用的序列、模块适配、并行布局和卷积通信辅助功能，
不注册独立的注意力模块族。具体算法实现仍位于各自的算法目录中。

| 算法 | 说明 |
| --- | --- |
| [Grouped Query Attention](grouped_query_attention/README.md) | 分组查询注意力 |
| [Gated Grouped Query Attention](gated_grouped_query_attention/README.md) | 带输出门控的分组查询注意力 |
| [Multi-head Latent Attention](multi_latent_attention/README.md) | 多头潜变量注意力 |
| [Gated Multi-head Latent Attention](gated_multi_latent_attention/README.md) | 不使用位置编码、带输出门控的多头潜变量注意力 |
| [DeepSeek Sparse Attention](deepseek_sparse_attention/readme.md) | DeepSeek-V3.2 稀疏注意力 |

## 文件结构

每种算法独立成目录，`<name>` 表示算法目录名：

| 文件 | 职责 |
| --- | --- |
| `<name>.py` | 本地注意力计算 |
| `<name>_family.py` | 公共工厂、实现选择和并行布局声明 |
| `cp_<name>.py` | 上下文并行通信与调度 |
| `__init__.py` | 导出公共入口 |
| `README.md` / `readme.md` | 使用方式和配置参数 |

以 Grouped Query Attention 为例：

```text
grouped_query_attention/
├── __init__.py
├── grouped_query_attention.py         # GroupedQueryAttentionModule
├── grouped_query_attention_family.py  # GroupedQueryAttention
├── cp_grouped_query_attention.py      # CPGroupedQueryAttentionModule
└── README.md
```

## 命名规范

- **目录和主文件**：使用算法全拼的 `snake_case`，如 `grouped_query_attention`；门控变体加 `gated_` 前缀。
- **公共工厂**：使用 `PascalCase` 算法名，如 `GroupedQueryAttention`。它是工厂函数，统一从
  `hyper_parallel.components.modules` 使用。
- **本地实现类**：在算法名后加 `Module`，如 `GroupedQueryAttentionModule`。
- **上下文并行实现类**：再加 `CP` 前缀，如 `CPGroupedQueryAttentionModule`；门控变体相应使用 `Gated` / `CPGated` 前缀。
- **内部函数和方法**：使用 `snake_case`，私有辅助函数加 `_` 前缀，如 `_build_local`。

## YAML 配置通用规则

YAML 指向公共工厂：`accelerator.cp_size` 为 1 时选择本地实现，大于 1 时选择上下文并行实现。

- 替换完整的注意力模块，`match` 写到 `self_attn` 一级。
- 算法参数写在 `replace.implementation` 下，与 `_target_` 同级；来源模块和运行时上下文由框架提供。
- 作业拓扑写在 `accelerator` 中，各算法的并行参数按对应文档配置。

下面以 Grouped Query Attention 为例，使用 4 张卡进行上下文并行：

```yaml
accelerator:
  tp_size: 1
  cp_size: 4
  sequence_parallel: false

plan_overrides:
  - match: "model.layers.*.self_attn"
    replace:
      kind: module
      implementation:
        _target_: hyper_parallel.components.modules.GroupedQueryAttention
        ulysses_degree: 2
        overlap: true
```

`ulysses_degree: 2` 将 4 路上下文并行分为 2 路 Ulysses 与 2 路 KV AllGather，
`overlap: true` 开启通信重叠。参数说明见 [Grouped Query Attention 文档](grouped_query_attention/README.md)。
