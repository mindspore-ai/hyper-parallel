# Kimi-K2.6 模型族适配

[English](integration.md) | **简体中文**

[返回训练指南](README.zh.md)

## 注册身份

| 项目 | 值 |
|---|---|
| HyperParallel 族名（`models/<family>/` 目录名、注册身份） | `kimi_k26` |
| HF `config.model_type`（K2.6 checkpoint 实际值） | `kimi_k25` |
| HF `architectures` | `KimiK25ForConditionalGeneration` |
| 实现来源 | transformers 原生实现（`transformers.models.kimi_k25`），`trust_remote_code: false` |

K2.6 checkpoint 沿用 K2.5 的原生实现与配置命名，因此目录名与注册身份取 `kimi_k26` 以对应用户
可见的产品版本，而 HF 侧的 `kimi_k25` 拼写由 `hyper_parallel/models/registry.py` 的
`_FAMILY_DIR_ALIASES`（`kimik25 → kimi_k26`）映射到本族，`get_model_adapter()` 对
`kimi_k26` / `kimi_k25` / `KimiK25ForConditionalGeneration` 三种拼写都能解析。

## 目录结构

```text
models/kimi_k26/
├── __init__.py                      # get_adapter_spec() 入口
├── adapter/
│   ├── __init__.py                  # KIMI_MODEL_TYPES：运行时守卫接受的 model_type 拼写
│   ├── chunk_loss.py                # 训练用 Chunk Loss 前向替换（绑定到 model.forward）
│   ├── distributed/
│   │   └── context_parallel.py      # CP 输入切分：inputs_embeds 切窗 + 全局 position_ids + 偏移 mask
│   ├── data/
│   │   └── transform_fn.py          # Omni 数据变换（原生 image_grid_thw 协议：eager + packing）
│   ├── validation/
│   │   └── cropped_model.py         # 模型构建入口 build_cropped_kimi_vlm（recipe 的 model._target_）
│   └── registration.py              # ModelAdapterSpec：sharding_rules + loss
└── recipes/
    ├── train.yaml                   # 4 超节点 / 512 卡主线配方（实测最优形状）
    ├── run_kimi_k26_4sn.sh          # 多机启动脚本（默认 64 节点 × 8 卡）
    └── prepare_kimi_vlm_data.py     # 生成演示图文数据（离线自测用）
```

数据侧变换也在本族目录内：`adapter/data/transform_fn.py`（`KimiVLMChatTransform` /
`KimiOmniTransform` / `KimiPackingOmniTransform` + 两个 builder），与
`models/deepseek_v41/adapter/data/` 的放置方式一致，由 recipe 的
`dataset.data_transform._target_` 直接引用（`ModelAdapterSpec` 不注册数据适配，无需改注册）。

## 适配内容

- **分片规则**（`adapter/registration.py`）：嵌套文本塔是 MLA，但低秩下投影
  （`q_a_proj` / `kv_a_proj_with_mqa`）保持复制（LoRA rank 维不切分），上投影
  （`q_b_proj` / `kv_b_proj`）沿 head 维列切；视觉塔与 `mm_projector` 全复制——其原生注意力
  在完整 hidden 维上重排 head，TP 切分不可组合。因此本族不能套用 DeepSeek-V3 的 MLA 命名规则。
- **Chunk Loss**（`adapter/chunk_loss.py`）：以原子前向替换的方式把分块交叉熵绑定到
  `model.forward`，避免全序列 logits 驻留；只在 `chunk_loss_targets` 在场时生效。
  目前要求 `tp_size=1`：词表分片（TP）下的 chunk loss 由核心 `ChunkedCausalLMLoss`
  在 `bind_model` 时直接拒绝。配方通过 [KimiChunkedCausalLMLoss](adapter/loss.py) 从
  Omni 的原始标签推导下一 token 的目标，也可直接使用已有的 `shift_labels`。
- **CP 输入切分**（`adapter/distributed/context_parallel.py`）：dataloader 按 `dp_rank` 切样本，
  同一 DP 组内 CP 各 rank 拿到同一样本并各自跑视觉塔；文本塔前把 `inputs_embeds` 切成
  `[cp_rank*L, (cp_rank+1)*L)` 窗口，RoPE `position_ids` 用全局窗口重建，并下发 4D 偏移
  causal+padding mask。`cp_size <= 1` 时是 no-op，可无条件调用。
- **模型构建**（`adapter/validation/cropped_model.py`）：校验 `config.model_type` 属于
  `KIMI_MODEL_TYPES`，截断文本塔层数 / 专家数后经 HyperParallel 模型构建入口并行化，并支持
  `freeze_patterns` 在 plan/FSDP 推导前冻结模块（视觉塔冻结可一并去掉其反向与激活）。
- **EP 计算工厂**（`adapter/distributed/ep_compute.py`）：家族自组合的
  `deepseekv3_sigmoid_group_shared` 原型（与 `models/qwen3_moe/adapter/distributed/` 同放置），
  经框架公开的 `build_ep_compute` 组合，因此框架层不需要任何改动。`fix_router: true` 切到
  框架的**均衡路由**（`routing.make_balanced_router`，由骨架统一解析）：真实路由的 token
  倾斜实测会把最忙的 rank 堆到均值的 6.3 倍（也是显存被打爆的主因），均衡路由让每个
  destination rank 拿到完全相同的 slot 数，使同口径对比能在 2–3 步内收敛。
  ⚠️ 该模式是 **benchmark-only**：忽略 gate 分数，`loss`/`grad_norm`/专家命中分布无意义，
  正确性与收敛性验证必须置 `false`（此时回落框架的 sigmoid-group 真实路由）。
- **数据变换**（`adapter/data/transform_fn.py`）：走原生 processor 的
  `image_grid_thw` 协议，支持 eager（pad 到 `max_seq_len`）与 token-budget packing 两种形态。
