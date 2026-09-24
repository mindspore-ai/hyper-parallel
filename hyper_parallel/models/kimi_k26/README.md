# Kimi-K2.6 模型族适配

Kimi-K2.6 是 Moonshot 的多模态（图文）MoE 模型，其文本塔是 DeepSeek-V3 风格的 MLA + MoE 结构，
视觉塔与 projector 独立。本目录是 HyperParallel 对该族的适配层：模块替换工厂、CP/分片规则、
损失适配、数据变换入口与训练 recipe 都收敛在这里。

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
│   ├── validation/
│   │   └── cropped_model.py         # 模型构建入口 build_cropped_kimi_vlm（recipe 的 model._target_）
│   └── registration.py              # ModelAdapterSpec：sharding_rules + loss
└── recipes/
    ├── train.yaml                   # 4 超节点 / 512 卡主线配方（实测最优形状）
    ├── run_kimi_k26_4sn.sh          # 多机启动脚本（默认 64 节点 × 8 卡）
    └── prepare_kimi_vlm_data.py     # 生成演示图文数据（离线自测用）
```

数据侧变换在 `hyper_parallel/data/omni/kimi_transform.py`（`KimiVLMChatTransform` /
`KimiOmniTransform` / `KimiPackingOmniTransform`），经 `hyper_parallel.data.omni` 导出。

## 适配内容

- **分片规则**（`adapter/registration.py`）：嵌套文本塔是 MLA，但低秩下投影
  （`q_a_proj` / `kv_a_proj_with_mqa`）保持复制（LoRA rank 维不切分），上投影
  （`q_b_proj` / `kv_b_proj`）沿 head 维列切；视觉塔与 `mm_projector` 全复制——其原生注意力
  在完整 hidden 维上重排 head，TP 切分不可组合。因此本族不能套用 DeepSeek-V3 的 MLA 命名规则。
- **Chunk Loss**（`adapter/chunk_loss.py`）：以原子前向替换的方式把分块交叉熵绑定到
  `model.forward`，避免全序列 logits 驻留；只在 `chunk_loss_targets` 在场时生效。
  目前要求 `tp_size=1`：词表分片（TP）下的 chunk loss 由核心 `ChunkedCausalLMLoss`
  在 `bind_model` 时直接拒绝；配方对应地只使用预移位目标（Omni batch 的
  `labels_are_shifted` 默认为 `True`，其发布的 `shift_labels` 即 chunk loss 所需目标）。
- **CP 输入切分**（`adapter/distributed/context_parallel.py`）：dataloader 按 `dp_rank` 切样本，
  同一 DP 组内 CP 各 rank 拿到同一样本并各自跑视觉塔；文本塔前把 `inputs_embeds` 切成
  `[cp_rank*L, (cp_rank+1)*L)` 窗口，RoPE `position_ids` 用全局窗口重建，并下发 4D 偏移
  causal+padding mask。`cp_size <= 1` 时是 no-op，可无条件调用。
- **模型构建**（`adapter/validation/cropped_model.py`）：校验 `config.model_type` 属于
  `KIMI_MODEL_TYPES`，截断文本塔层数 / 专家数后经 HyperParallel 模型构建入口并行化，并支持
  `freeze_patterns` 在 plan/FSDP 推导前冻结模块（视觉塔冻结可一并去掉其反向与激活）。
- **数据变换**（`hyper_parallel/data/omni/kimi_transform.py`）：走原生 processor 的
  `image_grid_thw` 协议，支持 eager（pad 到 `max_seq_len`）与 token-budget packing 两种形态。

## recipe 使用

`recipes/train.yaml` 是**主线规模：4 超节点 / 512 卡**的配方，形状
`cp 1 × dp_shard 512（= edp 4 × ep 128）`、`GBS 512 = 512 × MBS 1`、序列长度 8192；
它是本族在 4SN 上实测的最优配置：最小步时 **18.2617 s**，padded 吞吐 **228,873 tok/s**，
峰值 allocated **39.19 GB**（无 activation swap——`edp=4` 已把专家参数摊薄到每卡 1/4）。

```bash
# 每个节点执行一次，NODE_RANK 0..NNODES-1
NNODES=64 NODE_RANK=0 MASTER_ADDR=<rank0-ip> MASTER_PORT=6100 \
    bash hyper_parallel/models/kimi_k26/recipes/run_kimi_k26_4sn.sh /path/to/Kimi-K2.6
```

前提与说明：

- 需要 Kimi-K2.6 多模态 checkpoint（config + processor + 权重同目录）。本地/离线场景由启动脚本
  通过 dotted override 把 `model.*`、`dataset.model_assets.*` 指到该目录并置
  `local_files_only=true`；配方默认值则指向公开的 `moonshotai/Kimi-K2.6`。
- `trust_remote_code` 在 model 与 `dataset.model_assets` 两处都保持 `false`，走 transformers
  原生 `kimi_k25` 实现；VLM trainer 会把 `model.trust_remote_code` 转发给 `AutoProcessor`。
- 演示数据由 `recipes/prepare_kimi_vlm_data.py` 生成（默认 8 条，脚本内用 1024 条）；
  真实数据（COCO 等）用 `--dataset.data_path` 覆盖。
- 其它规模（1SN/2SN/3SN）需要同步调整 `accelerator.ep_size` 与 `fsdp_config` 的
  `dp_shard_size` / `edp_shard_size`，启动脚本对非 512 卡的 world size 默认拒绝启动。

## 测试

| 测试 | 覆盖内容 |
|---|---|
| `tests/ut/auto_models/test_kimi_k26_chunk_loss.py` | Chunk Loss 绑定、幂等性、守卫与数值一致性 |
| `tests/ut/auto_models/distributed/test_kimi_k26_cp_input.py` | CP 输入切分契约（窗口、position_ids、mask） |
| `tests/ut/data/vlm/test_dynamic_padding.py` | 动态 padding / 打包路径 |
| `tests/ut/data/vlm/test_image_max_pixels.py` | 图像像素预算与 media token 数 |
| `tests/ut/data/vlm/test_truncate_mode.py` | 截断策略与序列长度推断 |
| `tests/ut/data/vlm/test_get_batch_cp.py` | CP batch 切分 |
