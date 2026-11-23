# Qwen3-Omni 模型族适配

[English](README.md) | **简体中文**

Qwen3-Omni 是阿里的任意到任意多模态 MoE 模型（文本 / 图像 / 音频 / 视频输入，带可选的语音生成
talker）。其 thinker 内嵌视觉塔、音频塔与稀疏 MoE 解码器；训练时关闭音频输出
（`enable_audio_output=false`），仅 thinker 参与训练。本目录是 HyperParallel 对该族的适配层：
stock modeling 修复、派生缓冲物化、视觉 FSDP2 单元声明、音视频数据变换与训练 recipe 都收敛在这里。

## 注册身份

| 项目 | 值 |
|---|---|
| HyperParallel 族名（`models/<family>/` 目录名） | `qwen3_omni` |
| HF `config.model_type`（checkpoint 实际值） | `qwen3_omni_moe` |
| HF `architectures` | `Qwen3OmniMoeForConditionalGeneration` |
| 实现来源 | transformers 原生实现（`transformers.models.qwen3_omni_moe`），`trust_remote_code: false` |

HF checkpoint 的 `model_type` 带 MoE 后缀而族目录不带；`hyper_parallel/models/registry.py` 通过
`_FAMILY_DIR_ALIASES`（`qwen3omnimoe → qwen3_omni`）完成映射，`get_model_adapter()` 对
`qwen3_omni` / `qwen3_omni_moe` / `Qwen3OmniMoeForConditionalGeneration` 三种拼写都能解析。

## 目录结构

```text
models/qwen3_omni/
├── __init__.py                      # get_adapter_spec() 入口
├── adapter/
│   ├── __init__.py                  # QWEN3_OMNI_MODEL_TYPES：运行时守卫接受的 model_type 拼写
│   ├── model_patches.py             # stock Transformers 修复（训练前向、packed audio、patch-embed
│   │                                #   Linear 路由、deepstack 2-D mask、可选 NPU 融合 MoE）
│   ├── materialization.py           # 派生位置缓冲（rotary inv_freq、音频 sinusoids）的重建
│   │                                #   recipe——stock CPU fp32 值，切分前声明
│   ├── omni_randinit.py             # 随机初始化构建（HyperAutoModelForMultimodalLM.from_config）
│   ├── omni_moe_fused_npu.py        # 可选 NPU 融合 MoE 内核（env HYPER_OMNI_MOE_FUSED_NPU=1）
│   ├── omni_moe_packed_experts.py   # 可选 NPU packed-expert MoE 路径
│   ├── data/
│   │   └── transform_fn.py          # Qwen3OmniTransform：音频/视频/图像 SFT 记录变换
│   └── registration.py              # ModelAdapterSpec：materialization + stock_model_patches + fsdp_wrap_modules
└── recipes/
    └── train.yaml                   # 单机 8 卡 SFT recipe
```

## 适配内容

- **stock 模型补丁**（`adapter/model_patches.py`，经 `stock_model_patches` provider 声明）：安装的
  Transformers 存在破坏 Qwen3-Omni 训练的缺陷——只有生成用包装而无训练 `forward`、packed audio
  死分支、NPU 反向段错误的 patch-embed Conv3d（路由为等价 Linear——前向逐位一致、权重存储
  梯度相同）以及 deepstack 3-D mask 联合分支崩溃。每个补丁都守卫在缺陷的 stock 形态上，已修复的
  modeling 不受影响，且整套补丁幂等。
- **派生缓冲物化**（`adapter/materialization.py`，经 `materialization` provider 声明）：
  `to_empty()` 会抹掉 checkpoint 未持久化的非持久缓冲（thinker/视觉 rotary `inv_freq`、音频
  sinusoid `positional_embedding`）。provider 注册捕获 stock `__init__` CPU fp32 精确值的重建
  recipe；通用 dtype 路径在 `model_init_dtype` 转换后恢复 fp32 来源的可重建缓冲，保持与直接
  `from_pretrained` 模型的逐字节一致（stock Transformers 用显式 `.float()` 计算它们）。
- **视觉 FSDP2 单元**（`fsdp_wrap_modules` provider）：omni 的 `visual` 分支不在 HF decoder 容器
  约定之内；`patch_embed` 与 `merger` 被声明为显式 FSDP2 子单元，使它们的 all-gather hook 在
  自己的 `__call__` 上触发。
- **数据变换**（`adapter/data/transform_fn.py`，`Qwen3OmniTransform` +
  `build_qwen3_omni_transform`）：音频/视频/图像 SFT 记录走共享 VLM 数据管线
  （`hyper_parallel.data.vlm`）；recipe 通过 `dataset.data_transform._target_` 直接引用，无需
  改注册。
- **Compile 容器**：族在模型类上声明 `get_compile_layers()`（`thinker.model.layers` /
  `talker.model.layers`），满足 `distributed.compile` 的模型侧契约。

## 已知硬件验证边界（单机 910B2 实测，2026-11）

- **stock patch-embed `Conv3d` 在 NPU 反向段错误，已路由为等价 `Linear`**：家族补丁把视觉
  patch embedding 改写为数学等价的线性形式（前向逐位一致、权重存储梯度相同）。Ascend 上
  不得恢复 stock 路径；补丁守卫在缺陷 stock 形态上，已修复的 modeling 不受影响。
- **30B 规模在 64G 卡上必须开 CPU offload**：FSDP2 全分片 8 卡但关掉 offload 时，bf16 参数 +
  梯度 + fp32 AdamW 状态连同激活超出 64G 预算。recipe 默认 `fsdp_config.enable_offload: true`；
  8 × 910B2（64G）实测开 offload 后单卡峰值约 9G，不开则 OOM。
- **`max_seq_len` 受显存而非正确性约束**：已验证 SFT 为 `max_seq_len: 1024`、
  `micro_batch_size: 1` / `global_batch_size: 8`。更长序列在该规模下预期可放但未验证；
  调大前先扩批。
- **可选 fused / packed NPU MoE 路径默认关闭**：它们改变专家计算路径（grouped-GEMM /
  打包专家），由 `HYPER_OMNI_MOE_FUSED_NPU=1` 门控。默认 eager 路径是数值验证过的配置；
  packed 路径验证的是步时性能（step 0 loss parity 逐位一致），未做端到端收敛验证。

## 依赖与资产前置

- **Transformers 提供原生模型架构，本族不携带 modeling 代码**。构建以
  `trust_remote_code: false` 从 Transformers 原生 `qwen3_omni_moe` 实现解析
  `Qwen3OmniMoeForConditionalGeneration`；checkpoint `auto_map` 的 remote code 刻意不用。
  Transformers 必须提供原生 `qwen3_omni_moe` 支持。
- **权重必须为 BF16**：集成对官方 `Qwen/Qwen3-Omni-30B-A3B-Instruct` BF16 版验证。加载为
  严格模式——key 或 shape 不匹配直接报错，这同时在线上验证原生实现与官方权重结构一致。
- **训练关闭音频输出**（`enable_audio_output: false`）：仅 thinker（视觉塔 + 音频塔 +
  稀疏 MoE 解码器）可训练。talker / code2wav 语音生成栈在本集成中仅用于推理。
- recipe 默认指向公开 Hugging Face 资源（`Qwen/Qwen3-Omni-30B-A3B-Instruct`）；本地或离线
  checkpoint 与数据集经命令行 dotted override 传入，launcher 即如此使用。

## 使用 Recipe

`recipes/train.yaml` 面向**单机 8 卡昇腾 NPU**：FSDP2 `dp_shard_size: 8`
（`tp = cp = ep = pp = 1`）、`GBS 8 = 8 × MBS 1`、`max_seq_len: 1024`、bf16、AdamW 1e-5 cosine。
端到端 ScienceQA SFT 示例把同一 recipe 与 launcher 打包在一起：

```bash
# 单机 8 卡。本地 / 离线资源经环境变量设置。
MODEL_PATH=/path/to/Qwen3-Omni-30B-A3B-Instruct \
DATA_PATH=/path/to/scienceqa/train.json \
CKPT_DIR=/path/to/checkpoints \
    bash examples/qwen3_omni_sft/run_train.sh
```

前置与注意事项：

- 需要 Qwen3-Omni-30B-A3B BF16 checkpoint（配置、processor 与权重同目录）及 ScienceQA 格式
  标注文件。`MODEL_PATH` / `DATA_PATH` / `CKPT_DIR` 不改 yaml 即可覆盖 recipe 默认值；
  `N_EPOCHS` 缩放 `training.train_iters`（launcher 注释按 2048 条子集算 `256 × N_EPOCHS`）。
- `trust_remote_code` 保持 `false`——本族面向 Transformers 原生实现。
- VLM 数据管线要求 `tp = cp = pp = 1`；该规模下并行计划仅 FSDP2 分片可变。

## 测试

| 测试 | 覆盖 |
|---|---|
| `tests/ut/auto_models/models/qwen3_omni/test_qwen3_omni_adapter_contracts.py` | 适配器契约：三种拼写身份解析、provider 声明、视觉 FSDP2 单元命名、补丁幂等、fp32 缓冲跨 dtype cast 恢复 |

## Recipes

单机 8 卡 SFT recipe 见 [recipes/train.yaml](recipes/train.yaml)。
端到端 ScienceQA SFT 训练示例见
[examples/qwen3_omni_sft](../../../examples/qwen3_omni_sft)。
