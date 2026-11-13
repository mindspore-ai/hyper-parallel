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

## Recipes

单机 8 卡 SFT recipe 见 [recipes/train.yaml](recipes/train.yaml)。
端到端 ScienceQA SFT 训练示例见
[examples/qwen3_omni_sft](../../../examples/qwen3_omni_sft)。
