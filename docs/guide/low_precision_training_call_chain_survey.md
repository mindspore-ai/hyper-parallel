# 低精训练调用链四方调研

> 调研范围：仅分析原始训练框架中的低精训练实现，不分析任何 NPU 适配仓或补丁层。
>
> - TorchTitan：`C:\Users\z00378171\lowP\torchtitan-main`
> - VeOmni：`C:\Users\z00378171\lowP\VeOmni-main`
> - Megatron-LM：`C:\Users\z00378171\lowP\Megatron-LM-main`
> - NeMo AutoModel：`C:\Users\z00378171\training_framework\Automodel-main`
>
> 结论先行：四者中，TorchTitan 的低精训练接口最接近“模型配置转换器”；Automodel 最接近“一次 eager 模块转换”；Megatron-LM 最接近“训练全局 FP8 上下文”；VeOmni 当前没有 FP8/MXFP8 训练实现，只有 BF16/FP16 mixed precision。

---

## 1. 统一观察口径

不要只看 YAML 或一个 `fp8.py` 文件。一次真实低精训练至少包含五层：

```text
用户配置
  -> 模型构建/转换点
  -> 低精模块或量化 tensor
  -> forward + backward 的 GEMM
  -> 并行、重算、checkpoint、optimizer 的协同
```

其中“低精”指 activation、weight 或 gradient 被 FP8/MXFP8 表示并参与低精 GEMM。参数以 BF16 保存、FSDP 以 BF16 all-gather、或设置 BF16 matmul accumulation 都属于 mixed precision，不等同于 FP8 训练。

| 框架 | 是否有 FP8/MXFP8 训练 | 主依赖 | 低精入口风格 | 是否覆盖 MoE grouped GEMM |
|---|---|---|---|---|
| TorchTitan | 有 FP8、MXFP8 | TorchAO | `ModelConverter` 改写模型配置树 | 有 |
| Megatron-LM | 有 FP8、MXFP8 | Transformer Engine | `fp8_autocast` 包裹 TE 模块执行 | 有，取决于 TE/模型 spec |
| Automodel | 有 FP8/QAT | TorchAO | `apply_fp8_to_model()` eager 转换模型 | FP8 通用入口主要覆盖 `nn.Linear` |
| VeOmni | 未发现 | PyTorch FSDP mixed precision | FSDP dtype policy | 不适用 |

---

## 2. TorchTitan：声明式 Converter 驱动

### 2.1 用户怎么开

用户在模型 recipe 的 `ModelSpec` 中添加 converter，而不是在训练循环中手工包 autocast。例如 Llama Float8 配置：

```python
model_registry(
    "debugmodel",
    converters=[
        Float8LinearConverter.Config(
            recipe_name="rowwise",
            filter_fqns=["output"],
            model_compile_enabled=True,
        )
    ],
)
```

DeepSeek-V3 同时声明两种 converter：

```python
converters=[
    Float8LinearConverter.Config(
        filter_fqns=["output", "router.gate"],
        model_compile_enabled=True,
    ),
    Float8GroupedExpertsConverter.Config(
        model_compile_enabled=True,
    ),
]
```

MXFP8 使用同一模式，只是换为 `MXFP8LinearConverter` 或 `MXFP8GroupedExpertsConverter`。

关键文件：

```text
torchtitan/models/<model>/config_registry.py
  -> torchtitan/components/quantization/float8.py
  -> torchtitan/components/quantization/mx.py
  -> TorchAO Float8 / MX training 实现
```

### 2.2 调用链

```text
Trainer.Config
  -> ModelSpec(converters=[...])
  -> ModelConvertersContainer
  -> ModelConfigConverter.convert(model_config)
  -> Float8LinearConverter.convert(model_config)
       遍历 Linear.Config
       -> 替换为 Float8Linear.Config
  -> 构建模型
       -> Float8Linear(TorchAOFloat8Linear)
  -> train step
       -> Float8Linear.forward/backward
       -> TorchAO 对 activation / weight / grad 做量化
       -> 低精 GEMM
```

这里的关键是：转换的对象首先是**模型配置树**，不是已经创建好的 `nn.Module`。因此模型构建时天然得到 `Float8Linear`。这避免了“替换完模块后 TP/FSDP 是否还能识别模块类型”的问题。

`Float8LinearConverter` 负责：

1. 校验 TorchAO 和硬件能力。
2. 用 `Float8LinearConfig.from_recipe_name()` 构造 TorchAO recipe。
3. 遍历 `Linear.Config`，按 FQN/filter 和 shape 条件替换为 `Float8Linear.Config`。
4. 模型真正实例化时，由 `Float8Linear` 调用 TorchAO 的 float8 实现。

### 2.3 MoE 为什么需要独立 converter

普通 Linear 不能覆盖 MoE expert 的 grouped GEMM。TorchTitan 对此单独有 `Float8GroupedExpertsConverter`：

```text
GroupedExperts.Config
  -> Float8GroupedExperts.Config
  -> 初始化时 torchao.quantization.quantize_(..., Float8TrainingOpConfig)
  -> grouped expert forward
  -> TorchAO grouped GEMM training op
```

转换器还会调整 token dispatcher 的 padding 对齐，以满足 FP8 grouped GEMM 对齐要求。这就是为什么 DeepSeek-V3 recipe 要同时配置 Linear 和 GroupedExperts 两类 converter。

### 2.4 简单性与代价

**对用户简单**：配置只表达“用哪个 converter，排除哪些 FQN”。

**框架内部并不简单**：

- Linear、grouped experts 各自有转换路径。
- `torch.compile` 被视为高性能低精训练的前提，尤其是 MoE。
- converter 顺序受约束：量化/QAT converter 必须在 LoRA 前。
- 真实量化和 GEMM 的大部分复杂度下沉到 TorchAO。

### 2.5 关键代码位置

| 作用 | 文件 |
|---|---|
| converter 基类和公共导出 | `torchtitan/components/quantization/__init__.py` |
| Float8 Linear / grouped experts | `torchtitan/components/quantization/float8.py` |
| MXFP8 Linear / grouped experts | `torchtitan/components/quantization/mx.py` |
| converter 顺序校验 | `torchtitan/models/utils.py` |
| Llama Float8 recipe | `torchtitan/models/llama3/config_registry.py` |
| DeepSeek-V3 Float8 recipe | `torchtitan/models/deepseek_v3/config_registry.py` |

---

## 3. Megatron-LM：TE module + FP8 context 驱动

### 3.1 用户怎么开

用户通过启动参数和 `TransformerConfig` 开启 FP8，例如概念上：

```text
--fp8 hybrid
--fp8-recipe delayed
```

核心配置字段位于 `megatron/core/transformer/transformer_config.py`：

```python
fp8: Optional[str]                 # e4m3 / hybrid 等 format policy
fp8_recipe: Optional[str]          # tensorwise / delayed / mxfp8 / blockwise / custom
fp8_amax_history_len: int
fp8_amax_compute_algo: str
fp8_wgrad: bool
fp8_output_proj: bool
fp8_dot_product_attention: bool
fp8_multi_head_attention: bool
```

其中 format 和 recipe 是两个概念：

- `fp8` 决定 E4M3/E5M2 的使用策略。
- `fp8_recipe` 决定 scale 的组织和更新方式，例如 current、delayed 或 MX block scaling。

### 3.2 调用链

```text
CLI 参数
  -> validate_args()
  -> TransformerConfig(fp8, fp8_recipe, ...)
  -> model spec 选择 TEColumnParallelLinear / TERowParallelLinear / TE MoE 模块
  -> TransformerBlock.forward 或 checkpointed forward
  -> get_fp8_context(config, layer_no)
  -> transformer_engine.pytorch.fp8_autocast(...)
  -> TE Linear / GroupedLinear forward + backward
  -> TE quantizer、scale/amax metadata、低精 GEMM
```

Megatron 自身主要完成两件事：

1. 把 `TransformerConfig` 翻译成 TE recipe 和 `fp8_autocast` context。
2. 让模型 spec 使用 TE 的 parallel Linear、attention、grouped GEMM 模块。

真正的 quantize、scale 更新、GEMM kernel、部分 activation recompute 协同由 Transformer Engine 管理。

### 3.3 delayed 与 MXFP8 的生命周期差异

`delayed` 不是每次只看当前 tensor。它有 amax history 和 scale 更新策略，因此 TE 模块需要维护 FP8 metadata；Megatron 的配置暴露 `fp8_amax_history_len`、`fp8_amax_compute_algo` 等字段。

MXFP8 的 scale 是 block 级动态值，metadata 形态不同。Megatron 对它还扩展到：

- `fp8_output_proj`：LM head 可使用 MXFP8 TE ColumnParallelLinear。
- `fp8_param_gather`：FSDP/distributed optimizer 可在 FP8 中进行参数 all-gather。
- `reuse_grad_buf_for_mxfp8_param_ag`：减少 MXFP8 参数 gather 的额外内存。

这说明 Megatron 的低精逻辑不是一个独立 `Linear` 替换器，而是贯穿 model spec、TE context、FSDP/DDP 和 optimizer 的训练体系。

### 3.4 简单性与代价

**对用户不算复杂**：常用场景是几个 CLI 参数。

**对框架耦合最深**：

- 需要 TE module，而非任意 `nn.Linear`。
- 需要模型 spec、重算路径、MoE、distributed optimizer 共享 FP8 语义。
- 需要维护 delayed scale/amax state 的保存、恢复和跨 rank 行为。

它适合 Megatron 生态，不适合直接搬到一个原生 HF/PyTorch 模型框架中。

### 3.5 关键代码位置

| 作用 | 文件 |
|---|---|
| FP8 context 和 recipe 分派 | `megatron/core/fp8_utils.py` |
| TE Linear、recipe 和 autocast 适配 | `megatron/core/extensions/transformer_engine.py` |
| FP8 训练配置 | `megatron/core/transformer/transformer_config.py` |
| CLI 校验与 FSDP 约束 | `megatron/training/arguments.py` |
| FP8 参数 gather | `megatron/core/distributed/distributed_data_parallel_config.py` |
| optimizer 的 FP8 参数处理 | `megatron/core/optimizer/distrib_optimizer.py` |

---

## 4. Automodel：eager 模块转换封装

### 4.1 用户怎么开

Automodel 向用户提供一个小而直接的 API：

```python
from nemo_automodel.components.quantization.fp8 import (
    FP8Config,
    apply_fp8_to_model,
)

model = apply_fp8_to_model(
    model,
    config=FP8Config(
        enabled=True,
        recipe_name="rowwise",
        filter_fqns=["lm_head"],
    ),
)
```

也可以通过 `HyperAutoModel*.from_pretrained(..., fp8_config=...)` 传入。在基础设施路径中，FP8 位于 PEFT 之后、分片之前。

### 4.2 调用链

```text
FP8Config
  -> HyperAutoModel.from_pretrained(..., fp8_config=...)
  -> apply_model_infrastructure(...)
  -> _apply_peft_and_lower_precision(...)
  -> apply_fp8_to_model(model, config)
  -> 构造 TorchAO Float8LinearConfig
  -> torchao.convert_to_float8_training(
       model, module_filter_fn=...
     )
  -> 原 nn.Linear 替换为 TorchAO Float8Linear
  -> 训练循环调用 Float8Linear
  -> TorchAO quantize + low precision GEMM
```

它和 TorchTitan 的最大差异在于转换时机：

- TorchTitan 在**模型 config tree** 层替换 `Linear.Config`。
- Automodel 在**已经构建的 model instance** 上调用 `convert_to_float8_training()`。

因此 Automodel 的适配对象是“任意已有 `nn.Module`”，而 TorchTitan 的适配对象是“遵循其 `Configurable` 协议的模型”。

### 4.3 选择规则与限制

Automodel 的 `_module_filter_fn`：

1. 排除 `filter_fqns` 中命中的名字。
2. 仅选择 `nn.Linear`。
3. 若 weight 任意维度不能被 16 整除则跳过。

这使接口很易懂，但也意味着其通用 FP8 入口不天然覆盖自定义 `GroupedExperts`、手写 `torch.mm`、fused MoE GMM 等非 `nn.Linear` 路径。若模型主要算力在 MoE expert grouped GEMM，仅转换普通 Linear 不能等价于端到端 FP8。

### 4.4 QAT 与 FP8 的关系

Automodel 的 `quantization/` 下还提供 QAT：

```text
QATConfig
  -> 创建 TorchAO QAT quantizer
  -> quantizer.prepare(model)
  -> 训练时 FakeQuant，可按 mode 开/关
```

这与 FP8 training 不同：

- FP8 training：训练中调用 TorchAO Float8Linear 的真实低精计算路径。
- QAT：典型目标是用 fake quant 模拟整数部署误差，核心是 `prepare()` 和 STE，不等于真实 FP8 GEMM。

### 4.5 简单性与代价

**封装最短**：一个 Config、一个 `apply_fp8_to_model()`、一个转换统计函数。

**可扩展性限制清晰**：

- 依赖 TorchAO CUDA/ROCm 后端能力。
- 默认围绕 `nn.Linear`。
- FSDP FP8 all-gather 选项直接暴露 TorchAO 的实现细节。
- FP8 转换异常当前会告警并返回原模型，调用方需要额外检查 `verify_fp8_conversion()` 才能防止静默回退。

### 4.6 关键代码位置

| 作用 | 文件 |
|---|---|
| FP8 Config、筛选、转换、验证 | `nemo_automodel/components/quantization/fp8.py` |
| QAT prepare 和 fake quant toggle | `nemo_automodel/components/quantization/qat.py` |
| QLoRA/BitsAndBytes 加载量化 | `nemo_automodel/components/quantization/qlora.py` |
| PEFT -> FP8 -> QAT 的调用点 | `nemo_automodel/_transformers/infrastructure.py` |

---

## 5. VeOmni：当前没有 FP8/MXFP8 训练调用链

对 `VeOmni-main` 搜索 `fp8`、`float8`、`mxfp8`、`torchao`、`transformer_engine`、`quantization`，未发现 FP8/MXFP8 训练配置、模块转换器或量化 GEMM 调用。

当前存在的是 FSDP mixed precision：

```text
train.accelerator.fsdp_config.mixed_precision
  -> MixedPrecisionConfig
       param_dtype / reduce_dtype / output_dtype
  -> torch_parallelize(..., mixed_precision=...)
  -> torch.distributed.fsdp.MixedPrecisionPolicy
  -> BF16/FP16 参数、归约、输出 dtype 策略
```

`MixedPrecisionConfig` 可选 dtype 为 `bfloat16`、`float16`、`float32`，并由 FSDP `MixedPrecisionPolicy` 消费。这并没有把 activation/weight quantize 为 FP8，也没有调用 FP8 GEMM。

VeOmni 还显式调用 `enable_high_precision_for_bf16()`，关闭 BF16 reduced-precision reduction。这是提高 BF16 数值稳定性的设置，方向与低精 GEMM 加速相反，不能当作 FP8 功能。

因此应将 VeOmni 在本报告中的定位写为：

| 能力 | 状态 |
|---|---|
| BF16/FP16 FSDP mixed precision | 有 |
| FP8 Linear training | 未发现 |
| MXFP8 training | 未发现 |
| FP8 MoE grouped GEMM | 未发现 |
| FakeQuant/QAT 通用入口 | 未发现 |

如果未来为 VeOmni 加 FP8，其现有模型 patch generation、FSDP2 和 parallel plan 是可复用的训练基础设施；但低精模块/算子生命周期仍需要新增，不能视为已有能力。

关键位置：

| 作用 | 文件 |
|---|---|
| mixed precision 配置 | `veomni/arguments/arguments_types.py` |
| trainer 创建模型与并行化 | `veomni/trainer/base.py` |
| FSDP MixedPrecisionPolicy | `veomni/distributed/torch_parallelize.py` |
| BF16 高精度 accumulation 设置 | `veomni/utils/helper.py` |

---

## 6. 四方调用方式的本质差异

```text
TorchTitan
  用户声明 converter
    -> 改模型 Config tree
    -> 构建 Float8/MXFP8 module
    -> TorchAO 执行

Megatron-LM
  用户声明 FP8 recipe
    -> TransformerConfig
    -> TE model spec + fp8_autocast context
    -> Transformer Engine 执行

Automodel
  用户传 FP8Config
    -> 已构建模型
    -> TorchAO eager module conversion
    -> TorchAO 执行

VeOmni
  用户传 FSDP mixed precision dtype
    -> FSDP MixedPrecisionPolicy
    -> BF16/FP16 常规计算
    -> 没有 FP8 quant/GEMM
```

| 问题 | TorchTitan | Megatron-LM | Automodel | VeOmni |
|---|---|---|---|---|
| 配置是否直接表达量化 recipe | 是 | 是 | 是 | 否，仅 dtype |
| 替换发生在何时 | model config build 前 | model spec + forward context | model build 后、sharding 前 | 无替换 |
| 主要低精执行者 | TorchAO | Transformer Engine | TorchAO | 无 |
| 普通 Linear | 支持 | TE Linear 支持 | 支持 | BF16/FP16 only |
| MoE grouped GEMM | 专用 converter | TE/Megatron spec 路径 | 通用 FP8入口不覆盖 | 无 |
| 是否需理解并行细节 | 配置隐藏大部分细节 | 需要，特别是 TE/FSDP/optimizer | 中等，TorchAO FSDP 参数暴露 | 仅 FSDP dtype |

---

## 7. 对 HyperParallel 新架构的启示

如果目标是“封装简单、能演进到 DeepSeek-V3/V4”，最值得参考的不是某个单文件，而是调用模型：

### 7.1 用户接口参考 TorchTitan

用户只声明一个低精 converter/配置，模型构建期应用：

```yaml
low_precision:
  enabled: true
  recipe: mxfp8
  exclude: [lm_head, "*router*"]
```

用户不应配置低精 GEMM 的 scale layout、TP local shard 或 FSDP 通信细节。

### 7.2 通用 Linear 与 MoE 必须拆开

借鉴 TorchTitan：

```text
LowPrecisionLinearConverter
  -> attention / dense MLP / shared expert 等普通 Linear

LowPrecisionGroupedExpertsConverter
  -> routed expert grouped GEMM
```

若目标包含 DSV3/DSV4，第二项从第一版就需要保留接口；否则 API 虽短，但只能得到“部分低精”。

### 7.3 不应复刻 Megatron 的全局 FP8 context

Megatron 的 context 设计合理，但前提是模型完全使用 TE module。HyperParallel 新架构面向 HF/PyTorch 模型和 Dual-mode DTensor，应该将低精作为模型转换器/组件，而不是训练循环的全局开关。

### 7.4 不应直接照搬 Automodel 的异常策略

Automodel 的接口可以借鉴，但目标框架建议：

```text
转换结果 = converted_fqns + skipped_fqns(reason) + backend manifest
```

默认遇到请求的层无法转换时失败；只有显式配置 `unsupported: bf16` 才允许回退。这样可避免用户以为启动了端到端 MXFP8，实际只有少数 Linear 被转换。

---

## 8. 结论

1. TorchTitan 的封装形态最适合作为目标：以 converter 表达意图，构建期替换，Linear 和 MoE 分开。
2. Megatron-LM 的能力最完整，但其复杂度来自 TE、model spec、optimizer、FSDP 的深度耦合；不能作为 HyperParallel 的直接代码模板。
3. Automodel 的 API 最短，适合借鉴 `Config + apply + verify` 的体验；但只靠 `nn.Linear` 转换不够覆盖 DSV3/DSV4。
4. VeOmni 当前是 BF16/FP16 FSDP mixed precision 框架，不应被列为已有 FP8 训练方案。
5. HyperParallel 应采用“TorchTitan 的 converter 入口 + Automodel 的易用 API + 独立 MoE converter”，并让 Dual-mode DTensor/FSDP 保持只负责并行和参数生命周期。
