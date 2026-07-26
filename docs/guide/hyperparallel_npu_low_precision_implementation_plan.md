# HyperParallel NPU 低精训练方案 — 落地计划

## Context（为什么做这件事）

HyperParallel `feat/trainer-dev-pr-1080` 重构把训练能力迁移到"声明式配置 + 单一职责组件"。低精训练在代码调研 §8 被明确列为基础闭环之后的能力,当前 `hyper_models/` 里所有 FP8/QAT 路径都是 stub(`_transformers/infrastructure.py:121-125` 只 `logger.warning`)。

已存在一份成熟设计文档 `docs/guide/hyperparallel_low_precision_training_design.md`,经评审整体质量高(Phase 0 修 NPU 骨架、planner 类型兼容点、meta tensor 处理、边界所有权都抓得准)。本计划**不推翻它**,而是落地经多轮敲定的**目录结构 + 三点调整**,并把它同步进设计文档,再交付 Phase 1 代码骨架。

**已确认的范围与决策**:
- 硬件后端:仅昇腾 NPU,直接对接 `torch_npu`,不做 GPU/torchao 后端抽象。
- 首推格式:MXFP8 E4M3(第一优先级)+ HiF8 tensorwise(并列纳入,MXFP8 验证后)。
- "支持低精"的唯一标准:forward/dgrad/wgrad 三段都实际命中 `npu_quant_matmul`。
- 交付侧重:设计哲学 + 架构 + 落地路径。

## 设计哲学(沿用设计文档,不重述)

低精是**构建期模型转换**而非训练循环开关(装在 infrastructure 的 `QAT/FP8` 槽位,plan 之前);format/scaling/target **独立建模**(当前仅开放 `mxfp8_e4m3↔mx_block`、`hif8↔tensorwise` 两对经源码校验组合,不宣称任意正交);低精层**只拥有 GEMM 边界**(master weight/optimizer/all-gather/梯度归约仍归现有 owner);Dense 与 MoE **架构上分路**;真三段 GEMM 才算支持。

## 目录结构(本轮定稿)

```
hyper_models/components/training/low_precision/
├── __init__.py          # re-export: LowPrecisionConfig, apply_low_precision, ConvertManifest
├── config.py            # LowPrecisionConfig: 纯 dataclass + __post_init__ 校验
├── converter.py         # apply_low_precision(model, cfg) -> (model, manifest)
├── linear.py            # NpuQuantLinear(nn.Module) + 私有 _NpuQuantLinearFn(autograd.Function)
├── npu_ops.py           # 唯一直接 import torch_npu 的生产文件;经 _get_npu_ops() 注入点访问
├── manifest.py          # ConvertManifest + require_complete()
└── testing/
    ├── __init__.py
    └── mock_npu_ops.py  # 无 NPU 环境的 fake 算子,仅测试注入,生产代码不 import
```

顶层字段仍在 `trainer/config.py`:
```python
from hyper_models.components.training.low_precision import LowPrecisionConfig
low_precision: LowPrecisionConfig = field(default_factory=LowPrecisionConfig)
```

### 相对设计文档 §4 结构表的调整(及 rationale)

1. **config 下沉到 `low_precision/config.py`,不留 `trainer/config.py`。** 与 `checkpoint/config.py:24`(纯 dataclass、无 Config+build、直接被 resolver 实例化)的先例一致。顶层 schema 文件不承担组件校验职责。

2. **`LowPrecisionConfig` 不套 Config+build() 模式。** 它是声明式转换配置,不构建 runtime 对象;做成普通 dataclass + `__post_init__` 校验(format↔scaling 组合合法性、fallback 取值),消费点就是 `apply_low_precision(model, config)`。

3. **不公开 `grouped_experts.py` 的 NotImplementedError 占位。** MoE 模块签名取决于真实模型的 token dispatcher / expert 权重布局 / `npu_grouped_matmul` 契约,现在定死接口反而可能在接 DSV3/V4 时被推翻。改为:converter 遇到 MoE 时在 manifest 记 `skipped["...experts"] = "grouped-experts-not-supported"`,`fallback: error` 时失败。等第一个真实 MoE 模型接入再加 `grouped_experts.py`。Dense/MoE 的"分路"通过 converter 分派 + manifest 记录体现,不靠空文件。

4. **新增 `testing/` 子包(照搬 `distributed/testing/` 先例)。** `npu_ops.py` 暴露薄访问接口(`NpuOps` 或私有 `_get_npu_ops()`),测试用 monkeypatch/inject `mock_npu_ops.py` 的 fake 实现。使无 NPU 环境可测 converter、dtype 分派、参数方向,同时生产包不依赖 test double。

5. **`linear.py` 明确二分。** `NpuQuantLinear(nn.Module)` 负责保持 `weight`/`bias`/FQN/dtype/device 语义,`_NpuQuantLinearFn(torch.autograd.Function)` 负责三段量化 GEMM;两者不混为一个概念。只有 `_NpuQuantLinearFn` 经 `npu_ops` 触达 `torch_npu`。

6. **不新增 `activation_checkpoint_mode` 配置项。** AC 启停仍由现有 `GradientCheckpointingConfig`(`trainer/config.py:60-64`)管理;低精只在 `linear.py` 的 autograd.Function 里**读取**实际重算状态,决定"重新量化"(v1)或未来缓存路径。避免两个 AC 配置轴悬着耦合。

### YAML 启用形态(受 resolver `_target_` 约束)

`resolver.py:244-245`:顶层 component group 出现即需 `_target_`。字段带 default_factory,不写则 disabled;启用时:
```yaml
low_precision:
  _target_: hyper_models.components.training.low_precision.LowPrecisionConfig
  enabled: true
  format: mxfp8_e4m3
  scaling: mx_block
  exclude_fqns: ["lm_head", "*.router.gate", "*.embed_tokens"]
  fallback: error
```

## 分层架构

```
配置层  LowPrecisionConfig(config.py, __post_init__ 校验) → trainer/config.py 顶层字段
转换层  apply_low_precision(converter.py) → (model, ConvertManifest)
        └ 在 apply_model_infrastructure 的 QAT/FP8 槽位调用, ShardingPlanner.plan() 之前
模块层  NpuQuantLinear(nn.Module) + _NpuQuantLinearFn(autograd.Function)
算子层  npu_ops._get_npu_ops() → torch_npu.npu_dynamic_mx_quant / npu_dynamic_quant / npu_quant_matmul
观测层  ConvertManifest(converted/skipped/kernel_contract) — v1 交付
        [后续] 运行期数值监控(amax/scale/溢出比例):需单独设计统计生命周期与输出链路,不在本轮骨架
```

planner 类型兼容:`NpuQuantLinear` 加 `_hp_linear_compute_kind = "npu_quant"` 标记,扩展 `ShardingPlanner` 识别该标记而非依赖"仍是 `nn.Linear`"。

## 落地路径

**Phase 0 — 先让 NPU 骨架可信(前置,非低精本身)**
- 引入共享 accelerator-device helper,替换新 `hyper_models/` 路径**全部** CUDA 硬编码。已确认的点(非仅 auto_model):
  - `_transformers/auto_model.py:96/154/254`:build/infra 的 device 选择。
  - `components/distributed/infrastructure.py:188`:mesh 初始化 `device_type = "cuda" if cuda.is_available() else "cpu"`——NPU 会误落 cpu,mesh 建错(最关键)。
  - `components/training/grad_accum.py:135/137-138/149/151-152/241`:6 处 `torch.cuda.*`,含 `.cuda()` 强制搬运与 `get_device_name`。
  - `components/training/signal_handler.py:56`:分布式信号量 device 选择。
  - `recipes/llm/train_ft.py:92/95-99`:Recipe 主设备二选一(影响 batch 搬运与 MFU),且 `initialize_distributed("nccl")` 硬编码后端——NPU 应为 `hccl`,需一并参数化。
  - 并审计训练路径其余 `torch.cuda.*` / 通信后端残留(以上为已 grep 确认项,非穷举)。
- `examples/training_skeleton` 跑通单卡 NPU BF16 forward/backward/optimizer smoke。
- (构建时序测试见 Phase 1——它依赖低精 converter 存在,Phase 0 只验证 NPU BF16 骨架。)

**Phase 1 — MXFP8 Dense MVP**
1. `low_precision/config.py`:`LowPrecisionConfig` + `__post_init__` 校验;顶层 `trainer/config.py` 加字段 + 更新 `__all__`;带注释 YAML 样例。**Phase 1 schema 即能力声明**:`format` 的 `Literal` 本阶段只暴露 `"mxfp8_e4m3"`(不含 `hif8`),`scaling` 只暴露 `"mx_block"`——未实现的格式连配置类型都不接受,比"接受后运行时拒绝"更干净。Phase 2 实现 HiF8 时再把 `hif8`/`tensorwise` 扩进 `Literal`。相应地,`__post_init__` 校验聚焦合法组合(`mxfp8_e4m3↔mx_block`)与 fallback 取值。
2. `manifest.py`:`ConvertManifest` + `require_complete()`。
3. `converter.py`:`apply_low_precision`——FQN 选择(exclude 优先、glob 全名匹配、只选 `nn.Linear`、weight 32 对齐)、模块替换、生成 manifest;MoE 记 skipped。
   **MoE 识别规则(先于 nn.Linear 选择)**:某些 MoE 实现的 expert 内部本身就是 `nn.Linear`,会被 Dense converter 意外替换,破坏"MoE 延后"边界。v1 须在遍历时**先**识别 expert 容器(已知 MoE 模块类型 / expert 容器属性 / 匹配 MoE FQN 标记如 `*.experts*`),命中则整体记 `skipped["...experts"] = "grouped-experts-not-supported"` 并**不下钻**其子 Linear;`fallback: error` 时失败。只有不在 MoE 子树内的 `nn.Linear` 才进入 Dense 转换。
   **meta tensor 契约(硬约束)**:converter 运行在 meta 模型构建阶段,只依据 `in_features`/`out_features`/`weight.shape`/`bias is not None` 等**静态元数据**做校验与替换,**绝不**在此 import/调用 `torch_npu`、读取 device capability、或访问 tensor 数据(meta tensor 无实际存储)。所有 torch_npu 调用与 capability check 延后到 `npu_ops` 的**首次真实 forward**(materialize 之后)。否则多卡构建会在 materialize 前失败。
4. `linear.py` + `npu_ops.py`:`NpuQuantLinear` / `_NpuQuantLinearFn`;MXFP8 forward/dgrad/wgrad 走 `npu_dynamic_mx_quant`(单轴为正确性基线,dual-axis 待 compile/重算测试后启用)+ `npu_quant_matmul`。
   **bias 契约**:`_NpuQuantLinearFn` 只算**无 bias** 的三段低精 GEMM;bias 加法放在 `NpuQuantLinear.forward` 里以普通 PyTorch `output + bias` 执行——这样 `bias.grad` 由 autograd 自然产生,且三个矩阵乘都保持低精。若目标 kernel 或模型限制 `bias=False`,converter 须对带 bias 的层记 skip reason,不静默丢弃 bias。
5. `infrastructure.py:124-125`:`fp8` warning → `apply_low_precision(model, low_precision_config)`(新增 `low_precision_config` 形参,避免与 GPU torchao 的 `fp8_config` 语义混淆;`apply_model_infrastructure` 只见透传后的形参,不持有 `cfg`),严格在 plan 之前。
6. plumbing:`auto_model.py` 沿 `from_pretrained→_build_model→apply_model_infrastructure` 透传;`train_ft.py` setup 从 `cfg.low_precision` 传入。
7. 加测试锁定"转换发生在 `ShardingPlanner.plan()` 和 optimizer 创建之前"的构建时序(依赖 converter 存在,故在本阶段而非 Phase 0)。

**Phase 2+**:生产安全(shape/dtype/device capability 校验、`fallback: bf16` 测试、TP/FSDP 组合逐项验证、AC recompute)→ HiF8 tensorwise(`npu_ops.py` 独立 `HiF8KernelPolicy`)→ MoE grouped GEMM(独立 converter + `grouped_experts.py`)。

## 关键文件

| 动作 | 文件 |
|---|---|
| 同步目录/调整进设计文档 | `docs/guide/hyperparallel_low_precision_training_design.md`(§4 结构表、§3 config 位置) |
| 新建子包 | `hyper_models/components/training/low_precision/{__init__,config,converter,linear,npu_ops,manifest}.py` + `testing/{__init__,mock_npu_ops}.py` |
| 顶层字段 | `hyper_models/trainer/config.py`(加 `low_precision` 字段 + `__all__`) |
| 接线钩子 | `hyper_models/_transformers/infrastructure.py:124-125` |
| 透传参数 | `hyper_models/_transformers/auto_model.py`(from_pretrained→_build_model→apply_model_infrastructure) |
| recipe 读取 | `hyper_models/recipes/llm/train_ft.py`(setup) |
| YAML 示例 | `examples/training_skeleton/train.yaml` |

**复用资产**:`checkpoint/config.py`(纯 dataclass config 先例)、`distributed/testing/`(测试子包先例)、`masked_ce.py:68-69`(fp32_upcast loss 高精累加沿用)、`hyper_parallel/core/fully_shard/api.py:615`(FSDP 参数生命周期)、resolver `_target_` 机制。

## 验证(端到端)

无 NPU 硬件时,Phase 1 交付 = 设计文档更新 + 可导入 config/converter 骨架 + mock 单测:
1. **Converter 单测**(mock 算子):FQN 规则/排除/异常/manifest/parameter identity。
2. **dtype 分派单测(仅 MXFP8)**:MXFP8 的 `dst_type`;tensor 方向;输出 shape。HiF8 的 config 校验与 dtype 分派测试放 Phase 2+ 与 HiF8 实现同批交付,避免"配置可接受、运行未实现"的假支持。
3. **fail-on-skip**:`fallback: error` 下被选却未转换的 target 在 optimizer 创建前报错。
4. **回归**:`examples/training_skeleton` 在 low_precision 关闭时行为不变(GPU/CPU 冒烟不受影响)。

NPU 环境真机验证按**由小到大**推进,便于异常定位(不从多卡起步):
1. **单卡优先**:tiny model 单卡跑通——trace 证明 forward/dgrad/wgrad 三段都命中 `npu_quant_matmul`;vs BF16 baseline 的 loss/梯度范数/关键层输出误差(warmup+稳定);dgrad/wgrad 走低精 path 且 optimizer 更新后可续训。
2. **TP=2 其次**:Dense 模型 local shape 满足 kernel 契约。
3. **后续矩阵**:更多卡、FSDP2、MoE 属能力矩阵,逐格标注硬件/CANN 版本与测试结果,不据未测路径推断。
容错项(单卡即可验):shape 非 32 对齐按 fallback;NaN/Inf(复用 `debug.check_nan_inf`);checkpoint 保存/恢复。真机数值验证统一标注"需 NPU 环境执行"。
