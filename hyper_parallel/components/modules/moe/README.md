# MoE 模块与训练特性

本页说明新的 `Gate`、`GroupedExperts` 和共享专家如何组合。
专家接口和张量约定见 [GroupedExperts 模块说明](grouped_experts/README.md)。
**本次保留 Qwen、DeepSeek-V4.1、Kimi 的旧模型路径，不自动替换原生 gate，也不修改其原生辅助损失。**
下面介绍公共 Gate 的开关语义；模型接入新 Gate 时须传入模型级训练配置。
旧 Router 仍按各模型自己的参数映射和实现执行。
仅替换 `mlp.experts` 不会把旧 router 的训练逻辑迁移过来。

## 职责与文件位置

| 内容 | 实现位置 | 职责 |
| --- | --- | --- |
| Gate | [moe/router.py](router.py) | logits、learned TopK、fix-router；返回 logits/weights/实际 IDs |
| 公共训练状态 | [GateModule](router.py) | 可选 aux-loss、选择 bias、mask/sequence metadata、并行资源绑定 |
| sequence aux-loss | [losses/moe_aux_loss.py](../../losses/moe_aux_loss.py) | 不依赖具体模型的 sequence balancing 公式 |
| 负载与更新 | [moe_utils.py](moe_utils.py) | backward 计数、跨组同步、step bias 更新 |
| step 调度 | [MoEMonitorCallback](../../../trainer/callbacks/moe_monitor_callback.py) | 遍历模型并在 optimizer step 后更新 |
| 共享专家 | [shared_expert.py](shared_expert.py)、[shared_expert_overlap.py](shared_expert_overlap.py) | 共享计算与普通 EP dispatch 的组合 |

旧 `modules/gate.py`、`router.py`、`shared_expert.py`、`moe_utils.py` 已删除；路由统一为一个 Gate family，实现在 `moe/router.py`；仓内引用迁到新路径，family 只注册一次。
用户继续从 `hyper_parallel.components.modules` 导入公开模块。新组件不依赖 Trainer；独立循环可直接调用同步更新函数。
V4.1 的 adapter 保留模态 mask 与双行计数，复用公共计数、偏置更新及 aux-loss 公式。
原生 gate 的调用接口和更新时机保持不变。

## 开关与执行顺序

以下三个路由训练字段位于 `model` 配置下，彼此同级；不要放入 `GroupedExperts` 的 `plan_overrides.replace.implementation` 中。
独立使用 Gate 时，也可在传给 `Gate(config=...)` 的配置对象中设置：

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `moe_router_force_load_balancing` | `False` | 保留 learned weights，替换实际派发 IDs |
| `moe_aux_loss_coeff` | `0.0` | sequence aux-loss 系数；0 关闭 |
| `moe_router_bias_update_rate` | `0.0` | aux-loss-free step 更新幅度；0 关闭 |

这些是 Megatron 的 Python 配置字段；对应 CLI 参数分别为 `--moe-router-force-load-balancing`、
`--moe-aux-loss-coeff`、`--moe-router-bias-update-rate`。本仓 YAML 使用下划线字段。
本次仅对齐参数名称，强制均衡沿用现有确定性目的模式，不改为随机 logits。

新配置统一使用以上名称；历史短名称和 `router_*` 字段作为读取兼容，
同一配置对象同时包含新旧字段时，以新字段为准。
HF 的 `router_aux_loss_coef` 属于 enclosing model 的原生 loss，不作为新 sequence aux-loss 的隐式启用系数。
两个系数默认均为 0；这保留 Megatron 参数名称，同时使本仓的新训练特性默认关闭。

系数须为有限非负数，开关须为 bool。构造 context 的 `module_configs`
按最近 owner 到根的顺序提供配置，优先于单独的路由维度配置。

```text
projection → 无偏 scores → 加 correction bias 选 learned IDs → 取 learned weights
                                                       ↓
                                 fix-router（开启时只替换 IDs）
                                                       ↓
                       最终 IDs → aux-loss / backward 负载计数 → experts dispatch
```

Gate 的权重归一化和 scaling 顺序保持不变。
correction bias 不进入最终 weights。fix-router 也不重新从固定专家位置 gather weights。
**aux-loss 和 aux-loss-free 均使用 fix-router 之后、真正送给专家的 IDs。**
这与上一版建议的“保留 learned IDs 统计”有意不同，遵循此次指定的真实派发口径。
固定路由改变训练算法，只适合负载/通信实验；训练精度对照应关闭它。

### fix-router

[balanced_expert_indices](router.py) 沿用已有 owner-major 均匀目的模式。
全局专家数 `E`、物理 EP owner 数 `P`、`C=E/P`，flatten slot `s` 的目的为：

```text
(s % P) * C + (s // P) % C
```

例如 `E=8,P=4` 为 `0,2,4,6,1,3,5,7`。要求 `P` 整除 `E`。
绑定时 `P` 从现有 EP mesh 取得，包含 TP-extend-EP；未绑定的单卡组件按 `P=1`。

### aux-loss

复用 sequence balancing 公式，每个 logical sequence 计算：
`E * sum(实际 assignment 比例 * 无偏归一化 score 的均值)`，再对非空样本平均。
`token_mask` 排除 padding。普通 `[B,S,H]` 输入按 B 区分样本；
packed/跨 rank 分片输入可传 `sequence_ids` 与 `num_sequences` 明确样本边界。

通过已有 `aux_loss_auto_scale(weights, coeff * loss)` 注入辅助梯度，forward weights 数值不变。
`last_aux_loss` 为 detach 后的原始指标，**不要再次加进 task loss**。
Trainer 原有 `aux_loss_scale_context` / `bind_aux_loss_scale` 继续负责梯度累积和 loss scaling。

### aux-loss-free

公共 Gate 使用 FP32 `bias[E]` 作为选择状态，不参与 optimizer 梯度。
`tokens_per_expert[E]` 为 int64、非持久化计数，不区分模态。
计数挂在实际被消耗的 weights backward 上，checkpoint forward 重放不会多记一次。
每个逻辑 optimizer step 后同步并执行：

```text
bias += rate * sign(total_assignments - E * expert_assignments)
```

随后清零计数，全零计数不更新。恢复 checkpoint 时保留 bias、重建零计数。

V4.1 在模型 adapter 中保留 `tokens_per_expert[2,E]`（文本、图像）以及 `bias` / `bias_vl`。
模型将文本、图像映射到独立计数区间，用同一个公共负载累计函数处理 `token_mask`，再调用模型侧的
`update_modality_bias`，由它分别复用 `update_router_bias` 更新两套偏置；缺失模态不更新。
公共训练状态不读取 `v41_vision_enabled`。
MegaGate 原有 `vision_enabled` / `image_mask` 融合算子接口保留，模态分别更新仍由模型负责。

```python
from hyper_parallel.components.modules.moe.moe_utils import update_router_bias
from hyper_parallel.models.deepseek_v41.adapter.aux_loss import update_modality_bias

update_router_bias(bias, counts, rate)  # 普通场景：counts[E]
update_modality_bias(bias, modality_counts, rate, vision_bias=bias_vl)  # V4.1：counts[2,E]
```

两个函数都只更新偏置；同步和 step 后计数清零仍由原调用流程负责。

## 与并行、FSDP 的组合

新 Gate 自己声明完整复制参数、无额外 token 边界和 `bind_parallel` 协议。
family 的布局提供者只声明所有权；公共 binder 自动调用 `GateModule.bind_parallel`，
在该方法中直接绑定既有 mesh 和统计 group，不创建新组。

| 内容 | 汇总范围 |
| --- | --- |
| sequence aux-loss | CP，以及实际切分 token 的 TP；不混合不同 DP 样本 |
| step 负载计数 | DP、CP，以及实际切分 token 的 TP |
| EP payload | 不再作为一个额外统计轴，避免与 DP/TP 重叠计数 |

SP 或 TP-extend-EP 使用 TP token 切片；传入完整 token metadata 时按相同 TP rank 对齐。
已是本地 shape 的 metadata 原样使用。`sequence_partition_groups=None` 使用绑定值，显式 tuple 可覆盖。
源 token 统计在 dispatch 前完成；不手工 unshard，不改变训练主流程的 FSDP 预取。

## 构造与使用

模型整改后的配置形式保留为：

```yaml
model:
  # 保留原模型 target；该模型须先完成新 Gate 的代码接入
  moe_aux_loss_coeff: 0.0001
  moe_router_bias_update_rate: 0.001
  moe_router_force_load_balancing: false

plan_overrides:
  - match: 'model.layers.*.mlp.experts'
    replace:
      kind: module
      implementation:
        _target_: hyper_parallel.components.modules.GroupedExperts
        megakernel: false
        moonep: false
```

模型在构造新 Gate 时传递模型级 config，维度和原有 routing 语义由模型自己映射：

```python
from types import SimpleNamespace
from hyper_parallel.components.modules import Gate

routing_config = SimpleNamespace(
    hidden_size=model_config.hidden_size,
    num_experts=model_config.num_experts,
    top_k=model_config.num_experts_per_tok,
    scoring_func='sigmoid',
    routed_scaling_factor=1.0,
    normalize_topk=True,
    initializer_range=0.02,
)
self.gate = Gate(config=routing_config, context={'module_configs': (model_config,)})
```

这部分属于后续模型整改；本次不为旧模型追加替换父层 forward 的 adapter。
单组件用户也可直接在 `routing_config` 中设置上述 `router_*` 字段，不需要上层 config 或 Trainer。

独立训练循环（`router` / `experts` 已构造并按需绑定）：

```python
from hyper_parallel.components.functional.aux_loss import aux_loss_scale_context, bind_aux_loss_scale
from hyper_parallel.components.modules.moe.moe_utils import sync_and_update_expert_bias

optimizer.zero_grad()
for hidden in microbatches:
    with aux_loss_scale_context():
        _, weights, ids = router(hidden)
        output = experts(hidden, ids, weights)
        loss = criterion(output)
        bind_aux_loss_scale(loss)
        (loss / len(microbatches)).backward()
optimizer.step()
sync_and_update_expert_bias(router)  # 逻辑 step 后一次；已绑定时使用绑定统计组
```

也可使用 `hyper_parallel.trainer.callbacks.MoEMonitorCallback` 遍历整个模型，
手动调用 `on_step_end()` 或注册 optimizer post-hook；三种调度方式只选一种。

## `moe_utils` 的职责

`router_sharding` 是本次新增，原 Gate 注册没有这项布局声明；旧模型主要使用模型外层切分模板。
它声明 Gate 的全部参数在计算轴复制，不创建 token 重新分布边界，FSDP 存储切分仍由主流程负责。
Gate 共用这一布局声明，通过 `bind_parallel` 绑定并行信息。

| 方法 | 输入和输出 | 用途 |
| --- | --- | --- |
| `local_router_metadata` | 全局或本地 mask/sample IDs → 本地 token 元数据 | 将 TP 切片前的元数据与当前 router token 对齐 |
| `router_sharding` | 最终 module → `ModuleShardingSpec` | 声明复制参数与非通信边界 |
| `validate_router_layout` | module、最终 sharding → 校验结果 | 防止用户覆盖为不完整参数所有权或额外 token 边界 |
| `_AccumulateRouterLoad` | weights、微批 counts、累加器 → 原 weights | backward 时计数，重计算 forward 不重复累计 |
| `accumulate_router_load` | 实际 IDs、mask、weights → 保留梯度的 weights | 根据 fix-router 后的 IDs 累计标准 `[E]` 专家负载 |
| `update_router_bias` | bias、全局整数 counts、rate → 原位更新 | 按低于/高于均值的符号更新选择偏置，兼容 DTensor |
| `sync_and_update_expert_bias` | router、既有统计 group → 更新并清零 | optimizer step 后汇总唯一统计轴，调用 router 的更新方法 |

`routing_scores`、TopK 选择、`balanced_expert_indices` 统一放在 `router.py`；序列辅助损失在
`components/losses/moe_aux_loss.py`；监控与 optimizer hook 在 `trainer/callbacks/moe_monitor_callback.py`。
工具文件不负责专家计算、dispatch/combine，也不重复规约已经覆盖的 EP group。
