# MegaKernel 单组件接入与组合训练示例

这里提供三条相互独立的接入路径：`MegaGate` 路由、`MegaMoeExperts` 专家执行和 `MegaMhc` 四流连接边界。
模型开发者可以只替换已有网络中的一个模块。mHC 是独立的连接结构，不是 MoE 的必需组件，
也不是普通残差加法的等价替换。

推荐顺序是[技术与适用性](../README.md#技术概览与适用性) → 单组件最小示例 →
已有模块与权重迁移 → 组合训练验证 → 高级调优。首次运行不用先理解 push/pull、通信切块或 SHMEM 生命周期。

## 环境准备

需要配套的 Torch/torch_npu、CANN toolkit/ops **9.1.0 或更高版本**以及 Ascend NPU。
Mega 路径使用本组件构建的官方 CANN 算子制品，示例的普通 mHC 对照使用明确的 Torch 公式，
无需额外的 Omni 注册包或 vendor OPP。原有网络自己的依赖仍由对应模型维护。

```bash
source /path/to/cann/set_env.sh
python -m pip install -e .
bash hyper_parallel/core/multicore/build.sh --soc-list ascend910b --jobs 24
source build/native/payload/hyper_parallel/core/multicore/lib/set_env.bash
```

目标芯片和构建环境选择见[构建说明](../docs/build.md)。Python、Torch、CANN 和 SoC 必须与制品匹配。

## 单组件最小示例

三个命令均可独立执行，默认使用一张 NPU，不建立 EP8 网络，也不导入其余组件的模型：

```bash
python -m hyper_parallel.core.multicore.examples.minimal_gate
python -m hyper_parallel.core.multicore.examples.minimal_moe
python -m hyper_parallel.core.multicore.examples.minimal_mhc
```

需要检查实际容量、计划命中和路由负载时，MoE 入口直接添加一个选项：

```bash
python -m hyper_parallel.core.multicore.examples.minimal_moe --diagnostic-output ./moe_execution
```

该命令保留一次前向/反向精度检查，并写出独立进程的 Host 执行报告。已有训练模型的接入点、
FSDP/优化器边界及完整采样校验见[训练接入与运行检查](../docs/mega_moe_integration.md)。

| 入口 | 最小输入及迁移 | 检查内容 |
| --- | --- | --- |
| [minimal_gate.py](minimal_gate.py) | `[1, 128, 512]`，从已有投影/修正偏置构造 Gate | logits、TopK ID/权重、输入与投影梯度；报告实际后端 |
| [minimal_moe.py](minimal_moe.py) | 四个本地 SwiGLU 专家，K=2，从 `GroupedExperts` 转换 | 输出、输入/路由梯度和 gate/up/down 梯度；保持输入形状 |
| [minimal_mhc.py](minimal_mhc.py) | 四流 shifted 状态，从已有 mHC 参数构造边界 | 五项输出、五个状态输入与四组参数梯度，及末层状态合并 |

公共输入参数为 `--tokens`、`--hidden-size`、`--device`。默认 NPU，Gate 另支持
`--device cpu` 检查 Torch 路径；选择 Gate 不等于一定启用原生 Route，输出会说明实际实现及原因。
MoE、mHC 的 Mega 路径要求 NPU。mHC 示例会查询芯片并报告所需的最小本地训练 token 数，
不要求调用方查芯片核数。真实硬件限制仍会报错，不会改变输入或数学定义来绕过。

单组件对照的默认误差阈值为 `rtol=0.02, atol=0.002`，ID 要求严格一致。
最小例子用于检查接口与梯度，不是性能 benchmark。

## 已有模块与权重迁移

### MoE：保留原网络的 Router、共享专家和输入形状

`MegaMoeExperts` 替换 token 分发、grouped SwiGLU 专家计算及回收。已有 Router 的评分、
负载均衡和辅助 loss 由原模型保留。下面给出一个完整的替换模块：

```python
import torch
from hyper_parallel.core.multicore import MegaGate, MegaMoeExperts

class RoutedMoe(torch.nn.Module):
    def __init__(self, source, ep_group):
        super().__init__()
        self.router = MegaGate.from_module(source.router)
        self.experts = MegaMoeExperts.from_module(
            source.experts, top_k=self.router.top_k, ep_group=ep_group,
        )
        self.shared_expert = source.shared_expert

    def forward(self, hidden_states):
        route = self.router(hidden_states)
        routed = self.experts(
            hidden_states, topk_ids=route.ids, topk_weights=route.weights,
        )
        return routed + self.shared_expert(hidden_states)

model.moe = RoutedMoe(model.moe, ep_group)
```

`hidden_states` 可保持 `[B, S, H]`，返回同形状；不需要在业务 forward 中记住返回前 reshape。
`GateOutput` 的 `logits`、`weights`、`ids` 具名，兼容原来的三元组解包。
首次接入不传 `tokens_per_expert`，内部会统计；已有 Router 提供精确计数时，可以作为优化复用。
此时计数必须是本 rank 按全局专家编号统计的 `[E]` histogram，调用方负责和当前 ID 一致。

`ep_group` 必须是模型选定的 EP 组；组大小由组件推导。不要为了方便把 DP、TP、CP、PP 的混合域都当作 EP。
未传组表示独立单 rank；已初始化多 rank 的分布式进程必须显式传组，不会自动猜成 WORLD。
可选 `ep_size` 仅作为兼容性/一致性检查，通常不必传。

`from_module` 支持已经 materialize 的本地 `GroupedExperts` SwiGLU 布局：

| 原参数 | 原形状 | Mega 参数与转换 |
| --- | --- | --- |
| `w1`，gate | `[local_E, I, H]` | 转置最后两维，写入 `gate_up_weight[..., :I]` |
| `w3`，up | `[local_E, I, H]` | 转置最后两维，写入 `gate_up_weight[..., I:]` |
| `w2`，down | `[local_E, H, I]` | 转置最后两维，写入 `down_weight` |

转换创建新的专家 Parameter，保留源 dtype/device 和梯度参与策略。
EP DTensor 取本地专家片段；FSDP 参数须先完整 materialize。
不兼容的布局或 gate/up 不一致的冻结策略会报错。其他专家数学定义、clamp 或布局，
需要在模型自己的 adapter 中明确转换，不能仅凭同名参数猜测。

### Gate：绑定原参数，保持评分函数

```python
from hyper_parallel.core.multicore import MegaGate

model.router = MegaGate.from_module(model.router)
route = model.router(hidden_states)
backend = model.router.backend_for(hidden_states)
```

源模块须提供 `weight: [E, H]`，以及 `top_k`、`scoring_func`、`routed_scaling_factor`。
后三项也可以显式传给 `from_module`。可选 `bias: [E]` 是 **TopK 修正偏置**，不是线性投影 bias；
`bias_vl: [E]` 为视觉修正偏置。已有参数直接绑定，梯度参与策略不变；缺少修正偏置时使用零偏置。

支持 `sqrtsoftplus`、`softmax`、`sigmoid`。NPU `sqrtsoftplus` 使用原生 Route，其余支持场景走 Torch。
不能为使用原生实现而改变原网络的评分函数。
`backend_for` 返回 `name` 与 `reason`，只查询选择，不加载 native 库，也不逐步打印。

只有构造时启用视觉偏置的模块才接收 `image_mask`；显式传 mask 却没有视觉偏置会明确报错，
不会忽略 mask，也不会在 forward 中新建参数。需要视觉路由时使用支持 `bias_vl` 的源模块，
或在新构造 `MegaGate` 时设置 `vision_enabled=True`。

### mHC：参数、精度与具名状态

源模块必须已有固定四流 shifted mHC，具备 `hidden_size`、`hc_eps`、`norm_eps`、`num_iters`，
以及以下参数布局。`from_module` 保留其参数对象及配置：

| 参数 | 形状 | 精度 |
| --- | --- | --- |
| `phi` | `[24, 4H]` | FP32 |
| `alpha` | `[3]` | FP32 |
| `bias` | `[24]` | FP32 |
| `norm_weight` | `[H]` | BF16 |

```python
import torch
from hyper_parallel.core.multicore import MegaMhc, MhcState

model.mhc = MegaMhc.from_module(model.mhc)
model.to(device=device, dtype=torch.bfloat16)
state = MhcState.initialize(hidden_states)
result = model.mhc(state)
state = result.next_state(block(result.hidden_states))
output = state.finish()
```

整网 dtype 转换会保护 `phi`、`alpha`、`bias` 的 FP32 原值，避免先降精度再转回造成不可逆舍入。
Gate 的修正偏置也保持 FP32；普通权重和 mHC `norm_weight` 使用 BF16。
源模块若已经把这些预测器转成 BF16，需要重新从原始 FP32 权重加载，不能通过 `.float()` 恢复丢失精度。

`MhcState` 的五个字段为 `previous_output`、`residual`、`pre_mix`、`post_mix`、`residual_mix`。
`MegaMhcOutput` 包含更新后的 residual、三个 next mapping 及当前 block 的 `hidden_states`。
`next_state(block_output)` 把当前 block 输出与 next mapping 组成下一边界的输入。
首层用 `MhcState.initialize`，末层用 `state.finish()`，后者会纳入最后一个 pending block 输出。
每层状态显式传递，可同时保留多份；低层五张量调用与元组解包保留兼容。

当前训练要求 H 为 128 的倍数且不超过 5760。Sinkhorn 固定 20 次，并非任意可调超参数。
调用 `MegaMhc.execution_requirements(device)` 得到该设备的本地训练 token 下限；
CP/SP 等并行切分后的输入也须满足。当前不会为了支持小形状而自动 padding 或切换数学定义。

迁移校验可以先运行 `minimal_mhc.py`：它以源模块的 Torch 公式和同一组参数、状态为参考，
检查五项输出及九项梯度。项目中的 `MhcBoundary.reference_forward` 展示这套独立公式。
不符合该状态协议的原 mHC，应由模型 adapter 明确转换，不能只更换类名。
公共名称为 `MegaMhc`，`HyperMegaMhc` 保留为兼容别名；底层 ABI 名称保持原样。

### Checkpoint 与优化器的推荐顺序

1. 构造原模型，按原有协议加载模型 checkpoint；确认原权重、评分函数和 mHC 配置正确。
2. materialize 待转换参数，在正确的 EP 归属下完成模块替换及布局转换。
3. 执行设备和精度迁移，再应用目标并行包装；确认 mHC FP32 控制参数及普通 BF16 参数。
4. 对最终 Parameter 集合构造优化器，先做同输入下的输出、梯度和单步更新对照。

Gate 和 mHC 的支持布局可以绑定原 Parameter；MoE 的合并/转置会创建新 Parameter。
**权重迁移不等于完整断点续训迁移。** 当前示例和 `from_module` 不自动转换旧 Adam 状态。
严格续训需要同时转换 FP32 主参数、`exp_avg`、`exp_avg_sq` 和相应分布式 checkpoint 元数据；
gate/up 的动量须按同样顺序转置、拼接，down 动量须转置，gate/up 的 step 状态还必须一致。
不能把原优化器 state_dict 直接装入重新合并后的 Parameter，并据此宣称续训等价。
组合示例从相同初始参数构建新的优化器，未覆盖真实训练 checkpoint 的优化器状态恢复。

### 高级参数持有：已有 FSDP 或参数管理器

模型 adapter 仍可使用 `create_parameters=False`，并在 forward 内传入当前已 unshard 的权重。
这条路径保留原参数的 autograd 连接，不需要把 FSDP 管理权转交给执行器：

```python
experts = MegaMoeExperts(
    hidden_size=H, intermediate_size=I, num_experts=E, top_k=K,
    ep_group=ep_group, create_parameters=False,
)

def expert_forward(hidden_states, ids, weights, gate, up, down):
    gate_up = torch.cat((gate.transpose(1, 2), up.transpose(1, 2)), dim=-1)
    down_weight = down.transpose(1, 2)
    return experts(
        hidden_states, topk_ids=ids, topk_weights=weights,
        expert_weights=(gate_up, down_weight),
    )
```

连续权重会复用存储；非连续或拼接权重可能产生逐次复制开销。已有模型优先使用其模型专属 adapter，
由 adapter 承接布局、参数生命周期和并行拓扑，不在通用模块中猜测任意 HF 模型的属性名。

## 组合训练验证

[model.py](model.py) 组装了一个带四流 mHC 的随机权重文本 MoE 网络，使用简化 GQA、
`sqrtsoftplus` Router、路由专家和共享专家。[gate.py](gate.py)、[moe.py](moe.py)、[mhc.py](mhc.py)
按组件对照普通实现和 Mega 实现。[benchmark.py](benchmark.py) 检查同初始参数下的输出、梯度及更新，
再测完整 optimizer step。它是组合示例，不对应已发布模型，也不代表 DSV4.1 整网验收。

只有这个组合 benchmark 使用固定 TP1/EP8。默认每 rank B1×4096、H5120、两层、E48、TopK8，
专家与共享专家 I1792。默认预热 3 步、测量 5 步。两个后端使用相同模型结构、输入、初始权重及优化器。
普通 mHC 路径为 Torch 公式，不依赖额外 Omni 制品；与此前 Omni 基线的计时不能直接比较。

```bash
CANN_ENV_FILE=/path/to/cann/set_env.sh \
bash hyper_parallel/core/multicore/examples/run_benchmark.sh \
  --mega-kernels moe,gate,mhc --output output/three_kernels.json
```

| 参数 | 可选值或默认值 | 含义 |
| --- | --- | --- |
| `--mega-kernels` | `moe`、`gate`、`mhc` 的逗号分隔子集；默认 `moe,gate,mhc` | Mega 路径启用哪些替换，common 保持普通实现 |
| `--accuracy-mode` | `components`，或 `full_step` | 组件诊断加训练验证，或所选组件整步直接对照 |
| `--warmup-steps` | 3，正整数 | 预热步数 |
| `--measured-steps` | 5，正整数 | 测量样本数 |
| `--output` | JSON 路径 | 保存验证范围、后端选择和计时 |

例如 `--mega-kernels gate` 或 `--mega-kernels moe,gate`。单选仅改变这个组合网络中的实现，
不会移除其原有 mHC 结构；只想迁移单个模块时，应运行前面的独立入口。
默认组件精度检查把 mHC 边界误差与离散 TopK 的整网分歧分开，
不把数值有限或成功完成 optimizer step 等同于 Mega mHC 下的整网自由路由逐元素一致。

多机参数由 `NNODES`、`NPROC_PER_NODE`、`NODE_RANK`、`MASTER_ADDR`、`MASTER_PORT` 指定，
总 rank 必须为 8。`PYTHON_BIN` 选择解释器，`NATIVE_ENV_FILE` 选择配套制品，
启动器会检查 editable 导入来自当前 checkout。所有 rank 正常完成反向后再销毁通信组。

## 高级执行调优与资源管理

下列参数描述执行策略，首次接入可全部省略，不属于网络结构：

| 参数 | 默认策略与用途 |
| --- | --- |
| `local_num_tokens` / `max_local_num_tokens` | 兼容接收并忽略，可同时传入；不预分配，也不限制输入 |
| `dispatch_mode` | 默认 `push`；`pull` 是另一种通信策略，用相同工作负载测量后再选择 |
| `initial_capacity_factor` / `capacity_growth_factor` | push 接收容量与增长策略；pull 不接收这两个选项 |
| `dispatch_split` / `combine_split` | 默认通信切块；由算子调优使用 |
| mHC `token_tile` | 默认调度切块；由算子调优使用 |

自动容量覆盖实际输入和接收负载，无需预判 token 数；增长后保留历史容量，不随小输入缩容。
自动模式与显式上限模式都支持不等长/空输入，要求 EP rank 同序参与，并仅支持 eager。
扩容涉及同步及 SHMEM restart；容量内的长度变化直接复用。

普通模型依靠默认的执行资源托管即可。显式共享仅用于兼容层的串行调用，
不得把共享 workspace 的多个调用并发执行。需要提前释放时，在所有相关 rank 上同序 `close()`，
确保无在途调用及待执行 backward。完整约束见[资源共享与关闭](../README.md#资源共享与关闭)。

性能评估还需要足够预热、固定源/制品、无外部设备进程和一致的工作负载。
短示例成功不代表稳态加速比、HBM 峰值或真实 checkpoint 的整网训练验收。
