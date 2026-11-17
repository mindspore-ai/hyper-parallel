# GroupedExperts 模块

本目录提供 `routed_grouped_experts` 模块族：默认成员拥有专家参数和 routed 算法，EP 成员复用相同数学，只替换 slot 的运输阶段。
模型 / MoE 负责 router、最终路由权重、aux-loss、共享专家及其 gate/mask，以及最终输出组装。

目录按算法与并行成员组织：

```text
grouped_experts/
  grouped_experts.py
  grouped_experts_family.py
  mega_grouped_experts.py
  ep/
    __init__.py
    ep_grouped_experts.py
    ep_mega_grouped_experts.py
    moonep_grouped_experts.py
```

新工厂的显式导入路径为：

```python
from hyper_parallel.components.modules import GroupedExperts
from hyper_parallel.components.modules.moe.grouped_experts import GroupedExpertsConfig
```

`hyper_parallel.components.modules.GroupedExperts` 统一导出本目录的新专家族工厂。
默认本地计算复用现有 NPU grouped-GEMM；Mega/MoonEP 已注册为成员，但执行仍是骨架，选择它们会在构建时明确报错。

## 维度与 slot

| 符号 | 含义 |
| --- | --- |
| `E` | 全局 routed 专家数量，对应 `config.num_experts` |
| `H` | token 输入和输出宽度，对应 `config.hidden_size` |
| `I` | 每个专家 gate/up 分支的中间宽度，对应 `config.intermediate_size` |
| `T` | 来源 rank 的 token 数，即 `hidden_states.shape[:-1]` 各维的乘积 |
| `K` | 每个 token 给定的路由 slot 数，即 top-k，要求 `K >= 1` |
| `S` | 来源 slot 数，`S = T * K` |
| `R` | 当前 rank 经 dispatch 接收的 slot 数，EP 下可以与 `S` 不同 |
| `E_local` | 当前 rank 拥有的专家数；local 为 `E`，普通 EP 为 `E / ep_size` |
| `C` | 本次调用的计算权重表行数；普通 EP 为 `E_local`，MoonEP 规划中还可包含远端专家临时副本 |
| `N_e` | 某个本地专家接收的 slot 数，所有本地专家的 `N_e` 之和为 `R` |

一个 slot 表示“某个 token 分配给某个 expert 的一次计算任务”。例如 `T=2、K=2` 时有四个 slot。
`expert-major` 表示把属于同一专家的 slot 连续排列；它不意味着每个专家的 slot 数相同。
零 token 专家保留在计数数组中，计数为零；零路由权重仍是一个需要执行的 slot，不表示 drop。

## 参数布局

| 参数 | 全局 shape | 含义 |
| --- | --- | --- |
| `gate_up_proj` | `[E,2I,H]` | FC1，第二维的前 `I` 行是 gate，后 `I` 行是 up |
| `down_proj` | `[E,H,I]` | FC2，将 SwiGLU 中间结果映射回 `H` |

对于专家 `e`，默认计算为：

```text
X_e:              [N_e,H]
gate_up_proj[e]:   [2I,H]
FC1 = X_e @ gate_up_proj[e].T               -> [N_e,2I]
gate, up = FC1.chunk(2, dim=-1)             -> 两个 [N_e,I]
activated = silu(gate) * up                 -> [N_e,I]
output = activated @ down_proj[e].T        -> [N_e,H]
```

参数采用 Transformers packed 布局。两组权重以最后两维的转置视图传给 [grouped_matmul](../../../functional/grouped_matmul.py)。
EP 沿专家维 `Shard(0)`，每个 owner 保留它负责的完整专家矩阵。

## 公开入口与初始化

| 接口 | 输入 | 输出 / 行为 |
| --- | --- | --- |
| `GroupedExperts(...)` | `config` 或 `module` 二选一；可选 `module_fqn/context/mesh/megakernel/moonep` | 通过既有构建器返回一个具体 `nn.Module` |
| `GroupedExpertsConfig(...)` | 必填 `num_experts/hidden_size/intermediate_size/num_layers`，可选概率模式和初始化标准差 | 不包含并行规模的只读算法配置 |
| `reset_parameters()` | 固定使用正态初始化，FC2 标准差按模型深度缩放 | 原地初始化 FC1 和 FC2 |

`GroupedExpertsConfig` 的默认概率模式为 `routing_weight_mode="before_fc2"`，初始化配置采用 Megatron 命名：

| 配置项 | 默认入参 | 作用 |
| --- | --- | --- |
| `init_method_std` | `0.02` | FC1 正态初始化与 FC2 深度缩放的基础标准差 |
| `num_layers` | 必填 | 模型的全局 Transformer 层数，在 PP 切分前确定 |

初始化对齐 Megatron 的 [init_method_normal / scaled_init_method_normal](https://github.com/NVIDIA/Megatron-LM/blob/a0ef64615cc3d2574ffbaac70f324ee365f5ed34/megatron/core/utils.py)
与 [TransformerConfig 默认标准差](https://github.com/NVIDIA/Megatron-LM/blob/a0ef64615cc3d2574ffbaac70f324ee365f5ed34/megatron/core/transformer/transformer_config.py)：

| 参数 | 默认分布（`L = config.num_layers`） | 例：`L=32` |
| --- | --- | --- |
| `gate_up_proj`（FC1） | 均值 0，std 为 `init_method_std`，默认 `0.02` | `0.02` |
| `down_proj`（FC2） | 均值 0，std 为 `init_method_std / sqrt(2 * L)` | `0.0025` |

直接构造必须显式提供 `num_layers`；独立单层模块也需要写明 `num_layers=1`。
模型构建器从模型配置解析全局 Transformer 总层数并向下传递，不能使用 PP 本地层数或专家数。
缺少层数时构造立即抛出 `ValueError`，提示通过 `GroupedExpertsConfig(num_layers=...)` 或替换 context 提供。
两组参数均在 `reset_parameters()` 中固定调用 `nn.init.normal_`，FC2 使用 Megatron 的深度缩放公式。
可以用 `init_method_std` 调整基础标准差：

```python
from hyper_parallel.components.modules.moe.grouped_experts import GroupedExpertsConfig

config = GroupedExpertsConfig(
    4, 128, 256, num_layers=32,
    init_method_std=0.01,
)
```

构造和物化后的无参 reset 使用相同配置。
device/dtype 遵循调用方的 Torch 构造上下文，支持 meta。

单模块调用不依赖 Trainer，可直接使用公共工厂。接管已有 packed 专家时传入全局层数：

```python
from hyper_parallel.components.modules import GroupedExperts

experts = GroupedExperts(module=hf_experts, context={"num_layers": 32})
```

EP 作业可向工厂传入已有的 `mesh`，由公共模块族流程完成布局和并行绑定。
只传 `context={"ep_size": ...}` 会选择 EP 成员，实际通信还需要有效的 mesh 和绑定。
使用 meta 构造时，调用方需要在物化后执行初始化或加载权重。

`GroupedExperts` 支持原生构造和 YAML 替换。`module=` 接受兼容的族成员或未分片的无 bias packed SwiGLU 专家。
布局一致时直接复用原参数，并保留 requires_grad 和 train/eval 状态。
所有 `module=` 来源共用参数绑定流程，没有单独的 `_adopt_packed_source` 分支；原生 HF 配置通过关键字匹配生成算法配置，族成员保留已有配置。

### 外部 forward

```python
output = experts(hidden_states, topk_ids, topk_weights)
```

| 输入 / 输出 | shape 与类型 | 含义 |
| --- | --- | --- |
| `hidden_states` | 浮点 Tensor `[...,H]` | 当前来源 rank 的 token，模块内部展平成 `[T,H]` |
| `topk_ids` | int32/int64 Tensor `[T,K]` | 模型已经选择的全局专家 ID，每个 ID 必须在 `[0,E)` |
| `topk_weights` | 浮点 Tensor `[T,K]` | 模型已经完成归一化与缩放的最终路由权重，保留 router 梯度 |
| `output` | 浮点 Tensor `[...,H]` | 已应用概率并完成 top-k 求和的 routed 输出，恢复来源 token 顺序和 shape |

三个输入必须在同一设备。模块不调用 router、不重新 softmax、不重新选 top-k，也不计算 aux-loss 或共享专家。
输出 dtype 与专家计算输出一致；`output` 模式将原始概率 dtype 传给融合 unpermute，累加与舍入遵循 NPU 算子契约。

## 算法阶段方法

阶段命名对齐 [Megatron Core 的 permute / unpermute](https://docs.nvidia.com/megatron-core/developer-guide/latest/apidocs/core/core.transformer.moe.moe_utils.html)：
`permute` 按专家分组并复制 top-k token，`unpermute` 恢复原 token 顺序并合并 top-k 输出。
本模块沿用仓内既有 NPU 融合封装；跨 rank 运输仍由 `dispatch` / `combine` 负责。

完整流程在正常 Module 调用内部执行，阶段之间直接传 Tensor、split 列表和 shape：

```text
forward(hidden_states, topk_ids, topk_weights)
  -> validate_and_flatten_inputs  校验输入并展平 token
  -> permute         NPU permute，并对齐专家 ID / 概率
  -> dispatch        local identity / EP 运输
  -> compute_experts 当前 owner 的专家计算，直接传入当前两组参数
  -> combine         local identity / EP 逆运输
  -> unpermute       NPU unpermute、概率处理与 top-k 求和
```

| 方法 | 输入 | 输出与含义 |
| --- | --- | --- |
| `validate_and_flatten_inputs(hidden_states, topk_ids, topk_weights)` | 外部三 Tensor | Tensor `[T,H]`；校验 shape/device/dtype 后展平 token |
| `permute(hidden_states, topk_ids, topk_weights)` | `[T,H]`、`[T,K]`、`[T,K]` | 分别返回激活 `[S,H]`、全局 IDs `[S]`、计数 `[E]`、可选概率 `[S]`、NPU 逆映射 `[S]` |
| `dispatch(hidden_states, expert_ids, tokens_per_expert, routing_weights=None)` | 来源激活 `[S,H]`、IDs `[S]`、计数 `[E]`、可选概率 `[S]` | 分别返回激活 `[R,H]`、计数 `[E_local]`、可选概率 `[R]`、send_splits、recv_splits、recv_inverse；local 的后三项为 None |
| `compute_experts(hidden_states, tokens_per_expert, gate_up_proj, down_proj, routing_weights=None)` | 激活 `[R,H]`、计数 `[C]`、两组显式权重与可选概率 | Tensor `[R,H]`；同一数学可用于本地参数或临时副本 |
| `combine(values, send_splits, recv_splits, recv_inverse)` | 专家输出 `[R,H]` 与本次 dispatch 返回的三项逆运输参数 | Tensor `[S,H]`；恢复来源 expert-major slot 顺序，不做 top-k 求和 |
| `unpermute(values, unpermute_indices, topk_weights, output_shape)` | `[S,H]`、NPU 逆映射 `[S]`、原始概率 `[T,K]`、原始 shape | Tensor `[...,H]`；调用既有 NPU unpermute，加权/合并并恢复 shape |
| `forward(hidden_states, topk_ids, topk_weights)` | 外部三 Tensor | Tensor `[...,H]`；直接组织上述阶段，Mega/MoonEP 覆盖此入口 |

`forward` 直接调用 `compute_experts` 并传入当前两组参数；计算入口在 grouped-GEMM 边界把 DTensor 转成本地 Tensor。
`to_local()` 仅返回当前本地张量，不执行通信、unshard、预取或资源保护。
unshard、等待预取和参数生命周期由现有 [HSDP hooks](../../../../core/fully_shard/hsdp_scheduler.py) 负责。
计算入口校验 `[C,2I,H]` / `[C,H,I]` 的完整专家矩阵，专家切分沿 EP 专家维。
全空输入绕过空 grouped-GEMM，通过空矩阵乘法保留输入与两组参数的梯度连接。
来源 token 为空时也绕过 NPU permute/unpermute，保留空输出及对应梯度连接。
正常训练调用 `experts(...)`，保留 FSDP 等 Module hooks 的生命周期。
专家成员不提供额外 `state_contract()`；local/EP 重选使用同一只读配置快照，并由现有框架校验参数和 buffer 的名称、shape，
复用原 Parameter，保留 requires_grad 与 train/eval 状态。普通 state_dict 的参数键不变。

### 概率只在约定位置应用一次

| `routing_weight_mode` | 加权位置 | 单个 slot 的数学含义 |
| --- | --- | --- |
| `before_fc2`（默认） | `compute_experts` 在激活后、FC2 前 | `FC2_e(p * SwiGLU(FC1_e(x)))` |
| `output` | `unpermute` 在 FC2 后 | `p * Expert_e(x)`，随后 top-k 求和 |

默认 `before_fc2` 把可微概率随 slot 运输到专家 owner；`output` 保留原始 `[T,K]` 权重，在来源侧恢复时应用。
恢复时 `before_fc2` 传同形状的全 1 矩阵，仅做 top-k 合并。
融合算子在 probs=None 时按 top-k=1 处理，因此多专家恢复不能省略这个形状信息，也不能重复传真实概率。
模型给定的概率归一化、缩放与加权位置均保持不变。

## 逐调用参数

阶段之间直接传递下列值，不再定义请求、batch、权重或 receipt 数据类，也不以字典替代这些封装。
多项结果按函数签名顺序分别返回和解包；原始 topk_weights 与 output_shape 保留在 forward 的局部变量中。

| 参数 | shape / 类型 | 含义 |
| --- | --- | --- |
| `hidden_states` | `[T,H]`、`[S,H]` 或 `[R,H]` | 当前阶段的 token / expert-major slot 激活 |
| `expert_ids` | `[S]` | 来源 slot 的全局专家 ID，按升序排列；接收后只需本地专家计数 |
| `tokens_per_expert` | `[E]`、`[E_local]` 或 `[C]` | 与当前计算权重行号对应的 slot 数，保留零计数 |
| `routing_weights` | `[S]` / `[R]` 或 None | 仅 before_fc2 携带的逐 slot 可微概率 |
| `gate_up_proj` / `down_proj` | `[C,2I,H]` / `[C,H,I]` | 本次计算的两组权重，直接作为 Tensor 参数传入 |
| `unpermute_indices` | `[S]` | NPU 返回的逆映射：原始展平 route slot 下标 -> expert-major 行号 |
| `topk_weights` | `[T,K]` | 原始 token/top-k 顺序的概率，用于来源侧 unpermute |
| `output_shape` | `tuple[int,...]` | 最终恢复的原始 token shape |
| `send_splits` / `recv_splits` | 长度为 EP 的 list | 按 EP group rank 顺序组织的发送 / 接收 slot 数 |
| `recv_inverse` | `int32 [R]` | 接收侧 NPU permute 返回的逆映射，从 peer-major 行号指向本地 expert-major 行号 |

这些参数不主动 clone 或 detach；梯度沿 Tensor 计算图返回。
逆映射和 split 必须取自同一次 dispatch，保留在本次 forward 中，不写入模块的 last_dispatch 缓存。
普通 EP 的计算权重仅包含本 rank 的 home 参数；MoonEP 后续可直接传入含临时副本的两组权重。

## EP 成员与通信接口

实现见 [ep/ep_grouped_experts.py](ep/ep_grouped_experts.py) 与 [EPContext](../../../../distributed/expert_parallel/ep_context.py)。

| 方法 | 输入 | 输出与含义 |
| --- | --- | --- |
| `EPContext(mesh, num_experts=E)` | 已有一维 EP mesh 和全局专家数 | 借用作业资源，要求 E 可被 EP 整除，按连续专家区间确定 owner |
| `EPGroupedExperts.bind_parallel(binding)` | 既有作业的 `ParallelBinding` | 返回 None；借用专家数匹配的 EPContext，不创建通信组 |
| `EPGroupedExperts.parallel_module_spec()` | 无 | 声明 grouped_experts.slots.ep 协议、直接参数签名及布局校验 |
| `validate_ep_layout(...)` | 模块、sharding 声明与逻辑布局上下文 | 检查完整参数/矩阵及仅专家维 Shard(0)，允许 dense TP |
| `EPContext.dispatch(hidden_states, expert_ids, routing_weights=None)` | 来源激活 `[S,H]`、升序全局 IDs `[S]`、可选概率 `[S]` | 激活 `[R,H]`、计数 `[E_local]`、可选概率 `[R]` 和三项逆运输参数 |
| `EPContext.combine(values, send_splits, recv_splits, recv_inverse)` | `[R,H]` 与同次 dispatch 的三项参数 | `[S,H]`；校验行数、split 长度及映射 shape/dtype/device，执行逆运输 |

每个 peer 的计数交换同时携带“是否运输概率”的标记，确保所有 rank 使用同一个可选概率运输协议。
已按全局专家 ID 分组的 slot 直接发往持有连续专家区间的目标 rank。NPU 接收后用 top-k=1 的 permute 按本地专家重排，回程通信前用配对的 unpermute 恢复接收顺序。CPU 等设备使用 Torch 排序和索引实现同一映射。
空发送、空接收仍参与相同集体顺序。空接收跳过本地融合重排，激活和必要概率使用现有可微 A2A。
通信层不应用概率、不计算专家，也不将多个 slot 归并为 token。
调用方在 dispatch 前展开并排序 slot，在 combine 后恢复 token 顺序并归并 top-k。

### 默认采用 TP-extend-EP

dense 层可以启用 TP；专家区域复用已有作业的派生 EP mesh，专家矩阵本身不再做 TP 切分，即 ETP=1。
例如 dense mesh 为 `dp=4,tp=2`、`ep_size=4` 时，既有 mesh 路径将 8 个 rank 重排为 `edp=2,ep=4`；
两个 EP 组分别覆盖 `{0,1,2,3}` 和 `{4,5,6,7}`，每个 rank 拥有 `E/4` 个完整专家。

族布局仍为两组参数各自 `{"ep": Shard(0)}`，不能再叠加 `{"tp": Shard(...)}`。
验证的是参数归属和矩阵布局，不限制整个作业的 tp_size。EPContext 借用 fsdp_moe_mesh 的 ep 轴，不另建或叠乘一个 TP 通信组。

调用方传入当前 rank 的本地 token/路由，hidden 的 `H` 必须完整；输出返回同一本地 token 分片。
dense/SP 边界的输入输出布局由模型声明，本模块不猜测 DTensor 布局或重复复制 token。
这里支持的是 TP-extend-EP，不包括将单个专家的 H/I 再切开的 ETP。

## MoonEP 接入骨架

[ep/moonep_grouped_experts.py](ep/moonep_grouped_experts.py) 定义 MoonEPGroupedExperts，继承规范参数和共享计算，
直接覆盖三 Tensor forward，把 token、IDs、概率和两组权重直接传给 runtime。它不先执行普通 permute，也不在 token-level combine 之后重复 unpermute。
[moonep_context.py](../../../../distributed/expert_parallel/moonep_context.py) 预留以下 runtime 接口；当前均明确抛出 NotImplementedError。

| 方法 | 输入 | 输出 / 职责 |
| --- | --- | --- |
| `plan(hidden_states, topk_ids, topk_weights)` | 原始 token 与已选路由 | 独立 opaque plan，保存 home owner、计算行号与运输关系，供 backward/recompute 复用 |
| `dispatch(hidden_states, topk_ids, topk_weights, plan, routing_weight_mode=...)` | token、路由、本次 plan 和概率模式 | 分别返回激活 `[R,H]`、计数 `[C]`、可选概率 `[R]` |
| `prefetch_weights(gate_up_proj, down_proj, plan)` | FSDP 已物化的两组 home 权重 | 两个 Tensor：包含临时副本的 `[C,2I,H]` / `[C,H,I]` |
| `combine(values, topk_weights, output_shape, plan, routing_weight_mode=...)` | 计算输出、原始概率/shape、本次 plan 和概率模式 | 恢复 output_shape，概率及 top-k 求和仅执行一次 |
| `reduce_grad(gate_up_grad, down_grad, plan)` | 两组计算表梯度与本次 plan | 分别返回 home-local 梯度；未来 autograd 桥在 FSDP 梯度归约前完成副本归还 |

parallel_module_spec 声明独立的 grouped_experts.forward.moonep 协议，并通过布尔选项 `moonep=True` 选择相应 context。
普通 EP 和 MoonEP 共享参数 owner 布局；临时副本不作为新参数加入 checkpoint/optimizer。
MoonEP 规划的 `prefetch_weights` 用于把其他 EP owner 的专家临时复制到计算 rank，和 FSDP 的存储 unshard/预取是不同职责；当前仅保留骨架。
当前工厂会在参数分配前拒绝 MoonEP 骨架，不会静默退回普通 EP；本期没有接入 MoonEP 外部依赖或实际 planner/通信。

## 模块族选型与布局

[grouped_experts_family.py](grouped_experts_family.py) 注册一个族、多个具体成员；每个模型位置仍有自己的 Experts 实例和参数。
公开选项为 `megakernel: bool = False`、`moonep: bool = False`。
是否使用 EP 从既有作业 ep_size 推导，不重复作为算法配置传入。
参数和计算沿用调用方的模型 dtype（如 BF16/FP16）。

| megakernel | moonep | EP | 成员 | 当前状态 |
| --- | --- | --- | --- | --- |
| false | false | 1 | 默认算法成员 / local | 默认本地计算已实现 |
| false | false | >1 | `EPGroupedExperts` / ep | 复用默认数学，slot 运输已实现 |
| true | false | 1 | `MegaGroupedExperts` / mega | 已注册，构建时报内核未实现 |
| true | false | >1 | `EPMegaGroupedExperts` / ep_mega | 已注册，构建时报内核未实现 |
| false | true | >1 | `MoonEPGroupedExperts` / ep_moonep | 已注册，构建时报 runtime 未实现 |

没有对应成员的组合直接拒绝，不静默回退。
Mega 覆盖 `forward`，后续以一次 core executor 调用融合运输和计算，避免重复执行普通 EP 阶段。

| 族内 provider | 输入 | 输出与含义 |
| --- | --- | --- |
| `validate_grouped_experts(...)` | config/source、成员与构建上下文 | 返回 None；检查规范来源、E%EP=0，并拒绝未实现成员 |
| `grouped_experts_sharding(...)` | 模块、FQN、成员与逻辑布局上下文 | `ModuleShardingSpec`；完整声明参数和边界 I/O 的计算布局 |

local 对每个参数声明 `{}`，EP 对两组参数声明 `{"ep": Shard(0)}`。
`in_src={}, in_dst={}, is_boundary=False` 表示显式阶段自己负责激活运输；FSDP 存储分片由既有训练框架处理。

## 使用示例

### 本地 NPU 调用

以下示例需要可用的 Torch / torch_npu 环境。

```python
import torch

from hyper_parallel.components.modules import GroupedExperts
from hyper_parallel.components.modules.moe.grouped_experts import GroupedExpertsConfig

config = GroupedExpertsConfig(4, 128, 256, init_method_std=0.02, num_layers=32)
experts = GroupedExperts(config=config).to(device="npu", dtype=torch.bfloat16)
experts.reset_parameters()  # 可选；FC1 std=0.02，FC2 std=0.0025

hidden = torch.randn(3, 128, device="npu", dtype=torch.bfloat16)
ids = torch.tensor([[0, 2], [1, 0], [2, 1]], device="npu")
weights = torch.rand(3, 2, device="npu", requires_grad=True)
output = experts(hidden, ids, weights)  # [3,128]
```

真实模型传入 router 已经确定的 IDs 和最终 weights，基础标准差和模型总层数从模型配置映射。
已有作业 mesh 时可传 `mesh=job_mesh`；实际 EP 执行需要正确的参数布局和资源绑定。
逻辑/meta mesh 仅用于规划，不能直接执行专家计算。

### YAML 替换兼容来源

```yaml
accelerator:
  tp_size: 2
  ep_size: 4

plan_overrides:
  - match: "model.layers.*.mlp.experts"
    replace:
      kind: module
      implementation:
        _target_: hyper_parallel.components.modules.GroupedExperts
        megakernel: false
        moonep: false
```

该片段适用于未分片的 bias-free packed 专家。原生构造则使用 `GroupedExperts(config=...)`。
YAML 请求经过既有 replacement 规划与 build_family，结合作业规模选择 local/ep，再进行布局和绑定。
模型 spec 也可提供默认替换声明，无需每层填写 YAML；直接通过公开工厂组合的族在托管构建时确定最终成员，不再叠加显式模块替换。
将 moonep 改为 true 可声明 MoonEP 请求，但当前骨架会在构建时明确拒绝；它与 Mega 的组合尚未开放。

## HF 模型接入

`validate_ep_layout` 检查的是最终族成员相对自身的 `sharding.params`，不是整个模型的参数表。
两组参数名相同也不足以证明兼容：还必须满足权重方向、无 bias SwiGLU、完整专家矩阵和仅 EP Shard(0)。

| 来源 | 原始专家结构 | 本次接入范围 |
| --- | --- | --- |
| 新族的 local / EP 成员 | `[E,2I,H]` / `[E,H,I]` | 复用完整专家参数 |
| Qwen3-MoE | Transformers 5.13.x packed SwiGLU | 当前 recipe 选择专家族，直接复用原参数和 checkpoint 键 |
| Qwen3-VL-MoE | Transformers 5.13.x packed SwiGLU，top-k `[T,K]` | 来源接口兼容；选择新族时保留文本塔原生 router |
| Qwen3.5/3.6-MoE、DeepSeek-V3/V3.2 | 新版 Transformers packed SwiGLU | 当前普通配方选择新族；移除同一路径上的旧父层 EP 方法替换 |
| Qwen2-MoE、DeepSeek-V2 | 新版 Transformers packed SwiGLU | 选择新族后由 planner 自动推导父层布局，可使用上面的显式替换规则 |
| Kimi K2.5/K2.6 native 文本塔 | 新版 Transformers DeepSeek-V3 packed SwiGLU | 当前 recipe 使用专家族；保留原生 router 和共享专家 |
| DeepSeek-V4.1 普通专家 | 相同 packed 权重，额外 clamp/FP32 激活 | 模型 adapter 注入原生激活；现有 MegaMoe 示例配置保持独立 |
| Kimi-K3 | 官方 remote 实现为 ModuleList，Situ 激活，可带 latent MoE | 尚未迁移旧 ModuleList，不注册新专家族替换 |
| Kimi Linear 官方 remote 实现 | ModuleList，每个专家持有 w1/w2/w3 | 当前普通 packed 接管入口不支持 |

packed 接口参见
[Qwen3-MoE 5.13.1](https://github.com/huggingface/transformers/blob/v5.13.1/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py)
与 [Qwen3-VL-MoE 5.13.1](https://github.com/huggingface/transformers/blob/v5.13.1/src/transformers/models/qwen3_vl_moe/modeling_qwen3_vl_moe.py)。

本次 models 接入按 Transformers 5.13.1 的专家接口核对，兼容接口相同的 5.11 来源：
`gate_up_proj [E,2I,H]`、`down_proj [E,H,I]`，父 MoE 传入 token `[T,H]`、专家 ID `[T,K]` 和最终概率 `[T,K]`。
普通 HF packed 来源已满足该权重布局，接管入口不执行 stack、cat、转置复制或重新初始化；它校验并注册原 Parameter，
从来源/祖先 config 取得模型层数与初始化配置，并兼容 forward 关键字名。
HF 的 `top_k_index/top_k_weights` 与当前内部 `topk_ids/topk_weights` 位置语义相同；由统一匹配函数处理，
不再根据来源 forward 签名动态绑定另一层入口。
旧版 ModuleList、dense 路由分数接口和已有 NPU 转置权重不作为公共模块的兼容基线。
DeepSeek-V3.2 的两份配方、Qwen3.6 的文本/图文及两种 CP 配方均改为直接选择新族，并去掉旧父层 EP 方法替换。
DeepSeek-V3 的 MLA 配方也加入普通专家族；原 router 和共享专家继续由 HF 父层执行。
父 MoE 布局由 planner 和默认模板统一推导，各模型 spec 不再注册相同的父层 provider。
旧类通过 `hyper_parallel.components.modules.grouped_experts.GroupedExperts` 显式访问；
其低精选择、量化工厂和 MegaMoe 路径保留。公共 `components.modules.GroupedExperts` 统一选择新专家族。

### 配置和 forward 关键字匹配

配置由 `_match_expert_config_value` 统一匹配，来源模块优先，祖先配置由近到远；支持对象和 Mapping。

| 内部字段 | 已识别的来源 key |
| --- | --- |
| num_experts | num_experts、num_local_experts、n_routed_experts |
| hidden_size | hidden_dim、hidden_size |
| intermediate_size | intermediate_dim、moe_intermediate_size、intermediate_size |
| num_layers | num_hidden_layers、num_layers |
| init_method_std | init_method_std、initializer_range |
| hidden_act | hidden_act、hidden_activation、mlp_hidden_act |

`forward` 调用 `match_grouped_experts_inputs`，保持三个位置参数的语义，同时接受以下关键字：

| 内部字段 | 兼容 key |
| --- | --- |
| topk_ids | top_k_index、topk_indices、selected_experts、router_indices、topk_idx |
| topk_weights | top_k_weights、routing_weights、topk_weight |

匹配只返回原 Tensor，不复制、不选 top-k、不把 dense `[T,E]` 分数转换成概率。
同一个字段被位置/规范关键字与别名重复提供、缺少输入或出现未知 key 时明确报错。
所有名称最终仍须满足 `[T,K]` 的 ID/概率契约。

### Transformers 版本与旧版兼容边界

仓库最低依赖保持 **`transformers>=5.11.0`**，不设置版本上限。本次普通专家接口以 **5.13.1** 为固定核对版本。
具体模型还需要该版本实际包含其原生实现；最低依赖满足不代表所有模型均可构建或已通过训练验收。
本轮核对所有下列 HF 原生专家类的 **5.13.1 固定 tag 源码**，不依赖旧版类名或 forward 签名的动态绑定。
下表中的历史运行记录来自仓内已有文档；源码核对和离线构建不表示本轮已通过完整模型或多卡训练验收。

| 模型 | 5.13.1 对接来源 | 普通专家族接入范围 | 仓内历史运行版本 |
| --- | --- | --- | --- |
| Qwen3-MoE | Qwen3MoeExperts | 主配方和训练示例选择新族 | 未统一固定 |
| Qwen3-VL-MoE | Qwen3VLMoeTextExperts | 按文本塔路径显式选择新族，父层布局自动推导 | 未统一固定 |
| Qwen3.5/3.6-MoE | Qwen3_5MoeExperts；Qwen3.6 复用 qwen3_5_moe 类型 | Qwen3.6 四份普通配方及文本/图文 spec 接入 | Qwen3.6 示例记录 5.13.0 |
| DeepSeek-V3.2 | DeepseekV32Experts | 两份普通配方接入 | 5.13.1 |
| DeepSeek-V3 | DeepseekV3Experts | MLA 普通配方接入；量化工厂保留 | 未统一固定 |
| DeepSeek-V2 | DeepseekV2Experts | 显式替换选择新族，父层布局自动推导 | 未统一固定 |
| Qwen2-MoE | Qwen2MoeExperts | 显式替换选择新族，父层布局自动推导 | 未统一固定 |
| Kimi-K2.5/K2.6 native | kimi_k25 外壳，DeepseekV3Experts 文本塔 | 当前主配方接入 | Kimi-K2.6 文档记录 5.13.0 |
| DeepSeek-V4.1 普通专家 | 仓内自研模型复用 DeepseekV4Experts | 专用 adapter 注入 clamp/FP32 激活；MegaMoe 示例不切换 | 未提供完整模型统一验收版本 |
| Kimi-K3 | 官方 remote 实现；没有 HF 原生 K3 专家类 | 旧 ModuleList 未迁移，不注册新专家族替换 | 官方 text config 记录 4.56.2，不能当作 5.13 运行验收 |
| Kimi Linear remote | 官方 remote 模型，ModuleList w1/w2/w3 | 未接管旧专家结构；整改为兼容专家后才可使用默认推导 | 官方配置记录 4.57.1，未完成 5.13 模型整改 |

旧版结论按具体 tag 和接入层级区分，不能把某个模型“曾可加载”理解为新专家族可直接替换：

| 模型 | 旧 4.57.1 | 此前的 5.11.0 |
| --- | --- | --- |
| Qwen2-MoE、Qwen3-MoE、DeepSeek-V2/V3 | 专家为 ModuleList，父层逐专家调用；与新族 packed 参数及三 Tensor 调用不兼容 | packed 专家契约兼容；V3 类名为 DeepseekV3NaiveMoe，专家接口保留兼容 |
| Qwen3-VL-MoE | 权重为 `[E,H,2I]` / `[E,I,H]`；调用顺序是 hidden、dense `[T,E]` 概率、IDs，输出含额外维度；不能通过关键字别名适配 | 已是 `[E,2I,H]` / `[E,H,I]` 和 top-k 三 Tensor，专家契约兼容 |
| Qwen3.5/3.6-MoE | 该 tag 不含 qwen3_5_moe 模型实现 | qwen3_5_moe 专家契约兼容；这不代表已验证整个 Qwen3.6 模型 |
| DeepSeek-V3.2 | 该 tag 不含 deepseek_v32 模型实现 | packed 专家契约兼容，类名为 DeepseekV32NaiveMoe |
| DeepSeek-V4.1 | 该 tag 不含所依赖的 deepseek_v4 实现 | DeepseekV4Experts 已存在；仍须专用激活 adapter，整模型跨版本未验收 |
| Kimi-K2.5/K2.6 native | 该 tag 不含 kimi_k25 | 仍无 kimi_k25；即使文本专家兼容，也无法构建此 native 整模型 |
| Kimi-K3 / Kimi Linear remote | 属于独立 remote 实现，旧 ModuleList 接口未接管 | 升级 Transformers 不会自动转换 ModuleList、训练分支或 checkpoint；仍需模型整改 |

上表的旧版“不兼容”指当前新族入口无法直接接管，或版本中缺少所需原生模型，
不表示模型使用其旧实现也不能运行。5.11 的专家契约相同项可复用本入口；不同模型的原生类是否存在，以及完整训练能力，需要分别确认。
MegaKernel 和低精实现继续走各自既有入口，未改为调用本普通专家族。

固定源码依据：
[Qwen3.5-MoE 5.13.1](https://github.com/huggingface/transformers/blob/v5.13.1/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py)、
[DeepSeek-V3.2 5.13.1](https://github.com/huggingface/transformers/blob/v5.13.1/src/transformers/models/deepseek_v32/modeling_deepseek_v32.py)、
[Kimi-K2.5 5.13.1](https://github.com/huggingface/transformers/blob/v5.13.1/src/transformers/models/kimi_k25/modeling_kimi_k25.py)、
[DeepSeek-V4 5.13.1](https://github.com/huggingface/transformers/blob/v5.13.1/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py)。
旧版接口参见
[Qwen3-MoE 4.57.1](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py)、
[Qwen3-VL-MoE 4.57.1](https://github.com/huggingface/transformers/blob/v4.57.1/src/transformers/models/qwen3_vl_moe/modeling_qwen3_vl_moe.py)。
Kimi-K3 的版本字段和当前接口见
[官方 config](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/config.json)
及 [官方模型源码](https://huggingface.co/moonshotai/Kimi-K3/blob/f831ab66814297da540d832a5235f8e904f29d06/modeling_kimi_linear.py)。

### 模型特有的激活适配

[DeepSeek-V4.1 adapter](../../../../models/deepseek_v41/adapter/distributed/moe_engram_expert_parallel.py)
提供 `replace_deepseek_v41_grouped_experts`。它复用 [deepseek_v41_swiglu](../../../../models/deepseek_v41/modeling_deepseek_v41.py)，
在 FP32 中执行 gate 上界 clamp、up 双边 clamp 和 SwiGLU，FC2 前恢复输入 dtype；非正 limit 关闭 clamp。
普通 `F.silu(gate) * up` 无法保留这些语义，因此由模型工厂通过 activation 回调注入，local/EP 共用。
这与 HF 原始逐专家循环未显式转 FP32 的实现存在精度路径差异，不能据此声称逐位对齐。

Kimi-K3 尚未注册新专家族替换。旧 w1/w3 合并与 w2 堆叠、checkpoint 映射、SituAndMul 激活、
父层的三 Tensor 调用及训练分支仍需在模型整改中一起完成。若存在 latent MoE，专家的 H 应是
`routed_expert_hidden_size`；前后投影、norm、router 和共享专家保留在父层。

Qwen3-MoE 和 Qwen3-VL-MoE 使用上面的族替换规则，后者按实际文本塔 FQN 调整 `match`。
接入 packed 专家时在 FC2 后应用路由权重。
新入口替代旧 grouped-expert 替换和 `*.mlp.forward` 的旧 EP 方法替换；同一专家路径只选择一种通信组织。
router、aux-loss、shared experts、共享 gate/mask 和 MoE 返回结构继续由原模型实现。

父 MoE 边界由 [planner.py](../../../../distributed/_builder/planner.py) 先根据完整参数角色识别，
再从待分配的参数中排除子模块族已经拥有的参数；子专家的独立所有权不会抹掉父层的 MoE 语义。
当专家所有者声明无外部 I/O 边界、完整专家矩阵，以及必要的 EP 通信协议时，planner 选择
[默认模板](../../../../distributed/_builder/default_templates.py) 中的 `moe_mlp_routed`。
判断复用已有布局和通信声明，不比较具体模型或模块族名称，也不要求模型 spec 注册父层布局 provider。
父层统一拥有尚未被子模块声明的 router、共享专家和其他参数，按本地 token 分区复制计算。

SP 开启时保留 TP token 分片；dense SP 关闭但同时启用 TP/EP 时，在 MoE 入口按 TP 切分 token、出口恢复，
避免在扩展 EP 域重复执行相同 token。CP token 分片保持不变，H 保持完整。
专家算法已产生完整输出，父层输出不再声明为 TP Partial，也不重复执行旧 EP 包装。
旧 `moe_mlp` 模板继续服务原有实现；未声明独立专家所有权和运输的路径不会选择新模板。

工厂在 sharding、backward 和 optimizer 构建前安装专家族。它直接使用 `experts` FQN，并复用 packed Parameter。
构建器在 PP 切分前一次向下传递配置，将全局层数保存在 `context["num_layers"]`，
并以只读的 `context["module_configs"]` 元组保存由近到远的 config 引用；不保存父 Module 或读取运行中的 token。
层数解析支持 `num_hidden_layers`、`num_layers` 和文本子配置。模型配置提供层数后，YAML 无需重复配置。
构建 packed 专家时按同一配置顺序读取 `init_method_std` 或 HF 的 `initializer_range`，均未提供时使用 0.02。
`initializer_range` 是来源 HF 配置字段，缺少全局层数会明确报错。
独立使用 adapter 时通过 `context["num_layers"]` 提供模型深度；直接构造时填写 `GroupedExpertsConfig.num_layers`。
这些配置用于新建权重或物化后的 reset；转换不会重新初始化已有来源权重。

### 构建器与默认边界模板

| 文件 | 职责与生效范围 |
| --- | --- |
| [grouped_experts_family.py](grouped_experts_family.py) | 选择 local/EP 成员，仅声明专家自身的参数所有权 |
| [planner.py](../../../../distributed/_builder/planner.py) | 保留完整 MoE 结构语义，再分配参数所有权；依据已有布局和通信能力选择父层模板 |
| [default_templates.py](../../../../distributed/_builder/default_templates.py) | `moe_mlp_routed` 声明父层复制参数、TP/CP token 边界、完整输出及本地执行 |
| [models/replacement.py](../../../../models/replacement.py) | 按模块路径收集来源及祖先 config，不解析模型类型、层数字段或多模态分支 |
| [family_layout.py](../../../../distributed/_builder/family_layout.py) | 保留原校验，模块族布局不能注入 local_compute_fn、inner_wrapper、inner_target 或 region_dispatch |
| [module_binding.py](../../../../distributed/module_binding.py) | 绑定普通 EP 或预留 MoonEP runtime，复用作业已有 EP mesh |

`components/modules/moe/sharding.py` 及各模型中重复的父层 provider 注册已删除。
模板只声明布局，不运行 router、不改写模型 forward；模型 adapter 继续处理激活等模型特有差异。
当前父层默认契约是 `hidden_states` 输入和首项 hidden 输出，共享专家在本地 token 分片上复制计算。
`region_dispatch=False` 由父层默认模板声明，沿用既有 local-region 执行机制；
它不再经过模块族布局 provider，因此 `family_layout.py` 无需为此修改 `injection_fields` 或放宽校验。

HF packed 来源使用 `output`，保留原模型在 FC2 后应用概率的位置。
这不改变直接构造 `GroupedExpertsConfig` 时 `before_fc2` 的默认值。两种位置在无 bias 的实数运算中等价，
有限精度舍入不同，不能把构建检查当成数值对齐验收。

## EP 声明与绑定的职责

`parallel_module_spec(self)` 没有外部入参；`self` 是已经构造并持有算法 config 的具体 EP 成员。
它返回声明，不执行通信或参数切分：

| 字段 | 含义与消费者 |
| --- | --- |
| `protocol="grouped_experts.slots.ep"` | 给当前 slot EP 执行协议命名，供布局/绑定一致性检查使用 |
| `binding_kind="ep"` | 请求作业已有 EP mesh 上的 EPContext |
| `stages["dispatch"]` | 声明真实类方法及其参数名：来源激活、专家 IDs、来源计数、可选概率 |
| `stages["combine"]` | 声明真实类方法及其参数名：专家结果、发送/接收 split 和接收侧逆映射 |
| `validate=validate_ep_layout` | 保存布局校验回调；构建器在拿到完整 sharding/context 后调用，不在此处执行 |
| `options={"num_experts": self.config.num_experts}` | 把全局专家数交给 binder，用于校验所有权并构造匹配的 EPContext |

未显式传入的 `version` 使用 `"1"`，`validate_materialized` 使用 None。
公共 binder 校验这些真实方法的签名、最终参数 ownership 和作业 mesh，然后调用 `bind_parallel(binding)` 注入 context。
`bind_parallel` 只保存专家数匹配的借用资源，并拒绝绑定不同的 owner；不新建通信组、不改写 forward、不执行 FSDP unshard。
初始化时 `ep_context=None` 使 meta / 离线构建不依赖分布式运行环境；真正的 dispatch / combine 使用绑定后的 context。

从 `module=` 构建时，`self.training = module.training` 保留来源的 train/eval 状态，包括 eval 下的 False；
从 `config=` 新建时，`nn.Module.__init__()` 已设置 `training=True`，因此不需要再调用一次 `train(True)`。

## 当前边界

默认实现是无 bias SwiGLU 的 NPU grouped-GEMM，允许 dense TP 配合作业的 TP-extend-EP 布局。
GPU、Mega 融合内核、MoonEP runtime、专家内部 ETP、任意 HF 来源和旧训练状态迁移尚未完成。
router、aux-loss、共享专家、padding mask 和通信 overlap 调度属于模型组装；本期三 Tensor 契约不支持 capacity/drop 哨兵。
本页说明当前代码接口与语义，不代表已通过本次数值、多卡训练或性能验收。
