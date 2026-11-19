# Unified Execution：MoE 数值一致性与 EP 实现

MoE 两端共用 HP 的 router、专家权重语义和专家计算。训练使用可微动态路由，推理使用
MC2 或 AllGather＋有效行 GMM。通信和路由算子可以不同，但必须保留每条路由的数值含义和最终合并规则。
完整模型构建、Attention、CP/PCP 与发布流程见 [统一执行说明](unified_execution_guide.md)。

## 数值契约

设本 rank 有 T 个 token、每 token 选择 K 个专家、hidden size 为 H、专家总数为 E，EP 度为 P：

| 张量 | shape | 含义 |
| --- | --- | --- |
| `states` | `[T,H]` | 进入 MoE 的本地 token 行 |
| `indices` | `[T,K]` | 原始 top-k 槽位顺序中的全局专家 ID |
| `weights` | `[T,K]` | router 输出的原始 FP32 权重 |
| `routes` | `[T,K,H]` | 每个槽位对应的未加权专家输出 |
| 输出 | `[T,H]` | 按固定顺序合并后的结果 |

每条 `(source rank, token, top-k slot)` 路由恰好计算一次；不丢弃拥塞专家的 token。
专家使用 HP `GroupedExperts` 的 EIO 权重布局、SwiGLU 和 grouped GEMM。
专家输出回到源 token 后，遵循 [ordered_combine()](../rl/unified_execution/ops/moe/combine.py) 的顺序：

```python
result = torch.zeros_like(routes[:, 0], dtype=torch.float32)
for slot in range(routes.shape[1]):
    product = routes[:, slot].float() * weights[:, slot, None].float()
    result = result + product
return result.to(routes.dtype)
```

不能提前把 router 权重转成 BF16，不能把各 owner 的局部加权和用 AllReduce 合并，
也不能未经验证把槽位循环替换成 `sum`/`index_add_`。这些操作在数学上等价，但可能改变舍入顺序。
MC2 使用原生 combine；它已在验收形状上与上述结果逐位对照，不要求两端调用同一个 Python 函数。

## 绑定和执行边界

[models/bindings.py](../rl/unified_execution/models/bindings.py) 的 `bind_expert_ownership()`
复用 HP 的本地专家绑定和模型族 router，返回 EP group、rank、world size 和本地专家数。

| 角色 | 入口 | 执行路径 |
| --- | --- | --- |
| 训练 | [unified_ep_compute_fn()](../rl/unified_execution/runtimes/hyper_parallel/moe.py) | router → `training_ep` → 固定顺序合并 |
| 推理 | [inference_ep_compute_fn()](../rl/unified_execution/runtimes/vllm/moe.py) | router → 注册的 `InferenceExpertExecutor` → MC2/AllGather |

角色在模型构建时确定；Actor 调用 `eval()` 计算概率不会切换为推理通信。
worker builder 从共享配方生成内部推理绑定，保留原 Actor 配方和指纹。
`validate_training_ep()` / `validate_inference_ep()` 检查真实闭包、router、专家归属、通信组和图注册对象，
而非只检查模块名或一张静态配置表。

## 专家域与权重所有权

训练继续使用 HP 的专家 mesh；CP 可以参与专家域，不另设一套专家分片。
推理在原生 `DP × PCP × TP` 的 rank 顺序上构建 HP mesh，EP 度必须整除该域：

```text
DP1 × PCP2 × TP2：rank 顺序 [0,1,2,3]
EP2 → [0,1] 与 [2,3]，每组各有完整专家集合的两个分片
EP4 → [0,1,2,3]，专家分片跨两个 PCP 平面
```

较小的 EP 自然落在 TP 内，较大的 EP 跨 PCP 或 DP；没有额外的“跨 PCP”开关。
PCP1 时退化为 DP×TP。推理把 PCP 折入 HP 的非 TP 构建轴，但 HP 模型 CP 仍为 1，
不改变 vLLM 调度器的 DP/PCP 语义。

一个 EP rank 拥有连续的 E/P 个专家。
[parallel.py](../rl/unified_execution/runtimes/vllm/parallel.py) 负责通信域，
[LogicalWeightLoader](../rl/unified_execution/runtimes/vllm/weights.py) 按同一布局解释权重。
逻辑 `experts.e.gate_proj/up_proj/down_proj` 转换到本地 EIO 的 gate/up 与 down 区域，
只复制本 rank 所有的专家。发布接收端覆盖 DP×PCP 副本坐标，并保留实际 TP 宽度。

更新使用原址 `copy_`，不替换 Parameter；图中的地址保持有效。EP1 的 `when: ep` 规则不生效，
继续原本的本地专家路径。当前不启用 vLLM 原生 EP/EPLB，也没有专家内部 ETP/PIK。

## 训练：动态 AllToAll 与可微还原

[training_ep.py](../rl/unified_execution/ops/moe/training_ep.py) 复用普通训练 EP 的
`_prepare_ep_dispatch()` 和 `_run_ep_local_experts()`：

```text
states / 原始 IDs / FP32 weights
    → 动态路由与变长 AllToAll
    → module.experts(...)：保留模块调用、FSDP hooks 和 autograd
    → 逆 AllToAll 返回路由结果
    → dispatch_order 的逆排列，恢复 [T,K,H]
    → ordered_combine(routes, 原始 FP32 weights)
```

逆通信后仍是 dispatch 排列，必须先还原槽位，才能按相同顺序合并。
这里不使用普通 dispatch 为自身计算准备的低精度权重副本。
训练行数不受 `mc2_capacity` 约束；反向检查输入、router 权重和专家参数的梯度正确性，
不要求跨 EP/CP 的梯度逐位相同。

## 推理：MC2

[mc2_ep.py](../rl/unified_execution/ops/moe/mc2_ep.py) 直接连接原生算子与 HP 专家计算：

```text
等长 [C,H] 输入 + 前缀 active mask
    → npu_moe_distribute_dispatch_v2
    → expert-major 缓冲 [M,H] + 设备上的专家有效行计数 [E/P]
    → GroupedExperts.forward_expert_major
    → npu_moe_distribute_combine_v2，使用原始 FP32 weights
    → 清零无效行并返回 [:T]
```

C 是 `mc2_capacity`，M 是接收缓冲的物理上界，不能把 M 当作有效计算行数。
各 rank 物理行数相同，使用 `global_bs=0`；设备侧 active mask 表示各自的有效前缀，允许空 rank。
无效行上的专家 ID 仍必须合法且行内不重复，不能使用 AllGather 的负数 sentinel。

dispatch 返回的辅助元数据、接收计数和 scales 原样交给 combine，并使用同一个通信组、ID 和 mask。
这些元数据的内部格式由原生算子拥有，适配层不自行解释或重排。
MC2 专用 HCCL group 与 HP EP 的成员及顺序相同，但独立于普通 HP collectives。

动态路由计数留在设备上，因此不需要在图捕获中把路由直方图搬回 CPU 来计算 split sizes。
固定接收缓冲与“让 GMM 计算所有填充行”不同：GMM 只消费设备计数指定的有效行。

## 推理：AllGather 与有效行 GMM

[allgather_ep.py](../rl/unified_execution/ops/moe/allgather_ep.py) 使用本次共同的 `bucket_rows=C`：

1. 将本地 states/IDs 补到 C 行，AllGather 得到 `[P*C,H]` 与 `[P*C,K]`。
2. 把 padding ID 映射到真实专家范围之外的 sentinel，调用 `npu_moe_init_routing_v2`，
   只打包本 rank 拥有的专家路由，并在设备上生成有效行计数。
3. HP GMM 计算有效行，通过安全逆索引恢复各路由；非 owner 的位置清零。
4. 等长 AllToAll 返回**未加权的槽位输出** `[P,C,K,H]`。
5. `owner_routes()` 根据 `expert_id // local_experts` 选择唯一 owner，随后运行 `ordered_combine()`。

这里仍有 AllGather 广播与稠密路由返回，通信量没有变成稀疏通信；优化的是有效计算及主机开销。
owner 选择是索引读取，不是跨 rank 的浮点求和，因此不会引入一棵新的规约树。
各 rank 的缓冲大小相同，使动态 ID、空专家和空 rank 可以进入同一张图回放。

## 后端选择与容量

### 构建时选择能力

[select_ep_backend()](../rl/unified_execution/runtimes/vllm/parallel.py) 在建模/捕获前检查当前适配器的能力：

| 配方 `inference_backend` | 行为 |
| --- | --- |
| `auto` | MC2 条件成立时启用 MC2，否则记录原因并使用 AllGather |
| `mc2` | MC2 条件不成立则启动失败；合法大 batch 仍由 AllGather 承接 |
| `allgather` | 固定使用 AllGather |

当前实现的是 A2 非量化 fullmesh：设备名包含 910B，dispatch/combine V2 接口可用，
BF16/FP16，hidden size 为 32 的倍数且不超过 10240，专家数不超过 1024 且可按 EP 均分，
`top_k <= min(16,E)`，`1 <= mc2_capacity <= 256`。
允许的 EP 度为 2–8、16、32、64、128、192、256、384，不使用“仅测过 EP2/4/8”作为白名单。
这些是当前代码的准入条件，不能代替新设备、规模或算子版本上的数值验收。

启动前还检查 HCCL 窗口，所需字节数为：

```text
4 * mc2_capacity * P * min(E/P, K) * H + 4 MiB
```

同一组核对后端、容量、调度边界、对齐参数与 `HCCL_BUFFSIZE`。
示例默认窗口为 512 MiB；代码不隐式增大窗口，缺省环境值按 200 MiB 检查。
通信开始后出错会传播失败，不切换后端重试。

### 执行时选择 bucket

`mc2_capacity` 只限定 MC2 的适用范围，既不是模型最大长度，也不是所有推理输入的上限。
`inference_row_bounds()` 从调度预算、实际图尺寸、decode 取整与 SP 对齐得到 `max_token_rows`。
PCP decode 可能复制行，不能把调度预算简单除以 PCP。

`InferenceExpertExecutor.select_backend_and_bucket()` 使用原生已同步的 **CPU 调度计数**，
或当前图 batch descriptor，确定共同的物理行数。它不读取设备上的专家路由计数：

```python
if backend == "mc2" and bucket_rows <= mc2_capacity:
    return "mc2", mc2_capacity
return "allgather", bucket_rows
```

例如 MC2 容量为 256，合法的 300 行 batch 由 AllGather 处理；仅当实际行数违反调度/图上界契约时
报错，不截断 token。chunked prefill 的开关本身不保证单次调度始终小于 256。

## 图执行的职责划分

`execute_inference_ep` 是 `hyper_rl::inference_ep` 自定义算子：Dynamo 只看到张量接口和 fake shape，
原生执行/捕获时再读取调度上下文，访问模型所属的 `static_forward_context` 中的 executor。
这样不会把第一次调用的后端选择固化进编译图；内部的原生通信和 GMM 仍进入 ACL 捕获。
实际图回放使用该图已经捕获的路径，匹配的图由 vLLM runner 选择。

eager 和图模式共用 MC2/AllGather 两条路径，没有额外的推理变长 AllToAll 分支。
图回放主要减少主机发射开销；未被图覆盖的 prefill 或尺寸仍由原生 runner 执行。
训练动态通信与推理图通信可以不同，数值契约不能因此放宽。

## 验证与性能边界

[EP 设备测试](../tests/st/test_unified_expert_parallel.py) 验证路由、逆排列、空 rank、拥塞专家、
有效计数变化、原址权重更新及动态图回放，并检查训练梯度。
[PCP 测试](../tests/st/test_prefill_context_parallel.py) 另外检查跨 PCP 的专家归属与发布。
[两份配置的运行矩阵](../examples/consistency/README.md#配置与验证矩阵) 检查完整模型实际采样概率。
组件图捕获成功不能替代完整模型的 sleep/wake、非零更新及第二步发布。

当前单机验收不等于跨机 MC2 或所有大 EP 组合已经通过。
性能需要同时看 T、K、H、E/P、路由倾斜、padding 和图覆盖率：MC2 减少广播通信，
AllGather 接口简单并覆盖较大 batch，但通信仍按物理上界搬运。
固定小 batch 下的图回放收益不能直接外推为整轮 RL 加速；不在热路径引入 CPU 路由统计来择优。
