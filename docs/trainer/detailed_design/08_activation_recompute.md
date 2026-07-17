# 08 激活重计算设计（双模式）

> 前置文档：05（双模式 DTensor 并行策略）、06（分布式基础设施，FSDP2 集成侧
> `_apply_activation_checkpoint`）。
> 相关组件：`hyper_parallel/core/activation_checkpoint/`（原语层）、
> `hyper_models/components/distributed/`（双模式组件）。

## 0. 文档定位与结论速览

本文档定义 `hyper_models/components/distributed` 双模式方案下的激活重计算（activation
recompute / checkpoint）体系，回答三个问题：

1. **声明式配置**：如何像 spec 一样声明"某些模块、某些算子"的重计算；
2. **通信的重计算**：双模式下 TP/CP/EP 三类通信在重算相位的行为与正确性；
3. **局部不重算**：`checkpoint_exclude_wrapper` 的策略定位与应用方式。

核心结论：

- 声明载体为 `ModuleShardingSpec.recompute`（模块粒度）+
  `RECOMPUTE_POLICY_REGISTRY`（算子粒度），与现有四字段/CP 注册表同构；
- checkpoint 区域默认划在 **compute 段**（boundary 通信之外）——TP 边界
  通信零重放；CP 的 K/V all-gather 默认保存在区域输入（gathered K/V 不
  重取）；EP 的 a2a/topk 默认 MUST_SAVE（路由决策不重算）；
- `checkpoint_exclude_wrapper` 是**区域级 MUST_SAVE**，与算子级 SAC 互补，
  用于通信段/副作用段/非确定段的精确排除；
- 两模式共用同一份重算路径（双模式架构约束的延伸）：production 重放
  local tensor 前向，validate 重放 DTensor dispatch，校验逻辑无副作用
  可安全重入。

## 1. 调研结论（现状与外部实践）

### 1.1 torchtitan

`torchtitan/distributed/activation_checkpoint.py`，策略类以 **transformer
block 为应用单元**（`layers.named_children()` 逐个包装），三档：

| 策略 | 机制 | 粒度 |
|---|---|---|
| `FullAC` | `checkpoint_wrapper` 整 block 重算 | 模块（block） |
| `SelectiveAC` | `create_selective_checkpoint_contexts(policy_fn)` 逐 aten 算子判定：compute-intensive + SDPA + **通信算子（c10d functional/a2a/DeepEP）MUST_SAVE 避免重复通信**；mm 每隔一个重算；`force_recompute_mm_shapes_by_fqns` 按 fqn 反查 Linear 权重形状强制重算 | **算子** |
| `MemoryBudgetAC` | compiler partitioner 按显存预算自动划分 | 编译器自动 |

关键细节：SAC 的 mm 计数按 `ctx.is_recompute` 分桶，保证前向/重算两遍
判定一致；`topk` MUST_SAVE（MoE 路由跨重算稳定）。

**spmd_types 后端**（titan 的 local-tensor SPMD runtime）与 AC **完全正交**：
AC 策略照旧按 block 应用，不关心 SPMD 后端形态。对我们的启示：重算相位
在前向执行语义（DTensor dispatch 或 local tensor）上各跑一遍即可，类型
系统不引入额外约束；唯一要求是 checkpoint 需要知道权重的切分元数据
（我们的 plan/DTensor 自带）。

### 1.2 NeMo AutoModel

titan SAC 的超集（`nemo_automodel/components/distributed/activation_checkpointing.py`）：

1. **子模块粒度** `apply_submodule_checkpointing`：按属性名启发式
   （`self_attn/mlp/input_layernorm/...`）包子模块——"某些模块"声明式的
   雏形，但属性名清单写死、无语义角色概念；
2. **SDPA backend 快照 context_fn**：重算时重钉前向的 SDPA backend 集合，
   否则两遍 save 集不同触发 determinism check；
3. **KV-sharing 避让**：DynamicCache 共享 K/V 的模型重算 attention 会
   二次写 cache → 必须排除；
4. compile 交互：SAC wrapper 在最外（compile OUTER, SAC INNER）。

### 1.3 hyper_parallel 现状

**原语层已齐备**（`core/activation_checkpoint/`，平台抽象 torch/MindSpore
双后端）：

| 原语 | 位置 | 语义 |
|---|---|---|
| `checkpoint(fn, *args, policy_fn=, context_fn=, swap_inputs=)` | activation_checkpoint.py | 非重入 checkpoint；`policy_fn` 走 `plat.create_selective_checkpoint_contexts`（SAC）；多 context_fn 组合 |
| `CheckpointPolicy` | 同上 | torch 四档 + **`MUST_SWAP`（扩展）**：前向换出、反向前取回 |
| `swap()` / `swap_wrapper` / `swap_tensor_wrapper` | 同上 + swap.py | 不换重重算、换出换入（SwapManager 协调异步拷贝） |
| `checkpoint_wrapper` | plat | 模块级 checkpoint 包装 |
| **`checkpoint_exclude_wrapper`** | plat（**仅 MindSpore 已实现**；torch 平台 NotImplementedError） | 区域级局部不重算（机制见 §5.1） |
| `RecomputeState` / `is_recomputing()` | recompute_state.py | contextvar 跟踪重算相位；invocation 级资源注册表（前向存、重算取、相位结束清理） |

**缺口**：FSDP2 集成侧 `_apply_activation_checkpoint` 仅存在于 06 文档
（block 粒度 full / 类名启发式子模块，未落地代码）；双模式组件
`hyper_models/components/distributed` **无任何重计算接入**。

## 2. 双模式重算的语义框架

### 2.1 重算相位模型

非重入 checkpoint（`use_reentrant=False`，两平台统一约定）：

```
前向：region(input) → output；只保存 input，区域中间激活全部释放
反向：重放 region(input) 重建中间激活 → 接 autograd
```

`RecomputeState`（contextvar）标记当前相位：`is_recomputing()` 在重放时
为 True；`get_resource(key, factory)` 提供**跨前向/重算的 invocation 级
存储**（前向 save、重算 pop、相位结束自动清理）——这是
`checkpoint_exclude_wrapper` 与一切"跨相位状态"的载体。

### 2.2 两模式的重放语义

| | production | validate |
|---|---|---|
| 区域输入 | local tensor（boundary 入口已重分布） | DTensor（boundary 入口已重分布 + from_local 包装） |
| 重放内容 | local tensor 前向（含 CP wrapper/EP compute_fn 的同一份代码） | DTensor dispatch 重放（含 out_src/out_dst 校验再跑一遍） |
| 参数 | 永久解包的 local tensor | DTensor；local-region 内 `_temp_local_params` 重入（context manager，安全） |
| 校验副作用 | 无 | out_src/out_dst 校验为纯比较，重入幂等 |

**架构约束延伸**：双模式"区域内计算路径逐指令一致"自然延伸到重算相位——
重算声明（区域划分、算子策略）两模式共用一份，不允许按模式分叉。差异
只在张量形态（local vs DTensor），由既有缝合层吸收。

### 2.3 RNG 与确定性

- `preserve_rng_state=True`（两平台默认）：区域内 dropout 重放一致；
- SDPA backend 快照 context_fn（吸收 AutoModel §1.2-2）：重算相位重钉
  前向的 backend 集合，避免两遍 kernel 选择不同导致 save 集不一致；
- 非确定算子（topk 并列、atomicAdd 类）不允许留在重算区域——用策略
  MUST_SAVE（§4.2）或 exclude_wrapper（§5）排除。

## 3. 通信的重计算（TP/CP/EP）

### 3.1 通信分类清单

双模式组件内有三类通信，所在位置不同，重算语义必须分别定义：

| 类别 | 位置 | 具体通信 |
|---|---|---|
| **TP 边界通信** | `PrecompiledBoundary`（wrapper 的入口/出口） | all-gather（in_dst 收敛）、reduce-scatter/all-reduce（out_dst 收敛）、SP 布局转换 |
| **CP K/V all-gather** | inner attention wrapper **区域内**（CP wrapper 注入在模块 forward 之前） | K/V 在 CP 组 all-gather；D-04 offset causal mask 构建 |
| **EP all-to-all** | local-region **区域内**（`_hf_native_ep_compute` / 用户 compute_fn） | dispatch/combine a2a（NCCL/HCCL 不等长零填充、gloo pad-to-max）；router topk |

### 3.2 总原则：checkpoint 区域划在 compute 段

现有 wrapper 是整体闭包（boundary 入口 → 模块前向 → boundary 出口）。
重计算要求改造为**三段式**，checkpoint 只包中段：

```
module.forward =
    boundary.redistribute_inputs(...)     # 段 1：TP 边界入口通信（区域外）
    checkpoint(                           # 段 2：compute 段（区域内）
        compute_segment,                  #   production: local 前向
        *redistributed_inputs,            #   validate:  DTensor dispatch 前向
        policy_fn=..., context_fn=...)
    boundary.redistribute_outputs(...)    # 段 3：TP 边界出口通信（区域外）
```

效果：

- **TP 边界通信永不重放**（段 1/3 在区域外，前向一次、反向不重算）——
  这就是 titan SAC 把通信算子 MUST_SAVE 的动机，我们用区域划分结构性
  地达到同一目的，不依赖算子策略的覆盖面；
- 区域输入 = 重分布后的激活（被 checkpoint 保存），语义与单卡 checkpoint
  完全一致；
- 两模式三段式结构相同：production 段间传 local tensor，validate 段间传
  DTensor。

**例外路径**：block 级整层重算（§4.3，wrapper 外再包一层）会连段 1/3
一起重放——语义正确（见 §3.3）但多一次边界通信，属于用通信换简单的
兜底档位。

### 3.3 TP：边界通信的重放正确性

若区域外通信被重放（block 级重算或用户显式整模块声明）：

| 通信 | 重放安全性 | 依据 |
|---|---|---|
| all-gather | ✅ 安全 | 确定性 collective，输入（区域外上游激活）在重放时已被上游区域重建或保存 |
| all-reduce / reduce-scatter | ✅ 安全 | 同上；求和顺序固定（同一进程组同一算法），逐位一致 |

结论：TP 通信重放**只有性能代价、无正确性风险**。三段式（§3.2）把该
代价降为零，block 级兜底档保留该代价（与 Megatron 整层重算行为一致）。

### 3.4 CP：K/V all-gather 的保存与重放

CP wrapper 在模块 forward 内做 K/V all-gather。**默认区域划分：checkpoint
划在 all-gather 之后**——

```
CP wrapper 内：
    K/V all-gather（区域外，前向一次）     ← gathered K/V 成为 checkpoint 区域输入
    checkpoint(q/k/v reshape + RoPE + SDPA + o_proj)（区域内）
```

- gathered K/V 作为区域输入被保存（显存代价 = 全量 K/V，与不开 CP 重算
  的常规 AC 相同）；
- 重放只重算 attention 计算，不再通信；
- D-04 offset mask 由 `q_len/kv_len/cp_rank` 确定构建，重放逐位一致；
- **双模式约束**：CP wrapper 两模式注入同一份代码（D-01''），区域划分
  自然一致。

**备选档（显式声明 `recompute_scope="module"`）**：整 attention（含
all-gather）重算。all-gather 重放确定、安全（§3.3 同依据），代价是每
层反向多一次 K/V all-gather。适用：gathered K/V 显存不可承受的长序列
场景（这本来就是 CP 的动机之一）——此时通信换显存是合理交易，故保留
为显式可选项而非默认。

### 3.5 EP：a2a 与路由的保存

MoE local-region 内部结构：`router(topk) → dispatch a2a → expert GEMM →
combine a2a`。两类重放风险，性质不同：

| 风险 | 后果 | 处理 |
|---|---|---|
| **topk 重算不稳定**（并列分数的 tie-break 非确定） | 重算后 token→expert 分配与前向不同 → 梯度错 | topk **MUST_SAVE**（titan 同款）；路由 logits/indices 作为下游区域输入保存 |
| **a2a 重放** | 浪费通信；带元数据协商的实现（不等长 a2a 的长度交换、DeepEP handle）重放可能产生不一致句柄 | a2a 输出 **MUST_SAVE**；即 checkpoint 子区域划在 **expert compute 段**，两个 a2a 都在区域外 |

推荐结构（内置 `_hf_native_ep_compute` 默认采用）：

```
local-region 内：
    router + topk（区域外，输出保存）
    dispatch a2a（区域外，前向一次）
    checkpoint(expert GEMM + SwiGLU)（区域内）   ← MoE 的显存大头
    combine a2a（区域外，前向一次）
```

用户自定义 `local_compute_fn` 时，框架无法自动划分子区域——提供两条
声明路径：SAC `moe` 策略（§4.2，topk/a2a/grouped_mm MUST_SAVE）或
`checkpoint_exclude_wrapper` 包通信段（§5.3）。

### 3.6 通信重放正确性判定准则（一般化）

凡通信段落在 checkpoint 区域内会被重放时，按以下准则判定：

1. **确定性 collective**（all-gather/all-reduce/reduce-scatter/broadcast）：
   重放安全，只有性能代价；
2. **带运行时元数据的通信**（不等长 a2a 的长度协商、DeepEP handle、
   依赖 stream 状态的异步通信）：必须保存输出或排除出区域；
3. **通信的上游决策**（router/topk/sampling）：决策结果 MUST_SAVE，
   决策本身不得留在区域内。

## 4. 声明式配置体系

### 4.1 模块粒度：`ModuleShardingSpec.recompute`

```python
ModuleShardingSpec(
    ...,
    recompute: Optional[str] = None,          # None / "full" / "selective" / "swap"
    recompute_policy: Optional[Union[str, Callable]] = None,  # "selective"/"swap" 的算子策略
    recompute_scope: str = "compute",         # "compute"(默认，三段式中段) / "module"(整模块含通信)
)
```

- 与四字段（`use_local_map`/`local_compute_fn`/`inner_target`/
  `inner_wrapper`）同构：模板给默认值，`plan_overrides` 用户对任意
  spec 声明或覆盖；声明互不嵌套原则延伸——recompute 声明不与
  inner-wrap/local-region 声明互相改写；
- `"full"`：compute 段整体重算；`"selective"`：compute 段内按
  `recompute_policy` 逐算子判定；`"swap"`：不重算，激活按策略换出
  （`MUST_SWAP` + SwapManager）；
- `recompute_scope="module"` 仅用于 §3.4 的 CP 备选档等显式场景。

**嵌套规则**（与 D-16 同一哲学，fail-fast）：一个模块的 recompute 区域
不得与另一 recompute 区域嵌套（外层 block 级 AC + 内层 spec recompute
同时命中 → ValueError，指引保留一层）；同一模块重复声明以用户覆盖为准
（替换语义）。

### 4.2 算子粒度：`RECOMPUTE_POLICY_REGISTRY`

仿 `CP_WRAPPER_REGISTRY` 的注册表模式：

```python
RECOMPUTE_POLICY_REGISTRY["default"]   # titan 风格：compute-intensive + SDPA
                                       # + topk MUST_SAVE；mm 每隔一个重算
RECOMPUTE_POLICY_REGISTRY["moe"]       # default + grouped_mm/a2a/topk MUST_SAVE
register_recompute_policy("my", policy_fn)   # 用户自定义
# policy_fn: (ctx, op, *args, **kwargs) -> CheckpointPolicy（含 MUST_SWAP 扩展）
```

- 复用现有 `plat.create_selective_checkpoint_contexts`（torch/MindSpore
  双后端已通）；
- mm 计数按 `ctx.is_recompute` 分桶（titan/AutoModel 同款，保证两遍判定
  一致）；
- 可观察性同款：解析结果回写 `spec._resolved_recompute_policy` + INFO
  日志（模块 fqn、策略名、来源）；
- 默认组合 context_fn：RecomputeState + SAC + **SDPA backend 快照**
  （§2.3），经 `checkpoint()` 的 `_compose_context_fns` 栈式组合（已有
  机制，零新增）。

### 4.3 block 粒度（06 衔接，本期不实现）

整 decoder layer 包 AC（wrapper 之外）：边界通信重放（§3.3，正确但有
代价）。06 文档的 `_apply_activation_checkpoint` 落地时：

- `True/"full"` → 维持整 block checkpoint（兜底档）；
- `"selective"` → **改为经本文档层 1 的 spec 字段实现**（planner 已知
  每个模块的语义角色，比 06 现稿的类名启发式 `("Attention","MLP",...)`
  准确得多）；block 级与 spec 级同时命中按 §4.1 嵌套规则 fail-fast。

### 4.4 应用点：Phase C 三段式改造

`_wrap_production_forward` / `_wrap_validate_forward` /
`_wrap_local_region_forward` 统一改造为三段式，`spec.recompute` 非 None
时中段经 `plat.checkpoint` 包装；CP inner wrapper 场景中段再按 §3.4 细
分（all-gather 前/后）。无 recompute 声明时生成的闭包与现状逐字节等价
（零开销原则）。

## 5. `checkpoint_exclude_wrapper`（局部不重算）

### 5.1 机制（MindSpore 现有实现）

`platform/mindspore/activation_checkpoint/checkpoint_exclude_wrapper.py`：

```
前向相位（RecomputeState 非 None 且非重算）：
    在 saved_tensors_hooks(pack=存真实 tensor.data) 下执行模块
        → 该区域的 autograd 保存张量不被外层 checkpoint 释放
    output 按 wrapper id 存入 invocation 级缓存（RecomputeState.get_resource）
重算相位：
    不执行模块，直接 pop 缓存的 output
相位结束/异常：
    RecomputeState 自动清理未消费缓存
```

本质：**区域级 MUST_SAVE**——区域内激活经 saved_tensors_hooks 落为真实
保存（不被外层 checkpoint 的 placeholder 机制释放），区域输出经
invocation 缓存跨相位传递。要求：PyNative 模式 + 外层非重入 checkpoint。

### 5.2 策略定位：与 SAC 的分工

| | SAC policy（算子级） | exclude_wrapper（区域级） |
|---|---|---|
| 判定单位 | aten 算子类型（全局生效） | 代码区域（精确到某一次调用） |
| 声明方式 | `recompute_policy` 注册表 | 包装具体模块/函数 |
| 适用 | "所有 SDPA 都保存"、"mm 隔一重算" 这类**类型性**规则 | "**这一段**通信/副作用代码不参与重算" 这类**位置性**规则 |
| 失效风险 | 算子集合猜不全（自定义算子漏判） | 区域划错（把大计算包进去 → 显存收益丢失） |

选择准则：能用类型规则表达的用 SAC；涉及通信句柄/副作用/缓存语义、或
自定义算子不在 SAC 视野内的，用 exclude_wrapper。

### 5.3 双模式组件内的应用场景

| 场景 | 用法 |
|---|---|
| **CP all-gather 段**（`recompute_scope="module"` 时） | exclude_wrapper 包 K/V all-gather 段 → 整 attention 重算但 all-gather 不重放（§3.4 备选档的优化） |
| **EP a2a 段**（用户自定义 compute_fn） | exclude_wrapper 包 dispatch/combine a2a → 自定义 MoE 无需手写 SAC 策略即获得 §3.5 结构 |
| **副作用段**（KV-cache/DynamicCache 写入、日志、metric 累加） | 重算会二次执行副作用 → exclude_wrapper 强制只跑一次（AutoModel KV-sharing 问题的通用解法） |
| **非确定算子段**（sampling、tie-break 敏感的自定义 router） | 排除后输出跨相位复用，与前向逐位一致 |
| **校验豁免**（validate 下 out_src 声明式重包装段） | 声明式 out_src 的 from_local 包装不参与重算判定 |

### 5.4 torch 平台移植路径

torch 平台当前 `NotImplementedError`。移植要件全部存在：

- `torch.autograd.graph.saved_tensors_hooks(pack, unpack)` 与 MindSpore
  `ms.saved_tensors_hooks` 语义对应；
- `RecomputeState`/invocation 缓存是平台无关的（core 层，已双平台共用）；
- 非重入 checkpoint：`torch.utils.checkpoint.checkpoint(use_reentrant=
  False)` 与 `plat.checkpoint` 已统一。

即同一 `CheckpointExcludeWrapper` 模式可平移到 torch（实现为
`torch.nn.Module` 的 forward 版本）。这是消除平台能力缺口的**必要项**
（§5.3 的场景在 torch 后端同样需要），列入实施计划 P1。

### 5.5 与双模式的组合注意

- 缓存的 output 形态跟随模式：validate 下区域内若为 DTensor 输出，缓存
  DTensor 对象（引用语义，不复制）；invocation 清理保证不泄漏；
- exclude 区域内**不得再有 boundary**（boundary 属于段 1/3，结构上就
  在 checkpoint 区域外）——嵌套检测在 apply 期 fail-fast；
- exclude_wrapper 与 SAC 可叠加：SAC 管区域内算子，exclude 管子区域，
  组合时 exclude 子区域对 SAC 不可见（其算子不经过 policy 判定——
  saved_tensors_hooks 在 dispatch 之下）。

## 6. swap（MUST_SWAP）与重计算的组合

- `recompute="swap"`：compute 段不重算，段内激活经
  `async_save_on_cpu(policy_fn)` 前向换出、反向前取回（SwapManager 异步
  拷贝 + stream 同步）；
- SAC policy 返回 `MUST_SWAP` 可与 SAVE/RECOMPUTE 混用：大体积低算耗
  激活（如 attention 输入）换出，小激活重算，贵算子（SDPA/mm）保存——
  三档混排是 SAC 的自然扩展；
- `group_swap=True` 参与 group copy 融合；
- swap 与 checkpoint 区域正交可叠加（`checkpoint(..., swap_inputs=True)`
  已支持区域输入换出）。

## 7. 双模式等价性验证方案

| 验证项 | 方法 |
|---|---|
| 数值等价 | 重算开/关分别对单卡参考对拍（production + validate 各一遍；梯度等价复用 `testing/grad_equiv.py` S5.3 工具） |
| 相位一致性 | SAC mm 计数前向/重算分桶断言；`is_recomputing()` 在自定义 compute_fn 内的可观测性 |
| 通信零重放 | mock/计数 collective：三段式下重算相位 all-gather/a2a 计数不增；block 级下计数翻倍（预期行为，量化开销） |
| RNG 一致 | 区域内置 dropout 的模型两遍对拍逐位一致 |
| exclude_wrapper | 副作用计数器（区域内自增）在前向+重算后恰好为 1；缓存 output 与前向逐位一致 |
| 嵌套 fail-fast | block 级 + spec 级同时命中 → ValueError；exclude 内含 boundary → ValueError |

## 8. 实施计划

| 期 | 内容 |
|---|---|
| P0 | `ModuleShardingSpec.recompute` 字段 + Phase C 三段式改造（production/validate 两路）+ `RECOMPUTE_POLICY_REGISTRY`（default/moe 内置）+ 数值/通信计数验证 |
| P1 | torch 平台 `checkpoint_exclude_wrapper` 移植（§5.4）+ CP all-gather 区域细分（§3.4 默认档）+ EP a2a 子区域划分（§3.5 内置 compute_fn） |
| P2 | swap 声明（`recompute="swap"` + policy MUST_SWAP）+ SDPA backend 快照 context_fn 内置 + block 级与 06 `_apply_activation_checkpoint` 衔接（§4.3，随 FSDP2 集成落地） |

## 9. 开放问题

1. `recompute` 字段入 spec 后，`ShardingTemplate` 是否给默认档位
   （如 mlp 默认 None、未来超大模型 moe 默认 "selective"）？初版建议
   全 None（零默认重算），默认档位等显存模型数据后再定；
2. CP `recompute_scope="module"` 备选档是否内置 exclude_wrapper 组合
   （§5.3 第一行）作为第三档 `recompute_scope="module_exclude_comm"`？
   倾向 P1 按实际需求决定，不预先枚举；
3. block 级 AC 与 spec 级嵌套的 fail-fast 检测点在 planner 还是 applier？
   倾向 applier Phase C（与 D-16 的 `_check_no_nested_overrides` 同层，
   plan 生成不受 FSDP 侧配置影响）。
