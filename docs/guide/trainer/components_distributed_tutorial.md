# hyper_models/components/distributed 双模式 DTensor 使用教程

> 适用组件：`hyper_models/components/distributed/`
> 设计文档：`docs/detailed_design/05_dual_mode_dtensor_parallel_strategy.md`
> 代码走读：`docs/refactor/guides/components_distributed_code_walkthrough.md`

本教程覆盖：**TP / CP / EP / FSDP 组合、production↔validate 双模式切换、
自定义模块四接口（`use_local_map` / `local_compute_fn` / `inner_target` /
`inner_wrapper`）**，并讲清 CP/EP 各自内置了哪些 wrapper、何时自动生效、
如何显式固定或整体替换。

---

## 目录

1. [组件概述](#1-组件概述)
2. [五分钟快速开始（TP）](#2-五分钟快速开始tp)
3. [核心 API 参考](#3-核心-api-参考)
4. [双模式：validate 与 production](#4-双模式validate-与-production)
5. [TP 教程](#5-tp-教程)
6. [CP 教程（含内置 4 个 CP wrapper 详解）](#6-cp-教程)
7. [EP 教程（含内置 EP wrapper 详解）](#7-ep-教程)
8. [FSDP 组合（接口契约）](#8-fsdp-组合接口契约)
9. [多维并行组合（mesh 布局）](#9-多维并行组合mesh-布局)
10. [自定义模块完整指南](#10-自定义模块完整指南)
11. [典型模型支持矩阵](#11-典型模型支持矩阵)
12. [排错索引](#12-排错索引)

---

## 1. 组件概述

`hyper_models/components/distributed` 是独立可用的 DTensor 分片组件，零依赖训练流程
（不 import `recipes/` / `models/` / `datasets/`），两步用法：

```
ShardingPlanner.plan(model, mesh, ...)   → ShardingPlan   # 编译期推导（6-phase）
apply_sharding_plan(model, plan, mesh)   → (model, tp_grad_info)  # 双模式应用
```

- **Planner**：遍历 `named_parameters()`，按命名规则（`ParamRole`）+ 语义
  模板（`TEMPLATES`：attention/mlp/norm/embed/lm_head/moe_gate/moe_mlp）
  自动推导每个通信边界的参数 placement 与 I/O 契约（`in_src/in_dst/
  out_src/out_dst`）。推导不满的地方用 `plan_overrides` 手写 spec 合并。
- **Applier**：Phase A 参数分片 → Phase C 包装 forward（PrecompiledBoundary
  通信缝合 + local-region / inner-wrap 注入）→ Phase D tied weights。
  同一个 plan 可按两种模式应用（§4）。

**用户代码恒工作在 local tensor 世界**：DTensor↔local 的缝合、边界通信
（all-gather/reduce-scatter）由框架负责。

---

## 2. 五分钟快速开始（TP）

完整可运行示例见 `examples/distributed/`（gloo/CPU 可跑，torchrun 启动）：

```python
import torch.distributed as dist
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_models.components.distributed import ShardingPlanner, apply_sharding_plan

dist.init_process_group("gloo")   # 或 nccl/hccl

# 1) 构建 device mesh：1D TP
mesh = init_device_mesh("cpu", (dist.get_world_size(),), mesh_dim_names=("tp",))

# 2) 编译期推导（零模型代码改动，任意 HF 风格命名均可）
planner = ShardingPlanner()
plan = planner.plan(model, mesh, tp_size=dist.get_world_size())

# 3) 应用分片（production 模式）
model, tp_grad_info = apply_sharding_plan(model, plan, mesh)

# 4) 正常训练/推理——前向输出与单卡逐位一致
out = model(input_ids)
```

运行：

```bash
PYTHONPATH=. torchrun --nproc_per_node=2 examples/distributed/tp.py
```

### 2.1 示例目录一览

[`examples/distributed/`](../../../examples/distributed/) 七个独立示例，均与单卡
参考做数值对拍：

| 示例 | 并行 | 演示点 |
|---|---|---|
| `tp.py` | TP=2 | 零配置自动推导 + 应用（本节代码的完整版） |
| `cp.py` | CP=2 | `shard_batch_for_cp` + 内置 `"sdpa_hf"` wrapper + D-04 causal 修正（§6） |
| `ep.py` | TP=2×EP=2 | HF 原生 MoE 零配置：D-09 堆叠 + D-10 TP-extend-EP + 内置 `_hf_native_ep_compute`（§7） |
| `tp_cp_ep.py` | TP=2×CP=2×EP=2 | 三维组合：cp-major 序列布局（§6.6）+ plan 内省断言（`_ep_stack`/`_needs_cp_attn` 等） |
| `nested_local_map.py` | TP=2（嵌套） | D-14 嵌套 spec：外层 local_map（根 fqn `""`）+ 内层 validate 孤岛，双模式对拍（§10.1） |
| `custom_local_compute_fn.py` | TP=2 | 自研 MoE：`plan_overrides` + `local_compute_fn` 注入自定义 compute（§10.3） |
| `custom_inner_wrapper.py` | CP=2 | 自研 attention：`inner_target` + 注册表命名 wrapper（§10.4） |

**先 validate 再 production**（推荐工作流，详见 §4）：

```python
# 同一份 plan，先跑 validate 校验契约/数值，再切 production 训练
model_v, _ = apply_sharding_plan(model, plan, mesh, validate_mode=True)
```

---

## 3. 核心 API 参考

### 3.1 `ShardingPlanner().plan(...)`

```python
plan = planner.plan(
    model, mesh,
    tp_size=1,               # TP 组大小（>1 激活 TP 维）
    cp_size=1,               # CP 组大小（>1 激活 CP 维，attention 注入 CP wrapper）
    ep_size=1,               # 扩展 EP 组大小（>1 且命中 HF 原生 MoE 时激活 EP）
    sequence_parallel=True,  # SP：norm 间激活按序列维 Shard(1)
    loss_parallel=False,     # lm_head 输出 Shard(-1)（vocab 并行 loss）
)
```

- 返回 `ShardingPlan`：`modules: {fqn: ModuleShardingSpec}` +
  `mesh_dim_names` + `special_handlers` + `tied_pairs`。
- size=1 的轴会被自动剔除（`plan.mesh_dim_names` 只含活跃轴）。
- `ShardingPlanner(plan_overrides={fqn: spec})`：用户手写 spec 合并入口
  （§10.1）。

### 3.2 `apply_sharding_plan(...)`

```python
model, tp_grad_info = apply_sharding_plan(
    model, plan, mesh,
    validate_mode=False,     # True=validate 模式；默认 production
)
```

- 返回 `(model, tp_grad_info)`；validate 下 `tp_grad_info is None`。
- `model` 也可以是 **PP 多 part 列表**（`apply_sharding_plan([part0, part1], ...)`）。

### 3.3 DeviceMesh 构建

```python
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh

# TP=8
mesh = init_device_mesh("npu", (8,), mesh_dim_names=("tp",))
# CP=2 × TP=4（cp 外层、tp 内层——TP 组连续，通信局部性最好）
mesh = init_device_mesh("npu", (2, 4), mesh_dim_names=("cp", "tp"))
# TP=2 × EP=4（EP 显式轴；EP-aware 自研模块取 mesh["ep"] 用）
mesh = init_device_mesh("npu", (2, 4), mesh_dim_names=("tp", "ep"))
# DP=2 × CP=2 × TP=2
mesh = init_device_mesh("npu", (2, 2, 2), mesh_dim_names=("dp", "cp", "tp"))
```

取子 mesh：`cp_mesh = mesh["cp"]`、`tp_mesh = mesh["tp"]`。

---

## 4. 双模式：validate 与 production

### 4.1 语义对照

| | production（训练/推理） | validate（校验） |
|---|---|---|
| 参数 | build 期**永久解包**为 plain local tensor | 保持 **DTensor** |
| 前向 | 纯 local tensor + PrecompiledBoundary 通信 | DTensor dispatch 传播 + out_src/out_dst **契约校验** |
| 反向 | local autograd（梯度落 local 分片） | local autograd（同左） |
| 返回值 | `tp_grad_info`（供 FSDP） | `None` |
| 用途 | 生产训练 | 分片正确性验证 / 调试新模型接入 |

### 4.2 推荐工作流：先 validate 再 production

```python
plan = ShardingPlanner().plan(model, mesh, tp_size=4, cp_size=2)

# Step 1: validate——DTensor dispatch 逐边界校验契约，数值与单卡对拍
model_v, _ = apply_sharding_plan(model, plan, mesh, validate_mode=True)
out_v = model_v(batch)
torch.testing.assert_close(out_v, ref_out, rtol=1e-5, atol=1e-5)

# Step 2: 同一份 plan 切 production——零 DTensor dispatch 开销
model_p, tp_grad_info = apply_sharding_plan(model, plan, mesh)
```

**架构约束（双模式等价的关键）**：凡 DTensor dispatch 无法表达数据相关
逻辑的模块（embedding mask、attention K/V gather、MoE all-to-all），两
模式注入**同一份 wrapper** 显式重建该逻辑——区域内计算路径逐指令一致，
因此双模式数值可达 kernel 级等价。

### 4.3 validate 的声明式豁免清单

- **attention（CP>1）**：`out_src` 为声明式——CP wrapper 出口按声明
  `from_local` 重包装（K/V all-gather 语义 dispatch 无法派生）；
- **MoE（local-region）**：`out_src` 为声明式（all-to-all 数据相关性）；
  `in_src` 契约仍由 boundary 正常校验；
- 其余模块（embed/norm/mlp/lm_head）：`out_src` 由 DTensor dispatch
  **派生校验**——这是 validate 的核心校验能力。

---

## 5. TP 教程

### 5.1 基础 TP（qwen/llama/glm 类模型零配置）

默认命名规则自动识别：`q/k/v/gate/up_proj` → COLWISE（`Shard(0)`），
`o/down_proj` → ROWWISE（`Shard(1)`），norm/embed/lm_head 各归其位。
融合权重也支持：`qkv_proj/fused_qkv/query_key_value`（FUSED_QKV）、
`gate_up_proj`（FUSED_GATE_UP）均按 `Shard(0)`。

```python
plan = ShardingPlanner().plan(model, mesh, tp_size=8)
```

### 5.2 SP（sequence_parallel）

`sequence_parallel=True`（默认）：norm 之间（embed 出口 → attention/mlp
入口）的激活按序列维 `Shard(1)` 切分，每 rank 只持有 `S/tp` 的激活，
通信量为 all-gather（进 region）+ reduce-scatter（出 region）。
关闭后激活全程 `Replicate`（通信变为 all-reduce 风格）。

### 5.3 loss_parallel

```python
plan = planner.plan(model, mesh, tp_size=8, loss_parallel=True)
```

`lm_head.out_dst = {TP: Shard(-1)}`：logits 按 vocab 维切分输出，配
vocab 并行 CE loss（上游 loss 侧消费）。默认 `False` 时 lm_head 出口
`Replicate`（全量 logits）。

### 5.4 tied weights

`embed_tokens.weight` ↔ `lm_head.weight` 共享存储时，planner 自动检测
（`plan.tied_pairs`），applier Phase D 归一化两端 placement（Shard 优先
于 Replicate），`tp_grad_info` 中 tied 对映射到同一 placement——无需
用户处理。

### 5.5 非标准命名 → ARCH_OVERRIDES

命名不命中默认规则时（如自研 `wq/wk/wv`），注册架构覆盖：

```python
from hyper_models.components.distributed.sharding_planner import ARCH_OVERRIDES
from hyper_models.components.distributed.param_role import ParamRole

ARCH_OVERRIDES["myarch"] = [
    (["wq", "wk", "wv"], ParamRole.COLWISE),
    ("wo", ParamRole.ROWWISE),
]
# config.architectures=["MyArchForCausalLM"] 或 model_type="myarch" 即生效
```

已内置 DeepSeek MLA 条目（§11）。

### 5.6 attention 前向里的显式 `num_heads`（D-17 头数改写）

模型 forward 里 reshape/split 的头数有两种写法：

```python
q = self.q_proj(x).view(b, s, -1, self.head_dim)              # TP 容错（推荐）
q = self.q_proj(x).view(b, s, self.num_heads, self.head_dim)  # 显式全局头数
```

TP colwise 切分后每 rank 本地只有 `num_heads/tp` 个头。组件按"前向看到
什么张量"自动适配（无需用户配置）：

| 模式 | 模块位置 | 行为 |
|---|---|---|
| production | 全部 head-sharded 模块 | **自动改写**模块缓存的头数属性为本地值（`num_heads // tp`） |
| validate | 普通 boundary（attention/mlp/norm） | **不改写**——DTensor dispatch 在全局逻辑形状上运行，显式全局头数天然正确，shape 自动推导 |
| validate | local-region（`use_local_map`/`local_compute_fn`/EP） | **自动改写**（区域内两模式都是 local tensor） |

要点：

- **检测**：plan 中 q/k/v 类投影（含 MLA 的 `q_b_proj`）在 TP 维 colwise
  `Shard(0)` 即命中，与模块类名无关；
- **改写清单**（transformers 全库调研）：Q 侧 `num_heads`/
  `num_attention_heads`/`n_heads`/`num_attn_heads`/`n_head`/`heads`/
  `num_head`，KV 侧 `num_key_value_heads`/`num_kv_heads`/`kv_heads`；
- **绝不动**：`config`（RoPE 的 `head_dim` 推导不受影响）、`head_dim`
  （不切的维）、`num_key_value_groups`（GQA 比值，TP 不变量）；
- **幂等**：原值存于 `module._hp_full_head_counts`，重复 apply 不会二次
  除法；非整除只告警不改写（planner 的 `validate_model_compatibility`
  通常会先在 config 层面 fail-fast）；
- 局限：forward 里**直接读 `config.num_attention_heads`**（而非
  `__init__` 缓存到 `self.*`）的模型不覆盖——HF 主流模型均为缓存式。

---

## 6. CP 教程

CP（Context Parallel）沿序列维切分激活。**参数不切**（CP 维参数恒
`Replicate`）；attention 是唯一需要通信的模块——K/V 在 CP 组内
all-gather，由 **inner attention wrapper** 在区域内部完成。

### 6.1 数据管道：batch 必须先按 CP 切分

```python
from hyper_models.components.distributed.cp_utils import shard_batch_for_cp

cp_mesh = mesh["cp"]
batch = shard_batch_for_cp(batch, cp_mesh)   # 每个 rank 取自己的序列 chunk
```

契约与切分策略：

- 输入键：`input_ids`/`labels`/`position_ids` `[B, S]` int64；
  `seq_lens`/`seq_lens_padded`（packing 场景，单独重算）；`qkv_format`
  透传；
- pad 到 `2*cp_size` 的倍数后按 rank 连续切片（`labels` pad `-100`，
  `position_ids` 递增 pad）；
- embed 的 CP 契约（D-05）：input 已被管道切好 → `in/out` CP 维
  `Shard(1)`，框架不会二次切分。

### 6.2 内置 CP wrapper：注册表四路

CP 激活（`cp_size>1`）时，applier 对每个 attention 边界解析并注入一个
inner wrapper。全部内置方案登记在 **`CP_WRAPPER_REGISTRY`**
（`sharding_applier.py`）：

| 注册表名 | 适用模块风格 | 机制 |
|---|---|---|
| `"sdpa_qkv"` | **NeMo/Megatron 风格**：存在 `inner_attention`（或 `attn`/`attention`）子模块，其 forward 签名为 `(q, k, v)` | 整体替换 inner 子模块 forward：K/V 沿序列维 all-gather → SDPA → 取本地 Q 段输出 |
| `"sdpa_hf"` | **HF 风格**：attention 模块自身 `forward(hidden_states)` 内部调用 `F.scaled_dot_product_attention` | **拦截** `F.sdpa` 调用：在调用点插入 K/V all-gather 与 D-04 mask 修正，不替换模块 forward |
| `"flex_qkv"` | NeMo 风格 + FlexAttention（`(q,k,v)` 签名） | 同 sdpa_qkv，但走 `flex_attention` 路径 + block_mask 修正 |
| `"flex_hf"` | HF 风格 + FlexAttention（内部调 `flex_attention`） | 拦截 `flex_attention` 调用点 |

四路共同语义：

- **K/V all-gather**：`flex_cp_allgather`（cp_utils.py），通信组取
  `cp_mesh.get_group()`（DeviceMesh 缓存组，不重复建组）；
- **D-04 causal 修正**：`is_causal=True` 且 `cp_mesh.size()>1` 时，把
  is_causal 替换为 **offset-aware 显式 mask**——torch 的 is_causal 在
  q_len≠kv_len 时按左上角对齐，对 rank>0 的 CP chunk 是错的；
- **双模式容错**：wrapper 内部对 DTensor 输入做 unwrap/rewrap，
  production（local）与 validate（DTensor）共用同一份代码。

### 6.3 缺省分派：启发式 2×2

未显式指定时按 **模块风格 × attention 实现** 分派：

```
                SDPA (config._attn_implementation="sdpa"/缺省)    FlexAttention
HF 风格          sdpa_hf                                          flex_hf
NeMo 风格        sdpa_qkv                                         flex_qkv
```

（模块风格判定：有 `inner_attention`/`attn`/`attention` 属性 → NeMo；
类名含 "Attention"/"SdpaAttention" 或结构持有 q/k/v_proj → HF。）

### 6.4 可观察性与安全网

- **日志**：注入时 INFO 打印（边界 fqn、target、wrapper 名、来源：
  启发式分派/显式指定/自定义 callable）；
- **回写**：`spec._resolved_inner_wrapper = "sdpa_hf"` 等，plan 内省可查；
- **发火检测（misfire detection）**：`sdpa_hf`/`flex_hf` 拦截路在首次
  forward 检查是否真的拦到了原语调用——若模块内部根本没调
  `F.sdpa`/`flex_attention`（启发式猜错），立即 `RuntimeError` 并给出
  修复指引，**杜绝静默数值错误**；
- **定位失败 fail-fast**：声明了 CP 但 inner attention 定位不到 →
  `ValueError`，提示用 `inner_target` 显式指定（§10.4）。

### 6.5 端到端示例（TP×CP）

```python
mesh = init_device_mesh("npu", (2, 4), mesh_dim_names=("cp", "tp"))
plan = ShardingPlanner().plan(model, mesh, tp_size=4, cp_size=2)
model, tp_grad_info = apply_sharding_plan(model, plan, mesh)

for batch in dataloader:
    batch = shard_batch_for_cp(batch, mesh["cp"])   # 数据管道 CP 切分
    logits = model(batch["input_ids"])
    loss = local_chunk_loss(logits, batch["labels"])  # D-07：loss 在本地 chunk 计算
```

注意 lm_head 的 CP 契约（D-07）：CP 序列 all-gather **只发生在 attention
内部 K/V**，boundary 层 CP 维恒 identity——loss 直接在本地 chunk 上算，
不做 CP gather。

> 可运行示例：`examples/distributed/cp.py`（causal attention + 双模式对拍）。

### 6.6 TP×CP 组合时的序列布局：cp-major 嵌套切分

当 TP 与 CP 同时开启时，`hidden_states` 的序列维（dim 1）同时挂在 mesh 的
`cp`、`tp` 两个轴上（`{TP: Shard(1), CP: Shard(1)}`）。这不是"各切一遍"，
而是**嵌套切分**，顺序由 mesh 轴顺序决定（`cp` 在 `tp` 之前）：

```
序列先按 cp_size 粗切（外层），每个 CP 大块内再按 tp_size 细切（内层）
chunk_id  = cp_rank * tp_size + tp_rank       # cp-major
每 rank 持有 S/(cp_size*tp_size) 的连续 token 段
```

**示例**（S=16, TP=2, CP=2，chunk=4）：

| mesh 坐标 | 持有 token 区间 |
|---|---|
| (cp0, tp0) | [0, 4) |
| (cp0, tp1) | [4, 8) |
| (cp1, tp0) | [8, 12) |
| (cp1, tp1) | [12, 16) |

这个布局由两步自然形成，用户无需关心：`shard_batch_for_cp` 先做外层 CP 粗切
（§6.1），embed/attention 出口的 TP reduce-scatter 再做内层细分。

**对 norm / pointwise 模块的含义**：以 norm 的 spec 为例——

```python
params={"weight": {TP: Replicate(), CP: Replicate()}},      # weight 全复制
in_src/in_dst/out_src/out_dst 全为 {TP: Shard(1), CP: Shard(1)}   # 全 identity
```

norm **自己不切分、零通信**：入口/出口 placement 完全相同，PrecompiledBoundary
不生成任何通信 op;RMSNorm 的归约维是 hidden 维 H（未被切分），直接在本地
`[B, S/(tp*cp), H]` chunk 上逐 token 计算，数值与全局序列**逐位一致**。所有
逐 token pointwise 模块（mlp、norm 等）同理天然兼容该布局。

> ⚠️ 自定义模块注意：boundary 层 CP 维必须保持 `Shard(1)` identity（R8）。
> 若把 in_dst 的 CP 维写成 `Replicate`，会触发全序列 reduce-scatter 并产出
> **tp-major** 布局（chunk_id = `tp_rank*cp_size + cp_rank`），与上下游的
> cp-major 布局错位，导致**静默数值错误**（设计文档 05 §12.2 D-06）。

> 可运行示例：`examples/distributed/tp_cp_ep.py`（TP=2×CP=2×EP=2 三维组合，
> 与单卡参考对拍；日志逐 rank 打印其持有的 cp-major token 区间）。

---

## 7. EP 教程

EP（Expert Parallel）有两条接入路径，**内置 wrapper 只在路径 A 生效**。

### 7.1 路径 A：HF 原生 MoE（自动识别 + 内置 wrapper 注入）

planner 识别 HF 原生 MoE 结构（`mlp.gate` router + `mlp.experts` 参数），
当 `ep_size>1` 时自动：

1. **参数侧**：expert 权重按 `{EP: Shard(0)}` 在**派生 expert mesh
   `(edp, ep)`** 上分片（D-10 TP-extend-EP：EP 组 = flatten 连续 ep_size
   个 rank，先跨完 TP 组再向 dp/cp 扩展；每 rank 持
   `num_experts/ep_size` 个完整 expert）。两种布局都支持：
   - per-expert 布局（`experts.0.gate_proj.weight`…，旧版 HF/自研）：
     Phase A 前置 `_stack_moe_experts` 堆叠为 `[E, ...]`（D-09）；
   - batched 布局（`experts.gate_up_proj [E,2I,H]` +
     `experts.down_proj [E,H,I]`，HF 2025 重构后）：天生 stacked 直接
     分片（D-11）；
2. **前向侧**：注入内置 wrapper **`_hf_native_ep_compute`**（经
   local-region 解析链环 2，见 §10.3）——通信流与 Megatron
   `MoEAlltoAllTokenDispatcher`（expert_tensor_parallel_size=1）同构：

   ```
   SP-in（本地 chunk）
     → router（MOE_ROUTER_ADAPTERS 按 arch 选路由语义）
     → dispatch all-to-all（token → 目标 expert rank）
     → 本地 expert 计算（SwiGLU，fused/分离三矩阵均支持）
     → combine all-to-all（结果回源 rank 加权求和）
   → SP-out
   ```

   a2a 按后端分派：NCCL/HCCL 用不等长 `all_to_all`（零填充）；gloo 用
   pad-to-max `all_to_all_single`。

3. **路由语义注册表 `MOE_ROUTER_ADAPTERS`**（ep_utils.py）：
   `default`（softmax+topk）、`qwen3moe`/`qwen3_moe`、`mixtral`、
   `deepseekv3`/`deepseek_v3`（sigmoid + e_score_correction_bias +
   group-limited）、`glm4moe`/`glm4_moe`。自研路由可注册新 adapter。

```python
# 用户侧零配置：HF 模型 + ep_size>1 即全自动
mesh = init_device_mesh("npu", (2, 4), mesh_dim_names=("dp", "tp"))
plan = ShardingPlanner().plan(model, mesh, tp_size=4, ep_size=8)
model, tp_grad_info = apply_sharding_plan(model, plan, mesh)
```

**约束（plan 时校验，fail-fast）**：

- `ep_size` 不超过且整除 dense 区域（dp × cp × tp）；
- `num_experts % ep_size == 0`；
- v1 暂不支持 pp>1 与 expert bias。

### 7.2 路径 B：EP-aware 自研 MoE 模块（自带 dispatcher）

模块自身 forward 已实现 EP dispatch（如 Megatron 风格
`MoEAlltoAllTokenDispatcher`）→ 声明 `use_local_map=True`（§10.2）走
local-region 骨架，框架只做边界缝合。此时需自行设置模块的 EP 运行时
状态（类比 Megatron `init_token_dispatcher`）：

```python
ep_mesh = mesh["ep"]
for layer in model.model.layers:
    layer.mlp.experts.expert_offset = ep_mesh.get_local_rank() * n_local_experts
    layer.mlp.ep_group = ep_mesh.get_group()
```

### 7.3 两条路径的选择

| | 路径 A（HF 原生 MoE） | 路径 B（EP-aware 模块） |
|---|---|---|
| 触发 | planner 自动识别 + `ep_size>1` | `use_local_map=True` / `local_compute_fn` |
| 前向 wrapper | **内置 `_hf_native_ep_compute`** | 模块自身 forward（或用户 fn） |
| expert 分片 | 框架 `{EP: Shard(0)}` | 模块自行管理 |
| 适用 | qwen3_moe/glm4_moe/deepseek_v3 等 HF 模型开箱即用 | 自研 dispatcher（DeepEP 等） |

> 可运行示例：`examples/distributed/ep.py`（per-expert 布局 + D-09 堆叠 + TP=2×EP=2 双模式对拍）。

---

## 8. FSDP 组合（接口契约）

TP/CP/EP 切完后的 **dense 参数**由上游 FSDP（`hyper_parallel/core/
fully_shard` 的 `fully_shard`，FSDP2 语义）再做数据并行分片。组件与
FSDP 的接口是 **`tp_grad_info`**——production 模式 `apply_sharding_plan`
的第二个返回值：

```python
model, tp_grad_info = apply_sharding_plan(model, plan, mesh)
# tp_grad_info: {param_fqn: (tp_placement, tp_mesh)}
```

契约：

- `tp_placement ∈ {Shard, Replicate}`：标记该参数的梯度在 TP 组内是
  分片还是全量——FSDP 据此决定梯度同步域（Shard 梯度不能按 Replicate
  语义 all-reduce）；
- **tied 对归一化**：`embed_tokens`/`lm_head` 共享存储时两端映射到同一
  placement（Shard 优先）；
- **expert 参数**（TP-extend-EP）：梯度为各 rank 不同的 local shard
  （不同 expert + 扩展 EP 组聚合的 token），标记 `Shard(1)`——不做 TP
  组同步，FSDP 也不应对其按 dense 语义 all-reduce。

上游集成点（在 recipes/训练侧实现，接口示意）：

```python
# 上游训练流程：apply_sharding_plan 之后、fully_shard 之前
model, tp_grad_info = apply_sharding_plan(model, plan, mesh)

for fqn, (tp_placement, tp_mesh) in tp_grad_info.items():
    param = get_param_by_fqn(model, fqn)
    register_grad_sync_semantics(param, tp_placement, tp_mesh)  # 上游接口

fully_shard(model, mesh=dp_mesh, ...)   # hyper_parallel/core/fully_shard
```

> 说明：`tp_grad_info` 从 **plan** 而非 DTensor 读取（production 下参数
> 已解包，plan 是唯一保留完整 placement 信息的地方）。

---

## 9. 多维并行组合（mesh 布局）

| 组合 | mesh | plan 调用 | 说明 |
|---|---|---|---|
| 纯 TP | `(8,)` `("tp",)` | `tp_size=8` | §5 |
| 纯 CP | `(4,)` `("cp",)` | `cp_size=4` | §6，数据管道配合 |
| TP×CP | `(2, 4)` `("cp", "tp")` | `tp_size=4, cp_size=2` | TP 组连续（内层） |
| TP×CP×EP | `(2, 2)` `("cp", "tp")` | `tp_size=2, cp_size=2, ep_size=2` | 示例 `tp_cp_ep.py`；EP 组从 dense 区域派生 |
| TP×EP | `(2, 4)` `("tp", "ep")` 或 `("dp","tp")` | `tp_size=2, ep_size=4` | D-10：EP 组从 dense 区域派生，无需专用 etp 轴 |
| DP×CP×TP | `(2, 2, 2)` `("dp", "cp", "tp")` | `tp_size=2, cp_size=2` | dp 轴供 FSDP |
| 全组合 | `(2, 2, 4)` `("dp", "cp", "tp")` | `tp_size=4, cp_size=2, ep_size=8` | EP 组跨 TP 组向 dp/cp 扩展 |

要点：

- **轴顺序即 rank 映射**：`mesh_dim_names` 最右轴连续。TP 通信最频繁，
  放最内层；
- `ep_size` 是**扩展 EP 组大小**（D-10），从整个 dense 区域
  （dp×cp×tp）派生 `(edp, ep)` expert mesh，不要求 mesh 里有 "ep" 轴；
  mesh 里显式放 "ep" 轴仅路径 B（自研 EP-aware 模块取 `mesh["ep"]`）需要；
- size=1 轴自动剔除，可写全 `(dp, cp, tp)` 再按需开 size。

---

## 10. 自定义模块完整指南

### 10.1 `plan_overrides`：手写 spec 合并入口

一切自定义声明都经 `ModuleShardingSpec` 字段，在 plan 时合并（替换语义：
命中 planner 已生成的 fqn 则整体替换，结构标记从模板补齐；未命中且与
所有派生边界无祖孙关系则插入并参与链式传播）：

```python
from hyper_models.components.distributed import ModuleShardingSpec
from hyper_models.components.distributed.sharding_config import TP, CP
from hyper_parallel.core.dtensor.placement_types import Replicate, Shard, Partial

spec = ModuleShardingSpec(
    params={"q_proj.weight": {TP: Shard(0), CP: Replicate()},
            "o_proj.weight": {TP: Shard(1), CP: Replicate()}},
    in_src={"hidden_states": {TP: Shard(1)}},
    in_dst={"hidden_states": {TP: Replicate()}},
    out_src={TP: Partial()},          # 标量简写 → 自动归一化为 {"output": ...}
    out_dst={TP: Shard(1)},
)
planner = ShardingPlanner(plan_overrides={"model.layers.0.self_attn": spec})
```

与上游 `out_dst` 不一致**不再检查**（D-14：链式校验已废除，各模块只
断言自身策略传播，端到端正确性由双模式数值对拍兜底）；fqn
拼写错误 → `ValueError` fail-fast；spec 缺 `in_src`（`in_dst` 非空时）
→ `ValueError`（D-14 全声明强制化，链式填充已废除）。

**嵌套 spec（D-14，05 §13）**：override 的 fqn 可以是其他边界的祖先/
后代——外层边界（如整个 decoder layer、整个 LM 的根 fqn `""`）与内层
边界共存，常用于"外层只做 I/O 缝合（`params={}` + `use_local_map`），
内层关键模块跑 validate 孤岛"（§13.4）。唯一约束是**参数唯一归属**：
每个参数只能被一个边界声明（外层不得声明内层边界子树的参数），冲突
即 `ValueError`。想定制某个叶子（如 `self_attn.q_proj`），仍应直接
覆盖它所属的边界（`self_attn`，替换语义，参数名相对于边界模块）。

> 可运行示例：`examples/distributed/nested_local_map.py`（外层 local_map
> + 内层 validate 孤岛，production/validate 双模式对拍）。

### 10.2 四接口总览（两个家族）

| 字段 | 签名 | 做什么 | 生效方式 |
|---|---|---|---|
| `use_local_map` | `bool` | **纯门控**：模块自身 forward 即数据相关逻辑 → 走 local-region 骨架 | local 解析链环 3 |
| `local_compute_fn` | `fn(module, *args, **kw) -> Tensor` | 替换 local-region 骨架**内**的计算函数（边界缝合/双模式不变） | local 解析链环 1 |
| `inner_target` | `str`（属性名 / `"self"`） | **纯位置**：指定 inner attention 子模块 | target 解析链环 1 |
| `inner_wrapper` | `str`（注册表名）或 `fn(target, cp_mesh) -> None` | **纯行为**：固定内置 CP wrapper / 全自定义接管 | wrapper 解析链环 1-2 |

**门控派生原则**：声明互不嵌套、不改写任何标记——设置
`local_compute_fn` 后**不需要也不应**再设 `use_local_map`；applier 的
解析链（`_resolve_local_compute_fn` / `_resolve_inner_wrapper`）解析非
None 即注入。

### 10.3 local-region 族：`use_local_map` 与 `local_compute_fn`

骨架结构（通信保留，§6/§7 的边界通信照常执行）：

```
输入(in_src) → boundary入口(in_src→in_dst 通信) → 【compute_fn】
            → 按声明 out_src 重包装 → boundary出口(out_src→out_dst 通信) → 输出
```

解析链（优先级递减）：

1. `spec.local_compute_fn` —— 用户自定义计算；
2. planner EP 注入意图（HF 原生 MoE + `ep_size>1`）→ **内置
   `_hf_native_ep_compute`**；
3. `spec.use_local_map` —— 纯门控，compute 即模块自身 forward。

**示例 A：自研 MoE 自带 dispatch（纯门控）**

```python
spec = ModuleShardingSpec(
    params={...}, in_src=..., in_dst=..., out_src=..., out_dst=...,
    use_local_map=True,    # 模块 forward 内含 a2a，骨架只负责缝合
)
```

**示例 B：注入自定义 dispatch（local_compute_fn）**

```python
def my_deep_ep_compute(module, hidden_states):
    """在 local tensor 世界实现自定义 EP dispatch（如 DeepEP）。

    契约：fn(module, *args, **kwargs) -> Tensor；输入输出均为 local tensor，
    布局与 spec 的 in_dst/out_src 声明一致。区域内部可自由使用显式
    process group 通信。"""
    topk_idx, topk_w = module.router(hidden_states)
    dispatched, recv_counts = deep_ep_dispatch(hidden_states, topk_idx,
                                               group=module.ep_group)
    expert_out = module.experts(dispatched)
    return deep_ep_combine(expert_out, topk_w, recv_counts,
                           group=module.ep_group)

spec = ModuleShardingSpec(
    params={...}, in_src=..., in_dst=..., out_src=..., out_dst=...,
    local_compute_fn=my_deep_ep_compute,   # 链环 1：直接生效，无需 use_local_map
)
planner = ShardingPlanner(plan_overrides={"model.layers.3.mlp": spec})
```

骨架四步里只有 compute 被替换；in/out 边界的 all-gather/reduce-scatter
照常执行（详见 §4.3 声明式豁免）。

> 可运行示例：`examples/distributed/custom_local_compute_fn.py`（自研 top-1
> MoE + 自定义 batched expert 布局；含融合 gate_up 不可直接 Shard 的坑说明）。

### 10.4 inner-wrap 族：`inner_target` 与 `inner_wrapper`

机制：**定位 inner 子模块 + 替换/包装其 forward**。机制本身通用（不限
CP），CP（K/V all-gather）是第一个内置域。双解析链：

- **target 链（纯位置）**：`inner_target` 指定 > `inner_attention`/`attn`/
  `attention` 属性 > 类名判定 > q/k/v_proj 结构兜底；
- **wrapper 链（纯行为）**：callable 自定义 > str 注册表名 > （声明了
  inner_target 或命中 attention 模板时）启发式 2×2 分派 > 不注入。

**示例 C：自动定位失败 → `inner_target` 指定**

```python
spec = ModuleShardingSpec(
    ..., inner_target="core_attention",   # 属性名；"self" 表示模块本身
)
```

**示例 D：启发式会猜错 → str 固定内置方案**

```python
# 自研 attention 是 HF 风格但内部调的是自研 sdpa 封装，
# 强制走 (q,k,v) 替换路而不是拦截路
spec = ModuleShardingSpec(..., inner_wrapper="sdpa_qkv")
```

可选名：`"sdpa_qkv"` / `"sdpa_hf"` / `"flex_qkv"` / `"flex_hf"`（§6.2
各自的机制与适用场景）。未知名 fail-fast 并列出可用名。

**示例 E：内置四路覆盖不到 → callable 全自定义**

```python
def my_flash_cp_wrapper(target, cp_mesh, *, spec=None, mesh=None,
                        mesh_dim_names=()):
    """契约：fn(target, cp_mesh, *, spec, mesh, mesh_dim_names) -> None。

    整体接管：原地替换 target.forward。内部自行完成 K/V all-gather
    （cp_mesh.get_group()）与 causal mask 修正。双模式注意：validate 下
    输入可能是 DTensor——做 unwrap/rewrap 容错（推荐），或在边界模块
    声明 use_local_map 让骨架先转 local。"""
    orig_forward = target.forward

    def cp_forward(q, k, v, **kwargs):
        was_dtensor = isinstance(q, DTensor)
        if was_dtensor:
            q, k, v = q.to_local(), k.to_local(), v.to_local()
        k = all_gather_along_seq(k, cp_mesh.get_group())
        v = all_gather_along_seq(v, cp_mesh.get_group())
        out = flash_attn_varlen(q, k, v, causal=True, **kwargs)
        return DTensor.from_local(out, ...) if was_dtensor else out

    target.forward = cp_forward

spec = ModuleShardingSpec(
    ..., inner_target="core_attention", inner_wrapper=my_flash_cp_wrapper,
)
```

`inner_target` 与 `inner_wrapper` 可独立使用：callable 缺省 target 为
自动定位结果，定位不到时退化为边界模块本身（不会 fail-fast——用户
callable 自己负责）。

**示例 F：注册命名方案（团队共享）**

```python
from hyper_models.components.distributed.sharding_applier import CP_WRAPPER_REGISTRY

CP_WRAPPER_REGISTRY["my_flash"] = my_flash_cp_wrapper
# 之后任意 spec 可写 inner_wrapper="my_flash"
```

### 10.5 两个家族怎么选

| 场景 | 接口 |
|---|---|
| 模块 forward 内含数据相关逻辑（a2a/gather/mask），保持模块不变 | `use_local_map` |
| 保持骨架（边界缝合/双模式），只换区域内部计算 | `local_compute_fn` |
| attention 的 inner 子模块定位不到/有歧义 | `inner_target` |
| 内置 CP 方案选错/要固定 | `inner_wrapper="..."` |
| 内置四路不满足（flash_attn_varlen 直调等） | `inner_wrapper=callable` |

> 可运行示例：`examples/distributed/custom_inner_wrapper.py`（非标准 inner
> 属性名 + `CP_WRAPPER_REGISTRY` 注册命名方案 + `_resolved_inner_wrapper`
> 回写断言）。

---

## 11. 典型模型支持矩阵

已用 transformers 仓（v5.12.1）真实模型类验证（planner 全链路）：

| 模型 | ParamRole 命中 | 边界推导 | 备注 |
|---|---|---|---|
| llama / qwen2 / qwen3 | ✅ 零 SKIP | ✅ | qwen3 的 q_norm/k_norm 归 NORM |
| qwen3_moe | ✅ | ✅ `moe_mlp` + EP | batched experts（D-11），router adapter 已注册 |
| glm4 | ✅ | ✅ | fused `gate_up_proj`；q/k/v bias 归 BIAS；两个额外 post norm 归 NORM |
| glm4_moe | ✅ | ✅ `moe_mlp` + EP | shared_experts 归 SHARED_EXPERT（EP 维 Replicate） |
| deepseek_v2 / v3 | ✅（ARCH_OVERRIDES） | ✅ | **MLA**：q_a/kv_a 下投影 `REPLICATED`，q_b/kv_b 上投影 COLWISE；sigmoid router adapter 已注册 |
| mixtral | ✅ | ✅ `moe_mlp` + EP | per-expert 布局走 D-09 堆叠 |

DeepSeek MLA 覆盖条目（D-14，`ARCH_OVERRIDES` 内置，v2/v3 两种拼写）：

```python
[(["q_a_proj", "kv_a_proj_with_mqa"], ParamRole.REPLICATED),  # LoRA rank 维不切
 (["q_b_proj", "kv_b_proj"], ParamRole.COLWISE)]              # 按 head 维切
```

---

## 12. 排错索引

| 报错/现象 | 原因 | 解法 |
|---|---|---|
| `ValueError: ... named_modules` | plan_overrides 的 fqn 拼写错误 | 对照 `dict(model.named_modules())` |
| `ValueError: ... nests inside / ancestor / nested` | override 的 fqn 与派生边界（或另一 override）构成祖孙嵌套 | 改为覆盖所属边界本身（同 fqn 替换），见 §10.1 |
| `PlacementMismatchError` | validate 模式 out_src/out_dst 校验失败，或参数分片声明与实际不符 | 对照报错中的模块名与 placement 检查 spec 声明 |
| `chain contract mismatch`（警告） | 相邻模块 out_dst ≠ in_src | 边上有 reshape/transpose 属正常；否则检查声明并跑 validate 对拍 |
| `ValueError: ... inner_target` | CP 声明了但 inner attention 定位失败 | `inner_target="<属性名>"` 显式指定 |
| `ValueError: ... CP_WRAPPER_REGISTRY` | `inner_wrapper` str 未注册 | 用四个内置名之一，或先注册 |
| `RuntimeError: 未拦到 F.scaled_dot_product_attention` | 发火检测：启发式选了 `sdpa_hf` 但模块内部不调 F.sdpa | 显式 `inner_wrapper="sdpa_qkv"`，或提供 callable |
| `ep_size (...) 必须不超过且整除 dense 区域` | EP 组超出 dp×cp×tp | 调小 ep_size 或扩大 dense 区域 |
| `num_experts (...) 必须整除 ep_size` | expert 数不能均分 | 调整 ep_size |
| `NotImplementedError: ... 仅支持 SwiGLU expert` | 内置 EP compute 只支持 SwiGLU 三矩阵 | 自研 expert 结构 → `local_compute_fn` |
| validate 下自定义 inner_wrapper 报 DTensor 相关错 | wrapper 未做 DTensor 容错 | unwrap/rewrap（示例 E），或边界模块声明 `use_local_map` |
| production 前向 view/reshape shape mismatch（显式 `num_heads` 写法） | 模块 q/k/v 未被识别为 colwise（命名非标准），头数改写（D-17）未命中 | `ARCH_OVERRIDES` 注册命名规则（§5.5）使 q/k/v 归 COLWISE，或 `plan_overrides` 手写 spec |
| `head-count adjustment: ... not divisible`（警告） | 模块头数属性不能被 tp_size 整除，已保持原值 | 调小 tp_size；确认该属性确为头数（否则可忽略） |
| 参数未分片且无报错 | 命名不命中默认规则（落 SKIP，只有 warning） | `ARCH_OVERRIDES` 注册架构规则（§5.5） |

---

## 附：运行测试

```bash
python -m pytest tests/components/distributed/ -q   # 300 例
```

单进程用例直接跑；多进程用例经 `run_dist`（spawn + gloo/CPU，macOS 可跑），
覆盖 TP/CP/EP 及两两组合的 plan golden、production 数值（vs 单卡参考）、
validate 校验与双模式等价。
