# hyper_parallel/auto_models/components/distributed 双模式 DTensor 使用教程

> **文档层级（2026-08-10）**：本教程与 [patch_injection_mechanism.md](../../patch_injection_mechanism.md)
> 是使用/机制语义的**现行口径唯一权威**；其余设计文档与代码串讲为辅助
> 材料，说法冲突时以这两份为准。
> 适用组件：`hyper_parallel/auto_models/components/distributed/`
> 设计文档：[auto_models_dual_mode_dtensor_design.md](../../auto_models_dual_mode_dtensor_design.md)
> 代码走读：[components_distributed_code_walkthrough.md](../../trainer/code_guides/components_distributed_code_walkthrough.md)；
> FunctionModule 机制走读：[function_module_autograd_walkthrough.md](../../trainer/code_guides/function_module_autograd_walkthrough.md)
> 场景配方：[examples/recipes/](../../../examples/recipes/README.md)（按 arch/拓扑可引用的 YAML 起点）

本教程覆盖：**TP / CP / EP / FSDP 组合、production↔validate 双模式切换、
自定义模块注入接口（`region_dispatch` / `local_compute_fn` / `inner_target` /
`inner_wrapper` / `inner_out_src`）、TP-local 属性整除（`tp_divide_attrs`，
§5.7）**，并讲清仓内为 CP/EP 提供了哪些参考
实现函数、如何显式注入、如何显式固定或整体替换。

**阅读路径**：§1-§2 跑通 → §3 建立双模式心智模型（含 region_dispatch
一分钟判断）→ §4 用内省工具看懂自己模型的 plan → §5-§9 按需深入
TP/CP/EP/FSDP/多维组合 → §10 自定义注入完整指南 → §11-§13 参考与排错。

---

## 目录

1. [组件概述](#1-组件概述)
2. [五分钟快速开始（TP）](#2-五分钟快速开始tp)
3. [双模式：validate 与 production](#3-双模式validate-与-production)（含 region_dispatch 一分钟判断 §3.4）
4. [看懂自己的 plan：内省与判定工具](#4-看懂自己的-plan内省与判定工具)（plan.explain() §4.1 与 check_dispatchable §4.2）
5. [TP 教程](#5-tp-教程)
6. [CP 教程（含仓内 4 个 CP wrapper 参考实现详解）](#6-cp-教程)
7. [EP 教程（含仓内 EP compute 参考实现详解）](#7-ep-教程)
8. [FSDP 组合（接口契约）](#8-fsdp-组合接口契约)
9. [多维并行组合（mesh 布局）](#9-多维并行组合mesh-布局)
10. [自定义模块完整指南](#10-自定义模块完整指南)（region_dispatch 公理 §10.1）
11. [核心 API 参考](#11-核心-api-参考)
12. [典型模型支持矩阵](#12-典型模型支持矩阵)
13. [排错索引](#13-排错索引)

---

## 1. 组件概述

`hyper_parallel/auto_models/components/distributed` 是独立可用的 DTensor 分片组件，零依赖训练流程
（不 import `recipes/` / `models/` / `datasets/`），两步用法：

```
ShardingPlanner.plan(model, mesh, ...)   → ShardingPlan   # 编译期推导（6-phase）
apply_sharding_plan(model, plan, mesh)   → (model, source_shard_info)  # 双模式应用
```

- **Planner**：遍历 `named_parameters()`，按命名规则（`ParamRole`）+ 语义
  模板（`TEMPLATES`：attention/mlp/norm/embed/lm_head/moe_gate/moe_mlp）
  自动推导每个通信边界的参数 placement 与 I/O 契约（`in_src/in_dst/
  out_src/out_dst`）。推导不满的地方用 `plan_overrides` 手写 spec 合并。
- **Applier**：Phase A 参数分片 → Phase C 包装 forward（PrecompiledBoundary
  通信缝合 + local-region / inner-wrap 注入）→ Phase D tied weights。
  同一个 plan 可按两种模式应用（§3）。

**用户代码恒工作在 local tensor 世界**：DTensor↔local 的缝合、边界通信
（all-gather/reduce-scatter）由框架负责。

---

## 2. 五分钟快速开始（TP）

完整可运行示例见 `examples/distributed/`（gloo/CPU 可跑，torchrun 启动）：

```python
import torch.distributed as dist
from hyper_parallel.core.dtensor.device_mesh import init_device_mesh
from hyper_parallel.auto_models.components.distributed import ShardingPlanner, apply_sharding_plan

dist.init_process_group("gloo")   # 或 nccl/hccl

# 1) 构建 device mesh：1D TP
mesh = init_device_mesh("cpu", (dist.get_world_size(),), mesh_dim_names=("tp",))

# 2) 编译期推导（零模型代码改动，任意 HF 风格命名均可）
planner = ShardingPlanner()
plan = planner.plan(model, mesh, tp_size=dist.get_world_size())

# 3) 应用分片（production 模式）
model, source_shard_info = apply_sharding_plan(model, plan, mesh)

# 4) 正常训练/推理——前向输出与单卡逐位一致
out = model(input_ids)
```

运行：

```bash
PYTHONPATH=. torchrun --nproc_per_node=2 examples/distributed/tp.py
```

### 2.1 示例目录一览

[`examples/distributed/`](../../../examples/distributed/) 十二个独立示例，均与单卡
参考做数值对拍：

| 示例 | 并行 | 演示点 |
|---|---|---|
| `tp.py` | TP=2 | 零配置自动推导 + 应用（本节代码的完整版） |
| `cp.py` | CP=2 | `shard_batch_for_cp` + 仓内参考 `"sdpa_hf"` wrapper + D-04 causal 修正（§6） |
| `ep.py` | TP=2×EP=2 | HF 原生 MoE 一行显式注入：D-09 堆叠 + D-10 TP-extend-EP 分片 + 仓内默认 `hf_native_ep_compute_fn`（§7） |
| `tp_cp_ep.py` | TP=2×CP=2×EP=2 | 三维组合：cp-major 序列布局（§6.6）+ plan 内省断言（`_ep_stack`/`_needs_cp_attn` 等） |
| `nested_local_map.py` | TP=2（嵌套） | D-14 嵌套 spec：外层 local_map（根 fqn `""`）+ 内层 validate 孤岛，双模式对拍（§10.2） |
| `multimodal_encoder_dp.py` | ViT dp=4 + LLM dp=2×tp=2×ep=4 | 多模态双 mesh：encoder_dp ViT（`params={}` 纯 FSDP 公民，dp 语义由 vit_mesh + 数据分配表达——plan 坐标系 = 单 dp 切片）+ 桥接边界 all-gather（out 边界 `Shard(0)→Replicate`，plan_overrides 注入）+ LLM 独立 plan/apply |
| `custom_local_compute_fn.py` | TP=2 | 自研 MoE：`plan_overrides` + `local_compute_fn` 注入自定义 compute（§10.4） |
| `custom_inner_wrapper.py` | CP=2 | 自研 attention：`inner_target` + 注册表命名 wrapper（§10.5） |
| `custom_autograd_function.py` | TP=2 | 自定义 autograd.Function：第三方宿主裸调用 → `__class__` 替换 + `FunctionModule` + plan_overrides（§10.8） |
| `plan_overrides_demo.py` | TP=2 | **plan_overrides 全场景**（YAML）：merge 注入 / 契约 DSL / 显式空 / `when` 条件 / insert 自声明——plan 内省逐场景断言（§10.2） |
| `perf_replacement.py` | TP=2 | **YAML 性能替换双通道**：朴素实现 → 用户高性能 kernel——`local_compute_fn` 工厂（骨架托管）与 `inner_wrapper` 原地替换（双模适配器托管）对比；YAML→plan_overrides 全链路 + 脱糖打印（§6.3/§10.4） |
| `programmatic_injection.py` | TP=2 | **编程式注入五形态**（不接 trainer/YAML）：装饰 callable 直传 / Target / 注册表名 / `@local_compute` 工厂直传 / 工厂 Target——一个 plan_overrides dict 全覆盖（§10.7） |

另有**场景配方库** [examples/recipes/](../../../examples/recipes/README.md)（2026-08-10）：
按 arch/拓扑直接可引用的 YAML 起点（llama_tp / llama_tp_cp / qwen3moe_tp_ep /
custom_ep_moe）——新用户的起点从"空白配置"变成"改一份能跑的最近配方"。

**先 validate 再 production**（推荐工作流，详见 §3）：

```python
# 同一份 plan，先跑 validate 校验契约/数值，再切 production 训练
model_v, _ = apply_sharding_plan(model, plan, mesh, validate_mode=True)
```

---

## 3. 双模式：validate 与 production

### 3.1 语义对照

| | production（训练/推理） | validate（校验） |
|---|---|---|
| 参数 | build 期**永久解包**为 plain local tensor | 保持 **DTensor** |
| 前向 | 纯 local tensor + PrecompiledBoundary 通信 | DTensor dispatch 传播 + out_src/out_dst **契约校验** |
| 反向 | local autograd（梯度落 local 分片） | local autograd（同左） |
| 返回值 | `source_shard_info`（供 FSDP） | `None` |
| 用途 | 生产训练 | 分片正确性验证 / 调试新模型接入 |

### 3.2 推荐工作流：先 validate 再 production

```python
plan = ShardingPlanner().plan(model, mesh, tp_size=4, cp_size=2)

# Step 1: validate——DTensor dispatch 逐边界校验契约，数值与单卡对拍
model_v, _ = apply_sharding_plan(model, plan, mesh, validate_mode=True)
out_v = model_v(batch)
torch.testing.assert_close(out_v, ref_out, rtol=1e-5, atol=1e-5)

# Step 2: 同一份 plan 切 production——零 DTensor dispatch 开销
model_p, source_shard_info = apply_sharding_plan(model, plan, mesh)
```

**架构约束（双模式等价的关键）**：凡 DTensor dispatch 无法表达数据相关
逻辑的模块（embedding mask、attention K/V gather、MoE all-to-all），两
模式注入**同一份 wrapper** 显式重建该逻辑——区域内计算路径逐指令一致，
因此双模式数值可达 kernel 级等价。

### 3.3 validate 的声明式豁免清单

- **attention（CP>1）**：`out_src` 为声明式——CP wrapper 出口按声明
  `from_local` 重包装（K/V all-gather 语义 dispatch 无法派生）；
- **MoE（local-region）**：`out_src` 为声明式（all-to-all 数据相关性）；
  `in_src` 契约仍由 boundary 正常校验；
- 其余模块（embed/norm/mlp/lm_head）：`out_src` 由 DTensor dispatch
  **派生校验**——这是 validate 的核心校验能力。

### 3.4 注入与 region_dispatch：一分钟判断

读到后面的 CP/EP 章节，你会在配置里遇到 `region_dispatch: false`——它
回答的问题是"注入的计算函数能否被 DTensor dispatch 穿透"，决定 validate
模式对该区域做**穿透真校验**还是**黑盒托管**。判断口诀：

> 注入物含通信原语 / 自定义 kernel / 数据依赖分支 → `False`（黑盒托管）；
> 纯 aten 标准算子 → `True`（穿透真校验）。
> 拿不准时用 `check_dispatchable(fn, example_inputs, mesh)` 在开发期探明（§4.2）。

规则只有两条：**声明注入（`local_compute_fn`/`inner_wrapper`）时必填、
无默认**；未注入的普通边界不写（默认可穿透，多写 `True` 反而 fail-fast）。
完整公理与语义见 §10.1；production 模式恒 local 直通，不受影响。

---

## 4. 看懂自己的 plan：内省与判定工具（2026-08-10）

**推荐学习路径**：不必先从概念建模心智模型——先 `plan.explain()` 看懂
自己模型的实际切分，再反推概念，快得多。

### 4.1 `plan.explain(fqn=None)`：plan 内省报告器

```python
plan = planner.plan(model, mesh, tp_size=2, explain=True)  # 末尾 INFO 打出报告
print(plan.explain())                    # 或独立调用；fqn= 只看单个边界
```

对每个边界打印：参数切分表（参数名 → placement）、**编译后的边界通信计划**
（哪个张量、从什么布局到什么布局、对应什么集合通信——all_gather /
reduce_scatter / 直通）、注入声明与解析结果（`region_dispatch` 的含义当场
可见）、TP-local 属性整除计划（D-17/D-18，auto/user 分列，§5.6/§5.7）、
特殊处理器清单。insert 模式契约写不全时，fail-fast 报错会附带按
forward 签名推导的**建议 spec 草稿**（改成你的布局即可用）。

### 4.2 `check_dispatchable(fn, example_inputs, mesh)`：region_dispatch 判定工具

```python
from hyper_parallel.auto_models.components.distributed import check_dispatchable
report = check_dispatchable(my_compute_fn, [x_local, w_local], mesh)
print(report)   # dispatchable=True/False + 首个失败算子 + 填写建议
```

写注入函数时拿不准 `region_dispatch` 填 True 还是 False？本工具用 DTensor
试跑注入函数、记录 dispatch 轨迹：全程无异常 → 建议 `True`；任一算子失败
（通信原语/未注册算子/数据依赖分支）→ 报告**首个失败算子**并建议 `False`。
把试错从 apply 期提前到开发期。注意它只判定"能否 dispatch"，布局正确性仍由
validate 模式的 out_src 真校验兜底。

---

## 5. TP 教程

### 5.1 基础 TP（qwen/llama/glm 类模型零配置）

默认命名规则自动识别：`q/k/v/gate/up_proj` → COLWISE（`Shard(0)`），
`o/down_proj` → ROWWISE（`Shard(1)`），norm/embed/lm_head 各归其位。
融合权重也支持：`qkv_proj/fused_qkv/query_key_value`（FUSED_QKV）、
`gate_up_proj`（FUSED_GATE_UP）均按 `Shard(0)`。

**rowwise Linear 的执行方式（含 bias，D-19/D-22，2026-08-12）。**
rowwise 投影（`o/down_proj`）的权重沿**输入维**切分（`Shard(1)`）：边界入口
把激活 all-gather 成 `Replicate`（SP 下），每 rank 用本地的权重列块做
matmul，产出 `Partial` 贡献，边界出口的归约通信（SP：reduce-scatter；
非 SP：all-reduce）把各 rank 贡献求和——这就是输出。bias 的处理按归属
分三种：

| bias 归属 | placement | 加法位置 |
|---|---|---|
| colwise 投影（`q/k/v/gate/up_proj`） | `Shard(0)`（随权重、与输出通道同切，D-19） | 区域内本地 matmul 后直接加——结果随后被 rowwise 消费，不经过边界归约 |
| **rowwise 投影（`o/down_proj`）** | **`Replicate`（不切——它是完整输出向量）** | **区域内抑制、边界归约之后恰好加一次（D-22，Megatron `RowParallelLinear` 同构）** |
| 命名未命中规则的 bias（norm/router 等） | `Replicate` | 区域内（这些边界输出非 Partial，本就正确） |

为什么 rowwise bias 必须后置：`F.linear` 会把 bias 融合加在 matmul 结果
上，若放任不管，边界出口的求和归约会把它**累计 tp_size 次**（输出 =
正确值 + tp×bias）。因此 planner 在 plan 期识别"权重 `Shard(1)` + 兄弟
bias + 边界 out_src 为 Partial"的 nn.Linear（与用户自声明 spec、非标准
命名如 `wo` 同样生效——判定锚定最终 spec 声明而非命名规则），apply 期
让区域内 forward 暂时不带 bias（bias Parameter 原地保留，state_dict /
optimizer 零影响），在边界出口归约完成后统一加回。该标记在
`plan.explain()` 的"后置 bias"行可见。两个保护性规则：owner 不是
`nn.Linear`（如 GPT-2 `Conv1D`、自研线性层）→ WARNING 并保持现状（请把
bias 移到边界通信后、改用 nn.Linear 或用 `local_compute_fn` 接管）；
**lm_head 带 bias**（权重沿输出维 `Shard(0)` 而 bias 无法随切）→ plan 期
直接报错"模板不匹配"，按报错提示用 `plan_overrides` 显式声明
`{"lm_head.bias": {TP: shard(0)}}`。

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
于 Replicate），`source_shard_info` 中 tied 对映射到同一 placement——无需
用户处理。

### 5.5 非标准命名 → ARCH_OVERRIDES

命名不命中默认规则时（如自研 `wq/wk/wv`），注册架构覆盖：

```python
from hyper_parallel.auto_models.components.distributed.sharding_planner import ARCH_OVERRIDES
from hyper_parallel.auto_models.components.distributed.param_role import ParamRole

ARCH_OVERRIDES["myarch"] = [
    (["wq", "wk", "wv"], ParamRole.COLWISE),
    ("wo", ParamRole.ROWWISE),
]
# config.architectures=["MyArchForCausalLM"] 或 model_type="myarch" 即生效
```

已内置 DeepSeek MLA 条目（§12）。

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
| validate | local-region（声明了注入的边界：`local_compute_fn`/wrapper/EP） | **自动改写**（`region_dispatch=False` 时区域内两模式都是 local tensor） |

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

### 5.7 TP-local 属性整除：`tp_divide_attrs`（D-18）

D-17 只覆盖框架已知的头数名。模型还有其他**随 TP 缩放的 int 实例属性**
时（自定义头数别名、`__init__` 里缓存的宽度尺寸等），用
`tp_divide_attrs` 显式声明，框架按与 D-17 完全相同的双模式时机
（forward 看到 local tensor 处）把它们整除改写为本地值：

```python
# 编程式（plan_overrides merge：只写该字段，契约继承推导）
ShardingPlanner(plan_overrides={
    "*.self_attn": ModuleShardingSpec(tp_divide_attrs=["hidden_size"]),
})
# YAML（trainer 路径）等价写法：
#   plan_overrides:
#     - match: "*.self_attn"
#       tp_divide_attrs: ["hidden_size"]
```

规则与防呆（全部 plan 期 fail-fast，报错点名 fqn 与属性）：

- 属性必须在模块实例上**存在且为 plain int**（bool 拒绝）、**>0 且整除
  tp_size**——用户显式声明不像 D-17 自动段那样"警告跳过"，写错即失败；
- **保护清单**：`head_dim`/`attention_head_size`/`head_size`（头维度不被
  切）、`num_key_value_groups`（GQA 比值，TP 不变量）、`training`/
  `dtype`/`device`、`_hp_` 前缀（框架内部存储）不可声明；
- **不要重复声明 D-17 头数名**（`num_heads` 等）——auto 段已自动覆盖，
  重叠即报错，从 YAML 删除即可；
- **幂等**：原值存 `module._hp_full_tp_local_attrs`，重复 apply no-op；
  属性已被改写且与当前 tp_size 不兼容（同模块被不同 tp 的计划重复
  apply）→ fail-fast；
- merge 语义同其他声明字段：不写继承 glob 声明，显式 `[]` 清空继承；
  同一 match 多条 YAML 声明时后者覆盖前者；
- `plan.explain()` 的 "TP-local 属性整除" 段把 auto（D-17）/ user（本
  字段）分列打印，声明结果当场可见（§4.1）。

---

## 6. CP 教程

CP（Context Parallel）沿序列维切分激活。**参数不切**（CP 维参数恒
`Replicate`）；attention 是唯一需要通信的模块——K/V 在 CP 组内
all-gather，由 **inner attention wrapper** 在区域内部完成。

### 6.1 数据管道：batch 必须先按 CP 切分

```python
from hyper_parallel.auto_models.components.distributed.cp_utils import shard_batch_for_cp

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

### 6.2 仓内 CP wrapper 参考实现：注册表四路

inner-wrap 机制是**通用的"织入/替换 inner forward"通道**：声明即应用，
本身不由 CP 门控。仓内为 CP 语义（K/V all-gather）提供了四个**参考实现
函数**——它们与用户自己的 `@inner_wrapper` 函数地位完全平等，框架对它们
**零特殊对待、零默认**（用不用、用哪个都由用户显式声明），只是顺手登记在
**`INNER_WRAPPER_REGISTRY`**（`cp_wrappers.py`）里可按名引用；四路内置
wrapper 的静态要求由 **`INNER_WRAPPER_REQUIREMENTS`** 表记录并在 apply
前置守门强制（D-20）：必须存在活跃 cp 轴（无 cp 轴声明它们会 fail-fast
——自定义 callable/Target 不受此限，`cp_mesh` 传 `None`、语义自负）、
`region_dispatch` 必须为 `False`（内含 CP 集合通信，误写 `True` 在 apply
前即报错并附建议 YAML）：

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
- **local-only**：wrapper 只面向 local 张量——validate 的 DTensor 解包/
  参数临时解包/输出重包由框架的**双模适配器**统一托管（§10.5.1），四路
  仓内参考 wrapper 体内零 DTensor 代码；
- **重包布局来自显式声明**：target=self（`sdpa_hf`/`flex_hf`）用边界
  `out_src` 契约；inner 子模块（`sdpa_qkv`/`flex_qkv`）必须在 plan 里声明
  `inner_out_src: "first_input"`（输出布局 == q 布局），未声明 fail-fast。

### 6.3 显式选择：无缺省分派（2026-08-04 改造）

**启发式 2×2 分派已删除**——框架不再替用户猜 wrapper，cp>1 时 attention
边界必须显式声明 `inner_wrapper`；缺声明会在 apply 前被
`_preflight_compute_injection` fail-fast（报错附可粘贴的配置片段）。
**行为变化（inner-wrap 泛化后）**：以往 cp=1 时写了 `inner_wrapper` 会被
静默忽略；现在声明即应用（自定义 wrapper 无 CP 也会真的包装），仓内参考
CP 方案在无 cp 轴时 fail-fast 并指引改用 `local_compute_fn`。
按 **模块风格 × attention 实现** 自行对照选择：

```
                SDPA (config._attn_implementation="sdpa"/缺省)    FlexAttention
HF 风格          sdpa_hf                                          flex_hf
NeMo 风格        sdpa_qkv                                         flex_qkv
```

（模块风格判定：有 `inner_attention`/`attn`/`attention` 属性 → NeMo；
类名含 "Attention"/"SdpaAttention" 或结构持有 q/k/v_proj → HF；
`cp_wrappers.is_hf_style_attention` 等 helper 可程序化判定。）

声明方式（三形态平权）：

```python
# ① plan_overrides glob merge（契约继承推导，只需注入字段；
#    inner_target 与 inner_wrapper 必须成对显式声明——自动定位启发式
#    已删除；解析结果回写 spec._resolved_inner_target 并打 INFO 日志）
ShardingPlanner(plan_overrides={
    "*.self_attn": ModuleShardingSpec(
        inner_target="self", inner_wrapper="sdpa_hf",
        region_dispatch=False)})   # 必填伴生声明：wrapper 内含 K/V all-gather
# 注意：inner 子模块路径（sdpa_qkv/flex_qkv 或自定义 inner_target）必须
# 额外声明 inner_out_src（框架对 inner 输出布局零推导零猜测），layout-
# preserving 的 attention 写 "first_input" 即可：
#   ModuleShardingSpec(inner_target="core_attention",
#                      inner_wrapper="sdpa_qkv",
#                      inner_out_src="first_input",
#                      region_dispatch=False)
# ② YAML（trainer 路径，plan_overrides 脱糖后走同一通道）
#    plan_overrides:
#      - match: "*.self_attn"
#        when: cp            # 激活条件自述必要性：cp_size>1 才应用（缺省=总是）
#        region_dispatch: false   # 必填：wrapper 内含通信，不可 dispatch
#        inner_wrapper:
#          _target_: hyper_parallel.auto_models.components.distributed.cp_wrappers.sdpa_hf_cp_wrapper
# ③ plan 后直接赋值（两字段都要补）：
#    spec = plan.modules["...self_attn"]
#    spec.inner_wrapper = "sdpa_hf"; spec.region_dispatch = False
```

YAML 形态（`plan_overrides`，2026-08-05 改名自 `sharding.injections`）
还有三个独有字段：`when`（激活条件 `"cp"`/`"ep"`——自述 CP/EP 注入的
必要性，条件不满足时跳过并打日志，一份配置跨拓扑复用）；契约字段
`params`/`in_src`/`in_dst`/`out_src`/`out_dst`（placement 字符串 DSL
`"replicate"`/`"partial"`/`"shard(N)"` + 哨兵 `"auto"`/`"none"`——insert
模式因此可纯 YAML 表达；脱糖期闭集文法 fail-fast，plan 期
`_validate_override_axes` 轴名拼写 fail-fast）；`tp_divide_attrs`
（D-18 TP-local 属性整除，§5.7——不写继承 glob 声明、显式列表覆盖、
`[]` 清空继承，同一 match 多条 YAML 声明后者覆盖前者）。

### 6.3.1 inner-wrap 也是性能替换通道（不限于 CP）

inner-wrap 泛化后（2026-08-05），`inner_wrapper` 不再由 cp_mesh 门控——
**声明即应用**，任何并行模式（含纯 TP/DP、无 CP）都会注入。它与
`local_compute_fn` 形成两档替换通道：

| | `local_compute_fn`（推荐） | `inner_wrapper` |
|---|---|---|
| 语义 | 整体接管边界计算 | 织入/替换 inner forward（可定位子模块） |
| 张量形态 | local-region 骨架托管：参数解包 + I/O 契约，kernel 只见本地张量 | 双模适配器托管（§10.5.1）：替换后的 forward 只见本地张量，重包按声明（out_src / inner_out_src） |
| 典型场景 | EP compute、整模块性能替换 | CP K/V gather、探针/日志织入、只换某个子模块 |

最小完整示例（纯算子替换，与 CP 无关——框架填入的 `cp_mesh` 为
None，签名里声明但不用）：

```python
@inner_wrapper
def flash_attention_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
    """契约：fn(target_module, mesh, tp_mesh, cp_mesh, ep_mesh) -> None——
    必选上下文缺一不可（用不到也声明）、不得有默认值、禁 *args/**kwargs；
    原地替换 forward，返回值必须为 None。"""
    def fast_forward(hidden_states, *args, **kwargs):
        return flash_attn_op(target_module.q_proj(hidden_states), ...)
    target_module.forward = fast_forward

plan_overrides = {"*.self_attn": ModuleShardingSpec(
    inner_target="self",                       # 成对必填：包装边界模块自身
    inner_wrapper=flash_attention_wrapper,     # callable 直传（最常用）
    region_dispatch=True)}                     # 纯算子无通信 → True（穿透真校验）
```

- 仓内四个 CP 参考方案仍**需要活跃 cp 轴**——无 cp 轴时框架填入的
  `cp_mesh=None`，参考 wrapper 自检 fail-fast；自定义 wrapper 不受限
  （收到 `cp_mesh=None`，语义自负）；
- 同一边界两通道同时声明会**叠加**（inner-wrap 先替换 forward，
  local_compute_fn 在区域内再被调用）——替换场景二选一；
- 端到端对比示例（含 YAML 形态）：`examples/distributed/perf_replacement.py`
  + `perf_replacement.yaml`（local_compute_fn 通道）+
  `perf_replacement_inner_wrap.yaml`（inner_wrapper 通道）。

### 6.4 可观察性与安全网

- **缺注入 fail-fast**：cp>1 而 attention 边界无 `inner_wrapper` → apply 前
  `ValueError`（`_preflight_compute_injection`，报错含具体 fqn 与可粘贴
  YAML 片段）；
- **日志**：注入时 INFO 打印（边界 fqn、target、wrapper 名、来源：
  注册表/自定义 callable/Target）；
- **回写**：`spec._resolved_inner_wrapper = "sdpa_hf"` 等（Target 形态回写
  target_path），plan 内省可查；
- **发火检测（misfire detection）**：`sdpa_hf`/`flex_hf` 拦截路在首次
  forward 检查是否真的拦到了原语调用——若模块内部根本没调
  `F.sdpa`/`flex_attention`（wrapper 型号选错），立即 `RuntimeError` 并给出
  修复指引，**杜绝静默数值错误**；
- **成对声明强制**：声明 `inner_wrapper` 必须同时显式声明
  `inner_target`（`"self"` 或子模块属性名；自动定位启发式已删除），缺失
  → `ValueError`；仅 `inner_target` 无 `inner_wrapper` 同样 fail-fast
  （定位不能代替方案选择）——两者均在 apply 前置守门（D-20）拦截；
- **内置 wrapper 静态要求**（D-20）：仓内四路 CP wrapper 声明了
  `region_dispatch: true`、或无活跃 cp 轴、或子模块 target 缺
  `inner_out_src` → apply 前 `ValueError`（报错附可粘贴的建议 YAML）；
- **inner 输出布局强制声明**：inner 子模块路径未声明 `inner_out_src` →
  apply 时 `ValueError`（报错给出可粘贴的声明写法）——框架对 inner 输出
  布局零推导零猜测；
- **注入纪律**：未装饰/种类不符/缺必选上下文/配置保留键/替换 forward
  入参与原 forward 不兼容 → import 期或 apply 时 fail-fast（§10.5.1）。

### 6.5 端到端示例（TP×CP）

```python
mesh = init_device_mesh("npu", (2, 4), mesh_dim_names=("cp", "tp"))
# HF 风格 attention 显式注入 "sdpa_hf"（glob merge，契约继承推导；
# inner_target="self" 必填——与 inner_wrapper 成对显式声明，包装边界
# 模块自身；region_dispatch=False 必填：wrapper 内含 K/V all-gather 通信）
plan = ShardingPlanner(plan_overrides={
    "*.self_attn": ModuleShardingSpec(
        inner_target="self", inner_wrapper="sdpa_hf",
        region_dispatch=False),
}).plan(model, mesh, tp_size=4, cp_size=2)
model, source_shard_info = apply_sharding_plan(model, plan, mesh)

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

EP（Expert Parallel）有两条接入路径，**仓内参考实现 compute 只覆盖路径
A，且需显式注入**（框架不做任何自动注入/默认——仓内函数与用户函数地位
完全平等）。

### 7.1 路径 A：HF 原生 MoE（自动识别 + 显式注入仓内参考 compute）

planner 识别 HF 原生 MoE 结构（`mlp.gate` router + `mlp.experts` 参数），
当 `ep_size>1` 时自动完成**参数侧**工作；**前向 compute 由用户显式注入**：

1. **参数侧**：expert 权重按 `{EP: Shard(0)}` 在**派生 expert mesh
   `(edp, ep)`** 上分片（D-10 TP-extend-EP：EP 组 = flatten 连续 ep_size
   个 rank，先跨完 TP 组再向 dp/cp 扩展；每 rank 持
   `num_experts/ep_size` 个完整 expert）。两种布局都支持：
   - per-expert 布局（`experts.0.gate_proj.weight`…，旧版 HF/自研）：
     Phase A 前置 `_stack_moe_experts` 堆叠为 `[E, ...]`（D-09）；
   - batched 布局（`experts.gate_up_proj [E,2I,H]` +
     `experts.down_proj [E,H,I]`，HF 2025 重构后）：天生 stacked 直接
     分片（D-11）；
2. **前向侧**：经统一 override 通道显式注入仓内参考 compute
   **`hf_native_ep_compute_fn`**（ep_compute.py 公开工厂，走 local-region
   解析链环 1，见 §10.4；**伴生声明 `region_dispatch=False`**——compute
   内含 all-to-all 通信原语，区域内不可 dispatch）——通信流与 Megatron
   `MoEAlltoAllTokenDispatcher`（expert_tensor_parallel_size=1）同构：

   ```
   SP-in（本地 chunk）
     → router（内嵌 default softmax top-k——路由是注入函数的一部分）
     → dispatch all-to-all（token → 目标 expert rank）
     → 本地 expert 计算（SwiGLU，fused/分离三矩阵均支持）
     → combine all-to-all（结果回源 rank 加权求和）
   → SP-out
   ```

   a2a 按后端分派：NCCL/HCCL 用不等长 `all_to_all`（零填充）；gloo 用
   pad-to-max `all_to_all_single`。

3. **路由内嵌纪律**：框架不决定用户的 router——spec 里没有路由提示字
   段，工厂也不接受函数类型的配置参数。仓内按 gate 形态提供三个参考工厂
   （共享同一 a2a 骨架，按需选一个注入即可）：
   - `hf_native_ep_compute_fn`：内嵌默认 `_softmax_topk_router`
     （softmax+topk，gate 返回 logits 的旧式形态）；
   - `hf_topk_router_ep_compute_fn`（2026-08-12）：gate 为 TopKRouter
     模块、forward 返回 `(logits, scores, indices)` 三元组的形态
     （HF 2025 重构后的 qwen3moe/mixtral 等）——直接注入，无需再写
     工厂；`qwen3moe_ep_compute_fn` 是按名引用的等价变体；
   - 路由语义不同的其他 MoE（DeepSeek sigmoid-group 等）**写自己的工厂**，
     路由选择写在函数体内——`MOE_ROUTER_ADAPTERS`（ep_utils.py）保留为
     公开工具库（`default`/`qwen3moe`/`mixtral`/`deepseekv3`/`glm4moe`
     等），用户工厂按名引用即可（示例见 ep_compute.py docstring 与
     `tests/.../test_dist_s4_ep.py` 的 `_qwen3moe_ep_factory`）。

```python
# 用户侧：一行显式注入（glob merge，params/契约继承推导）——零配置：
# expert mesh 由框架在 apply 时统一派生（与专家参数分片共享同一对象，
# 派生有 INFO 日志），经 ep_mesh 上下文传给工厂，用户只管使用
mesh = init_device_mesh("npu", (2, 4), mesh_dim_names=("dp", "tp"))
plan = ShardingPlanner(plan_overrides={
    "*.mlp": ModuleShardingSpec(
        local_compute_fn=Target(
            hf_native_ep_compute_fn,
            target_path="hyper_parallel.auto_models.components.distributed."
                        "ep_compute.hf_native_ep_compute_fn"),
        # 必填伴生声明：EP compute 内含 all-to-all 通信 → 区域内不可 dispatch
        region_dispatch=False),
}).plan(model, mesh, tp_size=4, ep_size=8)
model, source_shard_info = apply_sharding_plan(model, plan, mesh)
# YAML（trainer 路径）等价写法：
#   plan_overrides:
#     - match: "*.mlp"
#       when: ep
#       local_compute_fn:
#         _target_: hyper_parallel.auto_models.components.distributed.ep_compute.hf_native_ep_compute_fn
#       region_dispatch: false   # 必填：compute 内含 a2a 通信，不可 dispatch
```

缺注入时 apply 前 fail-fast（`_preflight_compute_injection`）：报错列出
具体边界 fqn，并给出上面这段可粘贴的配置——不会带着错误的静默数值
跑下去。

**约束（plan 时校验，fail-fast）**：

- `ep_size` 不超过且整除 dense 区域（dp × cp × tp）；
- `num_experts % ep_size == 0`；
- v1 暂不支持 pp>1 与 expert bias。

### 7.2 路径 B：EP-aware 自研 MoE 模块（自带 dispatcher）

模块自身 forward 已实现 EP dispatch（如 Megatron 风格
`MoEAlltoAllTokenDispatcher`）→ 声明 `region_dispatch=False`（§10.4，无
注入时模块 forward 即区域 compute）走 local-region 骨架，框架只做边界
缝合。此时需自行设置模块的 EP 运行时状态（类比 Megatron
`init_token_dispatcher`）：

```python
ep_mesh = mesh["ep"]
for layer in model.model.layers:
    layer.mlp.experts.expert_offset = ep_mesh.get_local_rank() * n_local_experts
    layer.mlp.ep_group = ep_mesh.get_group()
```

### 7.3 两条路径的选择

| | 路径 A（HF 原生 MoE） | 路径 B（EP-aware 模块） |
|---|---|---|
| 触发 | planner 自动识别 + `ep_size>1`（仅参数侧）+ **显式注入** | `region_dispatch=False`（+ 可选 `local_compute_fn`） |
| 前向 compute | **仓内参考 `hf_native_ep_compute_fn`（显式注入）** | 模块自身 forward（或用户 fn） |
| expert 分片 | 框架 `{EP: Shard(0)}` | 模块自行管理 |
| 适用 | qwen3_moe/glm4_moe/deepseek_v3 等 HF 模型一行注入即用 | 自研 dispatcher（DeepEP 等） |

> 可运行示例：`examples/distributed/ep.py`（per-expert 布局 + D-09 堆叠 + TP=2×EP=2 双模式对拍）。

---

## 8. FSDP 组合（接口契约）

TP/CP/EP 切完后的 **dense 参数**由上游 FSDP（`hyper_parallel/core/
fully_shard` 的 `fully_shard`，FSDP2 语义）再做数据并行分片。组件与
FSDP 的接口是 **`source_shard_info`**——production 模式 `apply_sharding_plan`
的第二个返回值：

```python
model, source_shard_info = apply_sharding_plan(model, plan, mesh)
# source_shard_info: {param_fqn: (tp_placement, tp_mesh)}
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
model, source_shard_info = apply_sharding_plan(model, plan, mesh)

for fqn, (tp_placement, tp_mesh) in source_shard_info.items():
    param = get_param_by_fqn(model, fqn)
    register_grad_sync_semantics(param, tp_placement, tp_mesh)  # 上游接口

fully_shard(model, mesh=dp_mesh, ...)   # hyper_parallel/core/fully_shard
```

> 说明：`source_shard_info` 从 **plan** 而非 DTensor 读取（production 下参数
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
| 多模态双 mesh | LLM: `(2, 2)` `("dp","tp")`；ViT: flatten 视图 `("encoder_dp",)` | 两次 plan + `plan_overrides` 桥接 | §9.1，示例 `multimodal_encoder_dp.py` |

要点：

- **轴顺序即 rank 映射**：`mesh_dim_names` 最右轴连续。TP 通信最频繁，
  放最内层；
- `ep_size` 是**扩展 EP 组大小**（D-10），从整个 dense 区域
  （dp×cp×tp）派生 `(edp, ep)` expert mesh，不要求 mesh 里有 "ep" 轴；
  mesh 里显式放 "ep" 轴仅路径 B（自研 EP-aware 模块取 `mesh["ep"]`）需要；
- size=1 轴自动剔除，可写全 `(dp, cp, tp)` 再按需开 size。

### 9.1 多模态双 mesh：encoder_dp ViT + LLM（桥接边界）

场景：多模态模型的 ViT 与 LLM 想用**两套并行拓扑**——例如 4 卡下 ViT 走
encoder_dp（dp=4，每 rank 编码不同图像，消除 TP 组内冗余编码），LLM 走
dp=2×tp=2（+EP）。两者是容器下的兄弟模块，共享同一进程组但 mesh 视图不同：

```python
llm_mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
vit_mesh = llm_mesh._flatten("encoder_dp")   # dense 区域的 1-D 视图（dp=4）
enc_rank = vit_mesh.get_local_rank()         # 数据管道按它分配各 rank 的图像子集
```

`vit_mesh` 的两个用途：数据管道按 `enc_rank` 分配图像 + 训练侧
`fully_shard(vision_tower, mesh=vit_mesh)` 的 FSDP 权重域。**它不进任何
plan**——坐标系约定（§11.1）下 plan 恒为单 dp 切片，ViT 对 DTensor 的唯一
需求是"编码完的特征要在 LLM 的 TP 组内 all-gather"，这是一次边界重分布：

```python
# ① 桥接：vision_tower 的 out 边界 Shard(0)→Replicate = tp 组内 all-gather
bridge = ModuleShardingSpec(
    params={},                          # ViT 参数零 DTensor 分片（纯 FSDP 公民）
    region_dispatch=False,              # ViT 内部不做 dispatch（自定义编码流程）
    in_src={"pixel_values": {TP: Shard(0)}},   # 单 dp 切片内：tp 组间特征分片
    in_dst={"pixel_values": {TP: Shard(0)}},   # identity
    out_src={"output": {TP: Shard(0)}},
    out_dst={"output": {TP: Replicate()}},     # ← tp 组内 all-gather
)
vit_plan = ShardingPlanner(plan_overrides={"": bridge},
                           derive=False).plan(  # 关闭模板推导（见下）
    model.vision_tower, llm_mesh, tp_size=2)   # 根 fqn ""（D-14）
apply_sharding_plan(model.vision_tower, vit_plan, llm_mesh)

# ② LLM：与纯 LLM 完全同构的独立 plan/apply
llm_plan = ShardingPlanner().plan(
    model.language_model, llm_mesh, tp_size=2, ep_size=4)
model.language_model, source_shard_info = apply_sharding_plan(
    model.language_model, llm_plan, llm_mesh)
```

要点与约束：

- **声明点必须在 vision_tower 的 out 边界**，不能是 LLM 的 in 边界：
  特征 → `inputs_embeds` 之间有 merge glue（按序列位置散射），all-gather
  是 dim 0 拼接，表达不了位置散射；gather 必须发生在 merge 之前；
- **ViT 子树必须关闭模板推导**：encoder_dp 下各 rank 数据不同，ViT
  内层任何 TP 集合通信（如行并行 all-reduce）会把不同样本的 partial
  求和——数学错误。`derive=False` 让 planner 只做声明装配：plan 只含
  plan_overrides 显式声明的 spec（全部 insert 模式，须完整自声明），
  取代 `plan.modules = {"": plan.modules[""]}` 的后处理剪除写法；
- **梯度语义**:fwd all-gather / bwd reduce-scatter-sum（边界通信 autograd
  感知）——TP 组内 LLM 计算是分片的，梯度求和恰好补全各 rank 自己图像块
  的完整梯度；ViT 参数梯度再由 dp=4 FSDP 域 all-reduce = 全局 batch 梯度；
- **数据双布局**：文本按 LLM dp=2 切（TP 组内相同），图像按 encoder_dp=4
  切（每 rank 编码本 shard 图像子集的 1/tp 份）；
- **token 对齐约束**:all-gather 是静态形状集合通信，各 rank 视觉 token 数
  必须一致——真实场景需数据管道做视觉 token 均衡/padding。

完整可运行示例（双模式对拍 + gather 探针）:
[`examples/distributed/multimodal_encoder_dp.py`](../../../examples/distributed/multimodal_encoder_dp.py)。

---

## 10. 自定义模块完整指南

### 10.1 `region_dispatch` 公理（2026-08-07 语义收敛）

**区域计算默认可 dispatch 穿透**——普通边界（无注入）validate 恒在
DTensor 上跑传播校验，无需任何声明；**一旦声明注入（`local_compute_fn`
或 `inner_wrapper`），`region_dispatch` 必填、无默认**——框架不知道注入
函数内部能不能被 DTensor dispatch，必须由作者显式回答：

- `False`（最常见）：注入内含通信原语（all-gather / all-to-all 等）或
  自定义 kernel / 数据依赖控制流 → 不可 dispatch。validate 下骨架/适配器
  把区域当**黑盒孤岛**：入口 to_local、出口按声明（out_src/inner_out_src）
  重包，区域内部跳过传播校验；仓内 CP wrapper 参考实现与
  `hf_native_ep_compute_fn` 均属此类（内含通信，**必须** `False`）；
- `True`：注入只是**纯标准算子替换/写法优化**（如融合 kernel 的等价
  einsum/sdpa/silu 写法，无通信、无数据依赖分支）→ 可 dispatch。
  validate **穿透**注入函数：DTensor 直接传入跑真实算子传播，出口
  placements 与声明逐点比对——**out_src/inner_out_src 从"声明重包"升级为
  "真校验基准"**，声明错了当场 PlacementMismatchError；dispatch 失败
  （函数内部有 dispatch 不了的算子）会带着教学化报错提示改回 `False`；
- 防呆：声明注入却未给 `region_dispatch` → apply 时 fail-fast；**未注入
  却写 `region_dispatch=True`**（冗余）→ fail-fast 请删除。
- production 完全不受影响：注入区域在生产模式恒为 local tensor 直通。

拿不准填什么时，用 `check_dispatchable` 在开发期探明（§4.2）；判断口诀
见 §3.4。

### 10.2 `plan_overrides`：统一 override 通道（merge/insert/glob）

一切自定义声明都经 `ModuleShardingSpec` 字段，在 plan 时合并。核心规则
一句话——**「不写继承，写了照办」**（2026-08-05 语义收敛）：**merge**
——命中推导边界时，未声明（`None`，即不写）的字段继承推导、写了的
字段按字段粒度覆盖（**显式空 `{}` 也是"写了"**：清空推导，如
`params={}` = 本边界不切参数的纯 I/O 缝合边界），哨兵 `"auto"`（显式
继承，自文档）/`"none"`（显式清空，同 `{}`）；字段粒度替换使推导参数
被丢弃时打 WARNING（列出参数名，不逐 key 合并）；**insert**——未命中
且与所有派生边界无祖孙关系则插入，至少声明一项契约（显式 `{}` 也算
声明），全部未声明或误用哨兵会 fail-fast；key 支持 **glob**
（`"*.self_attn"` 一条覆盖所有层）；内部标记 `_ep_size`/`_needs_cp_attn`
等恒继承）：

```python
from hyper_parallel.auto_models.components.distributed import ModuleShardingSpec
from hyper_parallel.auto_models.components.distributed.sharding_config import TP, CP
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

**契约字段取值语义一览**（`params`/`in_src`/`in_dst`/`out_src`/`out_dst`）：

| 写法 | merge（命中推导边界） | insert / `derive=False` |
|---|---|---|
| 不写（`None`） | 继承推导值 | 无可继承——全部字段未声明 → fail-fast |
| `"auto"` | 显式继承（自文档，与不写同义） | fail-fast：没有推导值可继承（报错点名原因） |
| `"none"` | 显式清空继承值（→ `{}` / `None`） | fail-fast：没有继承值可清空 |
| `{}`（显式空） | 本边界不切参数/无该项契约（参数保持复制） | **同左——两种模式下唯一的"清空"写法** |
| 具体 dict | 字段粒度替换（不逐 key 合并，丢弃推导参数会 WARNING） | 原样插入 |

心智模型一句话：**`"auto"`/`"none"` 是"相对推导值"的修饰词**——继承
它或清空它；`derive=False`（§9.1）关闭了模板推导，plan 里没有任何
推导值，两个哨兵失去作用对象，因此只剩"显式声明"一种写法：要切分
写具体 dict，不切写 `{}`（`params={}` = 本边界参数保持复制/纯 I/O
缝合）。这也是 insert 模式报错会附赠契约草稿（按模块 forward 签名
生成占位骨架）的原因——把"从零写契约"变成"改草稿"。

insert 完整自声明示例（模板未覆盖的自研模块，纯 I/O 缝合边界——
`params={}` 显式空即合法声明，本例 SP 布局下入口 all-gather、出口
切回本地 chunk）：

```python
plan_overrides = {"model.aux": ModuleShardingSpec(
    params={},                                # 本边界不切参数（保持复制）
    region_dispatch=False,                    # 内部自研流程不可 dispatch
    in_src={"x": {TP: Shard(1)}},             # 上游 SP chunk 到达
    in_dst={"x": {TP: Replicate()}},          # ← 入口 all-gather
    out_src={"output": {TP: Replicate()}},
    out_dst={"output": {TP: Shard(1)}},       # ← 出口切回 SP chunk
)}

与上游 `out_dst` 不一致**不再检查**（D-14：链式校验已废除，各模块只
断言自身策略传播，端到端正确性由双模式数值对拍兜底）；fqn
拼写错误 → `ValueError` fail-fast；spec 缺 `in_src`（`in_dst` 非空时）
→ `ValueError`（D-14 全声明强制化，链式填充已废除）。

**嵌套 spec（D-14，05 §13）**：override 的 fqn 可以是其他边界的祖先/
后代——外层边界（如整个 decoder layer、整个 LM 的根 fqn `""`）与内层
边界共存，常用于"外层只做 I/O 缝合（`params={}` + `region_dispatch=False`），
内层关键模块跑 validate 孤岛"（05 §13.4）。唯一约束是**参数唯一归属**：
每个参数只能被一个边界声明（外层不得声明内层边界子树的参数），冲突
即 `ValueError`。想定制某个叶子（如 `self_attn.q_proj`），仍应直接
覆盖它所属的边界（`self_attn`，merge 语义，参数名相对于边界模块）。

> 可运行示例：`examples/distributed/nested_local_map.py`（外层 local_map
> + 内层 validate 孤岛，production/validate 双模式对拍）。

### 10.3 注入接口总览（两个家族 + 一个伴生声明）

| 字段 | 签名 | 做什么 | 生效方式 |
|---|---|---|---|
| `region_dispatch` | `Optional[bool]`（**无默认——声明注入时必填**） | **区域 compute 是否可 dispatch 穿透**：`False` = 区域内含通信原语/自定义 kernel，不可 dispatch → validate 走骨架/适配器黑盒托管（to_local + 声明重包）；`True` = 纯标准算子可 dispatch → validate **穿透**注入函数跑真实 DTensor 传播（out_src/inner_out_src 成为真校验）；不注入的普通边界无需声明（公理：默认可穿透，§10.1） | 骨架/适配器分流 |
| `local_compute_fn` | **`@local_compute` 工厂** `fn(mesh, tp_mesh, cp_mesh, ep_mesh, [module], <配置键...>) -> compute_fn`（callable 直传或 Target 载体） | 替换 local-region 骨架**内**的计算函数（边界缝合/双模式不变）。**唯一形态 = 区域计算工厂**：apply 时 build 一次（mesh 家族必选、框架填充、未激活轴为 None，**用不用随你**——统一接口规范；`module` 可选），须返回 compute fn（入参与原 forward 匹配，无需再装饰） | local 解析链环 1 |
| `inner_target` | `str`（属性名 / `"self"`） | **纯位置**：指定 inner attention 子模块（单独声明无 inner_wrapper → fail-fast） | target 解析链环 1 |
| `inner_wrapper` | `str`（注册表名）、`@inner_wrapper` callable 或 **Target** | **纯行为**：固定仓内参考 CP wrapper / 全自定义接管 / Target 引用仓内公开函数；`region_dispatch=False` 时替换后的 forward 只面向 local 张量（双模适配器托管，§10.5.1） | wrapper 解析链环 1-3 |
| `inner_out_src` | `"first_input"` / `{axis: placement}` / `{name: {...}}` | **纯布局**：inner 子模块输出的重包声明——**`inner_target` 是子模块时必填**（未声明 apply 时 fail-fast，框架对 inner 输出布局零推导零猜测；layout-preserving 写 `"first_input"` 即可）；`inner_target="self"` 时不需要（用边界 out_src） | 适配器安装时 |
| `tp_divide_attrs` | `Optional[List[str]]` | **TP-local 属性整除声明**（D-18，§5.7）：forward 见 local tensor 时把列出的 int 实例属性按 tp_size 整除改写（D-17 头数名的用户扩展）；plan 期校验（存在/plain int/整除/非保护属性/不与 D-17 自动清单重复）；`[]` 清空继承的 glob 声明 | Phase 4.6 定稿，apply 时改写 |

**注入纪律（injection.py）**：所有注入函数必须带模板装饰器（仅两个：
`@local_compute` / `@inner_wrapper`）——import 期校验：**必选上下文缺一
不可**（`@local_compute` 须声明 mesh 家族 `mesh`/`tp_mesh`/`cp_mesh`/
`ep_mesh`；`@inner_wrapper` 在此之上还须声明 `target_module`）、上下文
参数不得有默认值、禁止 `*args`/`**kwargs`；apply 期强制检查：未装饰/
种类不符 fail-fast，配置键与保留上下文同名 fail-fast，工厂返回的
compute fn / 替换后的 forward 的入参与原函数不匹配 fail-fast。配置键
只携带数据值，不允许再传函数。

**门控派生原则**：声明互不嵌套、不改写任何标记——applier 的解析链
（`_resolve_local_compute_fn` / `_resolve_inner_wrapper`）解析非 None 即
注入；`region_dispatch` 只决定 validate 下该区域是"穿透真校验"还是
"黑盒托管"。

### 10.4 local-region 族：`region_dispatch` 与 `local_compute_fn`

骨架结构（通信保留，§6/§7 的边界通信照常执行）：

```
输入(in_src) → boundary入口(in_src→in_dst 通信) → 【compute_fn】
            → 按声明 out_src 重包装 → boundary出口(out_src→out_dst 通信) → 输出
```

重包装是**逐输出**的（D-21）：多输出模块按 `out_names` 把 `out_src` 的
每个声明映射到 tuple/list 对应位置、各自按声明布局 `from_local`，`None`
与已是 DTensor 的输出跳过，返回类型保持（tuple 进 tuple 出、list 进 list
出）；声明键不在 `out_names`、下标越界、或声明多输出而 forward 返回标量
→ fail-fast。

解析链（优先级递减；**EP 自动注入链路已删除，2026-08-04**）：

1. `spec.local_compute_fn` —— 显式注入：**`@local_compute` 区域计算
   工厂**（callable 直传或 Target 载体），apply 时 build 一次；mesh 家族
   必选上下文框架填充（`ep_mesh` 与专家分片共享同一对象）、用不用随你；
   仓内参考 `ep_compute.hf_native_ep_compute_fn` 即此形态。**伴生必填
   `region_dispatch`**（§10.1 公理）；
2. `spec.region_dispatch is False`（无用户 fn）—— compute 即模块自身
   forward（自研 EP-aware MoE 等"forward 内含通信、不可 dispatch"的
   模块走这里）。

环外防呆：`_ep_size>0`（专家已 EP 分片）而解析为 None → apply 前
`_preflight_compute_injection` fail-fast；HF 原生 MoE（per-expert/
batched 布局）的 `region_dispatch` 推导值已被 planner 清除为 None（其
forward 非 EP-aware），必须经环 1 显式注入 + 声明 `region_dispatch=False`。

**示例 A：自研 MoE 自带 dispatch（模块 forward 即 compute）**

```python
spec = ModuleShardingSpec(
    params={...}, in_src=..., in_dst=..., out_src=..., out_dst=...,
    region_dispatch=False,   # 模块 forward 内含 a2a 通信 → 不可 dispatch，骨架只负责缝合
)
```

**示例 B：注入自定义 dispatch（local_compute_fn）**

```python
@local_compute
def my_deep_ep_factory(mesh, tp_mesh, cp_mesh, ep_mesh):
    """在 local tensor 世界实现自定义 EP dispatch（如 DeepEP）。

    契约：@local_compute 是区域计算工厂 fn(mesh, tp_mesh, cp_mesh,
    ep_mesh, ...) -> compute_fn——mesh 家族四个必选（框架按名填充，
    未激活轴填 None，用不用随你）、不得有默认值、禁止 *args/**kwargs；
    可选锚点 module（边界模块）；其余具名形参是用户配置键（Target/YAML
    按名绑定，拼写 fail-fast）。返回的 compute_fn(module, *args) 是普通
    callable（无需再装饰），入参必须与原 forward 匹配（apply 时校验）；
    region_dispatch=False 时其输入输出均为 local tensor，布局与 spec 的
    in_dst/out_src 声明一致。"""
    ep_group = ep_mesh.get_group("ep")     # apply 时建组一次，闭包固定

    def compute_fn(module, hidden_states):
        topk_idx, topk_w = module.router(hidden_states)
        dispatched, recv_counts = deep_ep_dispatch(hidden_states, topk_idx,
                                                   group=ep_group)
        expert_out = module.experts(dispatched)
        return deep_ep_combine(expert_out, topk_w, recv_counts,
                               group=ep_group)
    return compute_fn

spec = ModuleShardingSpec(
    local_compute_fn=my_deep_ep_factory,   # 链环 1：直接生效（callable 直传）
    region_dispatch=False,   # 必填伴生声明：DeepEP dispatch 含通信原语，不可 dispatch
)   # merge 语义：params/契约空 → 继承 planner 推导结果
planner = ShardingPlanner(plan_overrides={"model.layers.3.mlp": spec})
# 也可以用 glob 一条覆盖所有层：plan_overrides={"*.mlp": spec}
# 需要配置键或 YAML 载体时用 Target：
#   local_compute_fn=Target(my_deep_ep_factory,
#                           target_path="my_pkg.my_deep_ep_factory",
#                           block_size=128)
```

**为什么 mesh 家族是必选声明、却又"用不用随你"**：骨架在 compute 之前
已经完成了全部布局转换（边界入口通信 + to_local + 参数解包），区域内是
纯 local tensor 世界——compute fn 每个 forward 都跑，不该在运行期从
mesh 派生通信域（每次 `mesh.get_group()` / `new_group` 都是浪费甚至
泄露）。所以 mesh 只在 **apply 期 build 工厂时**传一次（对象引用，零
成本）：需要通信组的（典型：从 `ep_mesh` 建 a2a 通信组）在工厂体里建组
一次、闭包固定，运行时零 mesh 开销；自含计算（融合 kernel 替换、自定义
expert 排布、用模块上已有 process group 的 dispatch）声明了不用即可——
mesh 家族是统一的接口规范，不是使用义务：

```python
@local_compute
def my_fused_swiglu(mesh, tp_mesh, cp_mesh, ep_mesh):
    """自含计算的工厂：mesh 家族必选声明（框架填充），本例不使用。"""

    def compute_fn(module, hidden_states):
        return module.down_proj(
            F.silu(module.gate_proj(hidden_states))
            * module.up_proj(hidden_states))
    return compute_fn

plan_overrides = {"*.mlp": ModuleShardingSpec(
    local_compute_fn=my_fused_swiglu,      # callable 直传
    region_dispatch=True)}   # 纯标准算子 → validate 穿透真校验（§10.1）
```

**`local_compute_fn` 只接受 `@local_compute` 工厂一种形态**（callable
直传，或工厂的 `Target` 延迟引用——需要配置键/YAML 载体时）。裸函数
（未装饰）或装饰器种类不符（如给工厂用 `@inner_wrapper`）都在 apply
时 fail-fast 并指明正确的装饰器。

骨架四步里只有 compute 被替换；in/out 边界的 all-gather/reduce-scatter
照常执行（详见 §3.3 声明式豁免）。

> 可运行示例：`examples/distributed/custom_local_compute_fn.py`（自研 top-1
> MoE + 自定义 batched expert 布局；含融合 gate_up 不可直接 Shard 的坑说明）。

### 10.5 inner-wrap 族：`inner_target` 与 `inner_wrapper`

机制：**定位 inner 子模块 + 替换/包装其 forward**。机制本身通用（不限
CP、**不限 attention**——任何模块的任何子模块都可声明，声明即应用），
CP（K/V all-gather）只是第一个仓内参考实现域。双解析链：

- **target 链（纯位置）**：`inner_target` **显式指定，无缺省**——
  任意属性名或 `"self"`（边界模块自身），拼错 fail-fast。声明
  `inner_wrapper` 时必须**成对**显式声明 `inner_target`（缺失 →
  `ValueError`）：曾经的 attention 域自动定位启发式
  （`inner_attention`/`attn`/`attention` 属性 > 类名判定 > q/k/v_proj
  结构兜底）已删除——inner-wrap 是与 CP/attention 解耦的通用机制，
  静默定位有包错目标风险，而包错目标是静默数值错误温床；
- **wrapper 链（纯行为）**：Target > callable 自定义 > str 注册表名 >
  不注入（**启发式分派已删除**；仅 `inner_target` 无 `inner_wrapper` →
  fail-fast）。

**示例 C：非标准属性名 → `inner_target` 指定**

```python
spec = ModuleShardingSpec(
    ..., inner_target="core_attention",   # 属性名；"self" 表示模块本身
    inner_wrapper="sdpa_qkv",             # 成对必填：inner_target 不能单独声明
    inner_out_src="first_input",          # 子模块目标必填：输出布局显式声明
    region_dispatch=False,                # 必填伴生声明（仓内 CP wrapper 内含通信）
)
```

**示例 D：显式固定仓内参考方案（无启发式缺省）**

```python
# 自研 attention 是 HF 风格但内部调的是自研 sdpa 封装（拦截路拦不到），
# 显式走 (q,k,v) 替换路——inner 子模块目标必须声明 inner_out_src
spec = ModuleShardingSpec(..., inner_target="core_attention",   # 成对必填
                          inner_wrapper="sdpa_qkv",
                          inner_out_src="first_input",
                          region_dispatch=False)   # 必填：wrapper 内含 K/V all-gather 通信
```

可选名：`"sdpa_qkv"` / `"sdpa_hf"` / `"flex_qkv"` / `"flex_hf"`（§6.2
各自的机制与适用场景——**四个参考实现都内含通信，一律
`region_dispatch=False`**）。未知名 fail-fast 并列出可用名。

**示例 E：仓内四路覆盖不到 → callable 全自定义（local-only）**

```python
@inner_wrapper
def my_flash_cp_wrapper(target_module, mesh, tp_mesh, cp_mesh, ep_mesh):
    """契约：@inner_wrapper fn(target_module, mesh, tp_mesh, cp_mesh,
    ep_mesh) -> None——mesh 家族必选、框架填充、只用需要的；禁止
    *args/**kwargs。

    整体接管：原地替换 target.forward。替换后的 forward **只面向 local
    张量**（零 DTensor 代码）——validate 的解包/重包由双模适配器托管
    （§10.5.1）；重包布局来自声明：target=self 用边界 out_src，inner
    子模块用 inner_out_src。"""
    orig_forward = target_module.forward

    def cp_forward(q, k, v, **kwargs):
        gk = all_gather_along_seq(k, cp_mesh.get_group())
        gv = all_gather_along_seq(v, cp_mesh.get_group())
        return flash_attn_varlen(q, gk, gv, causal=True, **kwargs)

    target_module.forward = cp_forward

spec = ModuleShardingSpec(
    ..., inner_target="core_attention", inner_wrapper=my_flash_cp_wrapper,
    inner_out_src="first_input",   # layout-preserving：输出布局 == q 布局
    region_dispatch=False,         # 必填：wrapper 内含 all_gather 通信原语
)
```

`inner_target` 与 `inner_wrapper` 必须**成对**出现（任一缺失 →
fail-fast）。`inner_target="self"` 时按 self 情形用边界 out_src 重包，
无需 `inner_out_src`。

**示例 F：注册命名方案（团队共享）**

```python
from hyper_parallel.auto_models.components.distributed.cp_wrappers import INNER_WRAPPER_REGISTRY

INNER_WRAPPER_REGISTRY["my_flash"] = my_flash_cp_wrapper   # 须已带 @inner_wrapper
# 之后任意 spec 可写 inner_wrapper="my_flash"——仍需成对声明 inner_target
# （包装自身写 "self"；子模块目标另需 inner_out_src）与 region_dispatch
```

### 10.5.1 双模适配器：inner wrapper 的黑盒托管（`region_dispatch=False`）

所有 inner wrapper（仓内参考/注册表/callable/Target）替换完 forward 后，
由 `_wrap_inner_attention` 统一安装**双模适配器**。`region_dispatch=False`
（含通信，最常见）时适配器做黑盒托管：

- **入口**：validate（任一入参是 DTensor）→ DTensor 入参 to_local +
  参数临时解包；production（无 DTensor 入参）→ 直通零开销；
- **出口重包**：placements 全部显式声明，框架零推导零猜测——
  target=self → 边界 `out_src`（多输出按 `out_names`）；inner 子模块 →
  `inner_out_src`（`"first_input"` / 显式 placement / 多输出 dict）；
- **validate 对 inner 区域跳过传播校验**：inner 是黑盒孤岛（原实现在
  CP 下本就无法正确 dispatch，不存在拿它当校验基准的合法性）。安全网
  在孤岛之外：重包接回后外层 dispatch 是真实传播（声明错了被边界
  out_src 校验抓住）+ `from_local` 全局形状一致性 + 双模式数值对拍。

**`region_dispatch=True`（纯标准算子 wrapper）时适配器改为穿透**：
validate 下 DTensor 直接传入替换后的 forward 跑真实算子传播，出口
placements 与 `inner_out_src`（或 `"first_input"` 规则）逐点比对——
声明即真校验；链断了（返回非 DTensor）或布局不符当场报错。production
行为不变（恒 local 直通）。

### 10.6 两个家族怎么选

| 场景 | 接口 |
|---|---|
| 模块 forward 内含数据相关逻辑（a2a/gather/mask），保持模块不变 | `region_dispatch=False`（模块 forward 即 compute） |
| 保持骨架（边界缝合/双模式），只换区域内部计算（含通信/自定义 kernel） | `local_compute_fn` + `region_dispatch=False` |
| 纯算子替换/写法优化（无通信、可 dispatch，想拿真校验） | `local_compute_fn`/`inner_wrapper` + `region_dispatch=True` |
| 要包装的不是边界模块本身而是其某个子模块 | `inner_wrapper` + `inner_target="<属性名>"` |
| 仓内参考 CP 方案选错/要固定 | `inner_wrapper="..."` + `region_dispatch=False` |
| 仓内四路不满足（flash_attn_varlen 直调等） | `inner_wrapper=callable` + `region_dispatch=False` |

> 可运行示例：`examples/distributed/custom_inner_wrapper.py`（非标准 inner
> 属性名 + `INNER_WRAPPER_REGISTRY` 注册命名方案 + `_resolved_inner_wrapper`
> 回写断言 + `inner_out_src` 声明）；`programmatic_injection.py`（五形态
> 编程式注入汇总）。

#### `inner_target="self"` + `inner_wrapper` vs `local_compute_fn`（`region_dispatch=False` 时）

两者在 `region_dispatch=False` 下跑的是**同一个 local-region 骨架**（边界
入口通信 → to_local + 参数解包 → compute → 按 out_src 重包 → 边界出口
通信）——inner-wrap 情形下 compute 就是"已被用户替换的 target.forward"，
双模适配器在骨架内直通。真正的区别在**替换姿态**：

| | `local_compute_fn` | `inner_wrapper`（含 `inner_target="self"`） |
|---|---|---|
| 姿态 | **整体接管**：区域内计算完全由 fn 提供 | **织入/拦截**：拿到 `orig_forward` 引用后替换 forward |
| 原 forward 的代码 | 不可复用（fn 里再调 `module.forward` 会递归进已包装的 forward） | 可复用——如 `sdpa_hf` 只拦截其中的 `F.sdpa` 调用，q/k/v 投影、RoPE、o_proj 等原代码原样保留 |
| 签名 | 工厂 `fn(mesh, tp_mesh, cp_mesh, ep_mesh, ...) -> compute_fn`（框架管理的区域计算；compute_fn 为 `fn(module, *args) -> Tensor`） | `fn(target_module, mesh...) -> None`（原地替换，双模适配器托管 DTensor 转换） |
| 目标 | 恒为边界模块区域本身 | 可指向任意子模块（`inner_target="<属性名>"`） |

**判据**：要**保留原 forward 的大部分代码、只换其中一个算子/插一段通信**
→ `inner_wrapper`（织入）；要**完全重写区域计算**（自研 kernel 序列、
与原实现无共享代码）→ `local_compute_fn`（接管）。只想包装边界模块
自身时两通道都能做，按上表姿态选；目标是子模块时只能走 inner-wrap。

### 10.7 编程式注入：不接 trainer / YAML 的集成方式

双模式 DTensor 能力可以脱离 trainer 单独使用——`ShardingPlanner` +
`apply_sharding_plan` 两个对象就是全部入口，注入全部是普通 Python 对象
（适合把本组件接入自研训练框架）：

```python
# region_dispatch 公理：声明注入就必须显式回答"区域内能否 dispatch"——
# 纯标准算子（无通信、无数据依赖分支）→ True（validate 穿透真校验）；
# 含通信原语/自定义 kernel → False（黑盒托管）。下面五形态都是纯算子示例。
overrides = {
    # ① @inner_wrapper 装饰的 callable 直传（最常用，不需要注册表）
    "model.layers.0.self_attn": ModuleShardingSpec(
        inner_target="self", inner_wrapper=my_wrapper, region_dispatch=True),
    # ② Target 延迟引用 + 数据配置键（按名绑定到工厂形参）
    "model.layers.1.self_attn": ModuleShardingSpec(
        inner_target="self", region_dispatch=True,
        inner_wrapper=Target(my_wrapper, target_path="my_pkg.my_wrapper",
                             block_size=128)),
    # ③ 注册表名（可选：按名共享 / YAML str 引用时才需要注册）
    "model.layers.2.self_attn": ModuleShardingSpec(
        inner_target="self", inner_wrapper="demo", region_dispatch=True),
    # ④ glob + @local_compute 工厂 callable 直传（merge 语义：契约继承推导）
    "*.mlp": ModuleShardingSpec(
        local_compute_fn=my_factory, region_dispatch=True),
    # ⑤ @local_compute 工厂 Target（配置键按名绑定；精确 key 覆盖 glob）
    "model.layers.2.mlp": ModuleShardingSpec(
        local_compute_fn=Target(
            my_factory, target_path="my_pkg.my_factory", block_size=256),
        region_dispatch=True),
}
# 注册表与 Target 都不是必需的——注入函数都可以直接传装饰后的函数对象；
# INNER_WRAPPER_REGISTRY 仅为 YAML 字符串引用/团队按名共享而存在；
# CP/EP 场景的注入内含通信 → 改传 region_dispatch=False（§6/§7）
plan = ShardingPlanner(plan_overrides=overrides).plan(model, mesh, tp_size=2)
model, source_shard_info = apply_sharding_plan(model, plan, mesh)
model(input_ids)   # production 训练 / validate 对拍（validate_mode=True）
```

可运行示例：`examples/distributed/programmatic_injection.py`（五形态一个
dict 全覆盖，production/validate 双模式对拍单卡，计数器逐一断言生效）。

### 10.8 自定义 autograd.Function 与第三方/HF 宿主

**问题**：边界机制的作用粒度是 `nn.Module.forward`。自定义
`autograd.Function` 以 `A.apply(...)` 裸调用（没有实例、不在模块树）时
**没有 FQN**，spec 无处挂载——框架对它完全不可见。

**解决**:`FunctionModule` 壳给它模块形态，之后走标准的 plan_overrides
流程：

```python
from hyper_parallel.auto_models.components.distributed import FunctionModule

self.a_fn = FunctionModule(A)       # 挂在宿主 __init__：获得 FQN
# forward 里：A.apply(x) → self.a_fn(x)

plan_overrides={"...a_fn": ModuleShardingSpec(
    params={},                      # Function 无参数
    region_dispatch=False,          # 必须：自定义 Function 不在 dispatch 覆盖范围
    in_src={"x": {TP: Shard(1)}},
    in_dst={"x": {TP: Replicate()}},     # 入口 all-gather
    out_src={"output": {TP: Replicate()}},
    out_dst={"output": {TP: Shard(1)}},
)}
```

- **契约 key 绑定**：单张量输入直接用壳的 `*args` 透传（单输入契约回退
  绑定到第 0 个位置参数）；多输入（如额外权重张量）子类化壳并给显式
  签名（`def forward(self, x, weight)`），契约 key = 形参名；
- **梯度链**：边界通信 autograd 感知（fwd all-gather / bwd
  reduce-scatter），A 自己的静态 backward 照常——布局归边界，计算归 A;
- **DX guard**:plan() 检测到树上有 `FunctionModule` 但无 spec 覆盖 →
  warning（未声明 = 静默无通信，必须显式选择）。

> **机制原理**:`FunctionModule` 三行实现（持有 Function 类、forward 透传
> `apply`）——挂载为宿主属性后经 `nn.Module.__setattr__` 登记进模块树，
> 获得 FQN，边界包装才有作用点；壳无参数、无状态，反向仍走 A 自己的
> 静态 `backward`（布局变换归边界通信的 autograd 段，数值计算归 A）。
> ③ 中 `__class__` 替换之所以可行：Python **方法存于类、状态存于实例**
> ——`forward` 沿实例的 `__class__` 指针解析，两个类都是 `nn.Module`
> 子孙、无 `__slots__`（实例布局兼容），赋值只拨动方法解析指针，参数与
> 父子引用零拷贝。它是 plan 之前的装配期结构变更（模块树可见、plan 可
> 内省），与 patch 被 forward 闭包引用的 `A.apply` 有本质区别。

**调用点在第三方/HF 代码里（不能改原文件）的决策顺序**：

```
通信能移到宿主边界？  ──是──→ ① plan_overrides + region_dispatch=False
                              作用于宿主模块本身（零代码改动；A.apply 在
                              local 区域内原样执行）
        │否
必须侵入宿主内部？    ──是──→ ② local_compute_fn 整体接管宿主 forward
                              （D-09 HF 原生 MoE 的先例：区域内部通信
                              自己写，骨架边界仍归框架）
        │否（A 必须是独立边界）
        └──→ ③ 子类化宿主类 + 实例级 __class__ 替换（不改第三方文件、
               权重零拷贝），纪律：pin 依赖版本 + 原类 vs 子类单卡
               smoke test（升级时大声失败）
```

框架不提供对裸函数调用的拦截（monkey-patch `A.apply`）：那会让通信脱离
precompiled boundary（production 不预编译、validate 不断言、plan 不可
内省、出错时静默）。

> 可运行示例：`examples/distributed/custom_autograd_function.py`——模拟
> 第三方宿主裸调用"全序列统计量"Function，演示 ③ 的完整纪律（smoke
> test → `__class__` 替换 → plan_overrides → 双模式对拍）；Function 的
> 语义故意选为布局错则数值必错，**对拍本身就是桥接探针**。
> 代码级走读：[function_module_autograd_walkthrough.md](../../trainer/code_guides/function_module_autograd_walkthrough.md)。

---

## 11. 核心 API 参考

### 11.1 `ShardingPlanner().plan(...)`

```python
plan = planner.plan(
    model, mesh,
    tp_size=1,               # TP 组大小（>1 激活 TP 维）
    cp_size=1,               # CP 组大小（>1 激活 CP 维，attention 注入 CP wrapper）
    ep_size=1,               # 扩展 EP 组大小（>1 且命中 HF 原生 MoE 时激活 EP）
    sequence_parallel=True,  # SP：norm 间激活按序列维 Shard(1)
    loss_parallel=False,     # lm_head 输出 Shard(-1)（vocab 并行 loss）
    explain=False,           # True 时 plan 末尾 INFO 打出内省报告（§4.1）
)
```

- 返回 `ShardingPlan`：`modules: {fqn: ModuleShardingSpec}` +
  `mesh_dim_names` + `special_handlers` + `tied_pairs`。
- **坐标系约定**:`plan.mesh_dim_names` 恒为 `tp/cp/ep` 子集，**永远不含
  dp 轴**——plan 描述单个 dp 切片内的布局与通信；dp 维的数据切分归数据
  管道、参数/梯度切分归 FSDP（05 §3.1.1）。在 `plan_overrides` 里声明 DP
  placement 会被 `plan()` fail-first 拒绝（教学式报错）。
- size=1 的轴会被自动剔除（`plan.mesh_dim_names` 只含活跃轴）。
- `ShardingPlanner(plan_overrides={fqn 或 glob: spec})`：统一 override 通道
  （merge 未写字段继承 / insert 完整自声明）
  （§10.2）；`derive=False` 关闭模板推导——plan 只含 plan_overrides
  显式声明的 spec（全部 insert 模式），用于自动推导语义错误的子树
  （§9.1 encoder_dp ViT 桥接）。
- 内省与判定工具：`plan.explain()`（§4.1）、`check_dispatchable`（§4.2）。

### 11.2 `apply_sharding_plan(...)`

```python
model, source_shard_info = apply_sharding_plan(
    model, plan, mesh,
    validate_mode=False,     # True=validate 模式；默认 production
)
```

- 返回 `(model, source_shard_info)`；validate 下 `source_shard_info is None`。
- `model` 也可以是 **PP 多 part 列表**（`apply_sharding_plan([part0, part1], ...)`）。

### 11.3 DeviceMesh 构建

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

## 12. 典型模型支持矩阵

已用 transformers 仓（v5.12.1）真实模型类验证（planner 全链路）：

| 模型 | ParamRole 命中 | 边界推导 | 备注 |
|---|---|---|---|
| llama / qwen2 / qwen3 | ✅ 零 SKIP | ✅ | qwen3 的 q_norm/k_norm 归 NORM |
| qwen3_moe | ✅ | ✅ `moe_mlp` + EP | batched experts（D-11），router adapter 已注册 |
| glm4 | ✅ | ✅ | fused `gate_up_proj`；q/k/v bias 随权重归 COLWISE（D-19）；两个额外 post norm 归 NORM |
| glm4_moe | ✅ | ✅ `moe_mlp` + EP | shared_experts 归 SHARED_EXPERT（EP 维 Replicate） |
| deepseek_v2 / v3 | ✅（ARCH_OVERRIDES） | ✅ | **MLA**：q_a/kv_a 下投影 `REPLICATED`，q_b/kv_b 上投影 COLWISE；sigmoid router adapter 已注册 |
| mixtral | ✅ | ✅ `moe_mlp` + EP | per-expert 布局走 D-09 堆叠 |

DeepSeek MLA 覆盖条目（D-14，`ARCH_OVERRIDES` 内置，v2/v3 两种拼写）：

```python
[(["q_a_proj", "kv_a_proj_with_mqa"], ParamRole.REPLICATED),  # LoRA rank 维不切
 (["q_b_proj", "kv_b_proj"], ParamRole.COLWISE)]              # 按 head 维切
```

---

## 13. 排错索引

| 报错/现象 | 原因 | 解法 |
|---|---|---|
| `ValueError: ... named_modules` | plan_overrides 的 fqn 拼写错误 | 对照 `dict(model.named_modules())` |
| `ValueError: ... nests inside / ancestor / nested` | override 的 fqn 与派生边界（或另一 override）构成祖孙嵌套 | 改为覆盖所属边界本身（同 fqn 替换），见 §10.2 |
| `PlacementMismatchError` | validate 模式 out_src/out_dst 校验失败，或参数分片声明与实际不符 | 对照报错中的模块名与 placement 检查 spec 声明 |
| `chain contract mismatch`（警告） | 相邻模块 out_dst ≠ in_src | 边上有 reshape/transpose 属正常；否则检查声明并跑 validate 对拍 |
| `ValueError: ... 未显式声明 inner_target / inner_target 未声明` | 声明了 `inner_wrapper` 但未成对声明 `inner_target`（自动定位启发式已删除） | 包装模块自身 → `inner_target="self"`；包装子模块 → `inner_target="<属性名>"` |
| `ValueError: spec.inner_target=... did not match` | `inner_target` 属性名拼错或目标无 forward | 对照 `dict(module.named_children())` 修正属性名 |
| `ValueError: ... INNER_WRAPPER_REGISTRY` | `inner_wrapper` str 未注册 | 用四个仓内参考名之一，或先注册 |
| `RuntimeError: 未拦到 F.scaled_dot_product_attention` | 发火检测：`sdpa_hf` 拦截路但模块内部不调 F.sdpa（wrapper 型号选错） | 改 `inner_wrapper="sdpa_qkv"`，或提供 callable |
| `ValueError: ...未声明 inner_wrapper...` | cp>1 的 attention 边界无显式注入（preflight 防呆） | 按报错里的 YAML 片段配置 `plan_overrides` |
| `ValueError: ...没有 local-region 计算来源...` | ep>1 的 MoE 边界无 `local_compute_fn` 且非 `region_dispatch=False`（preflight 防呆） | 注入 `hf_native_ep_compute_fn` 工厂（+ `region_dispatch: false`）/ 自研 EP-aware 模块声明 `region_dispatch: false` / 自定义 compute |
| `ValueError: ...region_dispatch 未显式声明...` | 声明了注入（`local_compute_fn`/`inner_wrapper`）但没给 `region_dispatch`（无默认值） | 纯标准算子 → `True`；含通信原语/自定义 kernel → `False`（§10.1 公理） |
| `ValueError: ...region_dispatch=True 冗余...` | 未注入的普通边界写了 `region_dispatch=True`（默认即可穿透） | 删除该字段 |
| `...dispatch 失败...请改声明 region_dispatch=False` | `region_dispatch=True` 但 validate 穿透时函数内部有 dispatch 不了的算子/通信 | 改回 `False`（黑盒托管），或把不可 dispatch 的部分改写成标准算子 |
| `ValueError: ...未声明的键...` | Target 配置了目标函数未声明的 kwargs 键（拼写错误） | 按报错列出的合法形参名改正键名 |
| `TypeError: ...缺少 @local_compute/@inner_wrapper 装饰器` | 注入函数未按纪律装饰 | 按种类加装饰器（§10.3） |
| `TypeError: ...缺少必选上下文参数` | mesh 家族/target_module 没声明全 | 补齐签名（用不用都要声明） |
| `TypeError: ...上下文参数 ... 不得有默认值` | 上下文参数写了默认值 | 删默认值（框架必然填充） |
| `ValueError: ...配置了框架保留上下文键` | Target/YAML 里配置了 mesh 家族/锚点 | 删除该配置键 |
| `ValueError: ...未声明 inner_out_src` | wrapper 作用于 inner 子模块但未声明输出布局 | 按报错提示写 `"first_input"` 或显式 placement（§10.5.1） |
| `ValueError: Invalid built-in CP wrapper plan ... requires region_dispatch=False` | 仓内四路 CP wrapper 内含通信，却声明了 `region_dispatch: true`（D-20 静态要求表） | 改 `region_dispatch: false`（报错附建议 YAML，§6.2） |
| `ValueError: ...tp_divide_attrs 必须是属性名列表 / 只能包含合法属性名 / 包含重复属性` | YAML `tp_divide_attrs` 形态错误（D-18） | 改为合法属性名列表（§5.7） |
| `ValueError: ...tp_divide_attrs attribute ... must exist and be a plain int / must be positive and divisible by tp_size` | 声明的属性在模块实例上不存在、非 int、或不能整除 tp_size | 确认属性名拼写与取值；不整除就调 tp_size 或删声明（§5.7） |
| `ValueError: ...tp_divide_attrs cannot adjust protected attribute` | 声明了保护属性（`head_dim`/`num_key_value_groups`/`training` 等） | 删除该属性——这些量不被 TP 切分（§5.7） |
| `ValueError: ...redundantly declares D-17 automatic head attributes` | `tp_divide_attrs` 与 D-17 自动头数清单重复 | 从声明中删除（auto 段已自动覆盖） |
| `ValueError: ...was already adjusted ... incompatible with tp_size` | 同一模块被不同 tp_size 的 plan 重复 apply | 每个模型实例只 apply 一次同一 plan |
| `ValueError: ...out_src declares output ... out_names does not contain / forward returned only N output` | local-region 多输出重包契约与 forward 实际输出不符（D-21） | 对齐 `out_src`/`out_names` 声明与 forward 返回值（§10.4） |
| `TypeError: ...不存在同名项 / 不是同序子序列 / 必填参数 ... 未被接收` | compute fn 入参与原 forward 不匹配 | 对齐形参名/顺序/必填项 |
| `TypeError: ...入参不兼容` | 替换后的 forward 接不住原 forward 入参 | 用 `*args/**kwargs` 透传或对齐签名 |
| `ValueError: 仓内 CP wrapper 参考实现 ... 需要活跃的 cp mesh` | 无 cp 轴却声明了四个仓内参考 CP 方案之一（inner-wrap 泛化后声明即应用） | 改用自定义 callable/Target（收 `cp_mesh=None`），或 `local_compute_fn`（§6.3.1） |
| `ValueError: spec.inner_target=... 只是定位` | 只给位置没给方案（启发式已删除） | 同时声明 `inner_wrapper` |
| `ep_size (...) 必须不超过且整除 dense 区域` | EP 组超出 dp×cp×tp | 调小 ep_size 或扩大 dense 区域 |
| `num_experts (...) 必须整除 ep_size` | expert 数不能均分 | 调整 ep_size |
| `NotImplementedError: ... 仅支持 SwiGLU expert` | 仓内参考 EP compute 只支持 SwiGLU 三矩阵 | 自研 expert 结构 → `local_compute_fn` |
| inner wrapper 想拿 DTensor / 想做框架级布局推导 | 双模适配器已托管全部 DTensor 转换，用户 wrapper 只见 local 张量 | 不需要也不应该写 DTensor 逻辑；输出布局用 `inner_out_src`/out_src 声明（§10.5.1） |
| production 前向 view/reshape shape mismatch（显式 `num_heads` 写法） | 模块 q/k/v 未被识别为 colwise（命名非标准），头数改写（D-17）未命中 | `ARCH_OVERRIDES` 注册命名规则（§5.5）使 q/k/v 归 COLWISE，或 `plan_overrides` 手写 spec；非头名的 TP 缩放属性用 `tp_divide_attrs` 声明（§5.7） |
| `head-count adjustment: ... not divisible`（警告） | 模块头数属性不能被 tp_size 整除，已保持原值 | 调小 tp_size；确认该属性确为头数（否则可忽略） |
| 参数未分片且无报错 | 命名不命中默认规则（落 SKIP，只有 warning） | `ARCH_OVERRIDES` 注册架构规则（§5.5） |
| `ValueError: ... 模板不匹配（典型：lm_head.bias）` | 权重沿输出维 `Shard(0)` 而 bias 未随输出通道同切（D-22 检查，典型：lm_head 带 bias） | 按报错提示用 `plan_overrides` 声明 `{"lm_head.bias": {TP: shard(0)}}`，或移除该 bias（§5.1） |
| `ValueError: ... D-22 后置加法要求 bias 保持 Replicate` | rowwise 兄弟权重下 bias 显式声明了非 Replicate 的 TP placement | 从 `spec.params` 移除该 bias 声明，或改为 `{TP: replicate()}`（§5.1） |
| `ValueError: ... rowwise bias 后置（D-22）v1 仅支持单输出边界` | 多输出边界含 Partial 归约且带 rowwise bias，框架无法归因 | 用 `local_compute_fn` 接管该区域，自行在归约后加 bias（§5.1） |
| `WARNING: ... 非 nn.Linear，框架不擅自修改其 forward 语义` | rowwise + Partial 边界上的带 bias 模块不是 nn.Linear（如 GPT-2 Conv1D），bias 会在归约中重复计数 | 把 bias 移到边界通信后 / 改用 nn.Linear / `local_compute_fn` 接管（§5.1） |

---

## 附：运行测试

```bash
python -m pytest tests/components/distributed/ -q   # 447 例
```

单进程用例直接跑；多进程用例经 `run_dist`（spawn + gloo/CPU，macOS 可跑），
覆盖 TP/CP/EP 及两两组合的 plan golden、production 数值（vs 单卡参考）、
validate 校验与双模式等价。
