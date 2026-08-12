# D-22：rowwise bias 边界后置加法（deferred bias）详细设计

> 上位文档：[hyper_models_dual_mode_dtensor_design.md](hyper_models_dual_mode_dtensor_design.md)
> （双模式 DTensor 总体设计，D-01'' ~ D-21）；本文是其第 22 号修订的专题设计。
> 使用口径：[components_distributed_tutorial.md](../guide/trainer/components_distributed_tutorial.md)。
> 日期：2026-08-12　状态：已定稿待实施

---

## 1. 问题定义

### 1.1 现象

带 bias 的 rowwise Linear（`o_proj` / `down_proj` / `.w2`）在 TP 下数值错误：
production 模式输出 = 正确值 + `tp_size × bias`。最小复现（2-rank gloo/CPU，
`nn.Linear(8, 4, bias=True)`，weight `{TP: Shard(1)}`、bias `{TP: Replicate}`、
out_src `Partial` → out_dst `Replicate`）：

```
max|production − 单卡| = 0.331124   ← 误差向量恰好等于 (tp−1)·bias
max|validate − 单卡|   = 0.000000   ← validate 完全正确
```

### 1.2 根因

`F.linear` 的 bias 是**融合**加在 matmul 结果上的。当前模板的执行序：

```
边界入口: hidden Shard(1) → Replicate（all-gather）
模块 forward（production：参数已解包为纯 local tensor）:
  F.linear(x_local, W[:, H/tp], b_full) → partial + b_full   ← bias 在此融合
边界出口: Partial → Shard(1)/Replicate（reduce-scatter / all-reduce，均为 sum）
        = Σ_ranks(partial_r + b) = 正确值 + tp·b             ← bias 被加了 tp 次
```

- **production 无修正**：参数解包后无任何 dispatch，raw `F.linear` 直接融合
  bias；
- **validate 恰好正确**：DTensor dispatch 的 elementwise add 有
  contribution-aware 修正（`parallel_elementwise.py`
  `_contributes_to_partial_output`——`Partial + Replicate` 时只有 TP 坐标 0
  的 rank 贡献 bias），所以 validate 对单单卡能对拍、且与 production
  **不一致**——双模式等价原则在此处存在破口；
- **测试缺口**：仓内对拍模型（llama/qwen/glm4）rowwise 位置全部无 bias，
  未被覆盖；受影响的是 OPT / GPT-NeoX / GPT-2 / BLOOM 这类 rowwise-bias
  模型。

### 1.3 设计不变量

1. **bias 在 TP 归约之后恰好加一次**（Megatron `RowParallelLinear` 同构：
   bias 不进 `F.linear`，all-reduce 之后再加）；
2. **双模式逐指令一致**：production 与 validate 走同一份"区域内无 bias、
   边界出口后加 bias"路径。bias 预除 tp_size 的方案因此被否决——它只适用
   于 production、破坏逐指令一致、bf16 下引入额外误差；
3. **参数身份不变**：bias 始终是原 FQN 上的注册 Parameter——state_dict /
   optimizer / checkpoint 零影响；
4. **框架零运行时推导**：defer 判定全部发生在 plan 期，apply 期按标记执行。

### 1.4 合法性论据

defer 在数学上合法 ⟺ 边界 out_src 的对应输出在 TP 维为 `Partial`——
Partial 契约本身保证"边界输出 = 各 rank 贡献之和"，纯线性求和与 bias
加法可交换。这把执行修正锚定在已有声明之上，而非新的启发式；out_src 非
Partial 的边界（无归约、或用户在区域内自行归约）**不适用也不启动**
defer。

---

## 2. 编译期：统一检测（plan 期 finalize pass）

### 2.1 落点：新增 `_finalize_deferred_biases(plan, model)`

与 `_finalize_tp_local_attr_plans`（D-18，Phase 4.6）同级，在 plan_overrides
全部合并之后运行。**检测基于最终 spec 声明 + 模型结构，而非 ParamRole**——
一条代码路径同时覆盖三类来源：

| spec 来源 | 覆盖方式 |
|---|---|
| 模板推导（attention/mlp） | `o_proj.weight` 等被声明 `{TP: Shard(1)}` → 命中 |
| merge override | params 继承推导值 → 同上命中；用户显式改 params 按用户声明判定 |
| **insert / `derive=False` 用户自声明 spec**（本设计明确覆盖点） | 用户写了 `{"proj.weight": {TP: Shard(1)}}` 即按 rowwise 处理——与命名无关，自研模块（如 `wo`/`dense`/`fc2`）同样生效 |

基于 spec 而非角色名的额外收益：对 `ARCH_OVERRIDES` 注册的非标准命名
（`wo`、`c_proj` 等）天然生效，无需角色表配合。

### 2.2 检测算法（逐边界、逐 bias）

对边界模块的每个参数 `X.bias`（相对路径，`model.named_parameters` 物理扫描，
**不以 spec.params 是否声明为前提**——bias 物理存在于模块上就会被
`F.linear` 融合，与是否声明分片无关）：

```
sibling = spec.params.get("X.weight")
skip（不 defer）⟺ 任一成立:
  ① sibling 未声明 / 无 tp 轴                —— 权重不切，无 Partial
  ② sibling 的 TP placement 不是 Shard(weight.ndim − 1)
                                             —— 非 contraction 维切分
                                                （colwise/embed/lm_head 走 §2.3）
  ③ spec.out_src 对应输出的 TP placement 不是 Partial
                                             —— 边界无归约，bias 本就只加一次
defer 前置校验（fail-fast）:
  ④ X.bias 若在 spec.params 中显式声明且 TP placement 非 Replicate
     → 报错：defer 要求 bias 保持 Replicate（归约后整体加一次），
       请移除该声明或改正 placement
defer 类型守卫（按已确认的决策：WARNING + 跳过）:
  ⑤ owner 模块不是 torch.nn.Linear 实例
     → WARNING（指明 fqn："Partial 归约会使 bias 被重复计数；
        该模块非 nn.Linear，框架不擅自改其语义——请将 bias 移至
        边界通信之后、改用 nn.Linear，或以 local_compute_fn 接管"），
        保持现状行为，不 defer
```

条件③的关键含义：用户自声明 spec 若 out_src 非 Partial（区域内自行
归约/不归约），bias 语义归用户 forward 自己负责，框架不干预——defer 严格
限定在"**框架管理的边界归约**"场景。

### 2.3 输出维切分 + bias 的模板不匹配检查（lm_head bias，fail-fast）

同一 finalize pass 顺带校验反向组合（已确认的决策：**直接报错**）：

```
sibling weight 的 TP placement 是 Shard(0)（输出维切分：colwise/lm_head/embed）
且模块上物理存在 X.bias，且 bias 的声明 placement ≠ Shard(同一输出维)
（Replicate、未声明或切错维）
→ ValueError（plan 期）：
  "<fqn>: 权重沿输出维 Shard(0) 切分时 bias 必须随输出通道同样切分，
   当前声明为 <placement>/未声明 —— 模板不匹配（典型：lm_head.bias）。
   请用 plan_overrides 显式声明 {\"lm_head.bias\": {TP: shard(0)}}，
   或移除该 bias。"
```

- 模板推导路径下 q/k/v/gate/up 的 bias 已由 D-19 归为 COLWISE `Shard(0)`，
  不会误触发本检查；
- 该检查把原先"lm_head 带 bias → 运行期 broadcast 形状崩溃"的远端报错
  提前为 plan 期的教学式 fail-fast。

### 2.4 存储与展示

```python
# sharding_config.py — ModuleShardingSpec 新增内部字段（用户不可配，
# 同 _needs_cp_attn / _tp_local_attr_plan 风格）
_deferred_bias_params: Tuple[str, ...] = field(
    default=(), init=False, repr=False, compare=False)
# 例：attention 边界 → ("o_proj.bias",)；mlp 边界 → ("down_proj.bias",)
```

- **merge 语义**：内部标记由 finalize pass 在合并后统一计算，天然不受
  override 影响（用户 spec 对象永不携带该字段）；
- **可观察性**：`plan.explain()` / dump 边界段新增一行：
  `后置 bias（D-22）: o_proj.bias（TP 归约后恰好加一次）`。

---

## 3. 运行期：apply 机制（`sharding_applier.py`）

### 3.1 bias 抑制包装（区域内"无 bias 化"）

```python
def _install_bias_suppression(owner):
    """让 owner.forward 在区域内跑无 bias 线性（bias Parameter 原地保留）。"""
    original = owner.forward

    @functools.wraps(original)
    def bias_free_forward(*args, **kwargs):
        bias = owner.bias
        try:
            owner.bias = None      # nn.Module 允许把已注册参数置 None
            return original(*args, **kwargs)
        finally:
            owner.bias = bias

    owner.forward = bias_free_forward
```

要点：

- **参数身份不变**：`owner.bias` 始终是原 Parameter（同 FQN、同对象），仅
  forward 期间对 `F.linear` 不可见；state_dict、optimizer、
  `named_parameters` 完全不变；
- **泛化性**：不假设 `F.linear`——只要 owner 的 forward 读 `self.bias`
  即生效（nn.Linear 及 HF 各色 Linear 子类均如此）；
- **双模式同构**：两模式同样安装。validate 下区域内 matmul 走 dispatch
  产出干净 Partial，out_src 校验对象不再混入 bias（原 contribution-zeroing
  路径被这条更干净的路径取代，行为等价）；
- 安装时机：Phase C 包装阶段、inner-wrap（Step 1）之前，两模式无条件执行。

### 3.2 边界后置加法

```python
def _add_deferred_biases(module, spec, output):
    """边界出口通信之后，每个 defer bias 恰好加一次（forward 期读取）。"""
    for param_path in spec._deferred_bias_params:    # 如 "o_proj.bias"
        owner = module.get_submodule(param_path.rpartition(".")[0])
        output = output + owner.bias
    return output
```

- **forward 期读取** `owner.bias`（非闭包捕获）：production 下是 Phase C
  入口解包后的 local tensor，validate 下是 DTensor——读取即当前形态，
  autograd 边自然建立；
- 多 defer bias（同边界多个 rowwise 出口）逐个加，数学上仍各恰好一次。

### 3.3 三个包装路径的插入点

| 路径 | 位置 | 张量形态 |
|---|---|---|
| `_wrap_production_forward` | `boundary.redistribute_outputs(outputs)` 之后、return 之前 | 纯 local add |
| `_wrap_validate_forward` | Step 6 之后：嵌套场景 output 仍为 DTensor（add 走 dispatch，`Shard(1)+Replicate→Shard(1)` 逐点正确）；最外层已 `to_local` | 两种形态天然兼容 |
| `_wrap_local_region_forward` | 边界出口 + 最终解包之后（覆盖 local-region 边界带 rowwise bias 的情形） | 纯 local add |

validate 的 Step 5（terminal out_dst 防御校验）保持在后置加法**之前**——
bias 是 Replicate，add 不改变 placement，校验语义不变。

---

## 4. 正确性论证

### 4.1 前向（SP，out_dst=Shard(1)）

```
区域内:   out_r = x_local_r @ W_rᵀ            （无 bias，纯 Partial 贡献）
边界出口: chunk_r = (Σ_ranks out_r) 的本地序列段  （reduce-scatter）
后置加:   y_r = chunk_r + b                    ✓ bias 恰好一次
```

nosp（out_dst=Replicate）：all-reduce 后各 rank 持全量输出，各加自己那份
（值相同的）bias 拷贝 → 全组一致 ✓。

### 4.2 梯度

bias 在 `tp_grad_info` 中保持 `Replicate` 条目（TP 组 all-reduce 语义），
不变：

- **SP**：rank r 的 `∂L/∂b` = 本地序列 chunk 的输出梯度之和 → TP
  all-reduce 后 = 全序列梯度 ✓（与单卡一致）；
- **nosp**：各 rank 局部图里 bias 恰好进入输出一次——defer 只移动加法
  位置，**不改变每 rank 的梯度贡献结构**，聚合语义与现状逐点一致；
- 对拍保障：`testing/grad_equiv.py` 双模式梯度等价工具覆盖（§6）。

### 4.3 与 D-19 的组合全集

Linear bias 的完整处理矩阵：

| bias 归属 | placement | 加法位置 | 依据 |
|---|---|---|---|
| colwise Linear（q/k/v/gate/up/w1/w3） | `Shard(0)`（随权重） | 区域内本地加，被后续 rowwise 消费，不过边界归约 | D-19 |
| rowwise Linear（o/down/w2，nn.Linear，Partial 契约） | `Replicate` | **边界归约之后加一次** | **D-22** |
| rowwise Linear（非 nn.Linear owner） | `Replicate` | 保持现状（WARNING 提示风险） | D-22 §2.2⑤ |
| 非 Linear bias（norm/router/未匹配） | `Replicate` | 区域内（输出非 Partial，本就正确） | 现状 |
| 输出维切分 + bias 声明不符（lm_head.bias 等） | — | plan 期 fail-fast（模板不匹配） | D-22 §2.3 |

---

## 5. 边界情形与限制

1. **多输出边界**：v1 限定 defer 只作用于单输出边界（attention/mlp 均
   单输出）。`_deferred_bias_params` 非空且 out_src 声明多输出 → plan 期
   fail-fast，指引走 `local_compute_fn` 自定义；
2. **用户自声明 spec（重点覆盖）**：检测基于最终 spec（§2.1/§2.2），
   insert / `derive=False` / merge 三种形态统一处理；用户声明的 bias
   placement 非 Replicate → fail-fast（§2.2④）；用户 out_src 非 Partial
   → 框架不干预（§2.2③）；
3. **非 nn.Linear owner**（GPT-2 `Conv1D`、自研线性层）：WARNING + 跳过
   defer（已确认决策），保持现状行为、不静默改语义；接入指引写在 WARNING
   与教程排错索引中；
4. **lm_head bias**：plan 期 fail-fast"模板不匹配"（已确认决策，§2.3），
   报错附可粘贴的 plan_overrides 写法；
5. **重复 apply**：抑制包装幂等（嵌套 suppress 仍无 bias）；后置加法挂在
   边界 wrapper 内，与现有"重复 apply 不保证"的基线一致，不引入新风险；
6. **性能**：损失 addmm 融合（bias 从融合 epilogue 变为单独 add），相对
   边界集合通信可忽略；per-forward 两次属性 setattr，纳秒级。

---

## 6. 测试计划

| 层 | 用例 |
|---|---|
| plan 单测 | 模板推导：`o_proj.bias`/`down_proj.bias` 被标记，`q_proj.bias`（COLWISE）、norm bias、无兄弟 bias 不标记；无 TP 不标记；out_src 非 Partial 不标记；**insert 用户 spec 的 rowwise+bias 同样被标记**；bias 声明非 Replicate → fail-fast；lm_head bias → "模板不匹配"fail-fast；非 nn.Linear owner → WARNING 且不 defer |
| 数值对拍（核心） | 带 bias 的 TinyLlama 变体（`attention_bias=True` + `mlp_bias=True`），TP=2 × SP 开/关 × production/validate vs 单卡逐位对拍（repro 脚本固化为 `tests/components/distributed/test_dist_s*_rowwise_bias.py`） |
| 梯度 | grad_equiv 对 bias 梯度做双模式 + 单卡三方对拍 |
| 状态兼容 | apply 后 `state_dict` key 集合与 apply 前完全一致（含 `o_proj.bias`）；optimizer 参数组无遗漏 |
| 回归 | 既有 388 例全绿（无 bias 模型行为不变） |

---

## 7. 改动清单

| 文件 | 改动 |
|---|---|
| `sharding_config.py` | `ModuleShardingSpec._deferred_bias_params` 内部字段；dump/explain 输出一行 |
| `sharding_planner.py` | 新增 `_finalize_deferred_biases(plan, model)`（Phase 4.6 同级）：统一检测 + defer 前置校验 + 模板不匹配检查（§2.2/§2.3） |
| `sharding_applier.py` | `_install_bias_suppression` + `_add_deferred_biases` 两个 helper；Phase C 安装抑制包装；三个 forward 包装路径各加一行后置调用 |
| 测试 | `test_s1_plan_arch.py`（标记/校验单测）+ 新增多进程数值/梯度对拍 |
| 文档 | 总体设计 §13 加 D-22 行、§5 BIAS 行与 §12.2 后新增交叉引用；教程 §5.1 bias 段改写为全矩阵（§4.3 表）+ §13 排错索引加两行（WARNING 与模板不匹配报错） |

## 8. 备选方案记录（已否决）

| 方案 | 否决理由 |
|---|---|
| bias 预除 tp_size | 只适用 production，破坏双模式逐指令一致；bf16 引入额外误差；validate 的 contribution-zeroing 路径反而要跟着改 |
| production 侧 patch `F.linear` 全局拦截 | 作用域不可控（嵌套/多边界相互污染），违背"显式注入"制度 |
| 新增 ParamRole.ROWWISE_BIAS | 角色是命名层的概念，defer 是执行层的修正；且对用户自声明 spec（无角色介入）无效——检测必须锚定 spec 声明（§2.1） |
