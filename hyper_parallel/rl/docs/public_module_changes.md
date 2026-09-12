# RL 对公共模块的修改说明

## 目的

本文面向 `hyper_parallel/rl/` 之外公共模块的 CODEOWNER，说明 Qwen3 RL 迁移修改了哪些文件、根因是什么、为什么不能
只在 RL 内解决，以及对原有接口和普通训练的影响。

候选代码基于 `upstream/master@b9fa61a980665bf9b5e00d16ba797a870c2d98c9`。公共实现不包含 rollout、GRPO 或
`consistency.enabled` 分支；所有修改都是 HyperAutoModel、TP/FSDP、DCP 或 optimizer 的通用语义修复。

## 修改总览

| 范围 | 文件 | 目的 |
| --- | --- | --- |
| Checkpoint | [`checkpoint_loader.py`](../../auto_models/_transformers/checkpoint_loader.py) | 对齐 Transformers 5.5.4，并保留 master replacement conversion |
| Materialization | [`infrastructure.py`](../../auto_models/_transformers/infrastructure.py) | `to_empty` 后保留 TP layout 与 tied Parameter identity |
| TP/FSDP setup | [`distributed/infrastructure.py`](../../auto_models/components/distributed/infrastructure.py) | Pure TP 使用 size-one FSDP 同步 replicated gradients |
| TP planning | [`param_role.py`](../../auto_models/components/distributed/param_role.py)、[`sharding_planner.py`](../../auto_models/components/distributed/sharding_planner.py) | Tied aliases 同时进入分类、shape 和 placement contract |
| TP unwrap | [`sharding/apply.py`](../../auto_models/components/distributed/sharding/apply.py)、[`sharding_applier.py`](../../auto_models/components/distributed/sharding_applier.py) | 保留 layout，并让 tied aliases 共享同一 Parameter identity |
| DCP | [`distributed_checkpoint/api.py`](../../core/distributed_checkpoint/api.py) | `no_dist=True` 不访问未初始化的 distributed runtime |
| Optimizer | [`adamw.py`](../../core/optimizer/adamw.py)、[`optimizer.py`](../../core/optimizer/optimizer.py) | 正确处理 CPU-offload device 与 DTensor state restore |
| Gradient clipping | [`clip_grad.py`](../../platform/torch/clip_grad.py) | Plain TP Parameter 使用保留的 layout 计算 global norm |
| 公共测试 | [`test_tied_parameter_sharding.py`](../../../tests/ut/auto_models/test_tied_parameter_sharding.py)、[`test_api.py`](../../../tests/ut/core/distributed_checkpoint/test_api.py) | 为公共 tied 与 no-dist 语义提供最小回归 |

此外修改根 [`.gitignore`](../../../.gitignore) 排除本地 RL artifacts，并在根 [`AGENTS.md`](../../../AGENTS.md) 增加 RL
局部开发规则的读取入口；两者没有运行时影响。

## Checkpoint 与 Materialization

### EP combine 的可选 family 回调

- 文件：`hyper_parallel/auto_models/components/distributed/ep_utils.py`。
- 必要性：Qwen3-MoE 的 HF 语义是在 FP32 中应用 router weights，随后转回 hidden dtype 并按 Top-K 维求和；公共 EP
  primitive 原本先将 router weights 转成 hidden dtype。仅包装 RL leaf 无法恢复已经丢失的 FP32 权重精度，而复制 dispatcher
  又会形成第二套 all-to-all。
- 修改：`ep_routed_forward` 增加可选 keyword-only `combine_fn=None`。显式回调接收原始 dtype 的 router weights；默认值
  仍走原先的 hidden-dtype weighting/index-add，现有调用无需修改。该函数属于 auto-model EP primitive，不在包根 `__all__`。
- 影响：Qwen3 的专用 combine 位于公共 `ep_utils.py`，供 Trainer factory 和 Hyper rollout 共用；不改变通信或
  默认梯度行为。默认/自定义 combine 的前向与 autograd 回归位于 `rl_tests/test_direct_reshard.py`。

### Transformers Checkpoint Loader

固定运行环境使用 Transformers 5.5.4，其 `rename_source_key` 参数为 `prefix`，且当前 transform 对象不提供可依赖的
`was_used()`。同时，最新 master 增加了 scope-aware high-performance replacement conversion。

最终处理为：

- 所有 checkpoint route 使用 `prefix`；
- 依据实际 renaming/converter pattern match 记录 used transforms；
- 多个 converter 共享 source pattern 时保留 master 的 scope selection；
- replacement base mapping 同样返回实际 used transforms，不引入版本探测或双接口 fallback。

为什么位于公共模块：模型加载发生在 RL Actor 构建之前。RL monkeypatch 无法可靠覆盖 replacement、普通 Trainer 和
checkpoint reload，并会形成第二套 loader。

接口影响：`CheckpointManager.load_checkpoint` 和其他公开签名不变；修改只作用于内部 conversion route 与报告。

### DeepSeek-V3 原生 Converter 的 Scope 兼容（2026-09-03）

- 修改文件：`hyper_parallel/auto_models/_transformers/checkpoint_loader.py`。
- 触发问题：Moonlight 使用 Transformers 5.5.4 原生 `WeightConverter` 加载 checkpoint 时，该对象没有 Hyper replacement
  流程动态附加的 `scope_prefix`，公共 loader 直接访问属性会在模型 forward 前抛出 `AttributeError`。
- 修改方式：内部统一通过兼容 helper 读取 `scope_prefix`；缺少该属性的原生 converter 按 unscoped converter 处理，已有
  Hyper replacement converter 的 scope-aware 选择保持不变。
- 公共修改必要性：converter 选择发生在 HyperAutoModel 通用 checkpoint route 中，早于 RL Actor/Trainer 构建；放入 RL
  会复制公共 loader 并遗漏普通 Trainer 和 checkpoint reload 路径。
- 接口影响：没有新增、删除或修改公开函数/方法签名；无 conversion 的直接 key load 路径不变，也没有引入 DeepSeek、
  Moonlight 或 RL 专用分支。

### Meta Materialization

`model.to_empty()` 会重新创建 Parameters。TP 参数在此前已经携带 layout，Qwen3 embedding/LM head 还需要保持同一
Parameter identity。Materialization 现在按完整 alias 名称保存/恢复 `_sharding_spec`；parameter alias 重新注册同一个
Parameter 对象，buffer alias 继续共享同一底层 tensor。

仅恢复 storage 会让 optimizer 仍持有 materialization 前的 Parameter alias，首次 checkpoint 导出时可能找不到当前模型
Parameter 对应的 optimizer state。保持 identity 同时保证 tied update、optimizer ownership 和 DCP 映射一致。

为什么位于公共模块：这是所有 HyperAutoModel TP+tied 模型的 materialization 语义，与 RL 或 consistency 无关。

## TP/FSDP 参数语义

### Tied Alias Planning

PyTorch `named_parameters()` 默认去重 tied Parameters，导致 `embed_tokens.weight` 与 `lm_head.weight` 中只有一个进入
classifier/planner。相关遍历改为 `remove_duplicate=False`，使两个公开 FQN 都获得 role、shape 和 placement。

### Parameter Identity 与 Layout

TP apply 结束后 production path 将 DTensor unwrap 为 local Parameter：

- 同一个 DTensor identity 只创建一个 local Parameter，所有 tied FQN 复用它；
- local Parameter 保留原始 `_sharding_spec`；
- tied replication 绑定同一 Parameter 对象，而不是只共享当前 storage。

只共享 storage 会在 FSDP 首次 unshard 后产生两个独立 buffer，optimizer update 后 embedding 与 LM head 随即分叉。

### Pure TP Gradient Reduction

Pure TP 中 row-parallel 输入会让 replicated norm 参数得到 rank-local Partial gradient。现有 FSDP source-layout path 已负责
该 reduction，但 runtime setup 过去在 `dp_shard=1` 时关闭 FSDP manager。现在 `tp_size>1` 也启用 size-one FSDP，复用
同一 reduction、scaling 和生命周期。

为什么不能放入 RL：这些问题在普通 Qwen3 TP 训练中同样存在，且必须在 FSDP/optimizer 看到参数之前解决。RL 侧 hook
会重复公共 distributed 逻辑并遗漏 checkpoint、clip-grad 和非 RL Trainer。

接口影响：`ParameterClassifier.classify`、`ShardingPlanner.plan` 和 distributed setup 的公开签名不变。行为变化只发生在
tied 参数或 `tp_size>1` 的路径。

## DCP、Optimizer 与 Clip-Grad

### DCP `no_dist`

`save/load(..., no_dist=True)` 的语义是无需 process group。实现不再查询 rank/world size，也不执行 barrier；默认
`no_dist=False` 的 collective 行为不变。

### AdamW 与 Optimizer Restore

- Functional AdamW 的 step tensor 跟随 `params[0].device`，避免 CPU-offload 参数与 NPU state step 混用。
- `ChainedOptimizer.load_state_dict` 在 `SkipDTensorDispatch` 中初始化 optimizer slots，保持与真实 local optimizer step
  相同的 storage path。

### Global Gradient Norm

Production TP unwrap 后 parameter/grad 可能都是 plain Tensor。Clip-grad 在不存在 DTensor spec 时读取 Parameter 保存的
`_sharding_spec`，使用 `Layout.placements` 和 mesh 计算 shard/partial group；普通无 layout Parameter 行为不变。

这些修改属于 DCP、optimizer 和 platform 的公共合同。若放入 RL，会复制 state restore、device 和 norm 算法。

## 接口与回归结论

对最新 master 与候选 tree 做 AST 签名比较，11 个公共实现文件没有新增、删除或修改公开函数/方法参数：

```text
public_signature_changes=none
```

行为隔离如下：

| 修改 | 不受影响的默认路径 |
| --- | --- |
| Checkpoint conversion | 无 conversion 的直接 key load |
| Tied planning/identity | 非 tied 模型 |
| Size-one FSDP | TP1 且无 DP/FSDP 的模型 |
| DCP no-dist | 默认 collective `no_dist=False` |
| AdamW device | 参数已位于当前 accelerator 的路径 |
| Clip-grad layout | 普通无 `_sharding_spec` Parameter |

最新 master 同步后，完整 HyperParallel-RL suite 与公共 tied、DCP、optimizer、clip-grad、parallel-dims 和 upstream
swap-optimizer 回归在同一 pytest 进程中为 `466 passed`。公共修改没有引入兼容 alias、运行时版本分支或 RL 专用配置。

## CODEOWNER 审查建议

### 训练与 Hyper rollout 共用 Qwen3 EP combine

`auto_models/components/distributed/ep_utils.py` 新增 `qwen3_moe_combine`，原样迁移 RL Hyper MoE 的
FP32 router weighting、hidden-dtype输出与Top-K顺序求和逻辑；RL 旧导出名继续指向该公共函数，不复制实现。
`ep_compute.py::qwen3moe_ep_compute_fn` 增加可选 `hf_combine=False`，RL 的 EP 训练显式传入True。
默认False仍走原有 aggregation，其他 archetype 不变；新增的 keyword-only 参数不破坏已有调用。
必要性：训练 compute factory 不能反向依赖 RL/vLLM，且训推需要同源 combine；放在 RL 中会造成循环依赖或重复算法。
这些函数是 auto_models 的组件 API，不是 `hyper_parallel.__all__` 根级导出。模型支持边界见 [MoE 模型](moe_models.md)。

审查公共修改时建议按以下顺序：

1. 检查修改是否只触发于 conversion、tied、TP、no-dist 或 offload 路径；
2. 检查 source layout 是否始终来自 HyperAutoModel planner，而不是 RL rollout metadata；
3. 检查 tied aliases 是否共享同一 Parameter identity；
4. 检查 `no_dist=False`、TP1 和普通无 layout 参数的原行为；
5. 联合运行公共 UT 与 `hyper_parallel/rl/rl_tests/test_master_qwen3_contracts.py`。
