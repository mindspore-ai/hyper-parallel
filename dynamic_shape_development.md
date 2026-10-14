# HyperParallel 静态图动态 Shape 开发说明

本文说明当前联合图的动态输入机制、BaseTrainer 接入和 NPU 文本训练入口。文件链接相对于仓库根目录；链接标注当前源码中的函数名与起始行，便于代码变化后重新定位。按尺寸生成专用 FX 图和缓存分派见 [尺寸特化开发说明](size_specialization_development.md)。

动态 shape 允许 batch、sequence 等用户输入维度变化，并复用前向与反向联合 FX 图。 “静态图”指算子拓扑被捕获；图中部分尺寸仍是 `SymInt`，执行对象是并行 Pass 改写后的 `GraphModule`。这项能力不等同于算子融合、NPU Graph capture、任意动态控制流或动态输出尺寸支持。

## 1. 配置与最小使用方式

标准 TextTrainer / BaseTrainer 使用现有配置入口：

```yaml
compile:
  enabled: true
  use_joint_graph: true
  dynamic: true
```

[CompileConfig:114](hyper_parallel/models/build_options.py#L114) 定义配置， [CompileConfig.selects_graph_compiler:137](hyper_parallel/models/build_options.py#L137) 用 `enabled and use_joint_graph` 选择联合图路径。 [BaseTrainer._build_graph_compiler:545](hyper_parallel/trainer/base.py#L545) 传递完整训练配置、模型和现有 mesh。无需另建训练循环；正常的 `scripts/train_lm.py` 入口即可使用这项功能。

| 配置 | 语义 |
| --- | --- |
| `dynamic: false`，无映射 | 默认静态捕获 |
| `dynamic: true`，无映射 | 尝试符号化所有用户张量维度 |
| 提供 `dynamic_arg_dims` | 仅符号化指定维度；映射优先于自动选择，即使 `dynamic: false` 也启用符号捕获 |
| `dynamic_arg_dims: {}` | 使用符号捕获机制，但所有用户维度保持静态 |

参数和 buffer 始终以静态尺寸捕获，开启自动动态模式不会把模型状态全部符号化。自动模式也受算子与 guard 约束；首次样本中的尺寸 0/1 可能被特化。

精确选择时，路径从 `forward_backward(**inputs)` 的顶层关键字开始：

```yaml
compile:
  enabled: true
  use_joint_graph: true
  dynamic_arg_dims:
    model_inputs.input_ids: [1]
    labels: [-1]
```

实际输入若包含随长度变化的 `attention_mask`、`position_ids`，也须标记对应维度。例如 `[B, 1, L, L]` mask 的后两维与序列长度关联；`[B, L]` mask 则标记末维。映射支持字典键、非负列表下标和属性路径；维度允许负索引，列表路径下标不允许负数。路径必须解析为已注册 pytree 中的 Tensor 叶子，校验见 [normalize_dynamic_arg_dims:29](hyper_parallel/compile/tracer/dynamic_shapes.py#L29) 与 [_resolve_input:46](hyper_parallel/compile/tracer/dynamic_shapes.py#L46)。

## 2. 独立编译器与 UT 对照

以下小代码说明接口；持续数值回归由 UT 承担：

```python
import torch
from hyper_parallel.compile import GraphCompiler, PassConfig


def train_fn(model, x, y):
    return (model(x) - y).square().mean()


model = torch.nn.Linear(4, 3)
compiler = GraphCompiler(
    model, train_fn,
    pass_config=PassConfig(fsdp_enabled=False),
    dynamic_arg_dims={"x": [0, 1], "y": [0, 1]},
    device=torch.device("cpu"),
)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
for batch, sequence in [(2, 5), (3, 7), (4, 3)]:
    loss, loss_dict = compiler.forward_backward(
        x=torch.randn(batch, sequence, 4),
        y=torch.randn(batch, sequence, 3),
    )
    optimizer.step()
    optimizer.zero_grad()
```

[GraphCompiler.forward_backward:218](hyper_parallel/compile/compiler.py#L218) 首次调用懒捕获，后续执行已有图并累积 `param.grad`。它返回 `(loss, loss_dict)`；梯度已经由联合图生成，调用方负责 optimizer step 和清梯度。 [GraphTrainer.train_step:135](hyper_parallel/compile/trainer.py#L135) 复用相同编译器，首次捕获后初始化其优化器。

[test_batch_and_sequence_reuse_graph:65](tests/ut/compile/test_dynamic_shapes.py#L65) 比较多种 shape 的 eager loss 和参数梯度； [test_grad_accumulation_across_shapes:112](tests/ut/compile/test_dynamic_shapes.py#L112) 检查跨 shape 累积； [test_optimizer_updates_match_eager:125](tests/ut/compile/test_dynamic_shapes.py#L125) 检查优化器更新与 eager 对齐。这些断言覆盖了小模型演示所需的数值契约。

## 3. 捕获、并行改写与执行调用链

```text
CompileConfig → BaseTrainer._build_graph_compiler → GraphCompiler
micro batch → forward_backward_step → model_inputs / labels / token_counts
首次调用 → 参数静态化 + 用户输入符号化 → make_fx 捕获 forward + autograd.grad
PassPipeline → 刷新 InputGuards → 执行联合 FX 图 → 累积 param.grad
后续调用 → 校验输入 → 执行同一图 → BaseTrainer 按训练节奏更新优化器
```

[GraphCompiler.compile:168](hyper_parallel/compile/compiler.py#L168) 组织以下步骤：

```python
joint_graph = trace_model_graph(self.model, self.train_fn, inputs, **trace_kwargs)
pipeline = PassPipeline.from_config(self.pass_config, self.parallel_plan)
pipeline.run(joint_graph.graph_module, **self._build_pass_kwargs())
if joint_graph.input_guards is not None:
    joint_graph.input_guards.refresh()
self._joint_graph = joint_graph
```

节选省略可选尺寸分派初始化；完整实现先校验分派配置，再执行可能修改真实参数的并行 Pass。 [trace_model_graph:508](hyper_parallel/compile/tracer/graph_tracer.py#L508) 分离模型状态和用户输入：

```text
输入：[参数与 buffer 的扁平序列] + [用户输入 pytree 的扁平序列]
输出：[loss] + [可训练参数梯度] + [loss_dict 的值]
```

[_make_fwd_bwd_fn:437](hyper_parallel/compile/tracer/graph_tracer.py#L437) 还原输入、临时替换模型状态，并在训练函数内部调用 `autograd.grad`。 [extract_module_state:183](hyper_parallel/compile/tracer/graph_tracer.py#L183) 对 tied state 去重； [_reparametrize_train_state:201](hyper_parallel/compile/tracer/graph_tracer.py#L201) 在捕获期间恢复共享权重的别名关系。 [run_traced_graph:651](hyper_parallel/compile/tracer/graph_tracer.py#L651) 每次读取当前模型状态，在 `torch.no_grad()` 下执行显式反向算子。 [GraphCompiler._accumulate_grads:526](hyper_parallel/compile/compiler.py#L526) 按状态 FQN 顺序映射梯度，并检查数量，避免错配或截断。

## 4. 符号输入与 helper 分工

[build_symbolic_inputs:66](hyper_parallel/compile/tracer/dynamic_shapes.py#L66) 建立共享的 FakeTensorMode / ShapeEnv：

```python
fake_mode = FakeTensorMode(allow_non_fake_inputs=True, shape_env=ShapeEnv(duck_shape=False))
fake_args = [fake_mode.from_tensor(value, static_shapes=True) for value in state_flat]
context = StatelessSymbolicContext(
    dynamic_sizes=[DimDynamic.DYNAMIC if dim in dims else DimDynamic.STATIC
                   for dim in range(value.ndim)]
)
```

用户每一维明确选为动态或静态，例如 `(2, 5, 4)` 可捕获为 `(s0, s1, 4)`。 `duck_shape=False` 避免不相关维度仅因首个样本值相同而产生相等约束；真正的关联由算子产生。同一 Tensor 经多个路径被选中时按对象身份合并动态维度，不修改用户真实张量上的 Dynamo 标记。对应回归见 [test_same_sample_sizes_are_independent:101](tests/ut/compile/test_dynamic_shapes.py#L101) 与 [test_alias_marks_merge_and_alias_changes_fail:209](tests/ut/compile/test_dynamic_shapes.py#L209)。

当前 helper 的职责已分开：维度选择和 guard 在 `dynamic_shapes.py`。状态重绑定、输入叶子检查、联合 callable 构造、前后向元数据整理仍是 `graph_tracer.py` 内的独立函数。这种拆分区分捕获基础能力与具体算子兼容，未引入另一套 tracer 或外部 MagiCompiler 依赖。 MagiCompiler 的动态维度选择思路可参考，但本实现入口是 `make_fx + autograd.grad`，单独调用 `mark_dynamic()` 不足以接入它。

## 5. 执行前的 shape guards

直接调用 FX GraphModule 时，Dynamo 不会替它检查 guards。 [InputGuards:115](hyper_parallel/compile/tracer/dynamic_shapes.py#L115) 保存用户输入的元数据、FakeTensor 和 ShapeEnv：

| 检查 | 保护的契约 |
| --- | --- |
| pytree 结构 | 输入键、嵌套结构和叶子类型保持一致 |
| 元数据与常量 | rank、dtype、device、layout、requires_grad 和 Python 常量保持一致 |
| Tensor 对象复用关系 | 原来同一对象的输入不能变成两个不同对象 |
| ShapeEnv 表达式 | 未标记的静态维度、关联尺寸、stride 和形状分支约束 |

[InputGuards.refresh:145](hyper_parallel/compile/tracer/dynamic_shapes.py#L145) 使用 `ignore_static=False`，并在 Pass 后重新生成约束。
[InputGuards.validate:149](hyper_parallel/compile/tracer/dynamic_shapes.py#L149) 在执行图和图内通信前检查输入。
对象复用检查不是任意 storage 重叠分析；guard 不持有首个样本的真实张量存储。
模型状态不参与用户 guard，FSDP 可合法地在 Pass 中把完整参数改成分片。

例如 `if x.shape[1] > 8` 只捕获首次走过的分支，后续跨边界会报 `ValueError`。
见 [test_python_shape_branch_is_guarded:145](tests/ut/compile/test_dynamic_shapes.py#L145)。
guard 失败不会自动重新捕获；尤其在模型已分片后，不能直接拿该状态重新建立完整模型图。
多卡输入仍需遵守各通信组的共同约束；本地 guard 不提供跨 rank 的失败协商。

## 6. token_counts 必须保持运行时输入

packing 改变有效 token 数；即使张量 shape 相同，忽略标签分布也会改变 loss 权重。
若训练 callable 从闭包读取首轮 Python 属性，后续属性更新不能自动变成新的图输入。
[BaseTrainer.forward_backward_step:714](hyper_parallel/trainer/base.py#L714) 显式传递计数：

```python
graph_kwargs["token_counts"] = {
    "current": self.current_token_counts,
    "step": self.step_token_counts,
}
loss, loss_dict = self.graph_compiler.forward_backward(
    model_inputs=micro_batch, labels=labels, **graph_kwargs,
)
```

[BaseTrainer.train_step:922](hyper_parallel/trainer/base.py#L922) 按 optimizer step 和 micro step 计算计数；
[_graph_train_fn:557](hyper_parallel/trainer/base.py#L557) 将其传给 [postforward:687](hyper_parallel/trainer/base.py#L687)。
[mean_global_loss:53](hyper_parallel/trainer/runtime/metrics.py#L53) 的捕获路径使用 Tensor collective 和 Tensor 加权公式：

```python
step_len = differentiable_all_reduce(step_token_counts[token_key], "sum", dp_cp_group)
backward_loss = cur_loss * cur_token_len / step_len * device_mesh.dp_size * device_mesh.cp_size
```

这里省略日志值的全局归约；捕获路径不通过 `.item()` 做数据依赖的计数分支。
计数是数值变化的标量 Tensor，不需要新增动态维度，也不能在尺寸特化时把数值固化。
[test_basetrainer_token_weights_are_runtime_inputs:354](tests/ut/compile/test_dynamic_shapes.py#L354) 检查计数改变时 loss 和梯度改变且继续复用图。

## 7. causal padding decomposition

常见 causal loss 对 labels 做 `pad(..., (0, 1), value=-100)`，再切片实现标签 shift。
部分 NPU 原生 `constant_pad_nd` 捕获路径曾将符号长度变成首轮长度，再经 CE 的尺寸关系约束 logits。
[trace_model_graph:508](hyper_parallel/compile/tracer/graph_tracer.py#L508) 仅在符号捕获时启用 PyTorch 自带 decomposition：

```python
decompositions = get_decompositions([torch.ops.aten.constant_pad_nd.default]) if symbolic else None
traced_graph = make_fx(
    _fwd_bwd_fn, decomposition_table=decompositions, **_MAKE_FX_KWARGS
)(*fake_args)
```

展开后继续传播符号长度，保持 pad、shift、交叉熵语义，未放宽 guard。
[test_causal_label_padding_preserves_dynamic_length:398](tests/ut/compile/test_dynamic_shapes.py#L398) 对照 eager loss 和梯度，并检查变长图复用。

## 8. 标准 FP32 随机初始化文本训练

保留的真实训练示例使用在线文本数据、TokenBatchLoader 和 dense packed attention。
[prepare_dynamic_text.main:22](hyper_parallel/compile/examples/automodel_text_graph/prepare_dynamic_text.py#L22) 生成确定性的变长 JSONL；
[动态训练 YAML:1](hyper_parallel/compile/examples/automodel_text_graph/train_online_lm_graph_dynamic_tp2_fsdp2.yaml#L1) 配置 TP2、FSDP2 和六次 optimizer step。
从仓库根目录执行，模型路径提供结构配置和 tokenizer 资产：

```bash
python -m hyper_parallel.compile.examples.automodel_text_graph.prepare_dynamic_text
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3 bash \
  hyper_parallel/compile/examples/automodel_text_graph/run.sh \
  hyper_parallel/compile/examples/automodel_text_graph/train_online_lm_graph_dynamic_tp2_fsdp2.yaml \
  --model.pretrained_model_name_or_path=/path/to/Qwen3-0.6B
```

[run.sh 使用说明:19](hyper_parallel/compile/examples/automodel_text_graph/run.sh#L19) 接受配置和 CLI 覆盖项，标准入口为 `scripts/train_lm.py`。
可设置 `PYTHON`、`MASTER_PORT`、`NPROC_PER_NODE`、`OUTPUT_DIR` 等常规 launcher 参数；动态能力本身不需要额外诊断入口或魔法配置。
YAML 明确使用 `model_init_dtype: float32`、`torch_dtype: float32`、`load_base_model: false`，运行的是随机初始化模型。
packing 将可变数量的原始文本拼成 `[1, packed_length]`，物理 batch 为 1，变化的是序列维度。
更换 tokenizer 或数据后，实际长度及其重复频率可能变化；动态 batch 维由 UT 另行覆盖。

## 9. 安装差异、并行与精度边界

已核对的安装差异集中在 SDPA：`whh_titan` 有 `torch_npu/_init/patches/sdpa_symbolic_patches.py`，SDPA 入口是 Python wrapper，直接调用 `npu_fusion_attention_v3`；`whh` 无该文件，SDPA 入口是 builtin。
builtin 经 C++ SDPA bridge 时可能把符号尺寸具体化；该安装 wrapper 则直接进入 `npu_fusion_attention_v3` 的符号 fake 路径。
`symbolic_npu_sdpa` 保留已有 Python wrapper，因此两种安装的捕获路径可能不同，升级后应按实际入口验证。
这里记录的是已确认的 SDPA 文件和入口差异，环境名称本身不是接口保证。

[_patch_engine_backward:114](hyper_parallel/compile/tracer/graph_tracer.py#L114) 与
[_non_strict_tracing_context:139](hyper_parallel/compile/tracer/graph_tracer.py#L139) 独立探测当前安装的内部 API；缺失 engine patch 时会告警，不能据此保证所有联合图捕获可靠。
这项探测不构成两个环境全部 tracer / autograd engine 补丁的差异判断。
升级或替换 torch / torch_npu 后，应先运行下面的 UT，再验证目标模型训练。

参数的静态尺寸使 FSDP 参数 all-gather 和梯度 reduce-scatter 可以与动态 activation 协作。
同一 TP 组需提供匹配的输入和通信形状；不同 DP 分片可在满足各自图约束时处理不同长度。
[BaseTrainer.step_optimizers_and_schedulers:877](hyper_parallel/trainer/base.py#L877) 仍承担训练期优化器更新。
当前 BF16 模型配套的独立 FP32 主参数优化器副本可能在首次图 FSDP 分片前按完整尺寸建立，分片同步问题尚未修复。
完整训练示例因此使用 FP32 参数；小算子的 BF16 能力验证不能推导为该优化器组合已经可用。
MoE 内部按 token 路由产生的数据依赖 shape 不会因输入维度符号化而自动支持。
动态 PP 在 [GraphCompiler.compile:168](hyper_parallel/compile/compiler.py#L168) 捕获前显式拒绝；CP、EP、sequence/loss parallel 的其他组合需各自验证。

## 10. 当前回归入口与证据范围

```bash
python -m pytest tests/ut/compile/test_dynamic_shapes.py \
  tests/ut/compile/test_size_specialization.py -q
```

当前 UT 以 CPU 数值对照和 mock distributed 覆盖输入符号化、guards、FSDP 改写、BaseTrainer 计数与尺寸分派。
历史开发运行还验证过多进程 FSDP、NPU causal loss / attention，以及 TP2 + FSDP2 的随机初始化模型变长训练。
这些记录是指定模型和安装组合的能力证据，不能当作仍维护的持续 ST 入口，也不表示收敛或吞吐验证。
完整模型尚不能从有限 loss 和完成 optimizer step 推导出逐参数 eager 数值对齐。
后续扩展优先检查：变化是否已成为显式图输入、guard 是否覆盖其约束，以及后端算子是否保持符号尺寸。
