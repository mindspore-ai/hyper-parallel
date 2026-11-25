# 独立 Loss/PPL 评估

[English](evaluation.md) | **简体中文**

`examples/training_demo/run_text.py` 使用 `run_mode: eval` 执行本功能。默认仍为 `train`。

支持单卡、DP 和 dense FSDP/HSDP，无需构建 optimizer 或 scheduler。
当前拒绝 TP/CP/EP/PP/SP、训练编译、activation checkpoint、activation swap、HyperOffload 和 worker recovery。

## 启动

修改 [配置示例](../../../examples/training_demo/qwen3/eval.yaml) 中的路径后执行：

```bash
torchrun --nproc_per_node=2 examples/training_demo/run_text.py \
  examples/training_demo/qwen3/eval.yaml
```

设置 `checkpoint.restore_from` 指定 DCP checkpoint；使用 `LATEST` 时同时配置 `checkpoint.checkpoint_dir`。
缺失 checkpoint 会报错。未指定恢复时使用模型构建入口加载的权重。Ascend 使用 `hccl`，NVIDIA 使用 `nccl`。

## Evaluation

### Loss 评估与生成式评估

`evaluation.mode` 默认是 `loss`。它把已有文本交给模型，在因果注意力约束下计算真实标签的
负对数似然（NLL），然后汇总 loss 和 PPL。模型处于 `eval()` 状态，关闭梯度，直接执行
`model.forward(..., use_cache=False)`；它不通过 `infer.generate` 逐 token 生成答案。

`mode: generation` 是预留配置，当前运行入口会抛出 `NotImplementedError`。
生成式评估需要先根据 prompt 生成答案，再与参考答案或判分规则比较。未来实现时应复用
predict 使用的 `infer.generate`，再增加判分和汇总逻辑，而不是维护另一套生成循环。

独立 loss 评估用于比较不同 checkpoint，例如分别加载训练第 1000、2000、3000 步保存的权重，
在同一份验证数据上计算 loss/PPL。比较时应固定数据范围、tokenizer、模板、截断与监督 mask、
统计口径和数值精度；设有 `max_batches` 时还应固定组批配置和 DP 拓扑，避免实际评估样本发生变化。
更低的 loss/PPL 表示在这份数据上预测真实 token 的能力更好，不代表所有下游任务表现都更好。
训练日志中的 loss 可能来自不同数据批次、混入辅助目标或使用不同归约口径，不能直接当作这项独立评分。

### 数据范围

`evaluation.dataset`、`evaluation.dataloader` 可以覆盖顶层数据配置。
`evaluation.split` 可选 `valid`、`test`、`dataset`，最后一种用于单个未切分 Dataset。
任务不会隐式选择训练集。

- `max_samples` 在选定 Dataset 的全局索引范围上生效，然后才做 DP 分片。
- Online Mapping 将请求上限截断到自然切分大小；`max_samples: null` 遍历整个自然切分，不读取训练迭代计划。多个加权来源采用无放回采样，权重决定顺序，某个来源耗尽后继续从剩余来源取样；固定 seed 时，改变上限仍保留相同前缀。
- Iterable 来源需要 `max_samples`；其有限前缀会物化后分片。
- Indexed 构建器需要明确样本计划，使用 `max_samples` 或构建器自己的显式 split sizes。
- `max_batches` 限制前向轮数。动态 token batching 的每轮样本数由 token budget 决定。
- 不丢弃不足整批的尾部，不通过重复有效样本补齐 DP。耗尽数据的 lane 使用零权重前向保持 FSDP 轮次一致。

例如：

```yaml
evaluation:
  mode: loss
  split: valid
  batch_size: 2
  max_samples: 1000
  max_batches: 20
  metrics:
    - _target_: hyper_parallel.evaluation.metrics.LossMetric
    - _target_: hyper_parallel.evaluation.metrics.PerplexityMetric
```

这会先选取最多 1000 条原始记录，再分给各 DP lane，每个 lane 最多执行 20 轮前向。
数据耗尽或达到轮数上限时结束；两个上限均为 `null` 时遍历选定的有限数据集。
`num_samples` 统计含有效监督的模型样本；packing 不会把多个样本算成一个 tensor row。
原始记录被 transform 切成多个窗口时，每个窗口分别计数，所以它不一定等于原始记录数。

### 动态组批与 padding

当前 eval 示例使用已有的
[`TokenBatchLoader`](../../../hyper_parallel/data/batching/build_dataloader.py)。
它按编码后的输入 token 总量组批，默认每个 DP lane 的预算为：

```text
token_budget = evaluation.batch_size × data_transform.max_seq_len
```

也可在选用的 dataloader 配置下显式设置 `token_budget`。在这种动态 loader 中，`batch_size`
用于源数据读取和默认预算计算，不是最终前向的样本条数上限。例如 `batch_size: 2`、
`max_seq_len: 1024` 得到 2048 token 的预算，可以容纳 2 条 1024 token 或 4 条 512 token 的样本。
此预算统计输入长度，与 loss 分母中的有效监督 token 数不同。

loader 将候选样本放入缓冲区，按顺序扫描：能放进本轮剩余预算的就选中，放不下的保留给后续轮次，
并继续检查后面的候选。它不会按长度排序，也不会为了凑满预算而切开一个模型样本。
原始记录是否预先切成多个模型窗口，由前面的 data transform 决定。
例如预算为 2048，缓冲区长度依次为 `900、700、600、400`，本轮选择 `900 + 700 + 400 = 2000`，
把 600 留到后面，剩余 48 不必填满。单条样本超过预算时允许独占一轮，因此预算不是绝对长度上限；
模型上下文和 collator 的长度约束仍需满足。

`min_buffered_samples` 控制候选缓冲区的最低样本数量，不是每轮样本数。通常同时达到候选数量和
token 总量阈值才开始组批；源数据耗尽时会排空缓冲区，不丢弃尾部样本。

配套的 `TextPackingCollator` 把选中样本拼成 `[1, packed_length]`，用 `cu_seq_lens` 记录样本边界。
默认只补齐执行所需的对齐长度，不补满 token 预算；显式设置 `pad_to_length` 才补到指定长度。
padding 标签设为 `IGNORE_INDEX`，不计入 loss/PPL 的有效 token 数。不同轮次可以有不同输入长度，
因此 `max_batches` 限制前向次数，不能换算成固定的样本条数。

### 指标与统计口径

当前内置 `LossMetric` 和 `PerplexityMetric`，默认输出名分别为 `loss`、`perplexity`。
两者可在同一次 eval 中配置，报告还包括样本数、有效 token 数、前向轮数、耗时和状态。

```text
loss = 所有有效 token 的 loss 总和 / 有效 token 总数
PPL = exp(全局平均 NLL)
```

例如两轮分别有 10、90 个有效 token，平均 loss 分别为 2、1，整体 loss 是
`(10 × 2 + 90 × 1) / 100 = 1.1`，不是 batch 均值的平均值 `1.5`。

统计先累计分子、分母，最后归约；不平均 batch 均值或 rank 均值。
内置 loss/PPL 与样本数共用一次数值归约，归约前交换统计 schema 与错误状态。
`compute()` 可以重复读取，不重新归约。CPU/CUDA 使用 FP64 累加；NPU 使用带补偿的 FP32 累加，
避免 torch_npu 的 FP64 转换回退导致精度丢失。样本数在设备上保持整数计数，通信使用 FP32 高低位对，兼容 HCCL，
归约后在 CPU FP64 中合并。该方式保持常量统计内存，不逐 batch 同步到主机。
补偿用于减小本地累加误差；集合通信仍对每个通道执行 FP32 求和，不保证精确的 FP64 运算。

loss 模块必须声明 `loss_normalization`：`token_mean`、`sum` 或 `global_mean`。
已归约的 `global_mean` 不会再次累加。`LossStatistics.is_lm_nll` 是布尔标记：仅当统计量为纯语言模型 NLL 时
设为 `True`，PPL 才允许对其取指数。普通目标函数或混入辅助损失的值保持 `False`，仍可作为 loss 输出。
loss 模块可通过 `statistic_is_lm_nll`（映射或按 loss key 取值的可调用对象）声明这一含义。
chunked causal-LM loss 返回单个局部 token 均值；当模型以非零系数把 router aux 项折入其中时，
它将 `foundation` 声明为 `False`，PPL 会被明确拒绝。
aux 系数缺省或为零时保留纯 NLL，可以正常计算 PPL。
全 mask 的 batch 贡献零权重，不会把空交叉熵的 NaN 传播到最终报告。

### 自定义指标

`evaluation.metrics` 接受可导入的
[`Metric`](../../../hyper_parallel/evaluation/metrics/base.py) 类路径，例如
`_target_: my_project.metrics.CustomMetric`。
它的生命周期是：

```text
reset() → update(statistics)（每轮）→ synchronize(context, device) → compute()
```

`update()` 接收的是具名 `LossStatistics`，包含分子、分母和 `is_lm_nll`，不包含 logits、标签或生成文本。
`compute()` 返回 `{"指标名": 数值}`，只读取结果，不应再次执行分布式通信。各指标的输出名必须不同。

如果新指标只需对已有 loss 结果做换算，可以继承 `LossMetric` 并仅覆盖 `compute()`；例如纯 NLL
除以 `ln(2)` 得到 bits per token，计算前仍需检查该统计的 `is_lm_nll`。原样继承 `update` 和
`synchronize` 时可共用内置归约，覆盖任一方法则执行该子类自己的累计和同步流程。
更一般的指标可直接实现 `Metric` 的四个生命周期方法。如果只需原始共享统计，还可显式让
`supports_shared_reduction` 返回 `True`，并实现 `set_reduced_statistics`。

Accuracy、F1、BLEU、ROUGE 当前没有内置，也不能仅添加一个指标配置就获得这些数据。
例如 token accuracy 需要单步前向额外产出预测正确数和有效 token 数；生成文本指标还需要生成与判分流程。
底层 `Evaluator` 可以接收自定义 step，但 `TextTrainer.evaluate()` 当前固定装配 `LossEvalStep`，
没有配置自定义 step 的 YAML 入口。这类扩展需要同时实现单步数据产出和对应的 Trainer 侧接入。

## 分层

`evaluation/` 不导入 Trainer。Trainer 创建 `LossEvalStep` 并通过构造参数传给 `Evaluator`；
评估器每轮调用 `step(round_index)`，得到统计量。step 在 Trainer 侧读取 batch、调用 model/loss_fn，
评估器只负责循环与汇总。这是运行时回调，不是对具体 Trainer 的导入依赖。
当前 loss 评估直接执行前向，不经过 `infer.generate`。
`get_build_steps(run_mode)` 返回需要构建的组件名称，不实际执行构建，也不是执行顺序。

## 验证入口

```bash
pytest -q tests/ut/evaluation
pytest -q tests/torch/evaluation/test_eval_e2e.py
```

系统测试使用真实两层随机初始化 FP32 小模型和合成数据；不能据此推导全尺寸模型、BF16、
量化或替换算子组合的正确性。系统测试需要匹配的真实设备与后端，CPU 测试不能替代 NPU 验收。
