# HyperParallel Eval 与 Predict 方案设计

- 状态：修订草案，待实现与验收；本文中的新增接口均为设计契约，不代表当前已实现。
- 修订日期：2026-11-08。
- 源码基线：`hyper-parallel` master @ `32e9f10e`。
- 适用范围：Torch-only 独立文本评估与离线批量生成；基准评测作为扩展。
- 约束来源：[AGENTS.md](../../AGENTS.md)、[项目规则](../../.agent/rules/project-overview.md)、[分布式规则](../../.agent/rules/distributed.md)、[测试规则](../../.agent/rules/testing.md)。

## 1. 目标与边界

本方案增加统一的 `run_mode: train | eval | predict` 入口，复用当前模型构建、数据处理、loss 与生成实现。

| 目标 | 交付行为 |
|---|---|
| G1 独立 loss 评估 | 加载指定权重，对固定数据范围计算全局 loss；具备纯语言模型 NLL 时计算 perplexity |
| G2 独立批量生成 | 从 txt/jsonl/json 读取 prompt 或 messages，批量生成，按输入顺序输出 JSONL |
| G3 分布式正确性 | 明确数据分片、前向通信、采样、停止和输出归属；支持范围与对应测试同时交付 |
| G4 可复现评测 | 记录模型、数据、模板、参数、采样和判分版本，支持对已有输出重新判分 |

首期面向 `TextTrainer`。VLM、PP、TP/CP/EP 生成按能力矩阵分期接入，不从训练支持情况推导推理支持情况。

本期不包含训练中周期评估、在线服务、continuous batching、paged KV、beam search、训练图编译模式下的评估。`EvaluateCallback` 继续保持现状；未来接入训练中评估时仍需设计迭代器、RNG、FSDP 和梯度状态隔离，不能承诺只增加一个触发点。

MindFormers 仅作为三态分发、评估循环与生成任务分工的概念参考；接口和实现以本仓库的 Torch 组件契约为准。

## 2. 源码基线与必须补齐的能力

| 当前实现 | 已有能力 | 本方案需要补齐的部分 |
|---|---|---|
| [TextTrainer](../../hyper_parallel/trainer/text_trainer.py)、[VLMTrainer](../../hyper_parallel/trainer/vlm_trainer.py) | 通过 `BaseTrainer.__new__` 创建基础对象，自行编排构建步骤 | 模式裁剪必须作用于实际的 Trainer 装配路径，不能只修改 `BaseTrainer.__init__` |
| [TrainerConfig](../../hyper_parallel/trainer/config/trainer.py) | Target/dataclass、YAML 和 dotted override | optimizer 当前必填，配置和模型构建校验也读取 optimizer；须按模式解除依赖 |
| [模型构建](../../hyper_parallel/trainer/base.py) | 原子构建、预训练权重、并行化、dtype | 复用构建入口，保留前向必需上下文，按模式处理训练专属选项 |
| [checkpoint callback](../../hyper_parallel/trainer/callbacks/checkpoint_callback.py) | `on_train_begin` 才执行 `restore_from` | 抽出独立 model-only 恢复入口，避免跳过 callbacks 后漏载训练权重 |
| [数据构建](../../hyper_parallel/data/text/build_dataset.py) | train/valid/test 三分构建 | 独立评估的数据范围不再由 train_iters 推导；当前 eval_iters=0 会得到零 valid/test 样本 |
| [batch runtime](../../hyper_parallel/data/batching/get_batch.py) | 从 iterator 读数据，执行 CP 切分、TP 广播，返回 model/loss inputs | 评估循环必须保留 iterator 和数据 owner 语义；非 owner 也须参与通信 |
| [ModelOutputLoss](../../hyper_parallel/components/losses/model_output.py) | `loss_fn(model_output=..., labels=...)` 返回 Tensor 或 loss 字典 | 单独统计有效 token，声明 loss 归一化与 NLL 语义，不假设返回 `(loss, n_tokens)` |
| [token 计数](../../hyper_parallel/trainer/runtime/loss_aggregation.py) | shift_labels、ignore index、loss_mask 对齐 | 在 Trainer step 内复用，不让 evaluation 反向依赖 Trainer |
| [generate](../../hyper_parallel/infer/generation.py) | prefill/decode、采样、缓存、每行 EOS、左 padding 去除 | 分布式采样/停止协议、结构化返回；保留单一生成循环 |
| [GenerationConfig](../../hyper_parallel/infer/utils.py) | 生成参数和运行时 group/cache/callback 混合存放 | 分离可序列化参数和运行时对象，消除配置重复定义 |
| [prefill mask](../../hyper_parallel/infer/utils.py) | 构建 `[B, 1, S, S]` additive mask | 长 prompt 的二次方显存必须纳入预算，长上下文需单独验收 |

本文优先修正上述契约，再扩展功能。已有 helper 可复用，不等于其训练期默认值适合独立任务。

## 3. 架构与依赖方向

```text
examples/run_text.py
    └─ parse_training_args → validate_run_config → TextTrainer.run
          ├─ train: 现有训练装配与循环
          ├─ eval:  共享模型/权重/资产 + eval 数据 + Trainer eval_step
          │           └─ evaluation.Evaluator → metrics/aggregation
          └─ predict: 共享模型/权重/资产 + GenerationRuntime
                      └─ predict.Predictor → infer.generate

benchmarks adapters → PredictionRecord 文件 → 外部 harness / 纯统计判分
```

依赖规则：

1. `trainer/` 负责配置解析、组件装配和模态适配；`evaluation/`、`predict/` 不 import `trainer/`，两者不互相 import。
2. `evaluation/` 接收 step、迭代计划和归约上下文；模型输入整理、token 计数、loss adapter 调用留在 Trainer step。
3. `predict/` 依赖 `infer/` 的公共接口和注入的 tokenizer/输入适配器，不构建模型、不解释训练配置。
4. `benchmarks/` 负责记录与格式转换、输出提取、结果统计；工具执行与 harness 放在外部脚本或外部项目。
5. 无梯度统计和控制通信使用 `torch.distributed`，遵循仓库的通信与 stream 规则。生成循环、采样和停止状态更新只在 `infer/` 实现。
6. 模型特性通过 model-owned adapter 声明。新增 capability/provider 属于本方案改动，不假定当前 `ModelAdapterSpec` 已有 `supports_generation`。

| 模块位置（新增或调整） | 职责 |
|---|---|
| `trainer/config/{trainer,evaluation,prediction}.py` | 模式、评估、预测配置与 Target 解析 |
| `trainer/runtime/run_mode.py` | 模式校验、构建计划、有效运行选项 |
| `trainer/runtime/evaluation.py` | eval step、既有 get_batch/loss 适配、统计域规范化 |
| `components/checkpoint/model_restore.py` | 不依赖 Trainer 的模型恢复策略，复用既有 checkpointer |
| `data/parallel/evaluation_sampler.py` | 有限评估范围、无重复采样、带有效性标记的尾批补齐 |
| `evaluation/{evaluator,aggregation,types}.py` | 有界循环、设备端统计、统一归约与结果 |
| `evaluation/metrics/` | Loss、Perplexity；后续文本生成指标 |
| `infer/{config,runtime}.py` | 生成参数、运行时组信息、采样同步与停止协议 |
| `infer/generation.py` | 原有生成循环的兼容扩展，不另建循环 |
| `predict/{predictor,io,types}.py` | 请求编排、输入编解码、结果记录与输出提交 |

以上为建议文件边界；实现时可合并小文件，但不得改变依赖方向。

## 4. 运行模式与组件装配

### 4.1 入口与兼容性

`TrainerConfig.run_mode` 默认 `train`，新增 `TextTrainer.run()` 按模式分发。`examples/run_text.py` 调用 `run()`；既有 `train_text.py` 继续调用 `train()`，收到非 train 模式时应明确报错，防止按错误的构建计划执行训练。

旧训练 YAML 的默认值与执行顺序保持兼容。跨区配置校验在 YAML 与 CLI override 全部应用后执行；分布式初始化前完成纯配置检查，模型/拓扑解析后完成能力检查。

`optimizer` 改为可选配置，train 模式仍要求它存在。`TrainerConfig.__post_init__` 中的 optimizer 相关验证、`BaseTrainer._build_model` 中的 `_validate_optimize_dtype` 只对训练装配执行。`training` 区暂保留 backend、seed 等共享环境字段；eval/predict 不要求 train_iters、global_batch_size 或学习率配置。

### 4.2 构建计划

`build_steps_for(run_mode, evaluation_mode)` 产生明确的步骤集合，由 TextTrainer 的实际装配路径消费；BaseTrainer 保留原子构建能力。VLMTrainer 在接入前拒绝 eval/predict，不能悄悄执行训练装配。

通过 `__new__` 装配时，将非活动组件显式初始化为 None，避免通用清理或诊断访问不存在的属性。eval 的 collate 构建不得继续调用训练梯度累积数量计算。

| 步骤 | train | loss eval | generation eval / predict |
|---|---|---|---|
| 配置校验、分布式初始化、mesh、seed | 保留 | 保留 | 保留 |
| 构建模型、预训练权重、并行化 | 保留 | 保留 | 保留 |
| 独立任务 model-only restore | 不调用 | 调用 | 调用 |
| tokenizer / processor / chat template | 保留 | 按数据需求构建 | 必需 |
| dataset / transform / collate / get_batch | 训练管线 | 仅评估数据管线 | generation eval 构建 prompt/reference 管线；predict 使用 IO |
| loss_fn、bind_model、prepare_model_inputs | 保留 | 保留 | 不构建 loss |
| 前向上下文、dtype 策略 | 保留 | 保留前向必需项 | 保留前向必需项 |
| optimizer / attach_optimizer / lr_scheduler | 保留 | 跳过 | 跳过 |
| train_iters / 梯度累积 / backward context | 保留 | 跳过 | 跳过 |
| callbacks / graph compiler | 保留 | 跳过 | 跳过 |

独立任务不启用 activation checkpoint、activation swap 和训练编译包装：由模式构造有效 model-build options，记录其与输入配置的差异，不原地修改共享配置对象。显式请求 compile 的 eval/predict 首期报错；前向混合精度和 `loss_parallel` 上下文不能被视为训练专属功能删除。

模型集成诊断使用适合独立任务的 build-only 路径；要求 optimizer 或训练步骤回调的诊断模式首期拒绝。异常清理按分布式启动器约定执行，不在可能已有 rank 失败的 finally 中无条件 barrier。

### 4.3 资产来源

顶层新增 `model_assets: ModelAssetsConfig | Target | None`，使用既有 tokenizer/processor 配置类型：

- 优先使用显式顶层配置；为空时，兼容读取当前任务所选 DatasetConfig 的 `model_assets`。
- 两处均有显式配置且不同，校验时报冲突，不静默选择。
- predict 可完全不配置 dataset；此时从顶层 assets 构建 tokenizer。
- tokenizer 未指定模型路径时，沿用 `model.pretrained_model_name_or_path`；chat template、特殊 token、模型 revision 必须进入结果 manifest。

predict 不调用 `_build_data_transform`。从已有训练配置启动 predict 时，dataset 下的 assets 只作为兼容配置来源，不触发数据加载。

缺省的空 `ModelAssetsConfig()` 视为未配置；Target 按规范化 `_target_` 路径与参数比较，不使用对象身份判断是否相同。resolver 必要时保留配置来源，避免把自动填入的默认值误判为显式冲突。

配置 resolver 同步支持顶层 assets 及 evaluation 内嵌 DatasetConfig/DataLoaderConfig，复用现有专用解析规则；不能只增加 dataclass 字段而遗漏 Target 子树解析。

## 5. 权重加载契约

### 5.1 加载顺序

独立任务执行顺序：模型原子构建与预训练权重加载 → 并行化后的模型布局就绪 → 可选 `checkpoint.restore_from` 覆盖模型权重 → 模型/loss/生成能力校验 → 首次任务前向。

`restore_from` 沿用现有路径或 `LATEST` 语义。所有 rank 必须解析为同一个已完成 checkpoint；找不到、未完成或缺关键权重时失败，禁止回退到初始权重继续评估。manifest 记录最终解析路径、checkpoint step 和模型配置标识。

创建 restore skeleton 前必须冻结 state_dict 结构：PEFT 和任何注册参数或替换有状态模块的 binding 均先完成。恢复后的 loss/生成适配只能改变不影响权重结构的调用行为；现有只替换 forward 的 chunk binding 可在此后应用。

### 5.2 model-only restore

新增 `restore_model_weights(model, checkpoint_config, mesh_context)`，复用 `build_checkpointer()` 的路径发现和 `load()`，只提供模型 state skeleton，不创建 optimizer，也不读取 scheduler、dataloader cursor、训练 RNG 等状态。

当前 callback 的路径解析、模型 state 映射和恢复安装逻辑应抽取为共享函数；训练 callback 继续负责训练恢复策略。不能把 callback 包在一个带空 optimizer 的假 Trainer 中使用。

外部权重放行时保留 `before_checkpoint_load → dense load/install → after_checkpoint_load` 顺序和 dense/external 参数精确分工，剥离任何要求 optimizer 已存在的调用；不能只恢复 state_dict 中的 dense 部分。

独立任务固定为 model-only 语义；`restore_optimizer/restore_train_state/restore_dataloader_state` 属于训练恢复选项，在独立任务 manifest 中记录为未应用。checkpoint 的保存选项不触发写入。

| 权重类型 | 首期策略 |
|---|---|
| 标准完整模型 checkpoint | 支持；严格校验请求的模型 key、shape、dtype 与布局 |
| PEFT adapter-only checkpoint | 基座先就绪，按已有 adapter 映射恢复；仅允许声明的基座缺失，不全面放宽 strict |
| 模型外部参数 / host sidecar | 仅在 provider 支持无 optimizer 的完整恢复时放行；否则明确拒绝并列出缺失能力 |
| 跨拓扑恢复 | 沿用 DCP 的可重分片能力，但支持承诺以该模型、布局组合的 ST 为准 |

## 6. 独立评估数据语义

### 6.1 配置与范围

`evaluation.dataset`、`evaluation.dataloader` 可单独指定；缺省时回退到现有顶层配置。`evaluation.split` 为 `valid | test | dataset`：三分返回选 valid/test，单 Dataset 返回选 dataset，不隐式把训练集当验证集。

`evaluation.batch_size` 表示每个逻辑 DP lane 每次前向的样本数，独立于训练的 micro/global batch 和梯度累积。`max_samples` 是切分前全局有序样本范围上限；`max_batches` 是所有 lane 共同执行的前向轮数上限，均为正整数或 null。

- null 表示不施加该项上限；0 是非法值。
- 同时设置时先截取 max_samples，再应用 max_batches。
- 改变 DP/batch_size 会改变 max_batches 对应样本量；需要跨拓扑对比时固定 max_samples 或完整数据范围。
- 有限数据集允许全量评估；无确定结束边界的 iterable 必须提供上限，首期不支持通用流式评估。
- 独立任务不消费 `training.eval_iters`；该字段保留原有训练语义，不自动转换单位。

### 6.2 Dataset builder 适配

增加显式评估构建请求，将 split、seed、有限数据范围传入现有 builder。不得调用 `_get_train_valid_test_num_samples(training_config)` 来规划独立评估。

P0 使用有限 mapping/Indexed 数据和 fixed batching，拒绝 cyclic sampler、通用 iterable、动态 token batching/packing。Online mapping 按固定顺序遍历选定 split，首期要求一个 source record 转换为一个 ModelSample；像长 plaintext 展开多窗口这样的 transform 明确拒绝。以后支持一对多转换时，先建立有限 `(source_id, chunk_idx)` 索引，再做分片与步数规划。

batch_size、max_samples 和 num_samples 均按最终逻辑 ModelSample 计数，原始 source record 数可另报。Indexed 数据复用底层索引、split 和样本构建算法，增加“一遍有限样本”的模式：固定序列长度、尾窗口规则和 seed，不因请求样本数而重复多个 epoch；记录实际逻辑样本和有效 token 数。当前 `train_valid_test_num_samples` 可用于显式有界构建，但不能把 0 当作全量；全量 indexed 模式需在 P0 实现并测试。

仅构建选中的 split；若底层共享索引初始化不可避免，允许复用该初始化，但不创建 train dataloader 或推导训练迭代数。评估 shuffle 默认关闭，重复 evaluate 时使用全新 iterator/sampler 状态。

### 6.3 尾批、owner 与执行计划

现有 sampler 的默认 `drop_last=True` 会漏样本；直接设 False 可能让各 rank 前向次数不同。新增评估 batch plan，先确定全局样本 ID 与轮数，再分配各 lane 的槽位：

1. 真实样本在评估范围内恰好消费一次，保留稳定 sample ID。
2. 共享前向 collective 的 rank 执行同样的轮数；尾轮不足槽位以 dummy 补齐。
3. dummy 使用合法非空输入，携带 `valid_sample_mask=False`；step 把其 target/loss mask 排除，分子、分母和样本数贡献为零。
4. 不只依赖现有 `padding_flag`；有效性标记必须随 batch transport 正确传递，并与 causal shift、CP/TP 切分后的标签对齐。
5. get_batch 仍接收 iterator，返回 `(model_inputs, loss_inputs)`。非数据 owner rank 可以没有 dataloader，但必须按公共计划参与广播和前向。
6. owner 在进入当轮模型通信前发布数据可用状态；不允许某 rank 因局部 StopIteration 提前结束整个循环。

空数据集在前向前明确报错。真实样本全部被 mask 时，保留样本计数并标记无有效 token；若整个任务的指标分母为零，则失败并说明原因。

## 7. Evaluator 与指标契约

### 7.1 Trainer step

Evaluator 不直接调用模型的任意 forward。Trainer 注入 `eval_step(iterator, batch_plan_entry)`，负责：

1. 调用既有 get_batch，得到设备上的 model_inputs/loss_inputs。
2. 对齐有效样本标记，同步更新模型实际消费的 labels/shift_labels 与统计侧标签/mask，再调用 `count_loss_token(loss_inputs)`；CP 场景保留 shift_labels 语义。dummy 必须从 CE 本身和统计分母同时排除。
3. 应用 loss_fn 的 `prepare_model_inputs`（若存在），保留模型所需前向上下文。
4. 在 `torch.no_grad()` 下执行 `model(..., use_cache=False)`。
5. 按现有接口调用 `loss_fn(model_output=outputs, labels=...)`，规范化为命名统计项；及时释放 outputs/logits。

`bind_model` 属于构建步骤，不能每 batch 重复执行。loss-only 模型输出可没有 logits；仅输出 logits 的模型需要配置能够消费它的 loss_fn，默认 ModelOutputLoss 不自动补算 CE。

### 7.2 统计协议

每个 step 返回 `EvalStepResult`，包含以下设计字段：

| 字段 | 语义 |
|---|---|
| `statistics: dict[str, LossStatistics]` | 命名的可加统计项 |
| `LossStatistics.numerator` | 设备端标量，当前贡献域的 loss 总和 |
| `LossStatistics.denominator` | 设备端非负标量，和分子口径一致的有效 token 数或声明的权重 |
| `LossStatistics.kind` | `lm_nll` 或 `objective`；未声明 NLL 的项不能产生 PPL |
| `sample_count` | 排除 dummy 后的真实样本数，按数据 owner 计数 |

Trainer adapter 必须声明 loss 的归一化方式：local token mean、local sum 或已跨域归约。local mean 转为 `mean × valid_count`；已归约 loss 不能再次当作 local loss 加权。默认适配仅覆盖已验证的输出契约；自定义 loss 缺少归一化声明时拒绝启用相应指标。

例如现有 chunk loss 路径提供 `loss_sum/valid_token_count`，可由对应 adapter 直接产生 NLL 统计；其返回的训练 objective 可能还含 aux loss，不能替代 NLL。chunk loss 的 TP/PP/loss_parallel 限制继续由既有能力校验执行。

当 local count 为零时直接贡献零分子，不执行 `NaN × 0`。causal shift、IGNORE_INDEX 和 loss_mask 必须与真实 CE 使用相同的有效 target 集合。

### 7.3 聚合与 PPL

对每个指标先在设备上累计，再在评估结束时归约：

```text
loss = SUM(all unique contributions.numerator) / SUM(all unique contributions.denominator)
ppl  = exp(global_lm_nll)
```

`ReductionContext` 由 Trainer 根据 loss layout 构建，包含贡献域、归约组和副本结果传播规则。DP/CP 的不相交 token 贡献相加；TP/SP 已复制的统计只能计入一次。若某 loss 先在 TP/SP 形成了完整 NLL，则先规范化这一统计域，再跨 DP/CP 聚合，不能照搬训练 backward 的缩放系数。

P0 限 TP=CP=EP=PP=1，统计域为 DP；复合拓扑随 P2 的数值验收开放。无 group 不自动解释为 WORLD，单卡上下文直接跳过 collective。

累计分子至少 FP32，计数使用 int64；小规模参考计算使用 FP64。设备支持且长任务精度需要时可选 FP64 累计，分别归约不同 dtype。最终结果落盘前才转 Python 标量。禁止每 batch `.item()` 或为日志额外执行全局 loss 归约。

`LossMetric` 输出指定统计项的全局均值；`PerplexityMetric` 消费同一份 `lm_nll` 结果，不重复累积或通信。MoE router aux、MTP、正则项等保留为 objective，不能计入 PPL。无法提供纯 NLL 时仍可输出 objective loss，但配置 PPL 必须报错。

### 7.4 Metric 生命周期与结果

协议分为 `reset → update → synchronize → compute`。Evaluator 统一调用一次 synchronize；compute 是纯读取，重复调用不再 all-reduce，避免二次累计。每次 run 都 reset；多指标输出 key 冲突在运行前报错。

结果至少包含 metrics、num_samples、num_tokens、num_batches、实际权重来源和数据范围。非有限结果不写非标准 JSON 的 NaN/Infinity：记录 null 及明确 status；PPL 指数溢出时保留 NLL 并标记 overflow。

Evaluator 在 run 前保存各模块 training 标志，切换 eval，并在 finally 恢复原标志。使用 no_grad 作为基线；inference_mode 仅在 wrapper/cache 兼容测试后作为优化。该恢复不等同于完成训练中评估的状态隔离。

## 8. Predictor 与生成接口

### 8.1 参数与运行时对象

将现有 infer 配置拆分为同一来源的两层：

- `GenerationOptions`：可序列化采样参数，包括 max_new_tokens、temperature、top_k、top_p、do_sample、eos/pad token、repetition_penalty、use_cache。
- 兼容的 `GenerationConfig`：复用上述字段定义，保留既有程序化调用入口；process group、prefix cache、callable 等字段不进入任务 YAML。

建议先在 infer 内抽出公共参数 dataclass，由既有 GenerationConfig 继承；Trainer 的 `evaluation.generation`、`prediction.generation` 直接引用它。参数验证也由 infer 维护，不复制到两个配置类。特殊 token 使用显式 `auto` 值表示待解析，任务配置缺省为 auto；既有 GenerationConfig 的程序化默认值仍保持 EOS=2、pad=0，避免兼容性变化。

新增 `GenerationRuntime` 由 Trainer 注入：模型适配、logits layout/group、sampling owner/group、forward sync group、设备、RNG 和输出能力。运行时字段不由 YAML 的 Any 透传。现有 generate 拒绝额外 kwargs，新增 runtime 和结构化返回必须作为显式兼容扩展实现并测试。

既有 GenerationConfig 中的运行时字段在调用边界统一转换到 runtime；同时提供新旧两处配置且冲突时拒绝，不维持两套执行分支。

特殊 token 解析顺序为任务显式配置 → 模型 generation 元数据 → tokenizer；auto 在创建运行时 GenerationConfig 前解析为具体值，无法确定 pad/EOS 语义时明确报错。任务省略 eos_token_id 表示 auto，显式 null 表示禁用 EOS、仅按长度或自定义条件停止。兼容多 EOS ID 的任务需同时扩展停止与返回逻辑；未实现前不得仅接受 list 配置而仍做单值比较。

### 8.2 输入与批组织

- txt：每行一个原始 prompt；jsonl/json：每条是含 `id` 和 `prompt` 或 `messages` 的记录，两者互斥。
- 缺少 ID 时使用稳定的输入序号生成，重复 ID 在执行前报错；空输入和无有效 token 的记录报清晰错误。
- prompt 直接编码；messages 使用显式选定的 chat template，记录模板版本、generation prompt 与 thinking 等模板参数。不根据模型名在核心代码中分支。
- 左 padding 的 collate 位于 predict IO；attention mask 与位置必须正确，pad token 等于 EOS 时也不能通过扫描 padding 推断停止。
- `prediction.batch_size` 是每个逻辑 DP lane 的最大请求数，不要求总输入数或 batch_size 被 DP 整除。
- 可按 prompt token 长度分桶，受 `max_batch_tokens` 约束；最终按输入 ordinal 恢复顺序。该 token budget 表示 padded prompt tokens，不代表完整 KV 显存预算。
- 默认不静默截断。`prompt_tokens + max_new_tokens` 超过已验证模型上下文时失败；截断策略若以后增加，须显式配置并记录。

### 8.3 生成返回与输出记录

既有 `generate()` 默认仍返回 Tensor，内容是去掉左 padding 后的 prompt + completion。增加可选结构化返回 `GenerationResult`，包含 sequences、prompt_lengths、generated_lengths、finish_reasons；默认行为不变。

Predictor 根据每行真实 prompt_lengths 截取生成部分，不按整个 padded batch 的宽度切分。generated_lengths 与停止状态由生成循环提供，不能靠去除 pad token 猜测。

输出示例：

```json
{"id":"q-1","ordinal":0,"sample_idx":0,"generated_text":"答案是 123。","prompt_tokens":28,"generated_tokens":9,"finish_reason":"eos"}
```

可选保存原始输入、prompt/generated token IDs；默认 generated_text 仅含生成内容。finish_reason 至少区分 eos、length，以及生成循环实际触发的 stop；计数约定包含实际采样的 EOS，decode 时是否展示特殊 token 由输出配置控制，dummy 行不输出。事后文本截断单独记录后处理原因，不改写真实 finish_reason。

### 8.4 输出提交与重跑

每个逻辑数据分片只有一个 writer。分片先写本次任务独立临时目录，writer 关闭文件后提交 shard manifest；协调 rank 验证全部预期分片成功、所有预期 `(input_id, sample_idx)` 恰好出现一次，再按 ordinal/sample_idx 合并并原子提交最终文件。

P1 要求共享文件系统；跨节点非共享盘在支持独立传输机制前拒绝。已有最终输出默认报错，只有 `overwrite=true` 才替换；失败时保留带 run ID 的临时分片，不将其标为完成。

P1 不承诺断点续跑。B0 增加按 `(input_id, sample_idx)` 恢复时，必须匹配模型/权重、数据、模板、生成参数、base_seed、n 等不可变实验指纹，且已有样本不会被重新计入。batch plan、DP 分片和拓扑作为各次尝试的执行记录，在已验证能力与逐请求随机流条件下允许变化；不能对整个 manifest 做字节级相等检查。逐请求随机流或完整恢复原批次 RNG 状态是采样续跑的前置条件。

## 9. 分布式执行契约

### 9.1 角色与通信域

DP lane 是输入分片单位，不等于独立推理副本。FSDP 参数分片和 EP 路由可能使不同 lane 共享前向 collective。

| 角色/域 | 契约 |
|---|---|
| 数据 owner | 为一个逻辑输入分片读取记录；向同一请求的模型并行参与者提供一致输入 |
| sampling group | 共同处理同一组请求的 TP/CP rank；由一个 owner 采样 token，再广播 token、结束状态和原因 |
| forward sync group | 所有因 FSDP/EP/其他前向 collective 相互依赖的 rank；共同决定是否继续当前生成轮 |
| metric contribution group | 按真实样本/token 的唯一贡献归约，与生成同步域分开定义 |
| output writer | 每个数据分片一个 writer；TP/CP 副本不重复输出 |

各 group 从已解析 mesh 与模型能力派生，禁止以 global_rank%N 或缺省 WORLD 猜测。已验证拓扑采用显式组映射；复合布局可保守采用整个 stage 同步组，覆盖全部前向通信依赖并接受额外空跑开销。本期不实现通用通信图分析器，未建立映射的组合拒绝启用；EP 不简单归为“和 TP 一样处理同一请求”。

### 9.2 采样与停止

同一请求的 TP/CP rank 先按声明布局获得正确的 next-token logits，sampling owner 生成 next_tokens，再广播给其他参与 rank。固定 seed 只用于随机流复现，不替代广播。

共享 forward sync group 的生成执行：

1. 每个调度窗口先确定同步域内共同 batch 轮数；本地分桶产生更少批次、输入耗尽或某轮无真实请求时提供合法 dummy batch。所有参与 rank 按相同轮次进入 prefill；独立 DP 副本不强制共同轮数。
2. 每轮更新各请求的真实 finished 状态；采样组内状态一致。
3. 在一致位置对“是否仍有未完成请求”执行组内控制归约。
4. 仅当整个同步域结束，或共同达到生成轮数上限，才能退出。
5. 已完成 lane 继续以占位 token 执行必要前向，并维护约定的 cache、position 和 batch shape，不重新 prefill 无关 prompt，不累计输出；已完成行仍可能消耗模型计算和 KV 空间。
6. 局部 EOS、stop callback、输入耗尽、取消均不得直接 break 跳过其他 rank 仍需参与的 collective。

首批 FSDP 生成验收使用缓存开启、共同 max_new_tokens 和可处理 padding/dummy 的模型适配器；缓存降级为全量重算等路径须另测。固定形状 collective 或 EP 实现可能还要求跨 lane 一致的张量形状，runtime 必须校验并补齐。

### 9.3 能力矩阵

下表为计划交付范围，不是当前源码的支持声明。每项必须由真实模型、设备后端和拓扑对应的 ST 放行。

| 场景 | P0 loss eval | P1 predict | 后续条件 |
|---|---|---|---|
| 单卡文本模型 | 支持 | 支持 | loss/生成契约分别验证 |
| 完整权重的独立 DP 副本 | 支持 | 支持 | 无跨 lane 前向 collective；尾批完整且结果唯一 |
| Dense FSDP/HSDP，TP=CP=EP=PP=1 | 支持 | 拒绝 | P2 完成全域停止、dummy 前向和缓存验证后开放 |
| TP / SP | 拒绝 | 拒绝 | P2 loss 统计域、logits layout、owner 采样和输出去重一起验收 |
| CP | 拒绝 | 拒绝 | P2 验证真实模型 CP mask/cache/shard metadata；不能只凭 toy 示例放行 |
| EP 或复合并行 | 拒绝 | 拒绝 | P2 按具体模型/布局增加，验证路由通信和同步域 |
| PP | 拒绝 | 拒绝 | P3 先做 loss eval 的 forward-only schedule；predict 另行设计 |
| VLM | 拒绝 | 拒绝 | P3 增加 processor、视觉输入、模型前向和缓存契约 |

现有 [CP 示例](../../examples/generate/cp_generate_check.py) 和 [HF 对齐示例](../../examples/generate/hf_alignment_check.py) 用作测试模式参考，不作为全部模型已支持的证据。

PP loss eval 在最后 stage 计算统计，再向需要输出结果的 rank 传播标量。无需广播全量 logits；PP predict 还需要逐 token 反馈和缓存调度，不与 PP eval 合并承诺。

## 10. 配置 Schema 与校验

以下为新增配置区示例，须与实际模型和数据 Target 组合使用，不是可直接运行的完整 YAML。

```yaml
run_mode: eval
model_assets:
  tokenizer:
    _target_: transformers.AutoTokenizer.from_pretrained
    pretrained_model_name_or_path: /models/example

checkpoint:
  restore_from: /checkpoints/step_1000

evaluation:
  mode: loss                 # loss | generation
  split: valid               # valid | test | dataset
  batch_size: 2              # 每个逻辑 DP lane，每次前向
  max_samples: null          # 全局范围；null 为选定有限 split 全量
  max_batches: null          # 前向轮数上限，不是 optimizer step
  output_file: /outputs/eval.json
  metrics:
    - _target_: hyper_parallel.evaluation.metrics.LossMetric
      statistic: objective
    - _target_: hyper_parallel.evaluation.metrics.PerplexityMetric
      statistic: lm_nll
  generation: null
# dataset/dataloader 可放 evaluation 下，也可兼容复用顶层配置。
```

```yaml
run_mode: predict
model_assets:
  tokenizer:
    _target_: transformers.AutoTokenizer.from_pretrained
    pretrained_model_name_or_path: /models/example

prediction:
  input: /inputs/prompts.jsonl
  output_file: /outputs/predictions.jsonl
  batch_size: 4
  max_batch_tokens: 8192
  bucket_by_length: true
  overwrite: false
  return_token_ids: false
  generation:
    max_new_tokens: 256
    do_sample: false
    use_cache: true
    # eos/pad 缺省时按模型元数据与 tokenizer 解析。
```

`evaluation.mode=generation` 时要求 generation 和文本指标配置，不能同时启用 loss-only 指标。生成式评估由 Trainer 组合 infer 生成 callable 与独立 reference 数据，不把包含答案的 teacher-forcing 序列直接作为 prompt。

校验分为三个时点：

| 时点 | 检查 |
|---|---|
| 纯配置、分布式初始化前 | 模式、必填项、Target、字段单位、指标冲突、依赖、compile、输出覆盖策略 |
| 模型/mesh 解析后、任务开始前 | capability、checkpoint、loss 归一化、NLL、assets、特殊 token、所有通信组与 writer、共享存储 |
| 输入批构建后、前向前 | 非空 prompt、上下文长度、token budget、batch layout、参考答案隔离、dummy mask |

首期对未覆盖组合给出具体错误，例如“FSDP 生成尚未启用同步停止协议”，不能降级成未声明的运行方式。外部 metric 依赖缺失时，只有选用该 metric 才报错；不得静默跳过请求的指标。

## 11. 性能与显存设计

### 11.1 首期默认策略

| 优化 | 行为 | 验证指标 |
|---|---|---|
| 去掉训练态组件 | 不创建 optimizer/main params、scheduler、backward 包装 | 启动时间、首个前向前显存 |
| 设备端指标累计 | batch 内不做 host loss 同步，结束后统一归约 | eval tokens/s、统计 collective 次数 |
| loss-only 前向 | use_cache=False，及时释放输出；支持时使用模型原有 chunk loss | 峰值 allocated/reserved memory |
| 生成长度分桶 | 有界窗口内分桶，按 token budget 组 batch，输出恢复原序 | padding 比例、prefill/decode 吞吐 |
| 有界 IO | 分块读输入、流式写分片，不把全量 token/logits 留内存 | host RSS、输出完整性 |

### 11.2 容量模型与长上下文

标准 MHA/GQA 的未分片 KV 粗估：

```text
KV bytes ≈ batch × total_sequence_length × layers × 2 × kv_heads × head_dim × dtype_bytes
```

模型使用 MLA、滑窗、不同层缓存布局或复制 KV heads 时，须由适配器给出对应估算。TP/CP 后的每卡 KV 不能一律除以并行度；DP 仅分担请求，不降低单条请求的序列长度。

总预算还需包含参数常驻/临时 unshard、attention mask、prefill logits、通信 buffer 和 kernel workspace。当前 dense prefill mask 本身为 O(B×S²)：S=65536 时，仅单个 FP32 的 S×S 张量就占 16 GiB，实际构建还有临时分配。长生成与长 prompt 要分别测试。

当前 max_new_tokens 没有 64k 配置硬上限。64k 能力取决于模型上下文、attention/cache 路径和预算；长 prompt 支持必须先完成紧凑 mask 或融合 causal attention 适配及数值验证，不通过“放开校验”宣称支持。

### 11.3 后续优化及测量口径

FSDP decode 的逐层参数 all-gather 可能成为主要开销。P2 在显存允许时比较保留 unsharded 参数、reshard/prefetch 策略和不同推理布局，记录通信与显存权衡；不默认调用 empty_cache，也不以梯度 all-reduce 开关解决无梯度前向 OOM。

静态 batch 中已完成行仍参与 forward；按长度分桶先降低浪费，动态行压缩留到后续。多采样 n 不等于物理 batch size，应拆成有界 micro-batch。

性能报告固定模型/权重、设备、dtype、拓扑、输入/输出长度分布、seed 和 warmup。分别记录 prefill tokens/s、decode tokens/s、有效输出 tokens/s、峰值设备内存、启动耗时与输出 IO；不得用少评样本或更短输出换取表面吞吐。

## 12. 生成式评估与基准扩展

### 12.1 生成式评估

生成式 Evaluator 接收 prompt/reference batch、注入的生成 callable 和文本 metric。生成调用仍走 infer；参考答案不得进入 prompt。生成长度与截断行为进入评测元数据。

ROUGE 明确按样本聚合方式；corpus BLEU 汇总可加的匹配数、候选/参考长度后统一 compute，不能直接平均各 rank 的 BLEU。空输出保留为有效样本；mask/dummy 不进入分母。依赖按所选 metric 加载。

### 12.2 基准分层

基准适配输出标准请求，Predictor 输出 PredictionRecord，判分消费已落盘产物。AIME 文本判分和汇总统计可为纯函数；LCB/SWE 的容器执行有外部状态，须与结果统计分开。

| 基准 | 适配责任 | 判分契约 |
|---|---|---|
| AIME 25/26 | 题目 → messages；completion → 按版本固定的答案解析 | 整数 exact match、avg@k、maj@k、pass@k |
| LiveCodeBench | 固定 release/时间窗、官方 prompt/代码提取格式、输出转换 | 官方 harness 运行结果 → pass@k，保留失败与超时 |
| SWE-bench Verified | issue/repo 标识、外部生成或 agent backend adapter、patch 输出 | 官方 harness，报告 instance resolved rate |

可选 agent 在外部读取文件、执行命令、生成 patch；SWE-bench 核心接口接受 patch，不要求必须采用多轮 agent。工具协议、容器调度和上下文管理不进入 infer/Predictor。

### 12.3 多采样与复现

适配层展开 `(input_id, sample_idx)`，保留生成循环每行一个候选的语义。不同样本允许得到相同文本，不能去重后重采样，否则改变统计分布。

复现分为两级：

- P1 任务级复现：固定 seed、输入顺序、batch plan、拓扑和软件环境；manifest 记录这些条件，不承诺改变 batch/DP 后逐样本输出相同。
- B0 请求级随机流：新增 sampler 显式 generator 支持，按 base_seed、稳定 input ID、sample_idx 派生种子；sampling owner 为每个请求/候选维护独立 generator，不能让整个 batch 共用一条随机流。支持改变分桶、分片与续跑；即使随机流相同，batch shape、kernel、归约顺序或设备差异仍可能改变 logits 和输出，不承诺 token 序列在任意执行布局下位级一致。

greedy 下 n>1 默认拒绝，避免把重复生成误当作多次随机采样。采样参数、n、模板、模型 revision、数据 hash、提取器与 harness 版本均进入 manifest。

### 12.4 判分口径

AIME 按指定版本解析最终答案，统一整数格式后检查 `0 <= answer <= 999`；不对任意输出取模。超范围、无法解析、缺失输出视为失败，保留原始 completion 与解析状态。该范围来自 [MAA 官方说明](https://maa.org/maa-invitational-competitions/)。

对每题 n 个采样、其中 c 个正确：

```text
avg@n  = 各题 c/n 的平均值
pass@k = 各题 [1 - C(n-c, k) / C(n, k)] 的平均值，要求 n >= k
maj@k  = 对固定 k 个样本投票，比较最高票答案与标准答案
```

maj@k 的平票规则固定并版本化；本方案默认平票判失败。解析失败保留在样本预算中，不通过删除失败记录提高分数。avg@n 不等于 pass@n，greedy accuracy 与 sampled pass@1 分开标注。

LiveCodeBench 固定题集 release、日期窗、prompt、代码提取器和 [官方 harness](https://github.com/LiveCodeBench/LiveCodeBench) commit；超时、编译错误和错误代码计失败。环境故障作为运行错误报告，不静默减少分母。

SWE-bench 提交 `instance_id/model_name_or_path/model_patch`，遵循 [官方评测格式](https://www.swebench.com/SWE-bench/guides/evaluation/)。resolved 采用官方实例判定，不以测试用例通过百分比替代。报告区分已解决、未解决与基础设施错误，记录模型和 agent 配置以避免把 agent 差异归因于模型。

### 12.5 停止、会话与执行隔离

首版自定义 stop string 可以生成后截断，但必须将“实际生成 token”与“展示/计分截断文本”分开记录；代码 fence 或第一个 boxed 不默认等于正确停止位置。行级 stop 扩展在 infer 内合并 finished mask，并遵守分布式同步停止协议。

多轮会话首版全量重新编码与 prefill。prefix cache 复用只有在重新编码后的 token 前缀完全一致、缓存长度/位置/mask 兼容时才成立；chat template、工具消息重写与截断可能破坏前缀关系。

LCB/SWE 执行复用外部工具链，在隔离容器中限制 CPU、内存与时限；镜像和依赖准备完成后，测试执行阶段禁网。核心库不内嵌生成代码执行器。

## 13. 分期与验收

功能和对应正确性测试必须同阶段交付。P1 依赖 P0 的模式装配、assets 与 model-only restore；P2 按能力逐项放行，不等待一个包含所有拓扑的大 PR。

| 阶段 | 内容 | 必须通过的验收 |
|---|---|---|
| P0 | 模式/配置/资产解耦、权重恢复、有限 eval 数据、Evaluator、Loss/PPL；单卡/DP/Dense FSDP loss eval | checkpoint 恢复前后一致；完整样本覆盖；不同 token 数加权一致；全量 indexed 语义；旧 train 配置回归 |
| P1 | 文本 Predictor、prompt/messages、基础 manifest、GenerationOptions、结构化返回、单卡/独立 DP、输出提交 | 不需要 dataset/optimizer；HF greedy 对齐；非整除输入完整有序；prompt/completion 边界；默认拒绝未覆盖拓扑 |
| P2a | FSDP/HSDP 生成同步；TP/CP/EP 及复合布局逐项接入 | 不同 EOS 长度、空 lane、dummy、组内采样一致；真实模型数值/输出一致；超时保护验证不挂起 |
| P2b | 生成式评估、文本指标；已支持拓扑的统计扩展 | reference 不泄漏；分布式统计等于单进程参考；可选依赖与空输出语义 |
| P3 | PP loss eval 或 VLM，分别设计和交付 | PP forward-only 与标量统计；VLM processor/视觉输入/cache 的模型验收；不包含 PP predict |
| B0 | n>1、请求级 RNG、恢复指纹校验与续跑 manifest 扩展 | 固定环境下改变 batch 分桶/DP 请求分片，随机流稳定；重复/缺失候选检测 |
| B1 | AIME，依赖 P1+B0 | n=16 有界 micro-batch 出分；超范围答案拒绝；avg/maj/pass 与手算一致 |
| B2 | LCB，依赖 P1+B0 | 固定 release/时间窗，官方 harness 小集对齐，pass@1/pass@10 |
| B3 | SWE 格式/外部 backend adapter | 固定子集官方 resolved rate；模型、agent、harness 元数据齐全 |

B0-B3 不阻塞 P0/P1；使用未开放拓扑或长上下文时，必须先通过对应 P2/容量验收。每个合入 PR 遵循仓库单 commit 约定，阶段可按依赖拆分多个 PR。

## 14. 测试计划

UT 位于 `tests/ut/evaluation/`、`tests/ut/predict/`，配置/Trainer/数据/checkpoint 测试放在各自既有目录；ST 位于 `tests/torch/evaluation/`、`tests/torch/predict/`。遵循 [测试规则](../../.agent/rules/testing.md)，分布式 launcher 不导入 Torch 或 HyperParallel，worker 执行真实计算。

| 测试面 | 必测案例 |
|---|---|
| 构建与配置 | 默认 train；eval/predict 无 optimizer；predict 无 dataset；空缺省 assets 不冲突、等值 Target 不误判；真实冲突；dotted override；未支持组合早失败 |
| 权重 | 初始权重与训练 checkpoint 明显不同的模型；restore_from/LATEST；缺 key；adapter-only；外部状态拒绝；optimizer 未创建 |
| 数据覆盖 | N=0、N=1、N<DP、N 不整除 DP×batch；max_samples/max_batches；新 iterator 重复评估；全量 indexed 不重复 epoch；P0 一对多 transform/动态 packing 明确拒绝 |
| token 与 loss | 不同长度、ignore index、loss_mask、shift_labels、全 mask、dummy；local mean/sum；辅助损失不进入 PPL |
| 指标生命周期 | 多 batch 手算；一次 synchronize；compute 重复调用；reset；零分母；非有限值；重复 metric key |
| 生成接口 | 左 padding、pad=EOS、不同 prompt 长度、EOS 在首 token、max_new_tokens=0、长度终止、结构化返回与旧 Tensor 返回一致 |
| 分布式生成 | 同请求跨 TP/CP token 一致；FSDP/EP 不同结束步；尾批空 lane；错误 group/layout 被拒绝；collective 超时保护 |
| 输出 | ID 唯一且完整、顺序恢复、单 writer、原子提交、已存在输出、失败分片、B0 续跑 manifest 校验 |
| 生成指标/基准 | prompt/reference 隔离；BLEU 充分统计量；n 与 k；重复样本保留；无效答案；harness 环境错误与模型失败分开 |
| 性能 | warmup 后固定工作量，eval/predict 吞吐、峰值内存、通信次数；长 prompt 与长 decode 分别测量 |

DP 数值对比必须使用相同全局样本 ID 和有效 token 集合，不能只比较两个恰好相同的最终 loss。TP/CP 既测 toy 契约也测真实模型；greedy 对齐先确认相同权重、tokenizer、模板、EOS 和 mask，再比较 token 序列。

## 15. 实现前检查点

以下事项由实现 PR 提供证据，不作为默认可用假设：

| 检查点 | 关闭条件 |
|---|---|
| Indexed 全量有限范围 | 明确逻辑 sample 定义、尾窗口策略、有效 token 覆盖；固定 seed 可重复 |
| 自定义 loss 的统计域与 NLL | 每种支持的 loss adapter 声明归一化与辅助项，单卡/分布式对照通过 |
| FSDP 无梯度生命周期 | 连续 eval/generate 无参数/通信 buffer 泄漏，dummy 前向和 reshard 行为验证 |
| 模型外部权重 | 无 optimizer restore 能完整恢复外部参数；不支持时在前向前拒绝 |
| CP/EP 缓存与通信 | 模型 provider 声明有效组合，真实模型 prefill/decode 与停止测试通过 |
| 长上下文 | 参数、KV、mask、logits 和临时内存预算一致，代表长度实测通过 |
| 数据与输出持久化 | 全局样本范围、单 writer、输出完整提交、manifest 可追溯 |

本文交付设计与验收契约。实现完成前，不将计划中的并行组合、64k 长度或基准分数标记为已支持。
