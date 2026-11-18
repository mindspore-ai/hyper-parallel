# 通用 Trainer 监控指标诊断说明

适用范围：通用 Trainer，不涉及 RL。配套 [监控设计方案](design/trainer_monitoring_plan.md) 和 [使用与实现边界](trainer_monitor_usage.md)。
本文只列当前实现的诊断入口和输出字段，版本与设备验收限制见使用说明。
当前实现范围与本地/设备验收状态以 [使用与实现边界](trainer_monitor_usage.md) 为准。
开始诊断前核对运行 revision、实际输出字段和能力报告；未启用或不支持的采集项缺失，不自动判为训练故障。

## 使用前提

- 对齐 run、optimizer step、micro-step、rank、PP stage、module、采样窗口和实际输入；跨 rank loss 不同也可能来自不同数据。
- 区分 input/有效监督 token、local/global、裁剪前/后、规约前/后、逻辑量/物理 padding。不同分母和统计阶段不能直接比较。
- 全 rank 采集不等于每个 rank 都有所有指标：无 loss 的 PP stage、未持有的 expert/module 应标为不适用。
- 指标是定位线索，不能单凭一次尖峰判定 bug；先排除输入变化、预热、采样缺失和异步迟到，再关联相邻证据。
- 检查 `unavailable` 原因、`flops_coverage.complete` 和实际丢弃计数；未采样或尚未到达不等于数值零。monitor 不提供独立的通用 NaN/Inf 强制检查。

## 按现象分块排查

下面是给用户的排查提示，不是已经实现的自动告警。按出现的现象选择一块，给出具体检查动作，
不要只回答“可能是数据问题”或“建议看 profiler”。字段以该次运行的 JSONL 和代码为准；缺失时先检查采集覆盖。

### 先确认记录是否可用

查看 `<monitor.output_dir>/<run_id>/rank_<rank>.jsonl`，按记录中的 `step` 对齐，而非文件行顺序。
控制台的 `training metrics:` 是 rank 0 的逐 step 汇总，可能因异步传输晚到。
console/W&B 没显示某个 rank 的明细时，先查它的 JSONL。需要复现时，在短窗口内把相关项的
`sampler_steps` 设为 1；不必同时打开全部监控，也不要把全量采集当作无开销操作。

| 看到的现象 | 提示用户可以怎么做 |
| --- | --- |
| 找不到 local loss 或 EP 记录 | 先查对应 `enabled`、采样步是否命中，再找 `event=unavailable` 的 `reason`；核对该 rank 是否拥有 loss/该层，以及是否经过 compiled 路径 |
| 记录比基础训练日志晚出现 | 按原始 `step` 合并记录；异步摘要晚到不代表下一步的数据。运行中暂未出现不能直接判为丢失 |
| 摘要或远端输出丢失 | 查 `monitor/dropped_summaries`、`monitor_closed` 的丢弃计数和队列告警；先降低相关采样频率或减少 Table 输出，再比较同负载结果 |

### 启动慢或启动卡住

启用 `monitor.timing`，检查 `event=timing` 的 `phase`、`duration_seconds` 和 `outcome`。

| 耗时集中在哪里 | 提示用户可以怎么做 |
| --- | --- |
| `distributed_init` / `process_groups` | 对照各 rank 的进程启动日志，确认是否全部到齐、world size/rank 与启动配置是否一致；仍卡住时获取各 rank 调用栈，区分 rendezvous 等待和创建子组等待 |
| `pretrained_weights_load` / `model_build` | 先核对权重路径、分片数量和缓存状态；权重加载慢就检查存储读取，加载正常而构建慢则沿模型构建/并行包装阶段继续查。外层耗时包含内层，不能相加 |
| `tokenizer_processor_load` / `data_transform_build` | 核对资源是否命中本地缓存、构建 target 是否有下载或扫描操作；在同配置下单独复现对应构建函数。`data_transform_build` 只测转换对象构建，不能解释训练时 tokenize 速度 |
| `dataset_build` / `dataloader_build` | 检查数据文件数量、索引/缓存是否重建及 loader 的初始化逻辑；区分一次性扫描与每次启动重复处理 |

**当前证据边界：** 耗时记录在代码块退出后产生，且启动记录要等输出初始化后才写出。
启动卡住时可能完全没有这些 JSONL；不能把“最后一条已完成阶段”当作当前卡点。
此时需要启动日志、各 rank 调用栈，或针对可疑边界补充进入/退出证据，不能靠增大采样频率解决。

### 数据供应、模板和 packing

开启需要的 `monitor.datasets` 项；查看 `datasets` 事件和 `timing/host/get_batch`、
`timing/host/data_wait`（入口支持时）。

| 看到的现象 | 提示用户可以怎么做 |
| --- | --- |
| `get_batch` 变慢且 `data_wait` 同时增大 | 先检查数据读取、worker 异常/CPU 忙闲及预取状态；用同一批数据单独测读取、transform、collate。主进程等待时间尚不能区分这三者，不要直接调整 worker 数下结论 |
| `get_batch` 慢但 `data_wait` 稳定 | 检查该入口的 `batch_h2d_transport`、输入字节量与 batch 形状，再查看传输/广播调用；host 提交时间不能替代设备传输时长 |
| `counts.valid_tokens` 突降、`top_tokens` 被特殊 token 占据 | 对照同一步的 `input_lengths`、`supervised_lengths` 与 tokenizer 版本；复现该 batch 并检查 input IDs、shifted labels 和 loss mask。首个渲染模板只帮助确认格式，不能代表异常步样本 |
| `packing_failures` 非空 | `offset_order/offset_coverage/segment_crosses_row` 先查 `cu_seq_lens` 和行边界；`position_reset` 查隔离策略及 position IDs；`cross_segment_target` 查段尾 shifted label/loss mask。用可疑小 batch 复算监督位置和 loss |

CP 尾部 padding 已发现 `cross_segment_target` 误报，先对照真实样本边界；详见
[八卡验收复现](trainer_monitor_qwen36_validation.md)。

先看 `counts.packing_checks` 和 `unavailable` 再解释“零失败”；没有检查覆盖不等于通过。
记录目前以 step 聚合数据统计，不保存异常 micro-batch 的原文或完整 tensor；精确定位样本需要可复现的数据游标或另外保留输入身份。

### Loss、梯度和优化器异常

设置 `monitor.local_loss: true`、`monitor.local_grad_norm: true` 即可启用，采集跟随 `training.logging_steps`；
后者每个 global rank 汇总一个值，不按 module 拆分。跨 rank 比较时读取各自的 `rank_N.jsonl`，
按 `step`、`rank` 对齐，micro loss 还需对齐 `micro_step`；控制台和远端 sink 目前仅输出 rank 0。
先确认参数参与本步且未冻结，避免把无梯度当成梯度消失。

| 看到的现象 | 提示用户可以怎么做 |
| --- | --- |
| total/foundation loss 突升 | 先对齐异常步的 lr、有效 token、目标组成，再用 `local_step_loss` 找 rank、`local_micro_loss` 找 micro-step；复现对应输入检查模板、标签和 packing，不能仅凭不同 rank 的 loss 不同判并行错误 |
| micro loss 平稳但 step loss 不符合预期 | 用同一 loss 名称的 micro 原始均值与 token 数复算 `Σ(loss×tokens)/Σtokens`，检查零监督 micro 和 MTP 分支的分母；再对照 trainer 的全局加权，不能拿 micro 的算术平均替代 |
| global grad norm 尖峰或持续触发裁剪 | 联查同一步 loss、有效 token 和 lr；查看各 rank 裁剪前 local norm 的时间变化，再用同输入复现检查第一个异常梯度。不同 FSDP 分片 norm 不要求相等，当前 local norm 也不是规约前梯度 |
| Muon 更新异常但 loss/梯度暂未异常 | 开启 `optimizer.ns_norm`，按 parameter/transform/call 对齐 `ns_norm`，核对矩阵形状、NS 参数及更新缩放；必要时对可疑矩阵离线复算。这个值取自 NS 后、缩放前，不能直接当成实际参数更新量 |

### EP 负载倾斜

启用 `monitor.moe`；需要定位 expert 时再启用 `record_distribution`。
已定位疑似层后，可用 `monitor.moe.layers: [0, 5, 10]` 缩小采集范围；全模型排查时省略该字段。

| 看到的现象 | 提示用户可以怎么做 |
| --- | --- |
| `moe/max_vio` 增大 | 查逐层 `expert_load.summary.max_vio` 找到具体 module，再看 `counts` 中热点 expert ID；比较相同输入规模的多个采样步，排除小 batch 的自然波动 |
| 部分 rank 分布不同或 expert 长期为空 | 分开查看 `expert_load_local` 的 rank 范围与 `expert_load` 的聚合范围；核对输入分片、top-k、全局 expert ID 和副本去重。先确认聚合后是否仍偏斜，再查 router/bias/辅助项配置 |
| `dropped_assignments` 增加 | 核对 attempted/accepted、capacity 和 dispatcher 的 drop 策略；用同一批输入复现路由计数。若 step 同时变慢，再采集通信和 expert kernel 时间，不能由逻辑负载直接判定设备瓶颈 |

### Step 变慢与 MFU 下降

先比较相同模型、并行配置和输入几何的稳定窗口，排除预热和保存步。

| 看到的现象 | 提示用户可以怎么做 |
| --- | --- |
| tokens/s 下降 | 对照 step_time、输入/有效 token 及长度分布；数据等待增大时转到数据分块，否则比较 forward/backward、gradient_prepare_clip、optimizer 的 host 耗时，再对可疑阶段做短窗口设备 profiler |
| MFU 下降或缺失 | 查 `flops_coverage.complete/missing_blocks`、公式版本、设备数和输入长度平方和；`unavailable_peak` 时核实 SKU/dtype 对应的 dense 峰值配置。口径一致后再分析耗时，不用猜测峰值补出 MFU |
| 开监控后才变慢或显存升高 | 固定同一负载，先比较 monitor 关闭与开启，再分别启用可疑 recorder/远端 sink；记录稳定窗口耗时、设备峰值和队列丢弃。JSONL 写入/flush 仍是同步的，SDK 异步也不保证零开销 |

host 阶段可以嵌套，不能相加当作设备时间；host 时间正常也不能排除异步 kernel 或通信变慢。
现有显存值是累计峰值，不随释放下降；判断持续增长需要额外的时间序列或 memory profiler 证据。

### 权重保存、checkpoint 与续训

| 看到的现象 | 提示用户可以怎么做 |
| --- | --- |
| 保存步暂停明显 | 区分 `weights_save`、同步 `checkpoint_save` 和异步 `checkpoint_submit`；核对保存大小、路径及是否同时导出 HF 权重。`weights_save` 包含完成 barrier，要对照各 rank，不能直接把全部时长归因于本地写盘 |
| submit 很快但后续等待长 | 按原始提交 step 对齐 `checkpoint_persist`、`checkpoint_wait` 与 `outcome`；检查后端异常、目录权限、空间和 I/O。persist 记录尚未到达可能是后台未完成或监控队列未消费，结合保存后端状态判断 |
| resume 后 step/累计量异常 | 核对实际加载的 checkpoint、是否恢复 train state、dataloader 游标和 monitor extra state；旧 checkpoint 未保存的累计计数无法靠监控补回，不应直接判为重复消费 |

### 给用户的提示格式

每个相关分块写清：**观察到什么 → 优先检查方向 → 具体动作 → 用什么结果区分原因 → 当前缺少什么证据**。
只展示与本次现象有关的分块，没有证据时明确哪些内容是待验证假设。

例如：“step 120 的 `get_batch` 和 `data_wait` 同时增大，优先检查数据供应。
请核对该步长度分布、worker 日志和存储读取，再用同一批数据分别测读取/transform/collate。
如果只有读取耗时增长，继续查数据源；如果 transform 增长，检查模板/tokenize。
当前主进程等待计时还不能区分这两种情况。”这是提示格式示例，不是本次运行的实测结论。

## 当前字段索引

排查动作见上面的现象分块。此表用于在 JSONL 中找到对应证据，不列未实现的候选指标。
事件公共身份为 `run_id/rank/step`；有 micro、module、call 时再按相应字段对齐。

| JSONL 位置 | 含义与诊断用途 |
| --- | --- |
| `train_step.metrics.training/*` | 当前训练目标的 loss、lr、裁剪开启时的全局 grad norm；用于关联收敛、调度和裁剪异常，loss 名称取决于模型目标 |
| `local_micro_loss.values` | `[原始 loss 均值, 有效 token 数]`，按 `name/micro_step` 定位异常 micro |
| `local_step_loss.loss/tokens` | 当前 rank 按 token 加权的 step loss；用于复算累积权重 |
| `local_grad_norm.metrics.training/grad_norm_local_post_reduce_pre_clip` | 每 rank 一个规约后、裁剪前 L2；用于观察本地梯度整体变化，不按 module 拆分 |
| `ns_norm.values/matrix_owners/call` | Muon NS 后、缩放前矩阵范数及参数归属；`metrics.optimizer/ns_norm_mean/max/min` 是本次调用内汇总 |
| `train_step.metrics.data/step_tokens`、`data/consumed_tokens` | 当前步和累计有效监督 token 数；用于检查监督量与续训计数 |
| `train_step.metrics.data/step_samples`、`data/consumed_samples` | 当前步与累计训练样本计数；packed 行不能直接解释成原始 document 数 |
| `train_step.metrics.data/step_input_tokens` | 当前步输入计算规模；与有效监督 token 分开解释 |
| `datasets.counts` | 本采样步源 batch 的 rows、slots、input/valid tokens、target_slots、segments、packing_checks/failures 等；计数不跨 rank 聚合 |
| `datasets.input_lengths/supervised_lengths/segment_lengths` | min/max/mean、P50/P90/P99 和精确直方图；用于识别长度变化与长尾 |
| `datasets.top_tokens` | token ID/count/fraction；用于识别特殊 token 或重复模式，不能证明样本重复 |
| `datasets.padding_fraction/segments_per_row` | 有 padding mask 时的填充比例及平均每行 segment 数；无元数据不能推断利用率 |
| `datasets.packing_failures/unavailable` | 有界失败原因列表与缺失证据；完整失败次数看 `counts.packing_failures`，零失败须联查检查覆盖 |
| `expert_load_local.values`、`expert_load.counts` | 启用 `record_distribution` 后的本地/聚合 expert assignment 分布；按 module 定位热点 |
| `expert_load.summary` | max、mean、max_vio、empty_expert_fraction、attempted/accepted/dropped_assignments；用于观察路由倾斜和容量丢弃 |
| `expert_load_summary.metrics.moe/max_vio` | 当前 PP stage 选中且实际观测到的层的最大 max_vio；不能解释成全模型跨 stage 汇总 |
| `timing.phase/duration_seconds/outcome` | 启动、保存和恢复等已结束阶段的主机墙钟事件；不表示正在阻塞的阶段 |
| `train_step.metrics.timing/host/*` | step 内阶段主机耗时；数据等待、前后向、梯度准备/裁剪、optimizer、scheduler，不能替代设备 profiler |
| `train_step.metrics.performance/step_time`、`tokens_per_second`、`input_tokens_per_second` | step 时间、有效监督吞吐、输入吞吐；用于区分工作量变化与执行变慢 |
| `train_step.metrics.memory/device_max_allocated_gb`、`memory/device_max_reserved_gb` | 既有累计显存峰值；不直接表示当前占用或持续增长 |
| `flops_coverage.complete/blocks/missing_blocks/formula/peak` | 标准块覆盖、逐块 FLOPs 与估算口径；字段随是否完整而变化 |
| `train_step.metrics.performance/tflops_per_device`、`mfu_percent`、`mfu_window_percent`、`mfu_status` | 相应开关启用时的计算吞吐/MFU 或缺峰值状态；未覆盖完整模型时不输出部分模型值 |
| `unavailable.metric/reason` | 未支持或缺少观察点的原因；同原因可能只记录一次，要搜索整个 run |
| `train_step.metrics.monitor/dropped_summaries`、`monitor_closed.dropped_summaries/dropped_remote_records` | 摘要/远端队列丢弃计数；用于排查采集背压，不是训练数值错误 |

`log_first_rendered_template` 输出到模板 logger，不是 JSONL 指标；FSDP 顺序检查通过原有 runtime 检查报错。
当前不采集 post-clip norm、token 重复度、截断/丢弃 token 数、物理 dispatch padding 槽位、
NS 输入 norm/相对变化或队列峰值；worker transform/collate 也没有独立计时。
需要这些证据时单独复现或使用相应 profiler，不按“应有字段缺失”判断 monitor 故障。

## 文档与验证约定

- 每个新增指标必须提供：定义、单位、分母、采集时点、local/global 范围、可观察问题、联查证据、不可用条件及独立复算方式。
- 分布式诊断保留全 rank 身份；共享终端与远端汇总默认由 global rank 0 输出，禁止按日志可见性推断其他 rank 没有采集。
- 诊断文档与 [模型精度验证 skill](../.agent/skills/hf-model-precision-validation/SKILL.md) 共用本表，不在 skill 复制公式。
- 新字段验收需包含一例正常数据、一例能体现目标问题的受控异常，以及一例不可用/未采样数据；区分诊断提示与已有训练强制检查。
