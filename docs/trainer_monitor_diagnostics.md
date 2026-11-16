# 通用 Trainer 监控指标诊断说明

适用范围：通用 Trainer，不涉及 RL。配套 [监控设计方案](design/trainer_monitoring_plan.md) 和 [使用与实现边界](trainer_monitor_usage.md)。
本文描述指标的诊断用途；Recorder、异步链路及标准模块 MFU 已接入 `monitor_dev`。
当前实现范围与本地/设备验收状态以 [使用与实现边界](trainer_monitor_usage.md) 为准。
开始诊断前核对运行 revision、实际输出字段和能力报告；设计中的字段缺失不自动判为运行故障。

## 使用前提

- 对齐 run、optimizer step、micro-step、rank、PP stage、module、采样窗口和实际输入；跨 rank loss 不同也可能来自不同数据。
- 区分 input/有效监督 token、local/global、裁剪前/后、规约前/后、逻辑量/物理 padding。不同分母和统计阶段不能直接比较。
- 全 rank 采集不等于每个 rank 都有所有指标：无 loss 的 PP stage、未持有的 expert/module 应标为不适用。
- 指标是定位线索，不能单凭一次尖峰判定 bug；先排除输入变化、预热、采样缺失和异步迟到，再关联相邻证据。
- 正常缺失、unsupported、pending、dropped、incomplete 与数值零严格区分。已有 NaN/Inf、数据合法性等强制检查仍在原训练路径执行。

## 按现象分块排查

下面是给用户的排查提示，不是已经实现的自动告警。按出现的现象选择一块，给出具体检查动作，
不要只回答“可能是数据问题”或“建议看 profiler”。字段以该次运行的 JSONL 和代码为准；缺失时先检查采集覆盖。

### 先确认记录是否可用

查看 `<monitor.output_dir>/<run_id>/rank_<rank>.jsonl`，按记录中的 `step` 对齐，而非文件行顺序。
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

先看 `counts.packing_checks` 和 `unavailable` 再解释“零失败”；没有检查覆盖不等于通过。
记录目前以 step 聚合数据统计，不保存异常 micro-batch 的原文或完整 tensor；精确定位样本需要可复现的数据游标或另外保留输入身份。

### Loss、梯度和优化器异常

按需启用 `local_loss`、`local_grad_norm`；怀疑具体子树时设置 `local_grad_norm.modules`。
先确认模块参与本步且未冻结，避免把无梯度当成梯度消失。

| 看到的现象 | 提示用户可以怎么做 |
| --- | --- |
| total/foundation loss 突升 | 先对齐异常步的 lr、有效 token、目标组成，再用 `local_step_loss` 找 rank、`local_micro_loss` 找 micro-step；复现对应输入检查模板、标签和 packing，不能仅凭不同 rank 的 loss 不同判并行错误 |
| micro loss 平稳但 step loss 不符合预期 | 用同一 loss 名称的 micro 原始均值与 token 数复算 `Σ(loss×tokens)/Σtokens`，检查零监督 micro 和 MTP 分支的分母；再对照 trainer 的全局加权，不能拿 micro 的算术平均替代 |
| global grad norm 尖峰或持续触发裁剪 | 联查同一步 loss、有效 token 和 lr；按模块采集裁剪前 local norm 缩小范围，再用同输入复现检查第一个异常梯度。不同 FSDP 分片 norm 不要求相等，当前 local norm 也不是规约前梯度 |
| Muon 更新异常但 loss/梯度暂未异常 | 开启 `optimizer.ns_norm`，按 parameter/transform/call 对齐 `ns_norm`，核对矩阵形状、NS 参数及更新缩放；必要时对可疑矩阵离线复算。这个值取自 NS 后、缩放前，不能直接当成实际参数更新量 |

### EP 负载倾斜

启用 `monitor.moe`；需要定位 expert 时再启用 `record_distribution`。

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

## 指标与问题映射

表中的逻辑指标对应设计语义；尚未固定的字段名不视为已实现 API。实现时须为每个导出字段关联此表中的条目、单位与统计范围。

| 指标 / 统计项 | 可以观察或定位的问题 | 需要联查的证据与边界 |
| --- | --- | --- |
| total loss / foundation loss | 收敛异常、突升、数值非有限、目标组成变化 | lr、有效监督量、local loss、输入身份；两者不一定相等，先确认 auxiliary/MTP 的组成 |
| local step loss | 单 rank 数据异常、归一化或并行切分差异 | 对齐输入与有效分母；不同 rank 处理不同样本时差异正常，不能直接判定并行错误 |
| micro-batch loss | 某个 micro-batch 脏数据、梯度累积权重异常 | token 分布、packing 边界、micro-step ID；不能直接对不等长 micro loss 取均值替代整步 loss |
| router auxiliary loss | 路由均衡目标变化、辅助项尺度异常 | expert load/max_vio、辅助项系数与注入方式；该标量不直接说明 expert 物理负载，也可能仅供日志展示 |
| global pre-clip grad norm | 梯度爆炸、更新尺度异常、裁剪频繁 | loss、lr、裁剪阈值、并行聚合范围；确认实现是否包含所有 PP stage/专家 |
| local / module grad norm | 梯度消失、未参与训练的模块、局部异常梯度 | 参数是否冻结/参与本步、FSDP shard 大小与所有权；规约后本地分片不能冒充通信前梯度，不要求不同分片 norm 相等 |
| post-clip norm（可用时） | 裁剪是否符合预期 | pre-clip norm、阈值和共享参数去重；区分实际测量与由裁剪系数推算 |
| lr | warmup、调度器步数或 resume 恢复错误 | optimizer step、参数组、scheduler state；单个最大 lr 不代表所有组 |
| step samples / consumed samples | 数据重复消费、漏计、续训计数断裂 | dataloader 游标、输入 hash；packed 行数与原始样本数分开，预取未训练数据不算消费 |
| input tokens / consumed input tokens | 输入计算规模变化、吞吐分母错误 | mask、CP 切分及副本去重；prompt 即使不监督也参与计算 |
| valid tokens / consumed valid tokens | 监督稀疏、全 ignored batch、label shift 计数错误 | 实际 loss 分母和 loss_mask；不拿 input tokens 替代监督量 |
| padding / ignored 比例 | padding 浪费、模板/mask 异常、训练目标被意外屏蔽 | 明确各自分母；padding 与 ignored 有重叠，不是天然互斥类别 |
| input/valid 长度 min/max/mean、P50/P90/P99、直方图 | 长尾 batch、数据截断、不同 rank 输入量偏斜 | packed 行与原始 segment 分开；近似分位数须标注，max 增大不必然是数据故障 |
| token Top-N（ID/count/比例） | 重复文本、异常特殊 token、模板污染、tokenizer 不匹配 | 样本和 tokenizer 版本；padding 按 mask 排除，pad/eos 同 ID 时不能按 ID 全删；高频 token 本身不代表错误 |
| token 重复度（可用时） | 数据重复或生成式模板退化 | 明确定义是重复 token 比例还是样本重复；Top-N 不能替代样本去重证据 |
| packed 行数、原始样本数、每行 segment 数 | 拼接策略失效、原始样本计数错误 | packing metadata 与 dataloader；metadata 缺失不能估算成确定样本数 |
| segment 长度分布、填充率 | packing 效率低、长样本主导、槽位浪费 | 最大序列长度和实际 attention 策略；利用率高不代表边界正确 |
| 截断样本数 / 丢弃 token 数 | 截断过量、监督尾部丢失 | 原始长度与截断策略；只在有可靠 metadata 时输出 |
| packing 边界检查失败数 / 检查覆盖数 | offsets 越界/重叠、position 错位、跨段监督泄漏 | 按允许拼接或隔离策略检查；metadata 检查通过不能证明 attention kernel 正确，采样通过不代表每 batch 全覆盖 |
| 首步渲染模板 | 角色标记、特殊 token、格式或输入文本错误 | 对照实际 input IDs/labels；global rank 0 的首个限长样本不代表全数据集 |
| expert assignment 数 / 完整分布 | 路由热点、expert 长期闲置、不同层失衡 | top-k、层、expert 全局 ID、统计范围；同一个 token 可产生多个 assignment |
| expert load max / mean | 最繁忙 expert 与平均负载差距 | 有效接收量与路由分配量分别解释；不等同设备利用率或显存占用 |
| max_vio / 跨层 max_vio 最大值 | 最严重的路由负载不均衡及所在层 | `max/mean - 1`，空负载单独标记；先逐层算再汇总，不能混合各层 expert 计数 |
| 空 expert 数 | expert 利用不足或路由塌缩线索 | 小 batch 下自然可能为空；结合长期分布，不能凭单步判断塌缩 |
| dropped assignment 数 / 比例 | 路由容量限制、token 分配被截断 | capacity、分配/接收总数及 drop 实现；无 drop 统计能力不等于 drop=0 |
| 物理 padding 槽位 / 有效 assignment | expert kernel 填充开销、逻辑负载与实际工作量差距 | dispatch 策略和计数口径；不能用 padding 槽位替代有效负载 |
| Muon NS 输出 norm、可选输入 norm / 相对变化 | NS 数值异常、更新方向尺度异常 | 输入尺度、迭代次数、矩阵形状和 optimizer 类型；无通用固定健康阈值，非 Muon 不适用 |
| 组网耗时 | rendezvous、进程组/DeviceMesh 初始化阻塞 | 各 rank 阶段边界和运行日志；host 耗时不能直接定位网络设备故障 |
| 模型构建 / 并行包装耗时 | 初始化、模块替换或包装变慢 | 模型大小、初始化方式和各子阶段；避免嵌套阶段重复相加 |
| 预训练权重加载 / resume 恢复耗时 | 存储读取、转换或状态恢复变慢 | 文件大小、缓存命中和恢复组件；加载与完整续训恢复不是同一范围 |
| 数据集初始化 / tokenizer / map 耗时 | 数据准备成为启动瓶颈 | worker、缓存、样本量；与训练时等待 batch 分开 |
| batch_fetch / preprocess / collate 耗时 | 数据供应不足、CPU 预处理或组 batch 瓶颈 | 主进程等待与 worker 执行时间分别看；不能由等待时间推断 tokenize 耗时 |
| H2D host 耗时 | 拷贝提交/阻塞变化、数据准备问题 | pinned memory、设备流和数据大小；异步提交耗时不是传输执行时长 |
| forward/backward host 耗时 | 训练阶段变慢、序列/并行策略变化 | 输入几何、通信和 profiler；host 时间不能直接等同 kernel 时间 |
| grad clip / optimizer / scheduler host 耗时 | norm 通信、参数更新或调度开销变化 | 参数量、DTensor 布局和优化器种类；计时本身不加设备全同步 |
| 权重保存 / 完整 ckpt 保存、submit / complete | 保存暂停、后台 I/O 堆积、保存未完成 | checkpoint ID、写入量和完成状态；submit 返回不是落盘完成 |
| step_time / input tokens/s / valid tokens/s | 训练吞吐下降或监督效率变化 | 序列分布、阶段耗时、预热和采样；保持分母与时间窗口一致 |
| 既有 memory/device_max_allocated_gb、reserved_gb | 峰值压力、监控引入额外显存的线索 | 原有运行累计峰值不会随释放下降；reserved 高不等于泄漏，本方案不改既有显存采集 |
| 标准模块 FLOPs 分项 / coverage | 公式漏项、重复计数、模型组成变化 | 唯一块所有权、真实调用数、GDN 等特殊块；不完整覆盖不能报完整模型 MFU |
| TFLOP/s/device / MFU | 同口径下的计算吞吐下降、模型计算量与耗时不匹配 | 峰值 SKU/dtype、输入几何、全局设备数、公式版本；MFU 是估算，不等同硬件 profiler 利用率 |
| recorder pending/dropped/incomplete / 队列峰值 | 采集背压、异步迟到或退出丢失 | 原采样 step 与完成事件；这些是采集健康度，不是训练数值错误 |
| hook 调用次数 / CPU 耗时 / graph break（验收证据） | 监控回调过密、重计算重复记录、编译退化 | 无 hook/空 hook/同步/异步 A/B；不以 CUDA/NPU kernel 时间代替 Python 回调开销 |

## 文档与验证约定

- 每个新增指标必须提供：定义、单位、分母、采集时点、local/global 范围、可观察问题、联查证据、不可用条件及独立复算方式。
- 分布式诊断保留全 rank 身份；共享终端与远端汇总默认由 global rank 0 输出，禁止按日志可见性推断其他 rank 没有采集。
- 诊断文档与 [模型精度验证 skill](../.agent/skills/hf-model-precision-validation/SKILL.md) 共用本表，不在 skill 复制公式。
- 新字段验收需包含一例正常数据、一例能体现目标问题的受控异常，以及一例不可用/未采样数据；区分诊断提示与已有训练强制检查。
