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

## 四类诊断路径

1. **Loss 突升**：先对齐真实采样 step 与输入；看有效 token/packing → local/micro loss → pre-clip global/local norm → lr/optimizer 状态。异步记录缺失时先说明证据不足，不补零。
2. **EP 倾斜**：先定位逐层 max_vio → 查 expert ID 计数与空 expert → 核对 top-k/接收/drop/padding → 检查并行副本去重；要归因 step 慢还需通信或 kernel 证据。
3. **Packing 错位**：核对 offsets 和 segment 总覆盖 → 按策略看 position/attention 参数 → 对齐 shifted labels/有效监督量 → 用小 batch 直接复算 loss；不新建完整 S×S mask 做在线检查。
4. **MFU 下降**：先核对标准块覆盖/公式/峰值不变 → 核对实际 token 几何与设备数 → 比较 step_time 与各阶段 → 检查 hook/异步采集 A/B；不能直接认定算子算力下降。

## 文档与验证约定

- 每个新增指标必须提供：定义、单位、分母、采集时点、local/global 范围、可观察问题、联查证据、不可用条件及独立复算方式。
- 分布式诊断保留全 rank 身份；共享终端与远端汇总默认由 global rank 0 输出，禁止按日志可见性推断其他 rank 没有采集。
- 诊断文档与 [模型精度验证 skill](../.agent/skills/hf-model-precision-validation/SKILL.md) 共用本表，不在 skill 复制公式。
- 新字段验收需包含一例正常数据、一例能体现目标问题的受控异常，以及一例不可用/未采样数据；区分诊断提示与已有训练强制检查。
