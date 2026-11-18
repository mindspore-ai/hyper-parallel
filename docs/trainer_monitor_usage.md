# 通用 Trainer monitor 使用与实现边界

对应 [设计方案](design/trainer_monitoring_plan.md) 和 [指标诊断说明](trainer_monitor_diagnostics.md)。
适用范围：`monitor_dev` 的通用 Base/Text/VLM Trainer；版本化测试结果见文末。
本文区分已接入的路径与仍需设备验收的范围，不把 CPU 测试当作多卡性能验收。

## 配置和输出

完整默认 YAML 维护在 [设计方案](design/trainer_monitoring_plan.md#yaml)，实际字段定义见
[MonitorConfig](../hyper_parallel/trainer/config/monitor.py)。以下是排查 loss/梯度的配置示例，两项采集频次跟随外部 `training.logging_steps`：

```yaml
monitor:
  local_loss: true
  local_grad_norm: true
```

未指定项使用默认值；按需添加 `moe`、`datasets`、`optimizer.ns_norm` 或 `mfu`。
基础日志频次由 `training.logging_steps` 控制，local loss/norm 使用同一频次；其他项目的 `sampler_steps` 只控制采集。
关闭 `monitor.enabled` 后仍保留基础训练日志。

EP 按层采集只需增加 `layers`：

```yaml
monitor:
  moe:
    enabled: true
    sampler_steps: 1
    layers: [0, 5, 10]
```

层号从 0 开始，表示模型层号，不是第几个 MoE 层；省略或 `null` 表示全部层。
`layers` 必须是非空、无重复的非负整数列表。按模块路径中的 `layers.N`、`blocks.N` 或 `h.N`
匹配；Qwen3.6 对应 `model.layers.N.mlp`，PP 切分后应保留原始层号，不按本 rank 重新编号。
无法解析层号或本 rank 没有选中的已接入 MoE 层时，输出 `expert_layer_selection` 不可用原因。
筛选在模型绑定时完成，未选中层不分配 monitor 计数向量、不做 monitor 的 scatter 计数，
也不进入摘要规约、D2H 或任何 sink；原有路由和 EP 训练通信照常运行。
`moe/max_vio` 只汇总选中层，排查全模型负载倾斜时应省略 `layers`。

- 全局 rank 0 的 LoggingCallback 输出一行 `Training: 百分比|进度条| 当前步/总步数 | step=... 指标...`。
  tqdm 仅用于进度格式化，不带指标摘要、不独立刷新；终端与重定向输出相同，频次沿用 `training.logging_steps`。
- monitor 开启时，控制台是否输出由 `sinks` 中的 `console` 控制；关闭 monitor 时沿用基础控制台日志。
- JSONL 的 `train_step` 和远端指标不含进度条文本；未安装 tqdm 或无总步数时仍正常输出纯指标行。
- 开启 console 后，monitor 额外输出 `training metrics: step=... rank=0 ...`，每个命中
  `training.logging_steps` 的 step 恰好一行。内容为该步采样到的 local loss/norm、逐层 EP 标量、
  datasets 标量及 Muon NS；EP 保留层名，NS 保留 call 编号，不用后一个调用覆盖前一个。
- 所有输出中的 `rank` 均为 **global rank**，不是节点内的 local rank。控制台行首带 `step=N rank=0`；
  各 rank 的 JSONL 文件名为 `rank_N.jsonl`，每条记录带 `rank: N`、`step`，micro loss 另带 `micro_step`。
  W&B/TensorBoard 的局部指标名带 `/rank_0/`；目前远端 sink 只输出 rank 0，并非全 rank 汇总。
- 汇总等待本步已提交的异步摘要完成；不加设备同步，可能晚于后续 step 的基础日志，始终保留原 step。
  丢弃的摘要在该行计数，不等待不存在的结果；没有采集值时输出 `status=no_collected_metrics`，不复用上一步。
- 完整 expert 分布、Top-N、直方图、micro loss 和参数归属继续留在 JSONL。汇总只含 rank 0 已持有的结果，
  不为控制台收集其他 rank 的局部值，也不把这条汇总重复发送给 SDK。

控制台示例（省略其他指标）：

```text
Training:  15%|###                 | 3/20 | step=3 epoch=0 training/total_loss=10.5977
training metrics: step=3 rank=0 moe/max_vio=0.24 training/grad_norm_local_post_reduce_pre_clip=18.2 training/local_foundation_loss=10.5977
```

- 每个 rank 的诊断记录写到 `<output_dir>/<run_id>/rank_<rank>.jsonl`。
- `schema_version/run_id/event/rank` 标识记录；训练事件保留原始 `step`。
- 微步 loss、逐层 EP 等是独立诊断事件，不是另一条重复的 `train_step`。
- 异步摘要可能晚于本步基础日志到达，仍使用采集时的 step，不回填成下一步。
- 全部 rank 的细节保留在各自 JSONL。在线远端只由 rank 0 写入其所持有的全局/本地结果；
  不隐式增加跨 rank 细节上传通信，也不把 rank 0 的 local 数据命名为全局数据。

## 旧配置迁移

顶层 `debug`、`wandb` 及其配置类已删除，旧 YAML/CLI 键会报未知配置错误。

| 原配置 | 当前入口 |
| --- | --- |
| `debug.log_first_rendered_template` | `monitor.datasets.log_first_rendered_template` |
| `debug.check_dataset` | `monitor.datasets.log_level` |
| `debug.check_fsdp_runtime` | `monitor.check_fsdp_runtime` |
| `debug.check_nan_inf` | 删除：原来没有实际调用，不代表曾提供 NaN/Inf 检查 |
| `wandb.enabled/project/entity` | `monitor.sinks` 中添加 `wandb`；`monitor.wandb.project`；环境变量 `WANDB_ENTITY` |

上述数据日志与 FSDP 顺序检查均受 `monitor.enabled` 总开关控制。
FSDP 检查仍由原有 runtime 实现，按 micro-step 检查执行/预取顺序，不受 `sampler_steps` 控制；
默认关闭，开启后会安装执行跟踪 hook，可能影响性能。

## 采集位置

| 功能 | 接入位置 | 口径与限制 |
| --- | --- | --- |
| Local/micro loss | `BaseTrainer.postforward`，全局加权前 | 原始 micro 均值；step 用有效 token 加权；零分母不可用 |
| Local grad norm | `clip_grad_norm_`，Partial 规约后、裁剪前 | 复用 norm 输入视图；每 rank 一个 L2，覆盖本地全部有梯度的可训练参数，包含 expert；没有新增规约 |
| 不启用裁剪时的 local norm | `BaseTrainer.clip_grad_norm` | 读取已有 main_grad/grad；未完成 Partial 的值不冒充 post-reduce |
| EP | 共享 `MoE.forward`，原生 EP 的 `ep_routed_forward` / `ep_routed_dispatch` | `ExpertLoadRecorder` 固定注册 `monitor.moe.layers` 选中的层（默认全部）；真实 forward 计数，backward 重计算不计数 |
| Muon NS | `Muon._compute_batched_ns_outputs_for_tensors` | NS 后、学习率/RMS 缩放前的矩阵 Frobenius norm；保留 parameter、transform、call 与 matrix 范围 |
| 数据 | 通用 get-batch 的 CPU 源 batch、CP 切片和 H2D 前 | 本 rank 的原始源数据分布；精确 step Top-N，先合并全部 micro 计数再取 Top-N |
| 分块耗时 | 显式生命周期和 `statistics_phase` | 主机墙钟，非 kernel 执行时间；不新增逐块设备同步 |
| FLOPs | 初始化时的标准块清单 | Linear/attention/FFN/router/shared expert/LM head 组合；逐步只更新输入几何 |

Trainer 通过 `monitor_phase(..., observe=True)` 同时进入计时和采集作用域；backward 仅计时，不观察重计算。
常规指标采集没有新增 forward/backward hook；FSDP 顺序检查的可选 hook 见上节。`components.monitor` 是独立的注入接口，不导入 Trainer。
模型/优化器只借出当前有效 tensor；训练线程生成小摘要，异步 D2H 只持有摘要的独立存储。
D2H 完成 event 就绪后才读 CPU 缓冲。通信全部在训练线程上按固定层序执行。
传输队列上限 32、单摘要上限 65,536 元素；溢出通过 `monitor/dropped_summaries` 显式记录。
EP 合并所有层后一次提交，避免每层消耗一个传输槽位；关闭或未采样不创建计数 tensor。

## 数据与 EP 的解释

`data/*` 表示数量，`performance/*` 表示速率或耗时；`step_tokens`/`consumed_tokens` 为有效监督 token，
`step_input_tokens` 为参与计算的输入 token，当前没有累计 input token 字段。

数据事件包含输入/监督长度直方图与分位数、token ID/count/fraction、segment 长度、填充率和边界错误。
监督计数遵循 causal shift、IGNORE_INDEX、loss_mask，通用文本入口也传递 EOD mask 与 shifted-label 策略。
没有 padding mask 时不猜测 pad/eos ID；填充率标记不可用。
packing 只检查已有 offsets、位置和监督 metadata，不新建 S×S mask，也不声称验证了 attention kernel。
缺少 metadata、位置尚未生成或不支持的策略会列在 `unavailable`。
当前 CP 尾部 padding 可能触发 `cross_segment_target` 误报，必须先复查真实样本与 padding 边界；
复现及证据见 [Qwen3.6 验收报告](trainer_monitor_qwen36_validation.md)。

EP load 是 top-k **assignment 次数**。每层先求其独立 token 域的总计数，再计算 `max/mean - 1`；
`moe/max_vio` 为当前 PP stage 选中且实际观测到的层的最大值，不跨层平均掩盖异常。
共享 dropless MoE 的 attempted=accepted，drop=0；原生 EP 从已生成的 dispatch indices 统计 accepted，
用路由前 assignment 数减去 accepted 得到 drop，覆盖容量限制的分离 dispatch 路径。
物理 dispatch padding 槽位未采集，不能由 assignment 计数推算实际 kernel 填充量。注册层没有实际 observation 时标记 unavailable，不输出假零负载。
DP+CP 求和，sequence-parallel TP 再求和；TP 副本不重复累加。
当前 adapter 合约要求每层计数按完整的全局 expert ID 排列。
未接入这两种原生调用的其他 dispatcher、expert 分片向量、不同副本关系必须提供适配，不能直接套用这套归约。

## 耗时与续训

启动事件覆盖默认进程组、子组、模型构建、预训练权重加载、tokenizer/processor、dataset/dataloader 构建。
step 覆盖取数等待、数据传输提交、forward/backward、梯度准备/裁剪、optimizer、scheduler。
`get_batch` 包含其内部等待/传输阶段；这些嵌套计时不能相加当作 step 时间。
多 worker 的 transform/collate 执行时间尚没有独立 worker metadata，不能把主进程的等待时间解释为它们的计算时间。

保存区分：

- `weights_save`：完整 HF 权重导出、资产写入和完成 barrier。
- `checkpoint_save`：同步 DCP 保存调用。
- `checkpoint_submit`：异步提交返回。
- `checkpoint_persist`：提交到 persist future 完成，保留原始 step 和成功/失败状态。
- `checkpoint_wait`：调用方等待尚未完成保存的时间。
- `checkpoint_restore`：恢复训练状态。

checkpoint extra state 扩展保存 meter 的累计 token/sample 数与 run ID。
旧 checkpoint 没有 monitor 状态时不要求新增字段；它无法恢复旧作业未曾保存的累计量。
远端 run 在恢复完成后初始化。新作业可由外部 `WANDB_RUN_ID` 指定 ID，否则 rank 0 生成并共享。

## WandB / TensorBoard

不默认启用。需要相应渠道时安装可选依赖，并显式添加 sink：

```bash
pip install wandb tensorboard
```

```yaml
monitor:
  sinks: [console, jsonl, wandb, tensorboard]
  wandb:
    project: hyper-parallel
    mode: offline  # 本地 SDK run；改成 online 才上传
```

标量直接以 Python 数值传给 `run.log()`，横轴为 `global_step`，不解析终端字符串。
仅 Top-N 和 expert 分布转换为带 step/rank/module 的 Table，并与同一事件的标量合并为一次 `run.log()`。
普通诊断事件（如 micro loss 明细、FLOPs 块清单、不可用原因）完整保留在 JSONL，不再额外生成 W&B payload Table。
TensorBoard 标量使用相同原始数值。rank-local 标量的远端 key 附带 rank/module，避免混为同一曲线。
WandB 离线目录可用 SDK 的 `wandb sync <offline-run-directory>` 上传；JSONL 本身不是 SDK 离线 run 文件。
组织/用户归属通过外部环境变量 `WANDB_ENTITY` 设置；恢复时同一 run ID 使用 SDK resume。
远端输出队列上限 256 条，容纳多层 EP 在 step 边界集中产生的记录；它与设备摘要传输的 32 槽队列不同。
每个 sink 独立处理异常，WandB 失败不会跳过 TensorBoard 写入/关闭；同一 sink 的重复故障只首次告警。
队列满仍显式计数并保留本地 JSONL，不无限重试；退出等待有上限。
WandB artifact staging/cache 目录也必须可写，服务器共享 HOME 不可写时可由外部设置
`WANDB_DIR`、`WANDB_DATA_DIR`、`WANDB_CACHE_DIR`、`WANDB_CONFIG_DIR` 到作业目录。
每步全量上传逐层 Table 会增加 CPU/SDK 开销，异步线程不等于无性能影响；参见设备验收报告的对照结果。
不在远端启用时导入 SDK。

## MFU 完整性

`monitor.enabled=true` 时，`training.log_tflops` 与 `monitor.mfu` 共用标准块估算，分别控制 TFLOP/s 与 MFU 输出。
关闭 monitor 后，`training.log_tflops` 仍可使用环境计量 callback 的原有估算路径；不产生 monitor coverage 或 MFU。
计算版本为 `standard-blocks-v2-causal-triangle-gdn-megatron`：训练标准块使用 forward×3，causal attention 使用三角近似。
router、routed top-k 和 shared expert 分项计入；重计算、通信、optimizer、norm/激活/RoPE 不计入分子。
按实际 segment 的长度平方和计算 attention，包含不参与监督的 prompt 输入。

模型可提供 `monitor_flops_blocks` 清单，每项使用 `FlopsBlock` 的字段，唯一 name 表示唯一逻辑调用。
`backward_multiplier` 可声明实际梯度路径；不能把 frozen/LoRA 模型统一乘 3。
默认结构适配覆盖标准全注意力、带 output gate 的全注意力、GDN、gated FFN 和 routed/shared MoE。
Qwen3.6 MoE 纯文本路径已覆盖全部上述计算块；全注意力 Q 投影中的 output gate 独立记账，
优先由实际模块的投影维度确认，避免旧配置未写 gate 字段时漏算。初始化只读模块 metadata，训练中不增加 hook。
MLA、视觉额外计算、冻结/LoRA 或未知计算结构仍要求独立适配，不输出部分模型的 TFLOP/s 或 MFU。

### GDN 分块公式与 Qwen3.6 覆盖

记 `T` 为全局输入 token 数，`H` 为 hidden size，`q=Hk×D`、`v=Hv×D`，`Hk/Hv` 为 key/value head 数，
`D` 为相等的 key/value head dimension，`C` 为 causal convolution kernel size。
Qwen3.6 使用这个等维 GDN 结构。每层的 forward FLOPs 分解为：

| 块 | Forward FLOPs |
| --- | --- |
| QKV projection | `2×T×H×(2q+v)` |
| Z projection | `2×T×H×v` |
| Beta / alpha projections | 各 `2×T×H×Hv` |
| Depthwise causal convolution | `2×T×C×(2q+v)` |
| Gated delta-rule core | `8×T×Hv×D²` |
| Output projection | `2×T×v×H` |

forward+backward 为各项乘 3；core 使用 Q/K 重复后的 value head 数，卷积使用重复前的 QKV 通道数。
core 采用 [Megatron Bridge 的 GDN 标准估算](https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/src/megatron/bridge/training/utils/flop_utils.py)
中的 `8×T×Hv×D²` 约定；这是标准模型 MFU 的计算口径，chunk/WY/三角求解、尾块 padding、重计算的额外 kernel 工作不纳入。
norm、激活及逐元素门控也沿用本方案的忽略约定；没有将整个 GDN core 忽略或伪装成平方复杂度 attention。

按 [Qwen3.6-35B-A3B 公布的配置](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/main/config.json)
在 meta device 构建真实 HF text 模型，并从实际投影/卷积/expert 权重形状独立复算。
当 `T=8×4096`、`Σs²=8×4096²` 时，标准训练 FLOPs 为：

| 模型视图 | GDN / 全注意力层数 | 唯一计算块数 | FLOPs / step |
| --- | --- | --- | --- |
| 20 层 | 15 / 5 | 291 | 362,299,281,899,520 |
| 完整 40 层 | 30 / 10 | 581 | 624,611,725,148,160 |

以上是公式校验值，不是硬件实测吞吐。每层包含对应 attention、top-k routed FFN、router、shared FFN 与 shared gate，
整模型另含 LM head。checkpoint 中的 MTP 配置字段不等于实际执行 MTP；当前 HF `ForCausalLM` 路径没有 MTP 模块或调用，
因此不额外添加不存在的计算。两个视图的 coverage 都为 complete。

### 设备归一化

设备峰值不能仅凭 GPU/NPU 家族猜测。部署入口可设置 `HYPER_PARALLEL_MONITOR_PEAK` JSON，字段为
`sku`、`dtype`、`tflops`、`source`、`sparse: false`；必须来自实际 SKU/计算精度的厂商 **dense** 峰值。
缺少可信峰值时，MFU 状态为 `unavailable_peak`；仅在 `training.log_tflops=true` 时输出完整模型 TFLOP/s。
当前约定为同构设备；异构作业不得复用一个峰值。
每步 MFU 使用该步耗时，窗口 MFU 使用本进程训练窗口 `sum(F)/sum(dt)`，不平均各步百分比。

## 验收状态与已知限制

| 证据 | 版本与范围 |
| --- | --- |
| 2026-11-18 八卡输出复验 | `386640eb` + 工作区补丁 `9b6aca65e145`；8 NPU、20 层 Qwen3.6、EP2/CP2/FSDP8、20 steps，全部 monitor 开关及离线 SDK 开启，退出 0 |
| [此前 Qwen3.6 八卡验收](trainer_monitor_qwen36_validation.md) | `f49fa939`，2026-11-16 至 17 日；20 层纯文本视图、EP2/CP2/FSDP8、JSONL/离线 SDK、小模型保存恢复与 Muon |
| [早期 NPU 验收](trainer_monitor_npu_acceptance.md) | 2026-11-09 历史对照，不作为当前版本验收结论 |
| 配置迁移本地回归 | `386640eb`：配置迁移、每 rank 单一 local grad norm、模板/FSDP 开关及 monitor 训练回归；70 项测试、43 项子测试通过 |

本次输出复验确认 20 条进度日志、20 条 `training metrics:`，step 1–20 各一次；8 个 rank 的记录完整，
汇总标量与 JSONL 对账一致，摘要/远端丢弃均为零。TensorBoard 的 20 个 loss step、W&B offline 的
800 张 expert 表和 20 张 token 表已回读对账。FSDP 顺序检查开启；AdamW 无 NS 调用，MFU 无可信峰值，均明确不可用。

全开汇总每行 7,323–7,438 字符，主要是逐层 EP 标量；`datasets.log_level=debug` 另输出样本 DEBUG 日志。
本轮仍是短跑输出验证，没有关闭监控的配对性能基线，也没有执行大模型权重保存恢复。
代码快照、命令及原始日志位于服务器 `/home/chaoran/youchu_valid/hp-monitor-qwen36-20261118-udg5_3bm`，
其中 `manifest.json` 保存基准提交、补丁和快照哈希，`validation.json`、`sdk-validation.json` 保存对账结果。

此前设备报告的未通过项仍需处理：

- CP 尾部 padding 被误报为跨样本 target；不能直接据此判定监督泄漏。
- 完整 SDK 输出的 step P50 相对该次基线增加约 36%；尚未拆分各 sink 和 Table 的贡献。
- Qwen 同步 EP 入口的 capacity drop 分支未生效；monitor 记录实际计数，不代替 EP 算法验收。

完整 40 层、多机、其他 TP/PP 拓扑、GPU、长跑、在线 W&B 与断网恢复尚未完成设备验收。
小模型保存恢复不等于生产 EP/FSDP 模型及真实 dataloader 游标续训通过。
compiled 区域缺少观察点时记录不可用；外部 gradient owner 无适配时不发布只含 dense 部分的完整 local norm。
本地测试中使用的峰值仅验证 MFU 公式，不代表实测硬件 MFU。
