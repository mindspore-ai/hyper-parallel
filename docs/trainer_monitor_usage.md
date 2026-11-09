# 通用 Trainer monitor 使用与实现边界

对应 [设计方案](design/trainer_monitoring_plan.md) 和 [指标诊断说明](trainer_monitor_diagnostics.md)。
配置已接入 `monitor_dev` 的通用 Base/Text/VLM Trainer；设备显存采集逻辑保持原样。
本文区分已接入的路径与仍需设备验收的范围，不把 CPU 测试当作多卡性能验收。

## 配置和输出

将以下段落放入原训练 YAML。省略的字段使用设计文档中的默认值。
日志输出间隔继续由外部 `training.logging_steps` 控制；下面的间隔只控制采集。

```yaml
monitor:
  enabled: true
  sinks: [console, jsonl]
  output_dir: ./outputs/metrics
  timing: true
  mfu: false
  local_loss:
    enabled: true
    sampler_steps: 100
  local_grad_norm:
    enabled: true
    sampler_steps: 100
    modules: []
  moe:
    enabled: true
    sampler_steps: 100
    record_distribution: false
  optimizer:
    ns_norm:
      enabled: false
      sampler_steps: 100
  datasets:
    sampler_steps: 100
    token_distribution: true
    top_n_tokens: 20
    packing_stats: true
    packing_boundary_check: true
    log_first_rendered_template: false
```

20-step 验收时将需要观察的 `sampler_steps` 设为 1。
不配置 `monitor.logging_steps`、顶层 `monitor.sampler_steps`、`ranks`、`include_children` 或 local loss module。
关闭 `monitor.enabled` 后仍保留原来的基础训练日志，不注册新的采集 hook。

- 全局 rank 0 输出一条规范 `train_step`；TqdmCallback、刷新条和拼接逻辑已删除。
- 每个 rank 的诊断记录写到 `<output_dir>/<run_id>/rank_<rank>.jsonl`。
- `schema_version/run_id/event/rank` 标识记录；训练事件保留原始 `step`。
- 微步 loss、逐层 EP 等是独立诊断事件，不是另一条重复的 `train_step`。
- 异步摘要可能晚于本步基础日志到达，仍使用采集时的 step，不回填成下一步。
- 全部 rank 的细节保留在各自 JSONL。在线远端只由 rank 0 写入其所持有的全局/本地结果；
  不隐式增加跨 rank 细节上传通信，也不把 rank 0 的 local 数据命名为全局数据。

## 采集位置

| 功能 | 接入位置 | 口径与限制 |
| --- | --- | --- |
| Local/micro loss | `BaseTrainer.postforward`，全局加权前 | 原始 micro 均值；step 用有效 token 加权；零分母不可用 |
| Local grad norm | `clip_grad_norm_`，Partial 规约后、裁剪前 | 复用 norm 输入视图；L2、本地分片、含选中子树 expert；没有新增规约 |
| 不启用裁剪时的 local norm | `BaseTrainer.prepare_optimizer_step` | 读取已有 main_grad/grad；未完成 Partial 的值不冒充 post-reduce |
| EP | 共享 `MoE.forward`，原生 EP 的 `ep_routed_forward` / `ep_routed_dispatch` | `ExpertLoadRecorder` 固定注册全部层；真实 forward 计数，backward 重计算不计数 |
| Muon NS | `Muon._compute_batched_ns_outputs_for_tensors` | NS 后、学习率/RMS 缩放前的矩阵 Frobenius norm；保留 parameter、transform、call 与 matrix 范围 |
| 数据 | 通用 get-batch 的 CPU 源 batch、CP 切片和 H2D 前 | 本 rank 的原始源数据分布；精确 step Top-N，先合并全部 micro 计数再取 Top-N |
| 分块耗时 | 显式生命周期和 `statistics_phase` | 主机墙钟，非 kernel 执行时间；不新增逐块设备同步 |
| FLOPs | 初始化时的标准块清单 | Linear/attention/FFN/router/shared expert/LM head 组合；逐步只更新输入几何 |

默认没有新增 forward/backward hook。`components.monitor` 是独立的注入接口，不导入 Trainer。
模型/优化器只借出当前有效 tensor；训练线程生成小摘要，异步 D2H 只持有摘要的独立存储。
D2H 完成 event 就绪后才读 CPU 缓冲。通信全部在训练线程上按固定层序执行。
传输队列上限 32、单摘要上限 65,536 元素；溢出通过 `monitor/dropped_summaries` 显式记录。
EP 合并所有层后一次提交，避免每层消耗一个传输槽位；关闭或未采样不创建计数 tensor。

## 数据与 EP 的解释

数据事件包含输入/监督长度直方图与分位数、token ID/count/fraction、segment 长度、填充率和边界错误。
监督计数遵循 causal shift、IGNORE_INDEX、loss_mask，通用文本入口也传递 EOD mask 与 shifted-label 策略。
没有 padding mask 时不猜测 pad/eos ID；填充率标记不可用。
packing 只检查已有 offsets、位置和监督 metadata，不新建 S×S mask，也不声称验证了 attention kernel。
缺少 metadata、位置尚未生成或不支持的策略会列在 `unavailable`。

EP load 是 top-k **assignment 次数**。每层先求其独立 token 域的总计数，再计算 `max/mean - 1`；
`moe/max_vio` 为当前 PP stage 所持有层的最大值，不跨层平均掩盖异常。
共享 dropless MoE 的 attempted=accepted，drop=0；原生 EP 从已生成的 dispatch indices 统计 accepted，
用路由前 assignment 数减去 accepted 得到 drop，覆盖容量限制的分离 dispatch 路径。
物理 padding 槽位缺少出口时为 unavailable。注册层没有实际 observation 时标记 unavailable，不输出假零负载。
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
Top-N 和 expert 分布转换为带 step/rank/module 的 Table；其他诊断事件保留完整结构化 payload。
TensorBoard 标量使用相同原始数值。rank-local 标量的远端 key 附带 rank/module，避免混为同一曲线。
WandB 离线目录可用 SDK 的 `wandb sync <offline-run-directory>` 上传；JSONL 本身不是 SDK 离线 run 文件。
已有外部 `wandb.entity` 继续生效；恢复时同一 run ID 使用 SDK resume。
远端队列有界，失败告警、保留本地 JSONL，不无限重试；退出等待有上限。
不在远端启用时导入 SDK。

## MFU 完整性

`training.log_tflops` 与 `monitor.mfu` 共用标准块估算，分别控制 TFLOP/s 与 MFU 输出。
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
缺少可信峰值时只输出完整模型 TFLOP/s，并给出 `unavailable_peak`。
当前约定为同构设备；异构作业不得复用一个峰值。
每步 MFU 使用该步耗时，窗口 MFU 使用本进程训练窗口 `sum(F)/sum(dt)`，不平均各步百分比。

## 验收状态

2026-11-09 本地合并回归通过 533 项测试、746 个子用例，覆盖 Trainer、clip-grad、Muon、MoE、
checkpoint、数据入口、配置解析与原生 EP；其中既有 EP 用例的 Gloo 仅使用本机回环地址。
新增 Python 文件通过 Ruff E/F 检查，所有改动 Python 文件可编译，Markdown 格式、相对链接和 AGENTS 目录检查通过。

CPU UT 已验证：配置解析、loss/norm 不修改训练结果、重计算去重、Top-N/packing、异步就绪与原始 step、
输出去重、WandB SDK 接口 mock、checkpoint 累计量/run ID 恢复和标准块公式。
原生 EP 的两条 dispatch 入口、采样间隔、assignment drop、未执行层的缺失标记已有 CPU 用例；
Muon 原生 reshape 与自定义 transform 路径的参数归属、缩放前 norm 已用真实计算路径核对。
`tests/ut/trainer/test_monitor_training.py` 通过真实 TextTrainer 循环训练 tiny MoE 20 steps（每步两个 micro）：
监控开关前后参数 bitwise 一致，20 条或每 5 步共 4 条基础训练日志，40 条 micro loss、20 条 local loss/norm/EP/data。
同一测试文件还通过真实 HF hybrid MoE 的 20-step 训练：3 层 GDN、1 层带 gate 的全注意力，均包含 routed/shared experts。
每步 FLOPs coverage 完整、TFLOP/s/MFU 有输出，开关监控前后参数 bitwise 一致。
MFU 输出链路使用测试专用峰值验证归一化，不将 CPU 测试数值称为设备实际 MFU。

真实 WandB 0.31.0 offline SDK 已生成 `.wandb` run 与 token Table 文件，
TensorBoard 2.21.0 event 文件已重新读取并核对原始 step 和 scalar；本轮未上传在线 run。

尚未完成的设备验收：实际 Qwen3.6 多卡/多机、EP/CP/TP 拓扑、GPU/NPU event 与 pin-memory 行为、
采样开关吞吐/峰值显存对照和长跑无增长、在线 WandB 网络故障恢复。
compiled 区域不插入 Python 采集 graph break；缺失的 tensor 诊断明确 unavailable。
外部稀疏梯度 owner 没有适配时不输出仅含 dense 部分的“完整 local norm”。
这些边界不能因 CPU 回归通过而标记验收完成。
