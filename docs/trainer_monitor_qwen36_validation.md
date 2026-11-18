# Qwen3.6 MoE monitor 逐项验证

> 历史报告：以下结论只对应 `f49fa939`。其中 local grad norm 的模块筛选已在后续代码删除，不能照搬旧配置；当前入口见 [使用说明](trainer_monitor_usage.md)。

## 结论与环境

2026-11-16 至 2026-11-17，验证 `monitor_dev` 的实现提交 `f49fa939`。
**8 卡训练与多数监控功能通过，但整体验收未全部通过：完整 SDK 输出有明显耗时增长，packing 边界存在误报。**
本轮只增加测试和报告，未修改生产实现。此前的 [NPU 报告](trainer_monitor_npu_acceptance.md) 不替代本轮结果。

- 服务器：`chaoran@8.92.7.129`，8 × Ascend 910B3；正式验收全部使用 8 卡，此前停止的 4 卡结果不计入。
- 环境：`trainer_tp` 派生独立 venv、CANN 9.1.0；Torch 2.9.0、torch_npu 2.9.0.post6、Transformers 5.13.0。
  editable import 指向本次独立 `src`，未修改共享 conda 环境。
- 模型：Qwen3.6-35B-A3B 的既有 **20 层权重视图**，15 个 GDN + 5 个 full-attention，256 experts、top-k 8。
  使用 BF16、全量 activation checkpoint，EP2/CP2/FSDP8、expert EDP shard=4、TP1/PP1。
- 训练：原生 TextTrainer、在线 `decif-30k.jsonl`、序列上限 4096、global batch=8、micro batch=1，两个 micro/step。
- 输出：W&B 0.31.0、TensorBoard 2.21.0；W&B 只用 offline，没有在线上传。
- 小模型补验：真实 Qwen HF 类，4 层（3 GDN + 1 attention）、4 experts/top-k 2，8 个 NPU/HCCL rank。
  用于真实保存恢复与 Muon，参数为普通本地 Tensor，不替代完整 EP/FSDP 模型的持久化验收。

## 八卡作业矩阵

| case | 配置 | 结果 |
| --- | --- | --- |
| `disabled` | monitor 关闭，原生日志 | 20 步、退出 0，无 monitor JSONL |
| `disabled_precise` | monitor 关闭，仅测试入口将日志浮点格式改为 17 位有效数字 | 20 步、退出 0，作为精确数值基线 |
| `all_jsonl` | 所有 recorder 每步采样，console + JSONL | 20 步、退出 0，基础对账通过；packing 问题另列 |
| `all` | 加 TensorBoard + W&B offline，首次模板输出开启 | 20 步、退出 0，SDK 文件回读通过；性能和 packing 问题另列 |
| `sampled` | loss/EP 每 2 步、grad 每 3 步、data 每 5 步、外部日志每 4 步；关闭 timing/MFU/分布等字段 | 20 步、退出 0，触发 step 和字段缺省均吻合 |
| `small-final-checkpoint` | 同步及默认异步 DCP、HF 导出、AdamW、StepLR | 8 rank 通过；第 10 步保存恢复，后续 10 步精确一致 |
| `small-final-muon` | Qwen 的二维参数用真实 Muon；采集 NS norm、local norm，保存恢复 | 8 rank 各 20 步通过；每 rank 140 条 NS 记录 |

大模型三种 monitor 配置的已输出 step，`total_loss`、`foundation_loss`、`grad_norm`、`lr`
与 `disabled_precise` **逐值完全一致**；sampled 只对照有输出的 5 个 step。
普通 `disabled` 的日志为 6 位有效数字，不能用其显示舍入误差判断训练漂移。
大模型未导出最终权重；最终权重精确一致的证据来自本地 CPU 对照及小模型 checkpoint 续训。

## 分模块验收

| 模块 | 实际检查与结果 | 边界 |
| --- | --- | --- |
| 输出与 rank | 8 个 JSONL 文件、统一 run ID、事件 rank 正确；全局 train_step 仅 rank0；无 tqdm 输出 | 单机 8 卡，非多机 |
| local loss | 全量每 rank 40 条 micro loss、20 条 step loss；两个 micro 按 token 加权可重算；各 rank 加权可重算全局 loss | MTP、零 token 的特殊边界由 UT 补充 |
| local grad norm | 每 rank 20 次；`post_reduce_pre_clip`、有限非负；按 `model.layers.0` 采样 3 得到 6 次 | 独立 FP64 梯度重算在 CPU UT；NPU 未逐参数独立重算 |
| EP 负载 | 每 rank 400 条 local + 400 条 global；20 层、256 个 expert；所有 rank 的 global 一致，local 求和等于 global | 当前为 dropless；capacity 路径问题见下文 |
| EP 口径 | 每层 assignment 总数等于全局 input tokens × top-k 8；AC 重计算未重复计数；max/mean/max_vio/空 expert 比例独立重算一致 | 物理 dispatch slots 明确 unavailable |
| 数据计数 | 每组检查 160 条 datasets；input/supervised 长度直方图、均值、top-20 顺序及比例一致 | 无 padding mask，input 统计包含 CP 对齐槽位，padding_fraction=null |
| CP 数据归属 | 数据采集发生在 CP 切分前；8 rank 源数据总和除以 CP2 后等于全局 input/valid tokens | 不可直接把 8 rank 的 datasets 相加当全局样本量 |
| 数据累计与吞吐 | step_samples=8，20 步 consumed_samples=160；consumed_tokens=49,641；吞吐等于 step_tokens/step_time | 吞吐是有效训练 token 口径；input 吞吐另列 |
| packing | offsets、segment 长度等有记录；补充验收发现 146 条边界告警（含 CP 副本） | **未通过：padding 被误当作跨样本边界** |
| 首次模板 | `all` 真实生产模板入口只打印一次 | 其余 case 关闭；UT 覆盖配置兼容 |
| FLOPs/GDN | 每 rank 每步 coverage.complete=true；291 个标准计算块含 15 个 delta_rule 块，TFLOP/s 正常 | 这是公式估算，不是硬件 profiler 实测 FLOPs |
| MFU | 开关与外部 log_tflops 独立；硬件峰值未配置时为 unavailable_peak，没有伪造百分比 | 真实硬件 MFU 百分比未验；归一化公式与四种开关组合由 CPU 测试覆盖 |
| 分块耗时 | 每 rank 8 条启动 timing；组网、数据、tokenizer、权重加载等；step 的取数、前后向、clip、optimizer 等正常 | host_wall 计时，不等同纯设备 kernel 时间 |
| checkpoint | 同步/默认异步保存、等待、恢复、HF 权重保存耗时均有记录；异步完成仍标记原始 step=10 | 小模型实际存储；固定 batch，不包含真实 dataloader 游标恢复 |
| Muon NS | 每 rank 140 条、覆盖 20 step，参数 owner 有效、数值有限；local norm 与摘要落盘完整 | 只更新二维参数，非完整生产 Muon recipe；其他形状未更新 |
| TensorBoard | 原生 event 文件回读 20 个 loss step，与 JSONL 对照一致 | 未验证远程服务 |
| W&B | 原生离线 run；800 张 expert 表（20 step × 20 层 × local/global）及 20 张 token 表逐项对照 JSONL | 表格归属 rank0；未解码二进制历史、未在线 sync |
| 异步摘要/关闭 | 全量各 rank dropped_summaries=0、dropped_remote_records=0；关闭后记录完整 | 队列满、sink 故障隔离由 UT 注入验证，不等同长期压力测试 |

## 耗时与显存

以下为 rank0 的 step 6–20，排除前 5 步；单位秒。每组仅一次短跑，未进行统计显著性判断。

| 配置 | step P50 | step P95 | 相邻日志墙钟 P50 | 相对精确基线 P50 |
| --- | ---: | ---: | ---: | ---: |
| 关闭，普通日志 | 3.499 | 3.719 | 3.507 | -2.28% |
| 关闭，精确基线 | 3.580 | 3.863 | 3.589 | 基线 |
| 全采集 console + JSONL | 3.517 | 3.759 | 3.539 | -1.77% |
| 全采集 + TensorBoard + W&B offline | 4.879 | 5.191 | 4.901 | **+36.27%** |
| 分频采样，仅 JSONL | 3.514 | 3.665 | 不可比较 | -1.86% |

sampled 只在 step 8/12/16/20 留有预热后的耗时，样本口径与其他组不同，不能直接用于优劣排名。
JSONL 的差异在本轮基线波动范围内，不能据此声称加速或零开销。
完整 SDK 组存在明显开销，**不满足此前“不影响性能”的验收要求**；尚未拆分 TensorBoard/W&B/表格构造各自贡献。

rank0 最大 allocated：精确基线 51.0601578 GiB、开启后 51.0601883 GiB，相差 32 KiB；
reserved 均为 54.3515625 GiB。只能说明本轮峰值没有明显增长，不能证明长期无显存增长。

## 未通过项与定位

### 1. packing 将 CP 尾部 padding 误报为跨样本 target

两个全量 case 各发现 146 条 `cross_segment_target`（含 CP 副本，去重后 73 条）。
[`TextPackingCollator`](../hyper_parallel/data/batching/build_collate_fn.py) 为 CP 对齐追加独立 padding segment；
[`DatasetStatsRecorder._packing`](../hyper_parallel/trainer/runtime/monitor/datasets.py) 把所有中间边界都当作真实样本边界，
要求前一个位置的预移位 label 被忽略。单个样本末尾合法的 EOS target 因而被误报。

用真实 collator 独立复现：input=`[11,12,13]`、预移位 label=`[12,13,14]`（14 为该样本结束 target），
alignment=2 后 boundaries=`[0,3,4]`、label=`[12,13,14,-100]`，recorder 仍报 `cross_segment_target`。
不能把该告警直接解释成真实样本之间发生了 loss 泄漏。
下一步应明确真实 document 与 padding 的边界元数据，再校验有效 target；padding mask 和 position 元数据缺失也应一起核对。

### 2. 完整 SDK 输出开销明显

每步发送完整 expert 分布时，P50 增加约 36%，相邻日志墙钟也同步增加。
下一步分别对照 TensorBoard、W&B 标量、W&B Table，定位后台格式化、序列化和写盘对训练的影响。
本轮未修改实现、未降低全量采样频率以隐藏问题。

### 3. Qwen 同步 EP capacity 分支未生效

CPU 真实 Qwen EP adapter 设置 `HP_EP_CAPACITY_FACTOR=0.5`，仍全部接受 assignment，drop=0。
[`ep_routed_forward`](../hyper_parallel/distributed/expert_parallel/experts.py) 仅用 capacity 判断 fused permutation，
没有计算/传入 capacity keep mask；另一入口 `ep_routed_dispatch` 才执行 `_capacity_limited_slots`。
monitor 与实际 router 计数吻合，但该模型的真实 drop 分支不能判通过。本轮未修改 EP 算法。

### 4. 平台与旁路问题单列

- **macOS spawn 默认异步 DCP 失败**：真实 Qwen、单进程及两进程均在 `execute_async_persist → _save_impl → dist.get_rank()`
  报默认进程组未初始化；两个 rank 的 monitor 正确记录 `checkpoint_persist/step=10/outcome=failed`。
  显式 `use_gloo=True` 的两进程诊断对照通过，但不是默认分支修复。
  **Linux/NPU 本轮 8 卡默认异步保存及恢复通过**，两种平台结果不能混为一谈。
- **可选 codegen sidecar 失败**：缺少 Qwen GDN inline declaration；日志明确 `native training continues`。
  原生各作业均退出 0 并跑完 20 步，codegen 本身不判通过。

## 本地回归与证据

相关回归 **548 passed，358 subtests passed**；唯一 warning 为既有 CPU profiler warmup 提示。
最终 Qwen 文件单跑 **6 passed，8 subtests passed**；补齐 arg_mark 后有 5 条仓库未注册 pytest marker 的 warning。
Ruff、差异空白、文档目录检查通过。
新增对照位于 [test_monitor_training.py](../tests/ut/trainer/test_monitor_training.py)，覆盖精确更新一致性、
独立 FP64 norm/token 重算、分频/模块筛选、MFU 开关和真实 Muon。

```bash
python -m pytest -q tests/ut/trainer \
  tests/ut/components/modules/test_moe.py tests/ut/core/optimizer/test_muon.py \
  tests/ut/core/utils tests/ut/components/checkpoint tests/ut/core/distributed_checkpoint
```

服务器证据根目录：`/home/chaoran/youchu_valid/trainer-monitor-dev-20261117`。

| 位置 | 内容 |
| --- | --- |
| `eight-card/<case>/command.txt`、`train.log`、`exit_code.txt` | 大模型原始命令、日志、退出码 |
| `eight-card/<case>/metrics/<run_id>/rank_*.jsonl` | 全 rank 原始指标 |
| `eight-card/<case>/validation.json` | 基础训练与输出断言；不能替代下列 details 和性能结论 |
| `eight-card/details.json`、`comparison.json` | 数据/EP/表格补充对账、packing 失败、精确数值及耗时比较 |
| `eight-card/small-final-checkpoint/validation-rank-*.json` | 8 rank 同步/异步保存恢复断言 |
| `eight-card/small-final-muon/validation.json`、两个 `small-final-*.log/.exit` | Muon 对账、小模型原始日志及退出码 |
| `wandb/` | 原生 W&B offline run 和表格；TensorBoard 在对应 metrics run 目录内 |
| `run-eight*.sh`、`run-small-eight.sh`、`validate-*.py`、`compare-eight.py` | 本轮执行和断言脚本 |

本机额外证据：`/tmp/hp-monitor-qwen36-20261116/` 为 CPU 回归/SDK/EP，
`/tmp/hp-monitor-qwen36-20261117/` 为跨进程补验、padding 复现及 `remote-evidence/` 下载副本。
临时目录清理前需另行归档。作业均已结束，未终止其他用户的进程。

完整 40 层、其他 TP/PP 拓扑、多机、GPU、compiled、真实 dataloader 游标续训、长期性能/显存、
在线 W&B 同步及断网恢复未覆盖，不能因本轮八卡 20-step 完成而默认通过。
