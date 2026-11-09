# Trainer monitor 单机 8 卡验收

## 环境与范围

2026-11-09，`monitor_dev` 实现提交 `743963ae`，并应用本报告随附的输出队列/sink 隔离修复。
在 `chaoran@8.92.7.129` 的独立目录验收，未替换已有代码副本或修改共享 conda 环境。

| 项目 | 实际值 |
| --- | --- |
| 设备 | 8 × Ascend 910B3 |
| 环境 | `trainer_tp`，其上建立 `venv --system-site-packages`，当前 checkout editable install |
| CANN | `/home/chaoran/cann/cann-9.1.0` |
| 软件 | Torch 2.9.0、torch_npu 2.9.0.post6、Transformers 5.13.0 |
| 可选输出 | WandB 0.31.0 offline、TensorBoard 2.21.0，安装在独立 venv |
| 模型 | `Qwen3.6-35B-A3B-20L`：15 GDN + 5 full-attention 层 |
| 原始 recipe | `hyper_parallel/models/qwen3_6_moe/recipes/train_online.yaml` |
| 并行/批量 | 8 ranks，EP2、CP2、TP1、PP1，FSDP8，global batch 8，micro batch 1 |
| 训练 | BF16、AdamW、full activation checkpoint，纯文本在线数据，最大长度 4096 |

产物根目录：`/home/chaoran/youchu_valid/trainer-monitor-dev-20261109`。
各配置的完整命令保存在 `<case>/command.txt`，日志在 `<case>/train.log`，断言结果在 `<case>/validation.json`。
运行前使用设备锁和 `npu-smi` 空闲检查；遇到其他测试占卡时未强行启动或终止其他进程。

## 配置矩阵

| case | 步数 | 配置与验收重点 | 结果 |
| --- | --- | --- | --- |
| `disabled` | 20 | `monitor.enabled=false`，无 monitor 文件，控制台恰好 20 条 step 日志 | 通过 |
| `all` | 20 | 所有 tensor recorder 每步采样，EP 分布、datasets 各项、timing、MFU；console/JSONL/TensorBoard/WandB offline | 功能和记录完整性通过；性能另列 |
| `sampled` | 6 | loss/EP/data 每 2 步；grad norm 每 3 步且只选 `model.layers.0`；关 timing、EP 分布、token 分布/Top-N/packing 输出 | 通过 |
| `all_jsonl` | 20 | 与 `all` 相同采集项，只输出 console/JSONL，不启用 SDK | 通过 |
| NPU 小模型 | sync/async 各 2 步 | 真正执行 Muon、DCP 保存、HF 导出、恢复及其计时 | 独立脚本和仓库 ST 入口均通过 |

`all` 主作业使用 AdamW，所以 Muon NS 通过小模型独立验证。
主作业关闭大模型 checkpoint 写入；保存分支通过真实小模型 DCP/HF 路径覆盖，避免生成大体积重复权重。

## 记录完整性与口径

全功能作业每个 rank 的 JSONL 均包含：

- 40 条 `local_micro_loss`、20 条 `local_step_loss`、20 条 `local_grad_norm`。
- 400 条 `expert_load` 和 400 条 `expert_load_local`，即 20 层 × 20 steps。
- 20 条 datasets、20 条 FLOPs coverage，所有 291 个标准计算块完整，包含 15 个 GDN core。
- 设备摘要丢弃数为 0；rank 0 远端队列丢弃数为 0。

只有全局 rank 0 输出 20 条 `train_step`，没有训练 tqdm。TensorBoard 重新读取后 loss 的 step 恰好为 1–20。
WandB 生成原生 `.wandb` 文件和 1,769 个 Table 文件，没有在线上传。
输出校验采用原始结构化值，不从格式化终端字符串重建 SDK 标量。

EP 另做独立断言：8 个 rank 的全局直方图一致；8 个本地直方图求和等于全局；
每层 assignment 总数等于该 step 的输入 token 数 × 实际 top-k 8；所有层 dropped assignment 为 0。

累计监督 token 为 49,641，累计 sample 为 160。监控开关前后 20 步 loss 与关闭监控的终端精度结果一致。
小模型恢复单独验证保存前后权重逐位一致，NS 记录的参数归属正确，保存完成事件保留原始 step。

## 本轮发现并修复的问题

1. 原 32 条远端队列容纳不了一个 step 的多层 EP 突发记录，会丢远端指标。
   调整为有界 256 条，新增 40 层本地/全局记录集中入队用例；实跑重测无丢弃。
2. 原 WandB 异常会跳过同一事件的 TensorBoard 写入，关闭异常也可能跳过另一个 sink。
   分别隔离写入/关闭异常，重复错误只首次告警；故障注入 UT 验证 TensorBoard 继续获得训练指标。
3. 服务器默认 WandB artifact staging 目录不可写。验收显式设置作业独立的
   `WANDB_DIR`、`WANDB_DATA_DIR`、`WANDB_CACHE_DIR`、`WANDB_CONFIG_DIR`，未改变共享 HOME 权限。

首轮问题产物保留在 `all-first-run`；该轮不计作输出验收通过。通过结论来自修复后的 `all` 重跑。

## 性能与显存

使用 step 6–20，排除初始化和最早几步；同时统计相邻训练日志的墙钟间隔。

| 模式 | step time P50 | P95 | 日志墙钟间隔 P50 | P50 相对关闭监控 |
| --- | --- | --- | --- | --- |
| 关闭 monitor | 3.407 s | 3.942 s | 3.419 s | 基准 |
| 全量每步采样，仅 console/JSONL | 3.462 s | 4.300 s | 3.487 s | +1.63% |
| 全量每步采样，加 WandB/TensorBoard | 4.950 s | 5.811 s | 4.977 s | +45.32% |

三组 `device_max_allocated` 均约 51.0602 GiB，`device_max_reserved` 均约 54.2344 GiB，
本次短跑的峰值精度下未看到额外设备显存增长；这不代表固定摘要缓冲数学上为零，也不代替长跑泄漏检查。
三组 20 步的 loss 均与关闭监控的打印值一致。

**功能通过不等于性能目标通过。** 每步全量 Table 的 SDK 输出仍有明显代价，未达到设计的“无超出波动的退化”目标。
仅 console/JSONL 的结果表明本次主要额外开销与可选 SDK 输出组合相关，不能直接全部归因于 tensor recorder；
该对照没有继续分离 WandB 与 TensorBoard，也没有通过 profiler 证明具体瓶颈。
默认 tensor recorder 关闭/低频采样；不能把本轮每步全量 Table 压测配置当成无代价的生产配置。

各模式仅一次顺序运行，不能据此宣称统计意义的零开销或已完成性能优化。
原始数值和对齐断言见服务器根目录 `performance-comparison.json`。

## 明确未覆盖与正常缺失

- MFU 公式覆盖完整，但服务器没有提供可信的 SKU/BF16 dense 峰值，因此百分比为 `unavailable_peak`；
  完整模型 TFLOP/s 正常输出。没有用猜测峰值制造 MFU 百分比。
- 当前数据入口未提供 padding mask，采集点还没有对应 packing positions，且 metadata 检查不覆盖 attention kernel；
  `padding_mask`、`packing_positions`、`attention_kernel_validation` 明确标记 unavailable。
  这些字段不可判为“检查通过”；已有 offsets/监督边界、Top-N 和长度统计仍执行。
- 本轮不覆盖完整 40 层训练、多机、其他 TP/PP 拓扑、GPU、长期显存增长、在线 WandB 上传/网络恢复。
- SDK 隔离环境与共享 `ms-service-profiler` 的 OpenTelemetry 版本约束存在冲突提示；本轮未启用 profiler。
  可选 SDK 未装入共享 `trainer_tp`，本报告不宣称 profiler 与这些版本组合已兼容。

## 本地验证与复跑入口

本地相关回归：534 项测试、746 个子用例通过。新增 40 层输出突发/单 sink 故障隔离测试通过。
Ruff E/F、Python 编译、差异空白、Markdown 和 AGENTS catalog 分别检查。

统一 `autogit check` 未取得全绿：删除文件处理报错，备用检查还包含与当前 Torch-only/UT 规则冲突的旧规则、
Pylint 第三方类型推断误报，以及文档参数/复杂度检查项；不把分项验证描述为完整 lint 门禁通过。
`autogit commit` 不支持当前 GitHub remote，本轮使用普通 Git 提交/推送，没有改动 autogit 工具。

新增可复跑 NPU ST，正式入口在同一服务器通过：`1 passed`，耗时 48.02 s：

```bash
python -m pytest -q tests/torch/trainer/test_monitor.py
```

launcher 不导入 Torch/HyperParallel；8 个 worker 执行 Muon 和真实 sync/async DCP/HF 保存恢复。
测试通过后会自动清理其临时 checkpoint。完整 Qwen 配置命令和验证脚本保存在上述服务器产物目录。
